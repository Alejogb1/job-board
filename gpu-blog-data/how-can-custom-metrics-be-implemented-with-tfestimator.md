---
title: "How can custom metrics be implemented with tf.estimator?"
date: "2025-01-30"
id: "how-can-custom-metrics-be-implemented-with-tfestimator"
---
My experience building custom machine learning models within TensorFlow, particularly those using `tf.estimator`, has highlighted a frequent need to move beyond the standard metrics provided. While the built-in options like accuracy, precision, and recall are crucial, project-specific goals often necessitate nuanced performance evaluation. This isn’t simply a matter of visualizing loss; it’s about quantifying the precise characteristics of model behavior that directly align with a given problem. Implementing such metrics requires a specific approach to leverage the `tf.estimator` framework effectively.

Fundamentally, custom metrics in `tf.estimator` are created as functions that accept two primary arguments: `labels` and `predictions`. These arguments represent the true target values and the model's predicted outputs, respectively. These inputs should align with the data format being used in the model’s `input_fn`. Within this function, you define the logic for the custom metric calculation using TensorFlow operations. Crucially, the function must return a dictionary. This dictionary contains a `value` entry holding the computed metric's scalar value as a TensorFlow tensor, and a corresponding `update_op` which computes the value across the dataset. The presence of both allows for both metric calculation and accumulation over batches. The framework handles the aggregation and logging of metric values during training and evaluation, so the user only needs to provide metric computations and the logic for batch updates.

The key to this process resides within the `model_fn` argument, which is responsible for defining the computational graph of the model and the logic for the model. Within this function you will use the function we just discussed for custom metrics. These custom metric functions will be passed into `tf.estimator.Estimator` during its instantiation within a dictionary passed as the argument `eval_metric_ops`. This argument is only used during evaluation phases.

Let's demonstrate with a scenario where a model is attempting to classify images, and we need a metric that measures the average confidence of the model's top prediction. This isn’t a standard metric, as it doesn't measure correctness, but it does provides insight into how confident the model is in its predictions.

```python
import tensorflow as tf

def average_confidence(labels, predictions):
  """Calculates the average confidence of the model's top prediction.

  Args:
      labels: Ground truth labels (not used in this example but always passed in).
      predictions: Dictionary containing model predictions, expects 'probabilities' key.

  Returns:
      A dictionary containing the metric value and update operation.
  """
  probabilities = predictions['probabilities']
  max_prob = tf.reduce_max(probabilities, axis=1)
  mean_confidence, update_op = tf.metrics.mean(max_prob)
  return {'average_confidence': (mean_confidence, update_op)}


def my_model_fn(features, labels, mode, params):
  """Example model function.

  Args:
    features: Input features from input_fn.
    labels: Ground truth labels.
    mode: tf.estimator.ModeKeys mode.
    params: Dictionary of hyperparameters.

  Returns:
    tf.estimator.EstimatorSpec for training, evaluation or prediction.
  """

  # Model building logic (simplified)
  input_layer = tf.reshape(features, [-1, 784])
  dense1 = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu)
  logits = tf.layers.dense(inputs=dense1, units=10)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes']),
          'average_confidence': average_confidence(labels, predictions)
      }
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

In the above example the `average_confidence` function encapsulates the calculation and returns the mean and update operation, utilizing `tf.metrics.mean` which automatically handles the necessary accumulators. This function is then passed within the dictionary `eval_metric_ops` in `my_model_fn`, where it is defined to be part of the evaluation phase only.

Now, let’s consider a scenario involving a regression model, and we wish to calculate the mean absolute percentage error (MAPE), a crucial measure for comparing the accuracy of models predicting numerical values with widely varying magnitudes.

```python
def mean_absolute_percentage_error(labels, predictions):
    """Calculates Mean Absolute Percentage Error (MAPE).

    Args:
        labels: Ground truth labels (scalar values).
        predictions: Dictionary containing model predictions, expects 'predictions' key.

    Returns:
        A dictionary containing the metric value and update operation.
    """
    predicted_values = predictions['predictions']
    labels = tf.cast(labels, tf.float32)
    epsilon = 1e-7 # Avoid division by zero
    abs_percentage_error = tf.abs(tf.divide(tf.subtract(labels, predicted_values), tf.maximum(labels, epsilon)))
    mape, update_op = tf.metrics.mean(abs_percentage_error)
    mape = tf.multiply(mape, 100.0)
    return {'mape': (mape, update_op)}

def my_reg_model_fn(features, labels, mode, params):
  """Example regression model function.

  Args:
    features: Input features from input_fn.
    labels: Ground truth labels.
    mode: tf.estimator.ModeKeys mode.
    params: Dictionary of hyperparameters.

  Returns:
    tf.estimator.EstimatorSpec for training, evaluation or prediction.
  """

  # Model building logic (simplified)
  dense1 = tf.layers.dense(inputs=features, units=128, activation=tf.nn.relu)
  predictions = tf.layers.dense(inputs=dense1, units=1)
  
  predictions_dict = {'predictions': predictions}

  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)

  loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=predictions),
          'mape': mean_absolute_percentage_error(labels, predictions_dict)
      }
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

```

Here, `mean_absolute_percentage_error` directly computes MAPE by handling potential division by zero and then multiplying the result by 100 to express the error in percentages. The regression model and its associated metrics, including MAPE, are handled in the same fashion as the previous example. This example also illustrates how numerical labels must be cast into TensorFlow floating point tensors.

Finally, consider a slightly more involved metric – F-beta score, a generalized version of the F1-score that allows emphasizing either precision or recall. This is often used when one of these performance measures is more important than the other.

```python
def f_beta_score(labels, predictions, beta=1.0):
  """Calculates F-beta score.

  Args:
      labels: Ground truth labels (binary).
      predictions: Dictionary containing model predictions, expects 'classes' key.
      beta: A float value that controls precision and recall weighting.

  Returns:
      A dictionary containing the metric value and update operation.
  """
  predicted_classes = predictions['classes']
  labels = tf.cast(labels, tf.int64)
  precision, precision_op = tf.metrics.precision(labels, predicted_classes)
  recall, recall_op = tf.metrics.recall(labels, predicted_classes)

  numerator = (1 + beta**2) * precision * recall
  denominator = ((beta**2 * precision) + recall) + 1e-7
  f_beta = tf.divide(numerator, denominator)

  return {'f_beta_score': (f_beta, tf.group(precision_op, recall_op))}

def my_binary_model_fn(features, labels, mode, params):
  """Example binary classification model function.

  Args:
    features: Input features from input_fn.
    labels: Ground truth labels.
    mode: tf.estimator.ModeKeys mode.
    params: Dictionary of hyperparameters.

  Returns:
    tf.estimator.EstimatorSpec for training, evaluation or prediction.
  """

  # Model building logic (simplified)
  dense1 = tf.layers.dense(inputs=features, units=128, activation=tf.nn.relu)
  logits = tf.layers.dense(inputs=dense1, units=2)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          'f_beta_score': f_beta_score(labels, predictions, beta=0.5),
           'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
      }
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

In the `f_beta_score` function, we combine precision and recall, adjusting the `beta` parameter to control the trade-off between them. The `update_op` is a grouped operation that ensures the state of both metrics are updated. This showcases how metrics can be defined by combining multiple TensorFlow operations and leveraging existing `tf.metrics` functionality.

In summary, implementing custom metrics with `tf.estimator` is about defining your metric as a function that accepts labels and predictions, performing the computation using TensorFlow operations, returning both the current value and the update operation, and properly passing into the `eval_metric_ops` dictionary in the model function. The examples provided demonstrate different use cases involving various metric calculation complexities, providing a robust and generalized approach for creating customized measures of model performance.

For further learning, I highly recommend reviewing the official TensorFlow documentation on `tf.estimator` and `tf.metrics`, paying close attention to how `tf.metrics.mean` and similar functions work. Furthermore, exploring use cases on platforms such as GitHub and StackOverflow can give insight to more involved and creative metrics calculations. Additionally, it’s beneficial to investigate code examples related to model training, especially those dealing with custom input functions, as the data format is crucial to correctly implement metrics.
