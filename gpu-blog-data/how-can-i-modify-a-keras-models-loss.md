---
title: "How can I modify a Keras model's loss function to prioritize errors in the 99th percentile?"
date: "2025-01-30"
id: "how-can-i-modify-a-keras-models-loss"
---
Deep learning models, particularly those trained with gradient descent, typically optimize for average performance across a dataset. This can obscure critical failures occurring in the tails of the error distribution, such as the 99th percentile. My experience developing anomaly detection systems highlighted that these extreme errors, though numerically infrequent, often represent the most impactful problems. Modifying the loss function to explicitly prioritize these errors is crucial for robust performance in such cases. I will now detail a method for implementing this prioritization using Keras and custom loss functions.

The standard approach for loss function modification involves creating a custom function that leverages Keras’s backend API. This allows manipulation of the predicted and true values directly, performing computations to create a loss that reflects our desired error weighting. We aren't replacing an existing loss; we're crafting a new one designed for focused learning on high-percentile errors. The fundamental idea is to use a quantile or percentile as a threshold. We penalize errors above this threshold more heavily, thereby directing the gradient descent algorithm to focus on these critical errors. This is not as straightforward as simply using a standard quantile loss; we want to maintain some degree of optimization for errors across the entire distribution, albeit with differential weighting.

Let’s begin with a clear explanation of the logic. We need to calculate the model's errors, determine a threshold representing the 99th percentile of these errors, and then apply a custom penalty structure based on whether an error exceeds this threshold. The challenge is that we're operating within the Keras training loop, where we don't have access to the entire training dataset at once. Therefore, we'll have to compute the percentile on a per-batch basis. This is an approximation of the overall percentile but is generally effective for learning. Further, we need a differentiable function for the threshold determination and error-weighting, so we’ll be using a smoothed approximation for the absolute value which is used in error calculations as well as for the Heaviside step function.

Let me now provide code examples with accompanying explanations.

**Example 1: Basic Implementation with a Smoothed Heaviside Step Function**

```python
import tensorflow as tf
import keras.backend as K
from keras.losses import MeanSquaredError

def smoothed_heaviside(x, k=100.0):
    """Smoothed approximation of the Heaviside step function."""
    return 0.5 * (1 + tf.tanh(k * x))

def percentile_loss_v1(y_true, y_pred, percentile=0.99, smoothing_factor=100.0, error_scale=10.0):
    """Loss function penalizing errors above a batch-wise 99th percentile."""
    errors = K.abs(y_pred - y_true)
    threshold = tf.math.reduce_max(errors) * percentile  # Approximate 99th percentile of batch errors
    indicator = smoothed_heaviside(errors - threshold, k = smoothing_factor)
    weighted_errors = errors * (1 + error_scale*indicator)
    return K.mean(weighted_errors)

# Sample usage
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=percentile_loss_v1)

#Generate some dummy data
import numpy as np
X_train = np.linspace(0,10,1000).reshape(-1,1)
y_train = X_train*2 + np.random.normal(0,2,1000).reshape(-1,1)


model.fit(X_train,y_train, epochs=10, batch_size=32)
```

In this first implementation, the `percentile_loss_v1` function calculates the absolute errors between the true values (`y_true`) and the predicted values (`y_pred`). It approximates the 99th percentile by calculating a fixed percentage of the maximum error within the batch. We use the `smoothed_heaviside` function which approximates a step function. The error is then weighted such that any error above the approximate threshold is penalized more strongly, as weighted_errors = errors *(1 + error_scale*indicator), where the indicator function approaches 1 for values above the threshold and 0 for values below. I've added the error_scale to make the penalization stronger, a hyperparameter you can tune. This loss is then averaged over the batch. This approach is a simplification. Calculating the maximum error and assuming the percentile is a fixed fraction of it is a quick approach, but not accurate. The smoothing factor for the Heaviside function must also be tuned based on model performance.

**Example 2: Using a Batch-Wise Error Sort for Percentile Calculation**

```python
def percentile_loss_v2(y_true, y_pred, percentile=0.99, smoothing_factor=100.0, error_scale=10.0):
  """Loss function penalizing errors above the 99th percentile calculated via sorting."""
  errors = K.abs(y_pred - y_true)
  batch_size = K.shape(errors)[0]
  sorted_errors = tf.sort(errors)
  percentile_index = tf.cast(tf.math.floor(tf.cast(batch_size,tf.float32) * percentile),tf.int32)
  threshold = tf.gather(sorted_errors, percentile_index)
  indicator = smoothed_heaviside(errors - threshold, k=smoothing_factor)
  weighted_errors = errors * (1 + error_scale*indicator)
  return K.mean(weighted_errors)

# Sample usage (same model definition and data as before)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=percentile_loss_v2)

#Generate some dummy data
import numpy as np
X_train = np.linspace(0,10,1000).reshape(-1,1)
y_train = X_train*2 + np.random.normal(0,2,1000).reshape(-1,1)


model.fit(X_train,y_train, epochs=10, batch_size=32)
```

This second example, `percentile_loss_v2`, introduces a more robust approximation of the percentile. Instead of relying on the maximum error, it sorts the errors within the batch and selects the error at the calculated percentile index. This provides a more accurate representation of the actual percentile within each batch. The logic of the `smoothed_heaviside` function and error weighting remains the same. We explicitly get the batch size using `K.shape` and use a cast to ensure that the indexing is done correctly. This improves over example 1, however the `tf.sort` function can introduce overhead.

**Example 3: Hybrid Approach**

```python
def percentile_loss_v3(y_true, y_pred, percentile=0.99, smoothing_factor=100.0, error_scale=10.0, beta = 0.2):
  """Loss function using a weighted sum of MSE and percentile loss for smoothness."""
  mse_loss = MeanSquaredError()(y_true, y_pred)
  errors = K.abs(y_pred - y_true)
  batch_size = K.shape(errors)[0]
  sorted_errors = tf.sort(errors)
  percentile_index = tf.cast(tf.math.floor(tf.cast(batch_size,tf.float32) * percentile),tf.int32)
  threshold = tf.gather(sorted_errors, percentile_index)
  indicator = smoothed_heaviside(errors - threshold, k=smoothing_factor)
  weighted_errors = errors * (1 + error_scale*indicator)
  percentile_loss = K.mean(weighted_errors)
  return (beta * mse_loss) + (1 - beta) * percentile_loss

# Sample usage (same model definition and data as before)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=percentile_loss_v3)

#Generate some dummy data
import numpy as np
X_train = np.linspace(0,10,1000).reshape(-1,1)
y_train = X_train*2 + np.random.normal(0,2,1000).reshape(-1,1)

model.fit(X_train,y_train, epochs=10, batch_size=32)
```

The third implementation, `percentile_loss_v3`, introduces a hybrid approach. This function computes the Mean Squared Error (MSE) alongside the percentile-focused loss. Then, a weighted sum, controlled by the `beta` parameter, is used to combine both loss components. This blend helps prevent the model from overly focusing on the extreme tail errors at the expense of general performance. The use of `MeanSquaredError` ensures that the model continues to learn from errors below the percentile as well, providing a more balanced learning process. By blending, it avoids the situation where the model only tries to minimize errors above the percentile and ignores others. A good value for beta would need to be tuned.

These examples showcase different ways to modify the loss function. The optimal choice depends on the specific problem, data distribution, and desired trade-off between overall performance and addressing critical extreme errors. My experience has taught me that there is no one-size-fits-all approach; extensive experimentation and careful monitoring of the loss behavior during training is often necessary to obtain the best result.

For further exploration, I recommend delving into the following resources. Firstly, consult documentation and tutorials related to TensorFlow and Keras backend operations. A thorough understanding of these APIs is crucial when implementing custom loss functions. Secondly, explore research papers concerning robust loss functions. Many variants exist that may be more appropriate for specific problems. Thirdly, study work concerning anomaly detection, as there can be more direct implementations that can be used. Finally, familiarize yourself with different approaches to weighted sampling and data augmentation, which can further support the improvement of model performance on the tails of the distribution. This is not a quick implementation and often requires multiple tuning iterations to get to the required results.
