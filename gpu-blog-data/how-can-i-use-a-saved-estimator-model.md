---
title: "How can I use a saved Estimator model with Dataset API for prediction?"
date: "2025-01-30"
id: "how-can-i-use-a-saved-estimator-model"
---
Training a TensorFlow Estimator model and subsequently using it for predictions with the Dataset API requires careful orchestration of input functions and model restoration. Specifically, you must leverage the `tf.compat.v1.estimator.Estimator` class's `predict` method, coupled with a suitable input function that generates data in a format compatible with the trained model's input signatures. My experience working on large-scale recommendation systems has shown that this often presents challenges, especially when migrating from older graph-based training methodologies.

The core issue lies in ensuring that the input data pipeline, constructed using the Dataset API, precisely matches the expectation of the saved Estimator model. This includes adhering to correct tensor shapes, data types, and feature names. Any discrepancy can lead to runtime errors or, more insidiously, incorrect predictions. Successfully addressing this demands a deep understanding of both the Dataset API's data transformation functionalities and the Estimator's model input specification.

The `predict` method expects an input function. This function is the critical bridge between the saved model and the data stream created using the Dataset API. It must return a `tf.data.Dataset` object which generates batches of input data appropriate for the model. This is where the initial configuration of the model during training becomes very significant, as the input function must mirror that configuration when restoring and making predictions.

Let’s examine several code examples to illustrate this process.

**Example 1: Basic Numeric Feature Input**

Assume we have trained a simple Estimator model that expects a single numeric feature named 'feature_a'. The trained model has been saved to a directory. Below is a prediction function making use of a dataset:

```python
import tensorflow as tf
import numpy as np
import os

def prediction_input_fn(data, batch_size):
  """Input function for prediction, using tf.data.Dataset."""
  dataset = tf.data.Dataset.from_tensor_slices(data)
  dataset = dataset.batch(batch_size)
  return dataset

def predict_from_estimator(model_dir, data, batch_size):
  """Loads and predicts from the saved estimator model."""

  estimator = tf.compat.v1.estimator.Estimator(model_fn=None, model_dir=model_dir)

  input_function = lambda: prediction_input_fn(data, batch_size)

  predictions = estimator.predict(input_fn=input_function)

  return list(predictions)


if __name__ == '__main__':
    # create a dummy model and save it
    feature_column = [tf.feature_column.numeric_column('feature_a')]
    estimator = tf.compat.v1.estimator.LinearRegressor(feature_columns=feature_column, model_dir='dummy_model_dir')
    train_data = {'feature_a': np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), 'label': np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])}
    input_fn_train = tf.compat.v1.estimator.inputs.numpy_input_fn(x=train_data['feature_a'], y=train_data['label'], batch_size=2, num_epochs=10, shuffle=True)
    estimator.train(input_fn=input_fn_train)

    # Prepare data for prediction
    prediction_data = np.array([[6.0], [7.0], [8.0]])
    batch_size = 2

    # Call the predict function
    predictions = predict_from_estimator('dummy_model_dir', prediction_data, batch_size)

    # Print the predictions
    for idx, prediction in enumerate(predictions):
      print(f"Prediction {idx+1}: {prediction}")

    # cleanup
    os.system('rm -rf dummy_model_dir')

```
This example demonstrates the simplest case. `prediction_input_fn` takes numeric data, transforms it into a `tf.data.Dataset` and batches it. The crucial point here is that the shape of the input data passed to the prediction function (e.g., `prediction_data`) must match the shape of the input feature used during training. Also, `predict_from_estimator` uses a lambda function to wrap the `prediction_input_fn`.

**Example 2: Handling Multiple Features (Dictionary Input)**

Many real-world models require multiple features, often represented as a dictionary. The dataset must produce data in a dictionary format consistent with the feature columns defined during training. Let’s see how it changes the function:

```python
import tensorflow as tf
import numpy as np
import os

def prediction_input_fn(data, batch_size):
    """Input function for prediction with multiple features using tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    return dataset

def predict_from_estimator(model_dir, data, batch_size):
  """Loads and predicts from the saved estimator model."""

  estimator = tf.compat.v1.estimator.Estimator(model_fn=None, model_dir=model_dir)

  input_function = lambda: prediction_input_fn(data, batch_size)

  predictions = estimator.predict(input_fn=input_function)

  return list(predictions)


if __name__ == '__main__':
    # Create a dummy model and save it
    feature_columns = [tf.feature_column.numeric_column('feature_a'),
                       tf.feature_column.numeric_column('feature_b')]
    estimator = tf.compat.v1.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='dummy_model_dir')

    train_data = {'feature_a': np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
                  'feature_b': np.array([[6.0], [7.0], [8.0], [9.0], [10.0]]),
                  'label': np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])}

    input_fn_train = tf.compat.v1.estimator.inputs.numpy_input_fn(x=train_data, y=train_data['label'], batch_size=2, num_epochs=10, shuffle=True)
    estimator.train(input_fn=input_fn_train)

    # Prepare data for prediction
    prediction_data = {
      'feature_a': np.array([[6.0], [7.0], [8.0]]),
      'feature_b': np.array([[11.0], [12.0], [13.0]])
    }
    batch_size = 2

    # Call the predict function
    predictions = predict_from_estimator('dummy_model_dir', prediction_data, batch_size)

    # Print the predictions
    for idx, prediction in enumerate(predictions):
        print(f"Prediction {idx+1}: {prediction}")

    # cleanup
    os.system('rm -rf dummy_model_dir')
```

In this example, the input data during prediction now consists of a dictionary, mirroring the structure used during training. The `prediction_input_fn` remains largely the same, as the `tf.data.Dataset.from_tensor_slices()` is flexible. The vital modification is the form of the data passed to the function, which must replicate the `train_data` dictionary structure, excluding the label.

**Example 3: Sparse Features**

Handling sparse data, such as categorical features with a large cardinality, often requires special feature columns. If an Estimator is trained with embedding columns or indicator columns, the input function must represent the input data in accordance with the expected sparse structure. Here's an adaptation to example 2, assuming we now have a sparse category:

```python
import tensorflow as tf
import numpy as np
import os


def prediction_input_fn(data, batch_size):
    """Input function for prediction with sparse categorical features."""
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    return dataset

def predict_from_estimator(model_dir, data, batch_size):
  """Loads and predicts from the saved estimator model."""

  estimator = tf.compat.v1.estimator.Estimator(model_fn=None, model_dir=model_dir)

  input_function = lambda: prediction_input_fn(data, batch_size)

  predictions = estimator.predict(input_fn=input_function)

  return list(predictions)

if __name__ == '__main__':
  # Create a dummy model and save it
    categorical_feature = tf.feature_column.categorical_column_with_vocabulary_list(
        'category', vocabulary_list=['A', 'B', 'C', 'D']
    )
    indicator_column = tf.feature_column.indicator_column(categorical_feature)
    feature_columns = [tf.feature_column.numeric_column('feature_a'), indicator_column]
    estimator = tf.compat.v1.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='dummy_model_dir')
    train_data = {
        'feature_a': np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        'category': np.array([['A'], ['B'], ['C'], ['A'], ['B']]),
        'label': np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
    }

    input_fn_train = tf.compat.v1.estimator.inputs.numpy_input_fn(x=train_data, y=train_data['label'], batch_size=2, num_epochs=10, shuffle=True)
    estimator.train(input_fn=input_fn_train)

    # Prepare data for prediction
    prediction_data = {
        'feature_a': np.array([[6.0], [7.0], [8.0]]),
        'category': np.array([['C'], ['D'], ['A']])
    }
    batch_size = 2

    # Call the predict function
    predictions = predict_from_estimator('dummy_model_dir', prediction_data, batch_size)

    # Print the predictions
    for idx, prediction in enumerate(predictions):
        print(f"Prediction {idx+1}: {prediction}")
    # cleanup
    os.system('rm -rf dummy_model_dir')

```

The training data includes a categorical feature. During prediction, we mirror this data structure with the `prediction_data`. The Estimator manages the transformation from categorical data to numeric internally based on the defined `indicator_column` during training.

The crucial takeaway across these examples is the importance of feature parity. The structure of the input data used for prediction must be identical to the structure used to train the model. Any discrepancy in feature names, shapes, or types will result in either an error or incorrect model predictions.

For further study and deeper exploration, I would suggest consulting the official TensorFlow documentation. Specifically the sections concerning the Dataset API and the Estimator API. Additionally, reading through the source code for the `tf.compat.v1.estimator.inputs` module can clarify how to correctly format data for your input function. Also, study good practices for modular code design. The key concept lies in understanding the contract between the input function and the model during training and mirroring that during prediction.
