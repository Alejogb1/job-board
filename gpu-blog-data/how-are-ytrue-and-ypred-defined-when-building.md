---
title: "How are `y_true` and `y_pred` defined when building a custom Keras metric?"
date: "2025-01-30"
id: "how-are-ytrue-and-ypred-defined-when-building"
---
In crafting custom Keras metrics, the precise definition of `y_true` and `y_pred` hinges on the nature of your prediction task and the chosen output activation of your model.  My experience building anomaly detection systems frequently necessitates careful consideration of this, particularly when dealing with imbalanced datasets and varying loss functions.  Crucially, understanding the underlying data structures—tensors—and their dimensions is paramount.  `y_true` always represents the ground truth labels, while `y_pred` reflects the model's predictions.  However, their exact shapes and interpretations depend contextually.

**1.  Clear Explanation:**

`y_true` and `y_pred` are NumPy arrays or TensorFlow tensors passed to your custom metric function. Their shapes are determined by the model's output and the problem's dimensionality.  For a binary classification problem, `y_true` would typically be a 1D array or tensor of 0s and 1s (or -1 and 1 depending on the encoding) representing the true class labels for each sample.  `y_pred` would hold the model's probability predictions for the positive class, also a 1D array or tensor of floating-point values between 0 and 1 (or logits if using a different activation such as sigmoid).


For multi-class classification (with *C* classes), `y_true` can be a 1D array of integers representing the true class index (0 to *C*-1) for each sample, or a 2D one-hot encoded array of shape (number of samples, *C*), where a 1 indicates the true class.  `y_pred` then follows suit, with either class probabilities (a 2D array of shape (number of samples, *C*)) or predicted class indices (a 1D array). For regression problems, `y_true` and `y_pred` would both be 1D arrays or tensors containing the continuous target values and the model's predictions, respectively.  In cases involving multi-dimensional outputs, such as in image segmentation where we predict pixel-wise classifications,  `y_true` and `y_pred` would be tensors with additional spatial dimensions.


It's important to note that the shape of `y_pred` depends entirely on your model's final layer.  If you use a sigmoid activation for binary classification,  `y_pred` will contain probabilities.  If you use softmax for multi-class classification,  `y_pred` will be a probability distribution over classes.  Failure to consider these relationships often leads to shape mismatches and errors during metric computation.  Furthermore, batch processing means that these arrays will generally have a leading dimension representing the batch size.  Careful attention must be paid to handle potential broadcasting issues.


**2. Code Examples with Commentary:**

**Example 1: Binary Classification Accuracy**

```python
import tensorflow as tf
import numpy as np

def binary_accuracy(y_true, y_pred):
    """Calculates binary accuracy.

    Args:
        y_true: Ground truth labels (0 or 1). Shape (batch_size,).
        y_pred: Predicted probabilities. Shape (batch_size,).

    Returns:
        Binary accuracy.
    """
    y_pred = tf.round(y_pred)  # Convert probabilities to binary predictions
    correct_predictions = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

# Example usage:
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.2, 0.8, 0.9, 0.1, 0.7])
accuracy = binary_accuracy(y_true, y_pred).numpy()
print(f"Binary accuracy: {accuracy}")
```

This example shows a straightforward binary accuracy calculation.  Note the explicit rounding of predictions.  This highlights the importance of understanding the nature of `y_pred`; it's not necessarily the class label but a probability that must be thresholded.


**Example 2: Multi-class Categorical Crossentropy**

```python
import tensorflow as tf
import numpy as np

def categorical_crossentropy_metric(y_true, y_pred):
    """Calculates categorical crossentropy.
    Note: y_true must be one-hot encoded.

    Args:
        y_true: Ground truth one-hot encoded labels. Shape (batch_size, num_classes).
        y_pred: Predicted class probabilities. Shape (batch_size, num_classes).

    Returns:
        Categorical crossentropy.
    """
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss)


# Example Usage
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
loss = categorical_crossentropy_metric(y_true, y_pred).numpy()
print(f"Categorical Crossentropy: {loss}")
```

This demonstrates a custom categorical crossentropy metric.  It leverages TensorFlow's built-in function for efficiency, but crucially, underscores the expectation that `y_true` is one-hot encoded.  Ignoring this will produce incorrect results.


**Example 3: Mean Squared Error for Regression**

```python
import tensorflow as tf
import numpy as np

def mse_metric(y_true, y_pred):
  """Calculates Mean Squared Error.

  Args:
    y_true: Ground truth values. Shape (batch_size,).
    y_pred: Predicted values. Shape (batch_size,).

  Returns:
    Mean Squared Error.
  """
  mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
  return mse

# Example usage:
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.2, 1.8, 3.5, 3.8])
mse = mse_metric(y_true, y_pred).numpy()
print(f"Mean Squared Error: {mse}")
```

This example highlights the simplicity of defining a regression metric.  In this case, the shapes of `y_true` and `y_pred` are straightforward, but the accuracy of the result depends entirely on whether the predictions are reasonable given the dataset's properties.

**3. Resource Recommendations:**

TensorFlow documentation on custom metrics.  The official Keras documentation on building and using custom metrics.  A comprehensive textbook on machine learning with a focus on practical implementation using TensorFlow/Keras.  A well-regarded online course covering advanced topics in deep learning and model evaluation.  Lastly, numerous research papers on specific metric design choices related to different applications.  Careful review of these resources will provide a solid theoretical and practical foundation for developing robust custom metrics.
