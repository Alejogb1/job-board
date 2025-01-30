---
title: "How can I implement custom loss functions with weight arrays per batch in TensorFlow/Keras?"
date: "2025-01-30"
id: "how-can-i-implement-custom-loss-functions-with"
---
Implementing custom loss functions in TensorFlow/Keras that incorporate per-batch weight arrays requires a nuanced understanding of Keras's backend operations and the handling of tensor manipulations within the gradient tape context.  My experience building robust, production-ready deep learning models has highlighted the importance of careful tensor shaping and broadcasting to avoid common pitfalls.  The key is to leverage TensorFlow's broadcasting capabilities efficiently to seamlessly integrate these weights into the loss calculation.

**1. Clear Explanation**

The core challenge lies in correctly aligning the shape of your weight array with the shape of your prediction and target tensors.  If your weight array represents a unique weight for each sample within a batch, its shape must match the batch size dimension of your predictions and targets. Failure to align these dimensions will result in broadcasting errors or incorrect loss calculations.  Furthermore, the custom loss function needs to be differentiable for backpropagation to function correctly; this implies that all operations within the loss calculation must be differentiable with respect to the model's weights.

The workflow generally involves:

1. **Defining the weight array:** This array, usually a tensor, holds individual weights for each sample in the batch.  Its shape is crucial.  For a batch size of `B` and any other output dimension `O`, a common shape would be `(B,)` if each sample has a scalar weight or `(B, O)` if each output has its own weight.

2. **Integrating weights into the loss:**  The weight array is element-wise multiplied with the individual loss components calculated for each sample. This ensures each sample’s contribution to the overall loss is scaled according to its weight.

3. **Ensuring differentiability:**  All operations within the custom loss function must be differentiable.  TensorFlow’s built-in functions generally satisfy this condition.  Avoid non-differentiable operations like `tf.cond` or custom functions without defined gradients.

4. **Handling potential NaN or Inf values:**  It's essential to implement checks within your loss function to handle potential `NaN` or `Inf` values that might arise from numerical instability. These values can disrupt the training process.


**2. Code Examples with Commentary**

**Example 1:  Scalar Weight per Sample**

This example demonstrates a custom loss function where each sample in a batch has an associated scalar weight.

```python
import tensorflow as tf

def weighted_mse(y_true, y_pred, sample_weights):
    """
    Calculates weighted Mean Squared Error.

    Args:
        y_true: True labels (shape: (batch_size, ...)).
        y_pred: Predictions (shape: (batch_size, ...)).
        sample_weights: Weights for each sample (shape: (batch_size,)).

    Returns:
        Weighted MSE loss (scalar).
    """
    mse = tf.math.squared_difference(y_true, y_pred)
    weighted_mse = tf.reduce_mean(sample_weights * mse)
    return weighted_mse

# Example Usage
model = tf.keras.models.Sequential([ ... ]) # Your model definition
sample_weights = tf.constant([0.1, 0.5, 0.2, 1.0, 0.8], dtype=tf.float32) # Example weights
model.compile(loss=lambda y_true, y_pred: weighted_mse(y_true, y_pred, sample_weights), optimizer='adam')
```

This code efficiently leverages TensorFlow's broadcasting to multiply the `sample_weights` tensor with the element-wise squared differences.  The `tf.reduce_mean` then computes the average weighted MSE across the batch.


**Example 2:  Vector Weight per Sample**

Here, each sample has a weight vector, where each element in the vector corresponds to a particular output dimension.

```python
import tensorflow as tf

def weighted_mae(y_true, y_pred, sample_weights):
  """
  Calculates weighted Mean Absolute Error with per-output weights.

  Args:
      y_true: True labels (shape: (batch_size, num_outputs)).
      y_pred: Predictions (shape: (batch_size, num_outputs)).
      sample_weights: Weights for each sample and output (shape: (batch_size, num_outputs)).

  Returns:
      Weighted MAE loss (scalar).
  """
  mae = tf.abs(y_true - y_pred)
  weighted_mae = tf.reduce_mean(sample_weights * mae)
  return weighted_mae

# Example Usage
model = tf.keras.models.Sequential([ ... ]) #Your model
sample_weights = tf.random.uniform((5, 3), minval=0.1, maxval=1.0) # Example weights (5 samples, 3 outputs)
model.compile(loss=lambda y_true, y_pred: weighted_mae(y_true, y_pred, sample_weights), optimizer='adam')
```

This example extends the previous one by allowing per-output weights.  The broadcasting mechanism ensures each element in `y_true` - `y_pred` is multiplied by the corresponding weight.


**Example 3:  Handling NaN and Inf values**

This example incorporates checks for `NaN` and `Inf` values for robustness.

```python
import tensorflow as tf

def robust_weighted_loss(y_true, y_pred, sample_weights):
  """
  Calculates weighted MSE with NaN and Inf handling.

  Args:
      y_true: True labels.
      y_pred: Predictions.
      sample_weights: Sample weights.

  Returns:
      Weighted MSE loss, handling NaN and Inf values.
  """
  mse = tf.math.squared_difference(y_true, y_pred)
  weighted_mse = sample_weights * mse
  #Check for NaN and Inf values
  is_nan_inf = tf.math.logical_or(tf.math.is_nan(weighted_mse), tf.math.is_inf(weighted_mse))
  weighted_mse = tf.where(is_nan_inf, tf.zeros_like(weighted_mse), weighted_mse) # Replace NaN/Inf with 0
  return tf.reduce_mean(weighted_mse)


#Example Usage
model = tf.keras.models.Sequential([ ... ]) #Your model
sample_weights = tf.random.uniform((5, ), minval=0.1, maxval=1.0)
model.compile(loss=lambda y_true, y_pred: robust_weighted_loss(y_true, y_pred, sample_weights), optimizer='adam')
```

This improved version explicitly addresses potential numerical instability by replacing `NaN` or `Inf` values with zeros.  This ensures the training process doesn't crash due to these problematic values.


**3. Resource Recommendations**

For a deeper understanding of TensorFlow operations and tensor manipulation, I would recommend thoroughly reviewing the official TensorFlow documentation and the accompanying tutorials.  Focusing on topics like broadcasting, tensor shapes, and gradient tape will prove invaluable.  Furthermore, exploring advanced concepts like custom training loops and the use of `tf.function` for optimization can enhance your expertise.  Finally, studying examples of custom layers and losses within the Keras source code can provide valuable insights into best practices.  These resources, combined with practical experience, will allow you to confidently handle complex scenarios like integrating per-batch weight arrays into your custom loss functions.
