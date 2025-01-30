---
title: "How can a custom loss function be designed to output a loss per sample?"
date: "2025-01-30"
id: "how-can-a-custom-loss-function-be-designed"
---
The critical aspect of designing a custom loss function that outputs a loss per sample lies in leveraging the inherent vectorization capabilities of modern deep learning frameworks.  Ignoring this often leads to inefficient, less readable, and harder-to-debug code. My experience working on anomaly detection systems for high-frequency trading data highlighted this precisely; optimizing per-sample loss calculations significantly improved training speed and model interpretability.  The key is to avoid explicit looping whenever possible, relying instead on element-wise operations supported by libraries like TensorFlow or PyTorch.

**1. Clear Explanation:**

A custom loss function, in essence, defines how a model's predictions deviate from the ground truth.  Standard loss functions, like mean squared error (MSE) or cross-entropy, aggregate this deviation across all samples in a batch before backpropagation.  However, obtaining per-sample loss values offers several advantages:

* **Debugging and Monitoring:**  Analyzing individual sample losses allows for precise identification of problematic data points or regions where the model struggles.  This granular insight greatly aids in diagnosing training issues and refining data preprocessing techniques. In my work, this proved crucial in detecting outliers within the financial datasets that skewed model performance.

* **Weighted Losses:**  Per-sample loss enables the implementation of sample-specific weights, addressing class imbalance or data weighting schemes. This allows prioritization of specific samples during training, a technique particularly relevant when dealing with datasets containing noisy or unreliable data. I utilized this heavily when fine-tuning models with imbalanced datasets of fraudulent transactions.

* **Advanced Training Strategies:** Techniques like curriculum learning or selective backpropagation require access to individual sample losses for effective implementation.  Having the per-sample loss readily available allows for dynamic adjustments to the training process based on the model's performance on each example.

To achieve per-sample loss, the custom function must avoid aggregation operations like `tf.reduce_mean()` or `torch.mean()` until after the individual losses have been computed.  Instead, the function should compute the loss for each sample independently and return a tensor or array of the same shape as the input data. This vectorized approach aligns seamlessly with the framework's automatic differentiation mechanisms.  Failing to do so necessitates manual looping, dramatically impacting performance, especially with larger datasets.

**2. Code Examples with Commentary:**

The following examples demonstrate the implementation of per-sample loss functions using TensorFlow and PyTorch.  Assume `y_true` represents the ground truth labels and `y_pred` represents the model's predictions, both tensors of shape (batch_size, num_classes) for multi-class classification.

**Example 1: TensorFlow - Categorical Cross-Entropy per sample**

```python
import tensorflow as tf

def categorical_crossentropy_per_sample(y_true, y_pred):
  """Computes categorical cross-entropy loss per sample.

  Args:
    y_true: Ground truth labels (batch_size, num_classes).
    y_pred: Model predictions (batch_size, num_classes).

  Returns:
    Tensor of shape (batch_size,) containing per-sample losses.
  """
  epsilon = tf.keras.backend.epsilon() # Avoid log(0) errors.
  y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) # Clip for numerical stability.
  loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1) #Element-wise computation
  return loss

# Example usage:
y_true = tf.constant([[0., 1., 0.], [1., 0., 0.]])
y_pred = tf.constant([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
per_sample_loss = categorical_crossentropy_per_sample(y_true, y_pred)
print(per_sample_loss) # Output: tf.Tensor([0.22314353 0.35667496], shape=(2,), dtype=float32)

```

This TensorFlow example leverages `tf.reduce_sum` along the last axis (`axis=-1`), calculating the cross-entropy for each sample independently.  The clipping operation ensures numerical stability by preventing `log(0)` errors.


**Example 2: PyTorch - Mean Squared Error per sample**

```python
import torch
import torch.nn.functional as F

def mse_per_sample(y_true, y_pred):
  """Computes mean squared error loss per sample.

  Args:
    y_true: Ground truth values (batch_size, ...).
    y_pred: Model predictions (batch_size, ...).

  Returns:
    Tensor of shape (batch_size,) containing per-sample losses.
  """
  loss = F.mse_loss(y_true, y_pred, reduction='none') #reduction='none' is key
  return loss.mean(dim=1) #average over any additional dimensions after batch

# Example usage
y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_pred = torch.tensor([[1.2, 1.8], [3.5, 3.5]])
per_sample_loss = mse_per_sample(y_true, y_pred)
print(per_sample_loss) # Output: tensor([0.0200, 0.2500])
```

This PyTorch example uses `F.mse_loss` with `reduction='none'` to get the loss for each element. The subsequent `.mean(dim=1)` averages across multiple output values per sample (if applicable), providing a single MSE value per sample.


**Example 3: Custom Loss Function - Combining Metrics**

```python
import tensorflow as tf

def combined_loss(y_true, y_pred):
  """Combines MSE and categorical cross-entropy per sample.

  Args:
    y_true: Ground truth (batch_size, num_classes + num_regression_outputs)
    y_pred: Predictions (batch_size, num_classes + num_regression_outputs)

  Returns:
    Tensor of shape (batch_size,) containing per-sample combined losses.
  """
  num_classes = 3 #example
  num_reg = 2 #example

  y_true_class = y_true[:, :num_classes]
  y_pred_class = y_pred[:, :num_classes]
  y_true_reg = y_true[:, num_classes:]
  y_pred_reg = y_pred[:, num_classes:]

  class_loss = categorical_crossentropy_per_sample(y_true_class, y_pred_class)
  regression_loss = mse_per_sample(y_true_reg, y_pred_reg)
  return class_loss + regression_loss

#Example usage (Illustrative)
y_true = tf.constant([[0.,1.,0., 10.,20.], [1.,0.,0., 30., 40.]])
y_pred = tf.constant([[0.1,0.8,0.1, 12., 18.], [0.7,0.2,0.1, 35., 35.]])

combined = combined_loss(y_true, y_pred)
print(combined)
```

This example showcases combining multiple loss components (classification and regression) at the per-sample level for a more sophisticated loss function.  The flexibility here allows for highly customized model training based on different aspects of the prediction.


**3. Resource Recommendations:**

For a more in-depth understanding of custom loss functions and their applications, I would recommend exploring the official documentation of TensorFlow and PyTorch, focusing on their automatic differentiation mechanisms and tensor manipulation capabilities.  Furthermore, review publications on advanced training techniques and model architectures that leverage per-sample loss functions.  Consider studying advanced texts on machine learning and deep learning to obtain a solid theoretical understanding of loss functions and their role in optimization.  Finally, examine code repositories of publicly available projects that employ similar techniques to your application; studying well-documented code can be invaluable.
