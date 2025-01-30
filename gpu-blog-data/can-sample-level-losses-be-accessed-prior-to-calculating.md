---
title: "Can sample-level losses be accessed prior to calculating the final loss?"
date: "2025-01-30"
id: "can-sample-level-losses-be-accessed-prior-to-calculating"
---
In the realm of deep learning model training, particularly within frameworks like TensorFlow and PyTorch, access to individual sample losses before the aggregation into a batch loss is not a directly exposed, universally available feature. However, techniques exist to achieve a similar outcome, primarily by leveraging intermediate computations within the loss function or employing custom training loops. This ability to dissect sample-level losses can be critical for several purposes, such as identifying mislabeled data, focusing training on difficult examples, or debugging problematic model outputs for specific inputs.

My experience in developing a custom object detection model for a high-throughput imaging system highlighted this need. Initially, I faced a situation where the model performed exceptionally well on most samples but struggled with a specific subset exhibiting complex occlusion patterns. To diagnose this, I required access to sample-specific losses to pinpoint problematic instances rather than relying solely on the aggregated batch loss, which masked the detailed performance landscape.

The primary challenge stems from how most deep learning frameworks optimize for computational efficiency during training. Loss functions, while applied conceptually to each prediction and ground truth pairing, are often implemented as vectorized operations, processing an entire batch simultaneously. This design is optimized for GPU parallelism, resulting in a single scalar representing the overall batch loss rather than a series of individual sample-wise losses. The intermediate sample-wise losses are often computed internally, but not made readily accessible to the user.

To access these sample losses, we need to either modify the built-in loss functions (where possible) to retain these intermediary values or construct a custom training loop. One method involves modifying the loss function using a return statement that provides a list of the per-sample loss values, along with the actual batch loss used for the backward pass. This modification requires the understanding of the framework and the specific implementations of the loss functions. Another technique is implementing a custom training loop that includes a step that computes the loss for every sample within a batch iteratively. The framework's API does not expose individual sample loss during the typical training step.

Here are examples of how this can be done:

**Example 1: Custom Loss Function with Sample Loss Return (PyTorch)**

In this case, we modify a typical binary cross-entropy loss function to output individual sample losses alongside the batch loss.

```python
import torch
import torch.nn as nn

class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none') #reduction set to none

    def forward(self, predicted, target):
        per_sample_loss = self.bce_loss(predicted, target)
        batch_loss = torch.mean(per_sample_loss)
        return batch_loss, per_sample_loss # Returning both batch and sample loss

# Usage:
criterion = CustomBCELoss()
predictions = torch.sigmoid(torch.randn(10, 1))  # Example with 10 samples, 1 class
ground_truth = torch.randint(0, 2, (10, 1)).float()

batch_loss, sample_losses = criterion(predictions, ground_truth)
print("Batch Loss:", batch_loss)
print("Per-Sample Losses:", sample_losses)
```
The `BCELoss` is initialized with `reduction='none'`, which prevents the aggregation of the individual losses. The `forward` method calculates the `per_sample_loss`, and then the `batch_loss`. Both are returned for access. This allows us to examine the contribution of each data point to the overall loss. This example uses binary cross-entropy for demonstration, but the same approach can be used for other loss functions in PyTorch.

**Example 2: Sample Loss Calculation in a Custom Training Loop (TensorFlow)**

Tensorflow, with its eager execution, facilitates the implementation of a custom training loop. The following example manually iterates through each sample in a batch to obtain individual loss calculations.

```python
import tensorflow as tf

def sample_losses_tf(model, x, y, loss_fn):
    sample_losses = []
    for i in range(x.shape[0]):
        with tf.GradientTape() as tape:
          y_pred = model(tf.expand_dims(x[i], 0)) # Add a dimension for batch
          sample_loss = loss_fn(tf.expand_dims(y[i], 0), y_pred) # Calculate sample loss
        sample_losses.append(sample_loss.numpy())
    return tf.reduce_mean(sample_losses), tf.stack(sample_losses) #mean batch loss and individual losses


# Create a simple model and loss function.
model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid')])
loss_function = tf.keras.losses.BinaryCrossentropy()

# Example batch data
x_batch = tf.random.normal(shape=(5, 10)) # 5 samples, 10 features
y_batch = tf.random.uniform(shape=(5, 1), minval=0, maxval=2, dtype=tf.int32) # 5 samples, 1 target

# calculate losses.
batch_loss, sample_losses = sample_losses_tf(model, x_batch, y_batch, loss_function)

print("Batch Loss:", batch_loss)
print("Per-Sample Losses:", sample_losses)

```
In this approach, the custom `sample_losses_tf` function iterates through each sample in the batch using a for-loop. The `tf.GradientTape` is employed to capture gradients, though not directly utilized here. This tape is a context in which we execute model predictions and loss calculations, enabling gradient calculation if needed. The predicted values `y_pred` and loss calculation are generated by passing one instance of the batch to the model and loss function respectively. Note the use of `tf.expand_dims` to add the batch dimension to the single data samples. The sample loss is calculated for each item in the batch. This custom implementation provides a clear representation of the per-sample loss values. It can be slower than vector operations, so consider the performance implications of this operation in the overall training loop.

**Example 3: Obtaining Sample Loss via Intermediate Tensors (TensorFlow)**

In some cases, the loss calculation function, whether customized or built-in, may provide direct access to a tensor from which per-sample losses can be extracted.  This depends on the specific implementation. The below example calculates the individual losses through the returned `logits` tensor.

```python
import tensorflow as tf

# Example labels
labels = tf.constant([0, 1, 0, 1], dtype=tf.int32)

# Example logits representing pre-sigmoid outputs
logits = tf.constant([[1.0], [1.0], [-1.0], [-1.0]], dtype=tf.float32)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

sample_losses = loss_fn(labels, logits)

batch_loss = tf.reduce_mean(sample_losses)

print("Batch Loss:", batch_loss)
print("Per-Sample Losses:", sample_losses)
```
In this scenario, by setting `reduction=tf.keras.losses.Reduction.NONE` when instantiating the `BinaryCrossentropy` loss, the loss function returns the calculated loss for each sample, not the mean over samples. The sample losses can be calculated as the returned value from the instantiated loss function.

These examples demonstrate distinct methods to obtain sample-level loss information during training. The choice between a modified loss function and a custom training loop largely depends on the user's preference, the complexity of the loss being used, and the flexibility required. Modified loss functions can provide faster performance if the underlying framework supports per-sample calculations and the loss function's operation does not require explicit iteration. Custom training loops offer greater control and can accommodate more sophisticated analysis of sample-wise metrics.

For further understanding and practical implementation of these concepts, I recommend focusing on the following resources. First, thoroughly review the documentation of your chosen deep learning framework, specifically focusing on the implementation details of the built-in loss functions. Next, study examples of custom training loops for each framework and experiment with building simple ones to understand the mechanisms of gradient updates. Additionally, familiarize yourself with the concepts of gradient tape utilization and how it helps with custom gradients and calculations. Finally, explore community forums and user groups, particularly those within the chosen framework ecosystem, which can provide valuable practical advice and address specific scenarios. This focused study can significantly enhance your ability to diagnose model issues using sample-level loss information.
