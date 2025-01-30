---
title: "How can a TensorFlow Sequential model be trained with soft labels?"
date: "2025-01-30"
id: "how-can-a-tensorflow-sequential-model-be-trained"
---
Soft labels, representing probabilities across classes rather than hard one-hot encodings, often improve model generalization and calibration, particularly when data is noisy or ambiguous. I’ve personally observed this boosting performance in image classification tasks where object boundaries are blurry, or in text categorization where a document might reasonably belong to multiple categories to varying degrees. The technique, sometimes called label smoothing, is implemented by modifying the loss function calculation within the TensorFlow training loop.

At its core, training with soft labels involves replacing the traditional one-hot encoded target vectors with probability distributions. Instead of assigning a target of `[0, 1, 0]` to a data point belonging to class 1 (assuming a three-class problem), we might use `[0.05, 0.9, 0.05]`, implying a strong likelihood of belonging to class 1, but acknowledging a small probability of belonging to other classes. This softens the decision boundary, prevents the model from becoming overly confident, and encourages it to learn more robust representations. The key is that the total sum of probabilities within a given soft label vector is still 1.

The modification necessary in TensorFlow primarily affects the loss function. Instead of using categorical cross-entropy directly with one-hot encoded labels, we either manually calculate the loss using the soft labels or employ TensorFlow’s built-in utilities that facilitate label smoothing. Both methods effectively reduce the emphasis on precise classification of every data point, instead nudging the model to learn the relative probabilities across different classes.

Here are three approaches I've utilized for training a TensorFlow Sequential model with soft labels:

**Example 1: Manual Loss Calculation**

This first example demonstrates calculating the cross-entropy loss manually, allowing us to directly incorporate our soft labels. This is a good technique when you need precise control over the calculations. It requires an understanding of the underlying mathematical calculation of cross-entropy.

```python
import tensorflow as tf
import numpy as np

# Define a simple Sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Dummy data and soft labels
x_train = np.random.rand(100, 5).astype(np.float32)
soft_labels = np.random.dirichlet(alpha=(1, 1, 1), size=100).astype(np.float32) # Dirichilet produces probability distributions

# Training loop
def train_step(x, soft_y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.reduce_sum(-soft_y * tf.math.log(logits + 1e-7), axis=1) #Added a small constant for numeric stability
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

epochs = 500
for epoch in range(epochs):
  loss = train_step(x_train, soft_labels)
  if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')
```

In this example, `soft_labels` are generated using a Dirichlet distribution, which ensures each label is a valid probability distribution that sums to 1. Within the `train_step` function, we compute the cross-entropy loss manually using TensorFlow operations, specifically focusing on the logarithmic probability of the prediction under each soft label. This approach, unlike a built-in function, clearly exposes the cross-entropy calculation, giving flexibility if one wishes to modify the loss formula beyond basic cross-entropy. The small constant (`1e-7`) is added to prevent numerical instability from taking the log of zero. This example requires the labels to be in a probability vector form, not one-hot encoded form.

**Example 2: Using `tf.keras.losses.CategoricalCrossentropy` with `label_smoothing`**

This example utilizes a more direct, high-level approach using TensorFlow's built-in functions. The `label_smoothing` parameter within `CategoricalCrossentropy` provides a simpler way to introduce soft labels.

```python
import tensorflow as tf
import numpy as np

# Define a simple Sequential model (same as before)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Optimizer (same as before)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Loss function with label smoothing
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Dummy data and hard labels
x_train = np.random.rand(100, 5).astype(np.float32)
hard_labels = np.random.randint(0, 3, size=(100,)).astype(np.int32)

# Training loop
def train_step(x, y_true):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(tf.one_hot(y_true, depth=3), logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

epochs = 500
for epoch in range(epochs):
  loss = train_step(x_train, hard_labels)
  if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')
```

In this example, we are using a standard `CategoricalCrossentropy` loss function. The key difference is the `label_smoothing=0.1` parameter. This parameter automatically adjusts the hard labels, represented by one-hot encoded vectors within the loss calculation. Essentially, the one-hot encoded labels are softened as follows: if a one-hot encoded label is `[0, 1, 0]`, with a smoothing factor of `0.1`, the new target label effectively becomes `[0.05, 0.9, 0.05]`. This approach simplifies implementation when a predetermined smoothing factor is sufficient. This approach still requires the hard labels (categorical indices of the correct class), which it turns into a one-hot vector internally before applying label smoothing.

**Example 3: Custom Loss Function Class**

This approach is more modular and useful when combining soft labels with other loss modifications. The loss function is defined as a custom class, providing a reusable component within a more complex training pipeline.

```python
import tensorflow as tf
import numpy as np

# Define a simple Sequential model (same as before)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Optimizer (same as before)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Custom loss function class
class SoftLabelCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
      if self.smoothing > 0:
        y_true = (1 - self.smoothing) * y_true + self.smoothing / y_true.shape[-1]
      return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

loss_fn = SoftLabelCrossEntropy(smoothing=0.1)

# Dummy data and soft labels
x_train = np.random.rand(100, 5).astype(np.float32)
soft_labels = np.random.dirichlet(alpha=(1, 1, 1), size=100).astype(np.float32)


# Training loop
def train_step(x, soft_y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(soft_y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

epochs = 500
for epoch in range(epochs):
  loss = train_step(x_train, soft_labels)
  if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')
```

This example encapsulates the soft label handling into a reusable class. Within the `call` method, the input `y_true` (which is expected to be soft labels) is modified to incorporate a label smoothing if the smoothing factor is greater than 0. The loss function applies the `categorical_crossentropy` directly after smoothing. This encapsulates both manual label smoothing and standard cross-entropy calculations. This modularization makes it easy to modify or extend the loss function later. Unlike Example 2, this approach directly accepts the soft labels as the input `y_true`.

In my experience, manual loss calculation (Example 1) is necessary when you are working with advanced loss functions, as you will need to modify the cross entropy beyond basic label smoothing. The `CategoricalCrossentropy` class with the `label_smoothing` parameter (Example 2) is the most simple and efficient approach when a simple uniform smoothing parameter is sufficient. Custom loss function classes (Example 3) are useful for combining advanced loss manipulations, and they provide a good method for modularizing training when a specific training pipeline requires modifications.

For additional information, I recommend consulting the TensorFlow documentation on `tf.keras.losses` and exploring examples related to custom training loops. I've also found material covering advanced loss functions like focal loss and those utilizing Kullback-Leibler divergence to be useful, although they are beyond the scope of this response. Texts that cover information theory and machine learning with an emphasis on probabilistic methods would also be beneficial to understand soft labels and their role in regularization.
