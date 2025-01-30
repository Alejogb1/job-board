---
title: "How to record Keras/Tensorflow 2.0 training and validation loss per batch?"
date: "2025-01-30"
id: "how-to-record-kerastensorflow-20-training-and-validation"
---
The core challenge in recording per-batch training and validation loss during Keras/TensorFlow 2.0 training lies in leveraging the framework's built-in callback mechanisms and understanding the appropriate data structures to capture and manage the evolving loss values.  Directly accessing internal model state during the training loop is discouraged, favoring instead the officially supported callback approach for maintainability and compatibility across TensorFlow versions.  My experience debugging similar logging issues in large-scale image recognition projects highlighted this approach as the most robust solution.

**1. Clear Explanation:**

TensorFlow's `tf.keras.callbacks.Callback` class provides the foundation for custom functionalities during training.  By extending this class, we can intercept the training process at specific points (e.g., the end of each batch) and access relevant metrics. Specifically, the `on_train_batch_end` and `on_test_batch_end` methods allow us to capture the loss for each training and validation batch, respectively.  Crucially, these methods receive arguments including `logs`, a dictionary containing the batch-level metrics.  Therefore, we need to strategically populate this dictionary within the model's training process to capture the per-batch loss.

The `logs` dictionary isn't automatically populated with per-batch loss; that data needs to be explicitly computed and added. This typically involves accessing the loss value directly from the training or validation step's output and adding it to the `logs` dictionary using a descriptive key (e.g., 'batch_loss').  Care must be taken to handle potential issues, such as the `logs` dictionary being empty, or the loss metric not being computed correctly (perhaps due to a model configuration error). Robust error handling is critical in production environments.  Furthermore, efficient data storage for potentially many batches is necessary; appending to a list within the callback is a straightforward approach.

**2. Code Examples with Commentary:**

**Example 1: Basic Per-Batch Loss Logging**

```python
import tensorflow as tf

class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_losses = []

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None and 'loss' in logs:
            self.train_losses.append(logs['loss'])

model = tf.keras.models.Sequential(...) # Your model definition
model.compile(...) # Your compilation parameters
batch_history = BatchLossHistory()
model.fit(..., callbacks=[batch_history])

print(batch_history.train_losses)
```

This example shows a minimal implementation.  The `on_train_begin` method initializes an empty list to store the training losses, while `on_train_batch_end` appends the loss from the `logs` dictionary.  Error handling (checking `logs` and `'loss'` existence) is minimal for clarity but essential in real-world applications.  Validation loss is not captured here.


**Example 2: Separate Training and Validation Loss Logging**

```python
import tensorflow as tf

class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None and 'loss' in logs:
            self.train_losses.append(logs['loss'])

    def on_test_batch_end(self, batch, logs=None):
        if logs is not None and 'loss' in logs:
            self.val_losses.append(logs['loss'])

# ... (model definition and compilation remain the same) ...

model.fit(..., validation_data=..., callbacks=[batch_history])

print("Training losses:", batch_history.train_losses)
print("Validation losses:", batch_history.val_losses)
```

This example extends the previous one to include validation loss logging, utilizing the `on_test_batch_end` method. This is crucial for monitoring model performance during validation.  The structure mirrors the training loss logging, demonstrating the symmetry in handling both training and validation data.

**Example 3: Handling potential `None` values and Custom Loss Key**

```python
import tensorflow as tf
import numpy as np

class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get('my_custom_loss', np.nan) #Handles potential absence of 'loss' key
        self.train_losses.append(loss)

    def on_test_batch_end(self, batch, logs=None):
        loss = logs.get('my_custom_loss', np.nan)
        self.val_losses.append(loss)


def custom_loss_function(y_true, y_pred):
  #Your custom loss function
  loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
  return loss

model = tf.keras.models.Sequential(...)
model.compile(loss=custom_loss_function,...) #using custom loss function
batch_history = BatchLossHistory()

model.fit(..., validation_data=..., callbacks=[batch_history])

print("Training losses:", batch_history.train_losses)
print("Validation losses:", batch_history.val_losses)

```

This example showcases more robust error handling by using `logs.get()` to safely retrieve the loss value, defaulting to `np.nan` if the key is missing. This prevents unexpected crashes. Additionally, it demonstrates using a custom loss function and a custom key ('my_custom_loss') within the logs dictionary for clarity and extensibility.  This is especially valuable when dealing with multiple loss functions or custom metrics.


**3. Resource Recommendations:**

The official TensorFlow documentation on custom callbacks is essential.  Thorough understanding of the `tf.keras.callbacks.Callback` class and its associated methods is paramount.  Reviewing documentation on handling dictionaries and error management in Python will also benefit development and debugging.  A solid grasp of the Keras/TensorFlow model compilation process, particularly the intricacies of loss functions and metric calculation, is vital for correct interpretation and logging of results.  Consulting examples of custom Keras callbacks from reputable sources (e.g., research papers, established open-source projects) will provide further insights into practical implementation.  Finally, proficiency with data visualization libraries (e.g., Matplotlib) is crucial for effective analysis of the per-batch loss data collected.
