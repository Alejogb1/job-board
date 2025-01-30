---
title: "How can I apply custom callbacks in a TensorFlow 2.0 training loop?"
date: "2025-01-30"
id: "how-can-i-apply-custom-callbacks-in-a"
---
TensorFlow 2.0's `tf.keras.callbacks.Callback` class provides a flexible mechanism for intervening in the training process, offering hooks at various stages like the start and end of epochs, batches, and training itself. I’ve found it exceptionally useful for tasks ranging from dynamic learning rate adjustments based on validation loss to early stopping criteria beyond simple patience. The core idea revolves around subclassing `tf.keras.callbacks.Callback` and overriding its various lifecycle methods. This allows you to inject custom logic that interacts with the model and training data during the learning cycle.

A fundamental aspect to understand is that the Keras training process, including both model fitting and manual training loops with `tf.GradientTape`, ultimately invoke callbacks through a shared callback list. When using `model.fit()`, Keras internally manages this callback list, adding any you provide as arguments to a predefined set of core callbacks. With manual training loops using `tf.GradientTape`, you are directly responsible for instantiating and calling the methods on the `Callback` instances.

The callback class exposes a number of methods. Notably, `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`, `on_batch_begin`, `on_batch_end`, and similar methods prefixed with `test_` and `predict_` respectively, are the crucial touchpoints. These methods receive a `logs` dictionary as an argument, and in some cases, a batch or epoch number. This `logs` dictionary allows observation of metrics during training, including loss and any other metric calculated by the model. Crucially, it’s not just about observation; you can modify training behaviors. For instance, you can change the learning rate of the optimizer during `on_epoch_end`.

To elaborate, I will provide three practical examples that illustrate common use cases.

**Example 1: Custom Logging Callback**

This callback provides an extended logging functionality, printing the loss and any other metrics after each batch and after each epoch. I designed this initially when debugging a complex model, needing more granular output than the default Keras progress bar.

```python
import tensorflow as tf

class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs is None:
           return
        print(f"Batch {batch}: Loss = {logs.get('loss'):.4f}", end=" ")
        for metric_name in logs:
          if metric_name != 'loss':
             print(f"{metric_name} = {logs.get(metric_name):.4f}", end=" ")
        print()

    def on_epoch_end(self, epoch, logs=None):
         if logs is None:
           return
         print(f"Epoch {epoch}: Loss = {logs.get('loss'):.4f}", end=" ")
         for metric_name in logs:
          if metric_name != 'loss':
             print(f"{metric_name} = {logs.get(metric_name):.4f}", end=" ")
         print()
```

In this example, `on_batch_end` is invoked at the end of each training batch, and `on_epoch_end` after each epoch completes. `logs` is a dictionary containing the loss, along with any other metrics set by the Keras model. The methods retrieve the ‘loss’ value and any other available metrics and prints them in a formatted string. Using `logs.get('metric_name')` ensures a safe retrieval even if the metric is absent, preventing runtime errors. If a batch or epoch ends without having the `logs` dictionary updated (e.g. during testing and evaluation in specific cases), the function will return without performing any operation avoiding errors. This avoids needing additional logic to check if a callback is being run during training or evaluation.

**Example 2: Dynamic Learning Rate Scheduling Callback**

This callback implements a learning rate scheduler that reduces the learning rate by a factor of 0.5 if validation loss stagnates for two epochs. The method was adapted from several trial and error approaches, trying to balance faster convergence and stable training.

```python
import tensorflow as tf

class LearningRateSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=2, factor=0.5):
        super(LearningRateSchedulerCallback, self).__init__()
        self.patience = patience
        self.factor = factor
        self.wait = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
      if logs is None:
        return
      val_loss = logs.get('val_loss')
      if val_loss is None:
         return #do not execute if there is not validation data
      if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
      else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = old_lr * self.factor
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(f"Epoch {epoch}: Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                self.wait = 0
```

Here, `on_epoch_end` retrieves the validation loss from the `logs` dictionary. It tracks the best validation loss and increments a waiting counter if there is no improvement. If the waiting counter exceeds the specified patience, it retrieves the current learning rate from the optimizer, multiplies it by the specified factor, and sets the new learning rate. Using `tf.keras.backend.get_value` and `tf.keras.backend.set_value` allows direct access and modification of the optimizer's parameters. Checking if validation loss exists before making a comparison avoids errors when running on datasets that have no validation set.

**Example 3: Manual Training Loop with Callback Integration**

This example demonstrates how to utilize callbacks in a custom training loop using `tf.GradientTape`. I typically use manual loops when I need more granular control of the gradients or data flow.

```python
import tensorflow as tf

def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def manual_train_loop(model, optimizer, loss_fn, train_dataset, epochs, callbacks):
    history = {'loss': []}
    for epoch in range(epochs):
      callback_logs = {}
      for callback in callbacks:
        callback.on_epoch_begin(epoch, logs=callback_logs)
      epoch_loss = 0
      for step, (x_batch, y_batch) in enumerate(train_dataset):
          loss = train_step(model, optimizer, loss_fn, x_batch, y_batch)
          epoch_loss += loss
          batch_logs = {'loss': loss}
          for callback in callbacks:
             callback.on_batch_end(step, batch_logs)
      epoch_loss /= len(train_dataset)
      history['loss'].append(epoch_loss)
      callback_logs['loss'] = epoch_loss
      for callback in callbacks:
        callback.on_epoch_end(epoch, logs=callback_logs)
    return history

# Example usage:
model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid')])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.BinaryCrossentropy()
train_data = tf.data.Dataset.from_tensor_slices(([1,2,3,4], [0,0,1,1])).batch(2)
callbacks = [CustomLoggingCallback(), LearningRateSchedulerCallback()]
epochs = 5
history = manual_train_loop(model, optimizer, loss_fn, train_data, epochs, callbacks)

```

In the provided `manual_train_loop`, I explicitly invoke callback methods such as `on_epoch_begin`, `on_batch_end`, and `on_epoch_end`. It's important to note, for those unfamiliar with this, that you need to iterate through the callback list and call the methods explicitly. The `train_step` function handles the forward and backward passes. The example shows how the custom callbacks created earlier can be used within the loop, the key is to carefully create and pass along the logs dictionaries, ensuring that the data is accessible to the callbacks at the necessary point. By integrating all callbacks on the training loop, we can ensure consistent behaviour when comparing automatic training with manual training.

In summary, TensorFlow 2.0's custom callbacks offer extensive control over the training process. Understanding the lifecycle methods and how to utilize the `logs` dictionary allows for the implementation of intricate behaviors. When using callbacks, I always make sure to design them in an modular fashion, allowing me to re-use callbacks between models and training approaches.

For further learning, I would recommend reviewing the official TensorFlow documentation on callbacks. I have found that the examples given in the Keras documentation, particularly on how to create and use custom callbacks, to be an excellent starting point for experimentation. Other resources I have consulted and found useful in the past are tutorials that deal with optimizing training for deep learning models, specifically focusing on techniques such as callback-driven learning rate scheduling and metrics tracking.
