---
title: "How can I use tf.keras.callbacks.ModelCheckpoint without calling model.fit?"
date: "2025-01-30"
id: "how-can-i-use-tfkerascallbacksmodelcheckpoint-without-calling-modelfit"
---
The `tf.keras.callbacks.ModelCheckpoint` class is fundamentally designed to operate within the `model.fit` training loop, automatically monitoring a metric and saving the model at specified intervals. However, scenarios arise where one needs more granular control over the training process, perhaps involving custom training loops, or situations where `model.fit` is unsuitable. Fortunately, `ModelCheckpoint` provides the necessary hooks and properties to achieve model checkpointing without relying on the conventional training method.

The core understanding is that `ModelCheckpoint` doesn't inherently require `model.fit`; it needs access to the model and the ability to observe the training loop. By manually triggering the callback's methods at relevant points in a custom training loop, we can replicate the functionality. Crucially, the `ModelCheckpoint` object operates by observing a loss or metric and then conditionally saving the model. The internal machinery for these steps, however, is decoupled from the `fit` method.

First, `ModelCheckpoint` must be instantiated with the appropriate configuration. This entails defining the file path template, the monitoring metric, the save mode (`min` or `max`), and the save frequency. Once instantiated, the `set_model` method must be invoked to associate the callback with the model being trained. Without this association, the callback won't know which model to save, or which layers and weights to track.

The subsequent actions revolve around the training process. In a standard `model.fit` scenario, the callback is automatically invoked at the end of each epoch or after each batch, but when employing custom loops, these invocations must occur manually. The `on_epoch_end` (or `on_batch_end` if checkpointing per batch is desired) method is designed for this purpose. It expects a numerical value corresponding to the metric being monitored as input. This is where the custom logic plugs in. We need to compute the value of the metric, pass it to the callback, and allow it to perform the necessary checks and saving operations. This does *not* imply the entire loop has to be implemented from scratch - the Keras API and TensorFlow are flexible enough to provide utility functions.

The `ModelCheckpoint` keeps track internally the best epoch (or batch) which has been seen by the monitor and updates the filepath. The final save will happen when the on_train_end callback is invoked. The checkpoint object also has an `_save_model` method which could be used to make a manual save. However, that is discouraged because there might be other internal states which are not updated.

Here are three code examples illustrating these concepts:

**Example 1: Epoch-Based Checkpointing**

```python
import tensorflow as tf
import numpy as np

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Loss and optimizer
optimizer = tf.keras.optimizers.Adam(0.01)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Data generation
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, (100, 1))
y_train = tf.keras.utils.to_categorical(y_train)

# ModelCheckpoint setup
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint_ex1/best_weights',
    monitor='loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1
)

# Associate the callback with the model
checkpoint_cb.set_model(model)

# Custom training loop
epochs = 10
for epoch in range(epochs):
  print(f"Epoch: {epoch+1}/{epochs}")
  with tf.GradientTape() as tape:
    y_pred = model(X_train)
    loss = loss_fn(y_train, y_pred)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  # Manually invoke the callback at the end of each epoch
  checkpoint_cb.on_epoch_end(epoch, logs={'loss': loss.numpy()})

# Invoke the final callback
checkpoint_cb.on_train_end()

print("Training complete, best model saved.")
```

In this example, the `ModelCheckpoint` object is created, specifying to monitor the training loss and save the best model based on minimal loss. The callback is explicitly associated with the model. Instead of calling `model.fit`, a basic training loop using gradient tape is employed. Crucially, `checkpoint_cb.on_epoch_end` is called at the end of each epoch, passing the current loss value.  The `on_train_end` method is called at the very end to save the last model. The model weights will be saved to the designated path inside `model_checkpoint_ex1`. The `logs` parameter of the callback's method is what gives information to the model about the current epoch. Without the `logs` parameter the model won't be able to keep track of metrics.

**Example 2: Metric-Based Checkpointing and Custom Validation Loop**

```python
import tensorflow as tf
import numpy as np

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Loss and optimizer
optimizer = tf.keras.optimizers.Adam(0.01)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

# Data generation
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, (100, 1))
y_train = tf.keras.utils.to_categorical(y_train)

X_val = np.random.rand(50,5)
y_val = np.random.randint(0, 2, (50, 1))
y_val = tf.keras.utils.to_categorical(y_val)


# ModelCheckpoint setup
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint_ex2/best_weights',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

# Associate the callback with the model
checkpoint_cb.set_model(model)

# Custom training loop with validation
epochs = 10
for epoch in range(epochs):
    print(f"Epoch: {epoch+1}/{epochs}")

    # Training step
    with tf.GradientTape() as tape:
      y_pred = model(X_train)
      loss = loss_fn(y_train, y_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Validation step
    y_val_pred = model(X_val)
    accuracy_metric.update_state(y_val, y_val_pred)
    val_accuracy = accuracy_metric.result().numpy()
    accuracy_metric.reset_state()

    # Manually invoke the callback at the end of each epoch with the validation metric
    checkpoint_cb.on_epoch_end(epoch, logs={'val_accuracy': val_accuracy})

# Invoke the final callback
checkpoint_cb.on_train_end()

print("Training complete, best model saved.")
```
This example demonstrates monitoring a validation metric (`val_accuracy`) instead of the loss. A basic validation loop using `tf.keras.metrics.CategoricalAccuracy` is set up and the results are passed to `on_epoch_end`. This setup highlights that any relevant metric or value can be monitored by the callback as long as it is provided when the callback methods are invoked. The model weights are saved to the designated path inside `model_checkpoint_ex2`.

**Example 3: Batch-Level Checkpointing**

```python
import tensorflow as tf
import numpy as np

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Loss and optimizer
optimizer = tf.keras.optimizers.Adam(0.01)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Data generation
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, (100, 1))
y_train = tf.keras.utils.to_categorical(y_train)

# ModelCheckpoint setup
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint_ex3/weights-{batch:02d}',
    monitor='loss',
    save_best_only=False,  # Save every time for batch-level checkpointing
    save_weights_only=True,
    save_freq='batch',
    mode='min',
    verbose=0
)

# Associate the callback with the model
checkpoint_cb.set_model(model)

# Custom training loop with batches
batch_size = 20
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

batch_index = 0

for batch_x, batch_y in dataset:

    with tf.GradientTape() as tape:
      y_pred = model(batch_x)
      loss = loss_fn(batch_y, y_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


    # Manually invoke the callback at the end of each batch
    checkpoint_cb.on_batch_end(batch_index, logs={'loss': loss.numpy()})
    batch_index +=1

# Invoke the final callback
checkpoint_cb.on_train_end()

print("Training complete, model saved at batches.")
```

This final example showcases checkpointing at the batch level, not epoch level. Notice the `save_freq` parameter set to `batch`.  The `filepath` parameter in this example includes the batch number. Here, we are using a `tf.data.Dataset` to iterate over the dataset.  `checkpoint_cb.on_batch_end` is called to checkpoint at every batch. The `save_best_only` parameter has been set to `False` because we intend to save all weights. The model weights are saved to designated paths inside `model_checkpoint_ex3` with the batch number specified.

For further exploration of related concepts, I recommend reviewing the official TensorFlow documentation on custom training loops, `tf.GradientTape`, and the `tf.keras.metrics` module. Additionally, consulting resources discussing custom callbacks in Keras, particularly the method signatures and arguments, will prove beneficial. The book *Deep Learning with Python, Second Edition*, by François Chollet, provides an excellent overview of building and training neural networks in Keras. Lastly, the core TensorFlow API documentation is an invaluable reference for understanding the fundamental classes such as `tf.Variable`, `tf.train.Optimizer` and `tf.data.Dataset`. These provide the basis for building robust custom loops.
