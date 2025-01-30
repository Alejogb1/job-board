---
title: "How do I print the percentage of completion in TensorFlow?"
date: "2025-01-30"
id: "how-do-i-print-the-percentage-of-completion"
---
Achieving a reliable progress indicator during TensorFlow training, particularly when dealing with large datasets, necessitates understanding the internal mechanisms of the training loop and leveraging the framework's built-in tools for metric tracking and display. Directly printing a percentage completion based solely on epochs can be misleading because each epoch involves processing numerous batches of data. A more precise percentage requires knowledge of the total number of batches.

I’ve encountered the need for detailed progress feedback during complex model training, where vague output can be frustrating. Standard epoch-based printouts often don't adequately reflect the amount of data already processed. Specifically, in a recent project involving sequence-to-sequence learning on a large corpus of text data, a visual percentage indicator based on batch completion was critical for monitoring progress, diagnosing bottlenecks, and ensuring the process was advancing as expected.

The challenge lies in coupling the training logic, typically defined within a custom training loop or a `model.fit()` context, with a mechanism to track current batch progress relative to the total batches per epoch. TensorFlow provides several avenues to achieve this. One can utilize `tf.data.Dataset` properties, leverage `tf.keras.callbacks`, or incorporate custom logic within a custom training loop. I will outline approaches incorporating both the `model.fit()` API with callbacks and a custom training loop for more fine-grained control.

**Approach 1: Leveraging `tf.keras.callbacks.Callback`**

The `tf.keras.callbacks.Callback` class offers a structured and less intrusive method to intercept and augment the default `model.fit()` behavior. By subclassing the `Callback` class, one can define custom actions that execute at various stages of the training process, such as the start or end of an epoch, or at the beginning and end of each batch.

Here’s a specific example demonstrating how to calculate and print a percentage completion:

```python
import tensorflow as tf
import numpy as np

class PercentCompleteCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_batches, verbose=1):
        super().__init__()
        self.total_batches = total_batches
        self.verbose = verbose

    def on_train_batch_end(self, batch, logs=None):
        current_batch = batch + 1
        percent_complete = (current_batch / self.total_batches) * 100
        if self.verbose > 0:
          print(f"\rBatch: {current_batch}/{self.total_batches}  ({percent_complete:.2f}%)", end="")


# Generate dummy data
train_data = np.random.rand(1000, 10)
train_labels = np.random.randint(0, 2, 1000)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)

# Calculate total batches
total_batches = len(dataset)

# Create and pass callback
percent_complete_callback = PercentCompleteCallback(total_batches)
model.fit(dataset, epochs=5, verbose=0, callbacks=[percent_complete_callback])
```

In this example, `PercentCompleteCallback` calculates the percentage based on `current_batch` and `total_batches`. The `on_train_batch_end` method is triggered after each batch is processed. Using `\r` and `end=""` ensures a continuous, overwriting output on a single line instead of the default newline-separated print.  `verbose=0` disables default `model.fit` output, which is crucial to prevent output clutter. The key here is to obtain the total number of batches before fitting the model. This is accomplished using `len(dataset)` after creating the `tf.data.Dataset` instance.

**Approach 2: Custom Training Loop with Manual Percentage Calculation**

A custom training loop allows for granular control, which is often necessary for complex models or non-standard training procedures.  This method requires constructing a training loop manually, using `tf.GradientTape` for gradient calculation. While more verbose, it provides direct control over the execution flow.

```python
import tensorflow as tf
import numpy as np

# Generate dummy data
train_data = np.random.rand(1000, 10)
train_labels = np.random.randint(0, 2, 1000)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()


# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)

# Calculate total batches
total_batches = len(dataset)

epochs = 5

for epoch in range(epochs):
    for batch_index, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        current_batch = batch_index + 1
        percent_complete = (current_batch / total_batches) * 100
        print(f"\rEpoch {epoch + 1}/{epochs}, Batch: {current_batch}/{total_batches}  ({percent_complete:.2f}%)", end="")
    print()
```

Here, the percentage is calculated inside the loop for each batch iteration using the same formula as before. Instead of using a callback, I directly access the data using `enumerate` to obtain both batch index and batch data, and manually process each step in the training loop.  The key is determining the `batch_index` within the epoch loop. The percentage is calculated before the next batch is processed. This approach sacrifices some of `model.fit` convenience but enables much greater control.

**Approach 3: Using `tf.keras.utils.Progbar`**

Another readily available option is `tf.keras.utils.Progbar`, a utility specifically designed for progress display. This method streamlines progress tracking without extensive custom implementation.

```python
import tensorflow as tf
import numpy as np

# Generate dummy data
train_data = np.random.rand(1000, 10)
train_labels = np.random.randint(0, 2, 1000)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)

# Calculate total batches
total_batches = len(dataset)

epochs = 5

for epoch in range(epochs):
  print(f"Epoch {epoch + 1}/{epochs}")
  progbar = tf.keras.utils.Progbar(total_batches)
  for batch_index, (x_batch, y_batch) in enumerate(dataset):
      metrics = model.train_on_batch(x_batch, y_batch, return_dict=True)
      progbar.update(batch_index + 1, values=metrics.items())
```

This method utilizes `Progbar` for a clean display of the percentage, in addition to training loss and accuracy. We initialize the `Progbar` with `total_batches` and call `.update()` within the epoch loop. `train_on_batch` replaces the custom training loop, offering a middle ground between `model.fit` and the manual loop, but the update method of the `Progbar` must still be invoked within the batch loop.

In summary, for a percentage-based completion indicator in TensorFlow, calculating the total number of batches is crucial. You can utilize `tf.keras.callbacks.Callback` for minimal interference, implement a custom training loop for finer control, or leverage `tf.keras.utils.Progbar` for a simpler solution. Each approach provides a way to track progress based on batch completion, but choosing the appropriate method hinges on the required level of control and the complexity of the training pipeline.

For more in-depth understanding, consult the TensorFlow documentation specifically on `tf.data`, custom training loops, `tf.keras.callbacks`, and `tf.keras.utils.Progbar`. Explore tutorials on custom training loops and callback development. Also, consider studying examples of `tf.data` usage in production environments for inspiration.
