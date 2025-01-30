---
title: "How can I name checkpoint files using epoch or batch numbers with ModelCheckpoint() and integer `save_freq`?"
date: "2025-01-30"
id: "how-can-i-name-checkpoint-files-using-epoch"
---
Implementing custom checkpoint naming schemes with Keras' `ModelCheckpoint` callback, particularly when relying on an integer `save_freq` representing batch intervals, requires careful consideration of how Keras manages its internal epoch and batch tracking. The core challenge arises because `ModelCheckpoint` primarily relies on epoch-based information unless explicitly configured otherwise, and directly injecting batch numbers into the filename structure necessitates understanding the nuances of callback execution order and variable scope. This response details a method leveraging a custom callback to achieve batch-specific file naming, offering a solution that avoids manual file path manipulations.

The standard `ModelCheckpoint` callback, while powerful, assumes a primary use case of saving after each epoch or after a certain number of epochs. Its internal logic, tied to the `on_epoch_end` callback method, makes batch-level naming problematic. Specifically, attempting to directly access the batch number within the `ModelCheckpoint`'s filename formatting string yields unexpected results due to the callback's default operating scope. To circumvent this, I implement a custom callback that intercepts the training process and injects the batch count data before `ModelCheckpoint` executes its saving procedure. This strategy allows us to construct file names reflecting the exact batch at which the model snapshot was captured.

My approach hinges on creating a new callback, `BatchAwareCheckpoint`, that inherits from `keras.callbacks.Callback`. This custom callback maintains an internal counter representing the total number of batches processed during training. It overwrites the `on_train_batch_end` method, updating this counter after each batch completes. Crucially, it also uses `set_params()` to dynamically update the `ModelCheckpoint`'s filename template with a batch-specific variable, immediately prior to `ModelCheckpoint`'s execution. By doing this, I effectively pass the current batch count into the context where `ModelCheckpoint` will use it for file naming. This avoids complex inter-callback communication. This approach differs from standard techniques and addresses a very specific and often complex scenario.

Let's consider an example using TensorFlow with Keras:

```python
import tensorflow as tf
import os

class BatchAwareCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_callback):
        super(BatchAwareCheckpoint, self).__init__()
        self.batch_count = 0
        self.checkpoint_callback = checkpoint_callback

    def set_params(self, params):
         super().set_params(params)
         self.checkpoint_callback.set_params(params)

    def on_train_batch_end(self, batch, logs=None):
         self.batch_count += 1
         self.checkpoint_callback.filepath = self.checkpoint_callback.filepath.format(batch=self.batch_count)
         self.checkpoint_callback.on_train_batch_end(batch, logs)
```

In this initial section of the custom callback, I establish the class, inheriting the necessary callback base, initialize the batch counter and store a reference to the original `ModelCheckpoint` callback. The `set_params` method ensures that any relevant training parameters are properly passed to the original callback as well. The crucial method here is `on_train_batch_end`. This method increases the batch count and dynamically updates the original `ModelCheckpoint`'s filepath. It effectively injects the `batch` variable with the current batch count into the format string prior to `ModelCheckpoint` being invoked. This will dynamically update the `ModelCheckpoint`'s filepath and allows for the batch count to be used in the filename.

The next step involves configuring and implementing the custom and standard callbacks within a training loop:

```python
def create_and_train_model(save_dir):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = os.path.join(save_dir,"model-batch-{batch}.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq=50,
        save_weights_only=True,
        verbose=0
    )

    batch_aware_callback = BatchAwareCheckpoint(checkpoint_callback)

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    model.fit(x_train, y_train, batch_size=32, epochs=3, callbacks=[batch_aware_callback])

    return model

```

This example showcases the practical setup for utilizing the `BatchAwareCheckpoint`. The `create_and_train_model` function demonstrates a concise model definition and compilation procedure for a demonstration. Critically, I instantiate the `ModelCheckpoint` with the filename template, using the `{batch}` format specifier, which is crucial for the integration with the custom callback. The custom `BatchAwareCheckpoint` is then instantiated, taking the original checkpoint callback as an argument, and passed into the training process as a callback. The `save_freq` is set to 50, meaning the model weights are saved every 50 batches, as required. The MNIST dataset is loaded and preprocessed for training. The model is then trained with the specified parameters. This ensures that the model is saving not every 50 epochs, but rather every 50 batches, as specified, using the custom callback.

Finally, to demonstrate an alternative naming format and illustrate its usage, consider this modified `create_and_train_model` function:

```python
def create_and_train_model_alt(save_dir):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    checkpoint_path = os.path.join(save_dir,"model_batch_at_{batch:05d}.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq=50,
        save_weights_only=True,
        verbose=0
    )

    batch_aware_callback = BatchAwareCheckpoint(checkpoint_callback)

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


    model.fit(x_train, y_train, batch_size=32, epochs=3, callbacks=[batch_aware_callback])
    return model
```

This modified function demonstrates a slightly different filepath format specification. I have specified `{batch:05d}` to zero-pad the batch number to five digits, allowing for easier sorting and organization of the saved checkpoints. This demonstrates the flexible formatting capabilities of Python's string formatting. The other operations and setup remain the same as the prior example. The resulting file names after running would be something like `model_batch_at_00050.h5`, `model_batch_at_00100.h5`, and so on. This allows for clear and consistent naming of the saved checkpoint files.

For further study, I recommend reviewing the official Keras documentation on callbacks, particularly the `keras.callbacks.Callback` class and the `keras.callbacks.ModelCheckpoint` class. Investigating the implementation details of these classes through the TensorFlow source code will also improve understanding of how callbacks function. Lastly, exploring practical examples and tutorials on utilizing custom callbacks for specific training needs will help in creating more specialized and efficient workflows. Understanding the execution order and variable scoping is critical for a proper implementation when using the callback structure for non-standard use cases.
