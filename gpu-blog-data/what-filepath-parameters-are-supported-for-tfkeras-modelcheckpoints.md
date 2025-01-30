---
title: "What filepath parameters are supported for tf.Keras ModelCheckpoints?"
date: "2025-01-30"
id: "what-filepath-parameters-are-supported-for-tfkeras-modelcheckpoints"
---
The `tf.keras.callbacks.ModelCheckpoint` callback, in its handling of filepath parameters, exhibits behavior subtly dependent on the interaction between the provided string format and the underlying file system.  My experience troubleshooting inconsistent saving behavior across different operating systems and file structures led me to appreciate the importance of explicit path specification and the limitations imposed by wildcard characters.  Crucially, the `filepath` parameter does *not* support arbitrary shell-like expansions or globbing.  It relies on straightforward string formatting using Python's `str.format` functionality, potentially combined with OS-specific path separators.

**1. Clear Explanation:**

The `filepath` argument accepts a string template that determines the saved model's location and filename.  This template utilizes Python's string formatting capabilities, accepting several placeholders that are replaced during each epoch or at the end of training.  The key placeholders are:

* **`{epoch}`:**  Represents the current epoch number (starting from 0).
* **`{epoch:02d}`:**  Formats the epoch number with leading zeros to ensure consistent filename length (e.g., '00', '01', '10').  The `02d` specifies zero-padding to two digits.  More generally, `{epoch:0Xd}` uses `X` digits.
* **`{loss}`:**  Represents the training loss at the end of the epoch. The formatting is determined by the loss function's output and may include decimal places.  Similar placeholders exist for other metrics, depending on what is monitored (`monitor` parameter in `ModelCheckpoint`).
* **`{val_loss}`:** Represents the validation loss.  Again, the formatting is determined by the metric and may include decimals.
* **Other Metrics:**  If you monitor 'accuracy', 'val_accuracy' etc, then `{accuracy}` and `{val_accuracy}` will be available for string formatting.


The absence of a placeholder indicates a literal string that becomes part of the filename.  Therefore, carefully constructed strings are necessary to ensure consistent and correct file paths across different training runs. The framework does not interpret wildcard characters like `*` or `?` within the `filepath` parameter as shell-style wildcards for file selection.  Instead, they are treated as literal characters within the filename.  Furthermore, any operating system-specific path separators (e.g., `/` on Linux/macOS, `\` on Windows) must be explicitly included in the template.


**2. Code Examples with Commentary:**

**Example 1: Basic Saving**

```python
import tensorflow as tf

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(optimizer='adam', loss='mse')

checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch' #saves at every epoch
)

model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback])
```

This example shows a straightforward use of `{epoch:04d}` to create checkpoints named `cp-0000.ckpt`, `cp-0001.ckpt`, and so on within the `training_checkpoints` directory.  The `save_weights_only=True` parameter saves only the model's weights, making the checkpoints smaller.  The `save_freq='epoch'` saves the weights at the end of every epoch.  It's crucial to pre-create the `training_checkpoints` directory.  Failure to do so will likely result in an error.


**Example 2: Including Loss and Directory Structure**

```python
import tensorflow as tf
import os

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

checkpoint_path = "checkpoints/experiment_1/model_{epoch:02d}_{loss:.2f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=False,
    save_freq='epoch',
    monitor='val_mae', #Monitor validation MAE for saving
    save_best_only=True # Saves only the best weights based on validation MAE
)

model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[cp_callback])
```

This example demonstrates embedding the epoch number and the training loss (formatted to two decimal places) within the filename. The checkpoints are saved in a nested directory structure.  The addition of `save_best_only=True` and `monitor='val_mae'` showcases the conditional saving based on a validation metric.  Note the explicit path separation (`/`) which is crucial for cross-platform compatibility.  The `save_weights_only` is set to `False` implying that the entire model is saved, not just the weights.  This will create larger checkpoint files.


**Example 3: Handling Windows Paths**

```python
import tensorflow as tf
import os

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(optimizer='adam', loss='mse')

# Explicitly using os.path.join for Windows compatibility
checkpoint_path = os.path.join("C:\\", "checkpoints", "experiment_2", "model_{epoch:02d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch'
)

model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback])

```

This example highlights the importance of handling Windows paths correctly.  Using `os.path.join` ensures that the path separators are platform-independent, preventing errors on Windows systems.   This robust approach avoids hardcoding path separators, enhancing portability.


**3. Resource Recommendations:**

The official TensorFlow documentation on `ModelCheckpoint` is an indispensable resource.  Consult the documentation for details on all parameters and their interactions.  Exploring the Python `str.format` documentation will provide a thorough understanding of the formatting options available for the `filepath` parameter.  Finally, studying the examples provided in various TensorFlow tutorials focusing on model saving and loading will reinforce the practical aspects.  Understanding Python's `os` module, specifically `os.path.join` and `os.path.dirname`, is very beneficial for handling file paths correctly.
