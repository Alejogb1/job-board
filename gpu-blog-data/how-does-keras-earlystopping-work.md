---
title: "How does Keras EarlyStopping work?"
date: "2025-01-30"
id: "how-does-keras-earlystopping-work"
---
The core function of Keras' `EarlyStopping` callback is to monitor a training metric and halt the training process when that metric ceases to improve for a specified number of epochs, thereby preventing overfitting. Specifically, it observes the performance of the model on a validation set during training and interrupts the process when no improvement is observed.

The fundamental mechanism revolves around tracking a chosen monitor metric such as validation loss (`val_loss`) or validation accuracy (`val_accuracy`). `EarlyStopping` operates in a per-epoch fashion, evaluating this metric after each training epoch and comparing it to previously observed values. An “improvement” is defined based on the `mode` parameter: for `mode='min'`, improvement means a decrease in the metric (e.g., loss), whereas for `mode='max'`, improvement indicates an increase (e.g., accuracy). The `EarlyStopping` callback maintains an internal count of the number of epochs during which no improvement has been observed. This count is referred to as `patience`. If the `patience` threshold is reached, the training process is stopped using a `tf.keras.callbacks.TerminateOnNaN` operation. It doesn't modify the model's internal state; it merely stops the training loop.

The critical parameters that influence `EarlyStopping` behavior are:

*   **`monitor`**: The metric to be monitored. It must be a string representing one of the metrics computed by the model during training and validation, such as 'val_loss', 'val_accuracy', or a custom metric name.
*   **`min_delta`**: The minimum change in the monitored quantity to qualify as an improvement. This acts as a threshold to avoid early stopping due to negligible fluctuations in the metric. A common value is 0.0 for very precise metrics. If the improvement is lower than `min_delta`, it's not counted as an improvement.
*   **`patience`**: The number of epochs with no improvement after which training will be stopped.
*   **`verbose`**: Verbosity mode which prints a message about when early stopping occurs.
*   **`mode`**: Can be 'min', 'max', or 'auto'. Determines if the monitored quantity should be minimized or maximized. `auto` will attempt to infer from the monitored metric. It defaults to ‘auto’ which will infer from the metric.
*   **`baseline`**: Baseline value for the monitored quantity, considered as a minimum for stopping.
*   **`restore_best_weights`**: Whether to restore model weights from the epoch with the best value of the monitored quantity. Setting this to `True` is crucial for achieving the best model at early stop.

Now, let's examine how these parameters work in practice through specific examples.

**Example 1: Basic Early Stopping with Validation Loss**

Assume I am training a basic classification model on a dataset. I observe the validation loss decreasing during the first few epochs but beginning to plateau. Without EarlyStopping, this training might continue for a large number of epochs, potentially leading to overfitting.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Dummy model (replace with actual model)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(100,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback with patience of 3 and monitor 'val_loss'
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

# Dummy data
import numpy as np
x_train = np.random.random((1000, 100))
y_train = np.random.randint(0, 2, (1000, 1))
x_val = np.random.random((200, 100))
y_val = np.random.randint(0, 2, (200, 1))

# Training
history = model.fit(x_train, y_train,
                   validation_data=(x_val, y_val),
                   epochs=20, # Set to a number larger than your potential stop
                   batch_size=32,
                   callbacks=[early_stopping])

print(f"Early stopping triggered at epoch: {len(history.history['loss'])}")
```

In this snippet, the `EarlyStopping` callback is instantiated to monitor `val_loss`, aiming to minimize it. It has a `patience` of 3, meaning that if there's no improvement in the validation loss for three consecutive epochs, the training process will be terminated. I set `verbose` to 1 to be notified when early stopping occurred. It is worth noting that the provided epoch number in `model.fit` must be set higher than the expected early stop epoch. Running this code would likely show that training does not complete all 20 epochs.

**Example 2: Early Stopping with Validation Accuracy and `restore_best_weights`**

In this scenario, I'm focused on maximizing validation accuracy, and I want to revert to the model weights corresponding to the best accuracy achieved during training, instead of using the model weights of the last epoch before early stop.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Dummy model (replace with actual model)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(100,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback with patience of 5 and monitor 'val_accuracy'
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True, verbose=1)

# Dummy data (same as previous)
import numpy as np
x_train = np.random.random((1000, 100))
y_train = np.random.randint(0, 2, (1000, 1))
x_val = np.random.random((200, 100))
y_val = np.random.randint(0, 2, (200, 1))

# Training
history = model.fit(x_train, y_train,
                   validation_data=(x_val, y_val),
                   epochs=20,
                   batch_size=32,
                   callbacks=[early_stopping])

print(f"Early stopping triggered at epoch: {len(history.history['loss'])}")
```

Here, I monitor `val_accuracy` and set `mode` to `max`, as I’m aiming for the highest accuracy. Furthermore, by setting `restore_best_weights` to `True`, I ensure that, when training is terminated, the model will be reverted to the state when the highest validation accuracy was achieved, mitigating any slight performance degradation that may have occurred in the subsequent few epochs. This is a recommended configuration when aiming for the best model performance.

**Example 3: Using `min_delta` to Handle Minor Improvements**

In some scenarios, the validation loss might show marginal improvements that are practically insignificant. Using the `min_delta` parameter I can account for these marginal improvements, avoiding a premature early stop.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Dummy model (replace with actual model)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(100,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback with patience of 3, monitor 'val_loss', and min_delta of 0.01
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', min_delta=0.01, verbose=1)

# Dummy data
import numpy as np
x_train = np.random.random((1000, 100))
y_train = np.random.randint(0, 2, (1000, 1))
x_val = np.random.random((200, 100))
y_val = np.random.randint(0, 2, (200, 1))


# Training
history = model.fit(x_train, y_train,
                   validation_data=(x_val, y_val),
                   epochs=20,
                   batch_size=32,
                   callbacks=[early_stopping])

print(f"Early stopping triggered at epoch: {len(history.history['loss'])}")
```

In this example, `min_delta` is set to 0.01. This implies that only decreases of at least 0.01 in the validation loss are considered genuine improvements. If the loss decreases, but by less than 0.01, that epoch won’t reset the patience counter. This ensures that the training continues for a more suitable period, avoiding premature early stop.

**Resource Recommendations**

For further exploration and a more comprehensive understanding of the Keras callbacks, I suggest exploring the official Keras documentation. Specifically, review the API documentation for `tf.keras.callbacks.EarlyStopping`. Several online courses and tutorials also provide hands-on experience with the practical application of these techniques for better model training. Exploring different callback functions within the Keras API such as `ModelCheckpoint`, `ReduceLROnPlateau`, `Tensorboard` will provide a robust understanding of the Keras training process. Articles discussing best practices in neural network training also often discuss the implementation of EarlyStopping.
