---
title: "Do Keras callbacks like EarlyStopping utilize training history before a crash?"
date: "2025-01-30"
id: "do-keras-callbacks-like-earlystopping-utilize-training-history"
---
Keras callbacks, including `EarlyStopping`, operate by observing metrics generated *during* the training process. They are not directly accessing a persistent training history beyond what is available within the current training run. A critical point to understand is that these callbacks are reactive, acting on the output of each training epoch rather than consulting a stored record of past, potentially failed, sessions. My experience designing training pipelines for large language models has reinforced this understanding; loss, accuracy, and other monitored metrics are calculated on the fly, and callbacks make decisions based solely on these live values. A crashed training session, by its very nature, does not finalize or output metric data the callbacks can act upon in a subsequent session.

To explain further, the `EarlyStopping` callback, specifically, monitors a designated metric (typically validation loss) and terminates training when that metric ceases to improve over a specified number of epochs (the 'patience' parameter). The callback's logic is entirely contained within the scope of a single model.fit() call. It maintains internal state variables, such as the best metric value encountered so far and the number of epochs without improvement, but these are all reset whenever a new `model.fit()` call is executed. If a crash occurs mid-training, the information that the callback had accumulated is lost, along with the state of the model and optimizer, unless these were explicitly checkpointed (which is a separate process). The callback is not designed to, nor does it, have the mechanism to query persistent data from a previous, incomplete training attempt.

To clarify, consider a hypothetical scenario where you train a Keras model, and then experience a system failure before training concludes. If `EarlyStopping` was enabled, it would have recorded the best validation loss and number of epochs without improvement up until the point of failure, but this is transient data, not preserved between training sessions. Starting a new training session will effectively re-initialize the callback. It begins monitoring the metric from epoch 1 of this new session, unaware of previous attempts. Checkpointing of the *model state*, and separately, the *history* returned by the training is crucial to resume training from a specific point. Keras callbacks, themselves, do not persist state beyond a single training execution.

Let me illustrate this with code examples.

**Example 1: Basic EarlyStopping**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Simplified data for demonstration
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
x_val = np.random.rand(20, 10)
y_val = np.random.randint(0, 2, 20)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# First training attempt (hypothetically crashes before completion)
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[early_stopping])


# Second training attempt after crash, early_stopping is re-initialized
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[early_stopping])

print(history.history)
```

This first example illustrates a basic setup. We create a simple model and compile it. The `EarlyStopping` callback is instantiated with a 'patience' of 3, meaning training will stop if the validation loss doesn’t improve for three consecutive epochs. If the first attempt (commented out) hypothetically crashes before completion, the callback would have stored its internal state, *temporarily*. The subsequent `model.fit()` call initializes the callback anew. The print statement showing `history.history` reveals only metrics from the current training execution. No data pertaining to the hypothetical crashed attempt is included. The callback starts fresh, unaware of the lost training data and state.

**Example 2: EarlyStopping with ModelCheckpoint**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Simplified data for demonstration
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
x_val = np.random.rand(20, 10)
y_val = np.random.randint(0, 2, 20)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Training (potentially crash)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[early_stopping, checkpoint])

# Load model from checkpoint to start from best model
loaded_model = keras.models.load_model('best_model.h5')

# Further training
history_second_training = loaded_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[early_stopping, checkpoint])
print(history_second_training.history)
```

Here, I've added `ModelCheckpoint` alongside `EarlyStopping`. The `ModelCheckpoint` callback saves the model weights whenever an improvement is noted in `val_loss`. If a crash happens during the initial training, only the model’s last saved weights are available if checkpointing was enabled. While the `EarlyStopping` callback still starts fresh with each `fit()` call, loading the model from the checkpoint ensures you're restarting from a better point than starting the model training from scratch, and potentially resuming some level of previous learning. The second training's `history` will only reflect the metrics calculated within that `fit` call.

**Example 3: Visualizing Training History**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Simplified data for demonstration
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
x_val = np.random.rand(20, 10)
y_val = np.random.randint(0, 2, 20)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# First training run
history_first = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[early_stopping])

# Second training run
history_second = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[early_stopping])


# Plotting loss from both runs
plt.figure(figsize=(10,5))
plt.plot(history_first.history['loss'], label='First run training loss')
plt.plot(history_first.history['val_loss'], label='First run validation loss')
plt.plot(history_second.history['loss'], label='Second run training loss')
plt.plot(history_second.history['val_loss'], label='Second run validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss from different training runs')
plt.show()

```

This example demonstrates that while a history object is returned by the `model.fit()` call, each is tied to that specific call. The code shows the output of two training runs visually.  Notice that each `history` object (`history_first` and `history_second`) only reflects the training metrics for their respective calls, further evidencing the lack of continuity. The callbacks are operating independently each time the training process is initialized. The plotted histories have no knowledge of each other, regardless of whether early stopping occurred previously.

In conclusion, Keras callbacks like `EarlyStopping` do *not* utilize training history from previous, crashed sessions. They function within the scope of a single `model.fit()` call. For preserving progress across interruptions, utilizing `ModelCheckpoint` to save the best model weights, coupled with using the history returned from each call to fit, are the recommended approaches.  For further reading on these techniques, consult the official Keras documentation sections regarding callbacks and saving models, as well as general texts on deep learning which cover concepts of model persistence. These resources offer detailed explanations of these mechanisms.
