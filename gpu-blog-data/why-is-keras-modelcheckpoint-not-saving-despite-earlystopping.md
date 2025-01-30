---
title: "Why is Keras ModelCheckpoint not saving despite EarlyStopping working with the same monitoring argument?"
date: "2025-01-30"
id: "why-is-keras-modelcheckpoint-not-saving-despite-earlystopping"
---
The discrepancy between Keras' `ModelCheckpoint` and `EarlyStopping` callbacks, where the latter triggers but the former fails to save a model despite using identical monitoring metrics, often stems from a subtle misunderstanding of how these callbacks interact with the training process and the state of the model weights.  My experience troubleshooting this issue across numerous deep learning projects, particularly involving complex architectures and custom training loops, points to several potential root causes.

**1.  Understanding the Asynchronous Nature of Callback Execution:**

`EarlyStopping` and `ModelCheckpoint` are callbacks; they execute during the training process at specific points, based on the monitored metric.  However, their execution is asynchronous relative to the model's weight updates.  `EarlyStopping` checks the monitored metric after each epoch (or batch, depending on the `monitor` and `frequency` arguments). If the stopping criterion is met, it signals the training loop to halt.  Crucially, the weights *before* the final epoch are often *not* saved, even if they represent the best performance. `ModelCheckpoint`, on the other hand, performs a saving operation only if the *current epoch's* performance surpasses the previously saved checkpoint.  This is where the problem often arises: if `EarlyStopping` triggers before `ModelCheckpoint` has a chance to save the best weights observed so far, no model will be saved, even though the `EarlyStopping` callback successfully identified the optimal performance.

**2.  Inconsistencies in Metric Calculation and Reporting:**

Another frequent source of this problem is discrepancies between how the monitored metric is calculated within the model's training loop and how it's reported to the callbacks.   For instance, if you're using custom metrics, ensure they're properly implemented and that the values reported to the callbacks accurately reflect the model's actual performance.  A common oversight is a mismatch between the metric's name as defined in the model's `compile` method and the `monitor` argument in the callback.  Minor discrepancies in calculation, such as rounding errors, can lead to inconsistencies that prevent `ModelCheckpoint` from triggering while `EarlyStopping` still functions correctly.  In my experience, this is amplified when dealing with custom loss functions involving complex mathematical operations.


**3.  Incorrect Callback Configuration:**

A seemingly simple oversight is incorrectly specifying the `save_best_only` parameter in `ModelCheckpoint`. While the default is `True`, setting it to `False` will save a model after every epoch regardless of improvement, sometimes overwhelming storage capacity. However, if mistakenly set to `True`,  even if EarlyStopping triggers, no model is saved if the final epoch's metric is not the best observed overall during the training process.


**Code Examples and Commentary:**

Let's illustrate these issues with three code examples using TensorFlow/Keras:

**Example 1: Incorrect Metric Monitoring**


```python
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Define a simple sequential model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Incorrect metric name in ModelCheckpoint
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) #Note: 'accuracy' is the correct metric


checkpoint = ModelCheckpoint('best_model.h5', monitor='acc',  # Incorrect metric name
                             save_best_only=True, mode='max') #Should be 'accuracy'
early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max')


# Training process
model.fit(x_train, y_train, epochs=10,
          validation_data=(x_val, y_val),
          callbacks=[checkpoint, early_stopping])

```

In this example, the `ModelCheckpoint` is configured with an incorrect metric name ('acc' instead of 'accuracy').  This leads to `ModelCheckpoint` failing to monitor the correct metric, even though `EarlyStopping` functions correctly.  This demonstrates the importance of consistent naming across the model compilation and the callbacks.


**Example 2:  Asynchronous Execution and `save_best_only`**

```python
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

# ... (model definition as before) ...

checkpoint = ModelCheckpoint('best_model.h5', monitor='accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='accuracy', patience=1, mode='max')

# Training process
model.fit(x_train, y_train, epochs=10,
          validation_data=(x_val, y_val),
          callbacks=[checkpoint, early_stopping])

```

Here, even with the correct metric name, the asynchronous nature of the callbacks can cause problems.  If the best accuracy is achieved in an epoch *before* the last epoch and `EarlyStopping` triggers due to lack of improvement, `ModelCheckpoint` might not save anything because `save_best_only=True`. The solution might involve increasing the patience of `EarlyStopping`.


**Example 3:  Custom Metric Handling**

```python
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

# ... (model definition as before) ...

def custom_metric(y_true, y_pred):
    #Implementation of a custom metric
    return tf.reduce_mean(tf.abs(y_true-y_pred))


model.compile(optimizer='adam',
              loss='mse',
              metrics=[custom_metric])


checkpoint = ModelCheckpoint('best_model.h5', monitor='custom_metric', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='custom_metric', patience=3, mode='min')

# Training process
model.fit(x_train, y_train, epochs=10,
          validation_data=(x_val, y_val),
          callbacks=[checkpoint, early_stopping])
```

This illustrates the need for meticulous custom metric implementation and ensuring its correct interaction with the callbacks.  The `mode` parameter should accurately reflect whether the metric should be minimized or maximized for optimal performance.


**Resource Recommendations:**

For a thorough understanding of Keras callbacks, I recommend referring to the official Keras documentation.  The TensorFlow documentation also offers valuable insights into the workings of the TensorFlow backend.  Finally, studying examples from established deep learning libraries and reviewing related Stack Overflow discussions can provide practical experience.  Careful attention to detail and a systematic approach to debugging are critical to resolving these issues.
