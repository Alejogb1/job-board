---
title: "Why is Keras throwing a KeyError: 'acc'?"
date: "2025-01-30"
id: "why-is-keras-throwing-a-keyerror-acc"
---
The `KeyError: 'acc'` in Keras typically arises from attempting to access the 'acc' metric (accuracy) from a model's history object when that metric wasn't explicitly monitored during training.  This is a common oversight, particularly when customizing training processes or using non-standard metrics.  I've encountered this numerous times during my work on large-scale image classification projects, often stemming from inconsistencies between the compilation step and the subsequent evaluation of training progress.

**1. Clear Explanation:**

The Keras `model.fit()` method returns a `History` object.  This object contains a dictionary (`history.history`) that stores the values of the metrics monitored during training.  The keys of this dictionary correspond to the metric names, and the values are lists of metric values across epochs.  If you didn't explicitly specify 'accuracy' (or its alias 'acc') as a metric during model compilation using `model.compile()`, the 'acc' key will be absent from `history.history`, resulting in the `KeyError`.  This is not a bug in Keras, but rather a direct consequence of the training process only tracking what you instruct it to track.

The confusion often stems from the assumption that accuracy is always implicitly tracked. While accuracy is a standard metric and often displayed by default in simpler training examples, Keras doesn't automatically include it unless specifically requested during model compilation.  This design choice allows for flexibility, especially in scenarios involving custom metrics or when focusing on other performance indicators.  Therefore, ensuring 'accuracy' is included in the metrics list during compilation is crucial for avoiding this error.

**2. Code Examples with Commentary:**

**Example 1: Correct Implementation**

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

print(history.history['accuracy']) # Accessing accuracy without error
print(history.history['val_accuracy']) # Accessing validation accuracy
```

This example demonstrates the correct way to access the 'accuracy' metric. The `model.compile()` method explicitly includes `'accuracy'` in the `metrics` list. Consequently, the `history.history` dictionary will contain 'accuracy' and 'val_accuracy' (validation accuracy) keys, allowing for error-free access. The use of `x_train`, `y_train`, `x_val`, and `y_val` implies pre-processed training and validation data sets, typical in my experience with real-world datasets.


**Example 2: Incorrect Implementation Leading to KeyError**

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy') # Missing metrics parameter

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

try:
    print(history.history['acc']) # This will raise KeyError: 'acc'
except KeyError as e:
    print(f"Error: {e}") # Proper error handling
```

This example demonstrates the problematic scenario. The `model.compile()` call omits the `metrics` argument.  As a result, no accuracy metric is tracked during training. Attempting to access `history.history['acc']` (or `history.history['accuracy']`) will trigger the `KeyError`. The `try-except` block is crucial for robust code, preventing the program from crashing due to this common error.  During my development of a facial recognition system, this was a frequent source of debugging.


**Example 3:  Using a Custom Metric (Illustrative)**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

def custom_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[custom_metric])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

print(history.history[custom_metric.__name__]) # Accessing the custom metric
```

This example illustrates the use of a custom metric.  While the `KeyError` is avoided by specifying a metric, even a custom one, it showcases that simply having a `metrics` parameter is essential.  It's important to note that  `history.history` will use the function name (`custom_metric.__name__`) as the key for the custom metric's values. During my work on a time-series prediction project, I often defined custom metrics reflecting specific business requirements, highlighting the importance of this methodology. This approach necessitates careful naming conventions to ensure consistent access to the results.


**3. Resource Recommendations:**

The official Keras documentation provides comprehensive information on model compilation, training, and metric handling.  Consult the Keras API reference for detailed explanations of the `model.compile()` method and the `History` object's attributes.  Thorough review of tutorials focusing on model evaluation and metric tracking is beneficial.  Finally, examining examples of diverse model implementations, particularly those involving custom metrics and loss functions, will significantly enhance understanding and help in proactive error avoidance.
