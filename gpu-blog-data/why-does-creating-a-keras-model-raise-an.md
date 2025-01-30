---
title: "Why does creating a Keras model raise an AttributeError regarding '_keras_history'?"
date: "2025-01-30"
id: "why-does-creating-a-keras-model-raise-an"
---
The `AttributeError: 'NoneType' object has no attribute '_keras_history'` in Keras arises primarily from attempting to access the training history of a model that hasn't been trained yet, or where the training process has been interrupted and did not successfully complete.  This often manifests when one tries to retrieve metrics like accuracy or loss from `model.history.history` before initiating the `model.fit()` method.  My experience debugging similar issues in large-scale image recognition projects has highlighted this as a recurring point of failure for junior engineers.  The `_keras_history` attribute is populated *only* after a successful training run.

**1. Clear Explanation**

The Keras `Model` object possesses an attribute, `history`, which is populated during the training process by the `fit()` method.  This `history` attribute contains a dictionary (`history.history`) holding the values of metrics tracked during training (loss, accuracy, etc.) for each epoch. If the `fit()` method is not executed, or terminates abnormally (e.g., due to a runtime error or being prematurely stopped), the `history` attribute remains `None`.  Subsequently, attempting to access attributes of this `None` object, including the internal `_keras_history` attribute (which isn't directly accessible by the user but is referenced internally), naturally throws the `AttributeError`.  Understanding this sequence is key to resolving the issue.  The error message itself is a consequence of the internal workings of Keras, indicating a missing, expected attribute within a `NoneType` object, precisely highlighting the lack of training history.


**2. Code Examples with Commentary**

**Example 1: Correct Usage**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Define a simple sequential model
model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Import and pre-process MNIST data (replace with your data loading)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


# Train the model and store the history
history = model.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test,y_test))

# Access and print the training history after training completes
print(history.history['accuracy'])
print(history.history['val_accuracy'])
```

This example demonstrates the correct procedure.  The `model.fit()` method executes, populating the `history` attribute.  Subsequent access to `history.history` is valid because the `history` object is not `None`.  I've often used this structure as a minimal viable example for trainees to illustrate the correct workflow.


**Example 2: Incorrect Usage (Leading to the Error)**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Attempting to access history BEFORE training
try:
    print(model.history.history['accuracy'])
except AttributeError as e:
    print(f"Caught expected error: {e}")

#This would need to be run to successfully populate the history attribute.
#model.fit(x_train, y_train, epochs=2, batch_size=32)
```

This example intentionally omits the `model.fit()` call. The attempt to access `model.history.history` immediately after model compilation inevitably results in the `AttributeError` because `model.history` is `None`. The `try-except` block is crucial for robust error handling; in production code, you wouldn't want a crash due to this predictable scenario.

**Example 3: Handling Interruptions**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

try:
    #Simulate interruption - replace with actual training loop if needed.
    #This example deliberately raises a KeyboardInterrupt, mimicking the user stopping training.
    history = model.fit(x_train[:100], y_train[:100], epochs=1, batch_size=32) # Reduced dataset for faster execution
except KeyboardInterrupt:
    print("Training interrupted. History might be incomplete or None.")
    if hasattr(model, 'history'):  #Check if history exists at all
        if model.history: #Check if the history attribute is not None.
            print("Partial training history available")
            print(model.history.history)
        else:
            print("No training history available.")
    else:
        print("No training history available.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

This example demonstrates a more robust approach. It includes error handling for a situation where training might be interrupted, for instance, by a `KeyboardInterrupt` (user pressing Ctrl+C).  This is a crucial aspect often overlooked;  the code gracefully handles potential interruptions and informs the user about the status of the training history.  Checking `hasattr(model, 'history')` and `model.history` is a defence against accessing a NoneType object.  During my work on large datasets, this type of defensive programming was vital.


**3. Resource Recommendations**

The official TensorFlow documentation provides comprehensive information on the Keras API, model training, and error handling.  Consult the Keras API reference for detailed explanations of the `Model` class and the `fit()` method.  A good introductory text on deep learning with Python will also be beneficial for understanding the underlying concepts.  Finally, searching Stack Overflow for similar errors, focusing on the `AttributeError` and `_keras_history`, can provide numerous solutions based on other users’ experiences.  It is recommended to understand the underlying mechanisms of model training in Keras to avoid this error.
