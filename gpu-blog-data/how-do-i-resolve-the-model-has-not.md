---
title: "How do I resolve the 'model has not yet been built' error in Keras?"
date: "2025-01-30"
id: "how-do-i-resolve-the-model-has-not"
---
The "model has not yet been built" error in Keras stems from attempting to utilize a model before its architecture has been definitively defined and compiled.  This isn't merely a matter of instantiating a model class; the network's layers must be added, and the compilation step, specifying the optimizer, loss function, and metrics, must be completed.  Over the years, I've debugged countless instances of this, often stemming from subtle sequencing errors or misunderstandings about Keras's functional and sequential API.  My experience with large-scale image classification projects has made me acutely aware of this common pitfall.

**1. Clear Explanation:**

Keras, a high-level API for building and training neural networks, relies on a sequential or functional approach to model construction.  The sequential API, simpler for linearly stacked layers, requires that layers are added before any operations that depend on the model's structure, such as `model.summary()`, `model.fit()`, or `model.predict()`.  The functional API offers more flexibility for complex architectures involving branching or shared layers but requires a more explicit definition of the input and output tensors before compilation.  The "model has not yet been built" error is triggered when a method requiring a fully defined and compiled model is called prematurely. This usually means one of the following:

* **Missing layer addition:** The model instance has been created, but no layers have been added using `model.add()` (sequential API) or by defining the connections between layers (functional API).  The model is essentially an empty container.
* **Missing compilation:** Even with layers defined, the model lacks a specified optimizer, loss function, and metrics.  The `model.compile()` method is essential to define the training process.  Without it, Keras lacks the information to execute the training or prediction steps.
* **Incorrect layer ordering or connections (Functional API):** In the functional API, the input tensor must be clearly defined and propagated through the network correctly.  Incorrect connections or missing connections will prevent the model from being built.
* **Incorrect input shape definition:** The input shape must be specified accurately, either during layer definition or when calling `model.build()`.  A mismatch between the expected and actual input shape can lead to errors during model building.


**2. Code Examples with Commentary:**

**Example 1: Sequential API - Missing Compilation**

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# This line will cause the error because the model hasn't been compiled.
# model.fit(x_train, y_train, epochs=10)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10) #Now this will work
```

**Commentary:** This example demonstrates a common mistake: attempting to train the model (`model.fit()`) before compiling it (`model.compile()`). The `model.compile()` method is crucial; it informs Keras how to update the model's weights during training.  Without it, the `model.fit()` method fails, resulting in the "model has not yet been built" error, or a more descriptive variant pointing to the missing compilation step.  I've personally spent hours troubleshooting issues similar to this, mainly during rapid prototyping where I forgot this crucial step.

**Example 2: Functional API - Incorrect Input Shape**

```python
import tensorflow as tf
from tensorflow import keras

input_tensor = keras.Input(shape=(28,28,1))
x = keras.layers.Conv2D(32, (3,3), activation='relu')(input_tensor)
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Flatten()(x)
output_tensor = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=input_tensor, outputs=output_tensor)

# This would trigger an error if x_train doesn't match the input shape (28, 28, 1)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10)

#Correct input handling
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
```

**Commentary:** This showcases the functional API's flexibility but also its sensitivity to input shape.  The input tensor (`input_tensor`) must be correctly specified, and the data fed to the model (`x_train`) needs to match this shape precisely. Failure to do so will result in shape mismatches during model building, often manifested as the "model has not yet been built" error. In my experience with convolutional neural networks (CNNs), this is a very common error, particularly when dealing with image data where the dimensions (height, width, channels) must be carefully managed. The proper reshaping and normalization of `x_train` is critical for correct input.

**Example 3: Sequential API - Missing Layers**

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()

# This is missing layer addition; the model is empty.
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(x_train, y_train, epochs=10)

model.add(keras.layers.Dense(64, activation='relu', input_shape=(10,)))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=10) #Now the model is defined and compiled properly.
```

**Commentary:** This illustrates the simplest case – a completely empty sequential model.  Attempting to compile or train such a model will trigger the error because no computational units (layers) are present.  This is a fundamental mistake easily avoided by ensuring at least one layer is added to the model before any other operation. I've seen this happen particularly often when working with students new to Keras; they often forget this initial layer addition.


**3. Resource Recommendations:**

The official Keras documentation provides comprehensive details on both the sequential and functional APIs.  Referencing the documentation for layer-specific parameters and configurations is essential.  Furthermore, consult a reliable textbook on deep learning fundamentals.  These resources offer broader context on neural network architecture and training procedures, which are highly relevant to understanding and avoiding such errors.  Finally, explore introductory tutorials on Keras available online; these can help solidify understanding through practical examples.
