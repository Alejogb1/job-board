---
title: "What untracked functions are causing the 'WARNING:absl' message?"
date: "2025-01-30"
id: "what-untracked-functions-are-causing-the-warningabsl-message"
---
The root cause of "WARNING:absl:..." messages often lies in the interaction between your application's code and TensorFlow's (or related libraries using absl) internal logging mechanisms, specifically when functions or operations aren't explicitly tracked or managed by TensorFlow’s computational graph or its eager execution environment. These untracked elements can trigger default logging behavior, hence the warning.

In my experience, particularly during the transition from TensorFlow 1.x graph-based programming to 2.x eager execution, these warnings became more prominent. This shift, while increasing flexibility, often meant that certain utility functions, data processing steps, or even custom model components were operating outside the explicit management of TensorFlow's automatic differentiation and graph construction, generating these absl warnings. Essentially, the warning signals that TensorFlow is encountering an operation or function it didn't expect or doesn't have visibility into during its normal execution lifecycle.

The absl library (part of the Google Abseil project) serves as a foundational library for many Google projects, including TensorFlow. It provides logging infrastructure, among other things. The "absl" namespace in the warning message signifies that it originates from this library. When a TensorFlow operation is invoked, it is typically wrapped within a TensorFlow context allowing for features such as automatic differentiation, graph tracing, and optimization. However, if some function within your workflow, whether custom-built or borrowed from another library, bypasses this TensorFlow context, absl's default logging mechanism kicks in and produces a warning. This usually isn't indicative of a critical failure, but rather an indicator that best practices are not being followed to effectively manage and track all function calls within your TensorFlow environment, especially in situations where these calls might be a part of the gradients calculation.

A common scenario where such warnings occur is when data preprocessing is performed outside the scope of TensorFlow’s graph building or Tensor operations. Consider a custom data loading and augmentation process that incorporates functions from non-TensorFlow libraries, like PIL for image manipulations or SciPy for numerical transformations. These operations, while effective for data preparation, are not part of TensorFlow’s internal tracking. When these untracked operations are invoked during training, the absl logging kicks in. This is because the changes in data aren't recorded within the TensorFlow graph. Here's a demonstrative code example:

```python
import tensorflow as tf
from PIL import Image
import numpy as np

def load_and_augment_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    # Untracked operations using PIL and NumPy
    resized_image = Image.fromarray(image_array).resize((256, 256))
    resized_array = np.array(resized_image)
    return resized_array / 255.0

@tf.function
def process_image(image_path):
    image = tf.py_function(load_and_augment_image, [image_path], tf.float32)
    return image

image_path = "dummy_image.png" # Replace with your actual image path
dummy_image = np.random.rand(64,64,3).astype(np.uint8) # Create dummy image for testing
Image.fromarray(dummy_image).save(image_path)

processed_image = process_image(image_path)
```

In this snippet, `load_and_augment_image` utilizes PIL and NumPy. Even though `process_image` is wrapped by `@tf.function` for graph compilation, the internal `load_and_augment_image` function isn’t part of TensorFlow’s graph; therefore, the operations within it aren’t tracked. This typically triggers the absl warning during the first execution of the graph. To address this, you could integrate TensorFlow's image manipulation functions as much as possible, such as `tf.image.resize` rather than using PIL directly. If non-TensorFlow operations are indispensable, encapsulate them within a `tf.py_function` and ensure that the input and output types are specified, as demonstrated in this corrected code, which hides the absl warning by informing TensorFlow of the function’s execution. However, even `tf.py_function` can still trigger a warning if it is being re-traced often and is deemed inefficient. It is more of a bandage than a true fix.

Another common instance where untracked functions generate this warning lies in custom loss or metrics calculations that deviate from the built-in TensorFlow losses and metrics API. Suppose you have a custom loss function involving non-TensorFlow operations:

```python
import tensorflow as tf
import numpy as np

def custom_loss(y_true, y_pred):
    difference = y_true - y_pred
    # Untracked operations using Numpy
    squared_difference = np.power(difference,2)
    loss_value = np.mean(squared_difference)
    return loss_value

class MyModel(tf.keras.Model):
    def __init__(self, num_units=32):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(num_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)


model = MyModel()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = custom_loss(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
    
x = tf.random.normal((32,10))
y = tf.random.normal((32,1))

loss = train_step(x,y)
```

Here, the `custom_loss` function employs NumPy, and it's called inside a training step wrapped in a `tf.function`, therefore not tracked by the gradients tape properly. Because TensorFlow doesn’t have a clear picture of its computation, which is a pre-requisite for backpropagation, the absl logger defaults to providing a warning. To resolve this, the loss function should use TensorFlow operations exclusively. If a custom loss is truly necessary, derive it from `tf.keras.losses.Loss` or use `tf.function` with careful type management, which requires a deeper understanding of the `tf.function` specifics. A corrected version would include something like:

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_units=32):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(num_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)
    
class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        squared_difference = tf.math.square(difference)
        loss_value = tf.reduce_mean(squared_difference)
        return loss_value


model = MyModel()
optimizer = tf.keras.optimizers.Adam()
loss_object = CustomLoss()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_object(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
    
x = tf.random.normal((32,10))
y = tf.random.normal((32,1))

loss = train_step(x,y)
```

Finally, similar issues arise when dealing with external model evaluation or metric computation outside TensorFlow. If a custom evaluation function that computes metrics uses non-TensorFlow operations, then the warnings will occur. If using Keras, override its `metrics` parameter rather than using a function outside its scope. A model's custom layers are another example. If a layer definition includes calls to non-TensorFlow libraries, the absl warning may appear, even if the layers are used as part of a `tf.keras.Model`. These layers need to be wrapped in `tf.py_function` and/or the layer logic needs to be converted to use TensorFlow operations.

In summary, to effectively debug "WARNING:absl" messages, focus on understanding the scope of your operations within TensorFlow's execution model. Key areas to scrutinize include your custom data handling pipelines, custom loss or metrics functions, any utility functions or layers, especially those involving non-TensorFlow components such as NumPy, SciPy or PIL, particularly when used inside `tf.function`. Using TensorFlow APIs for equivalent computations or encapsulating non-TensorFlow operations with `tf.py_function` is often necessary. However, the most performant and accurate approach is to use TensorFlow's built-in operations. Additionally, it helps to check that training/validation loops and other code paths leverage TF's API appropriately, such as using `tf.keras.losses`, `tf.keras.metrics`, `tf.data.Dataset` classes and so on. For further reading, the official TensorFlow documentation is essential, particularly the sections on Eager Execution, the `tf.function` decorator, and the Keras API. Also, the Abseil C++ project documentation (although not directly Python) can give further understanding of the origins of the logging. The TensorFlow official GitHub repository contains valuable discussions about such warnings and how best to eliminate them.
