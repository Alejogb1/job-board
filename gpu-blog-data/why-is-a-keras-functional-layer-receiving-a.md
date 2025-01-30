---
title: "Why is a Keras Functional layer receiving a non-tensor input?"
date: "2025-01-30"
id: "why-is-a-keras-functional-layer-receiving-a"
---
Keras Functional API models, by design, expect and operate on tensors, the fundamental data structure for numerical computation in TensorFlow and Keras. Receiving a non-tensor input, such as a Python list or a NumPy array directly within the forward pass of a Functional layer signifies an improper interface with the underlying computational graph. This disconnect usually arises from misinterpreting how Keras, specifically the Functional API, manages data flow. I’ve encountered this in several projects, most notably when attempting to mix imperative NumPy operations directly into a network’s pipeline without proper conversion.

The core principle behind a Functional API model is that it constructs a symbolic computational graph. Each layer transforms the output of the previous layer, treating it as a symbolic tensor. When a layer receives non-tensor data, it means that the expected tensor input either hasn't been created correctly or hasn't been passed down to that particular layer. This can arise in a few scenarios, broadly categorized into issues concerning input placeholders, data type mismatches, and attempts at procedural manipulation within the graph definition.

Let's consider a common error, related to placeholder specification. When starting a Functional API model, one begins with `keras.Input`, which defines the *shape* and *datatype* of the expected tensor, creating a symbolic placeholder within the graph. If subsequent layers are not properly connected to this initial input tensor or a tensor derived from it, they’ll instead likely receive the raw data type used during initial testing. I observed this while rapidly prototyping an image processing model. My input layer was defined using `Input(shape=(256,256,3))`, but in my eagerness to move forward, I passed in a plain NumPy array of shape `(1,256,256,3)`, without ensuring it was wrapped into a tensor by TensorFlow, thereby bypassing the graph connections I had previously defined. This resulted in an error, because the first convolutional layer expected to receive a tensor derived from the symbolic graph defined by `Input`, not a raw NumPy object.

Here's a code example demonstrating this common pitfall:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Define the input shape
input_tensor = keras.Input(shape=(256, 256, 3))

# 2. Define the first conv layer
conv_layer = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(input_tensor)

# 3. Define the second conv layer
conv_layer_2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(conv_layer)

# 4. Define the model
model = keras.Model(inputs=input_tensor, outputs=conv_layer_2)

# 5. Create dummy data as a NumPy array, not a tensor
dummy_data = np.random.rand(1, 256, 256, 3).astype(np.float32)


# 6. Correctly transform to Tensor
dummy_data_tensor = tf.convert_to_tensor(dummy_data)

# 7. Attempt prediction - this causes the non-tensor input problem if using dummy_data
# prediction = model(dummy_data)
prediction = model(dummy_data_tensor)

print(prediction)
```
In this example, passing `dummy_data` directly to the model would cause an error during the forward pass of the first convolutional layer, because it expects a tensor derived from the symbolic graph. `tf.convert_to_tensor` remedies the issue, correctly converting the NumPy array into a Tensor before using it as an input to the model. It is essential to always feed the graph with tensors, either derived from `keras.Input` or using `tf.convert_to_tensor` after processing.

Another instance where I’ve seen non-tensor input issues occur is when attempting to perform pre-processing *inside* the model definition. For example, suppose you wish to standardize your input data. While you might be tempted to use NumPy for its vectorized nature, this breaks the graph construction since standardizing using NumPy outside of the Tensor environment transforms the data into a NumPy array rather than a Tensor. The following code segment exhibits this error.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Input shape and data
input_tensor = keras.Input(shape=(10,))
dummy_data = np.random.rand(1, 10).astype(np.float32)

# 2. Define a layer that should receive a tensor
dense_layer = keras.layers.Dense(units=5, activation='relu')

# 3. Incorrect Preprocessing within model definition - not creating a Tensor
mean = np.mean(dummy_data)
std = np.std(dummy_data)
scaled_data_np = (dummy_data-mean)/std

# 4. Correct Preprocessing using tensorflow
scaled_data_tensor = tf.math.divide(tf.math.subtract(tf.convert_to_tensor(dummy_data), tf.math.reduce_mean(tf.convert_to_tensor(dummy_data))), tf.math.reduce_std(tf.convert_to_tensor(dummy_data)))

# 5. Attempt inference
# This will error with non-tensor input using scaled_data_np
# output_np = dense_layer(scaled_data_np)
output_tensor = dense_layer(scaled_data_tensor)

print(output_tensor)
```
In this example, attempting to scale `dummy_data` using NumPy's arithmetic operators returns a NumPy array as `scaled_data_np`. Passing `scaled_data_np` to the dense layer will result in a 'non-tensor input' error, because the layer expects a TensorFlow tensor as an input. Instead, the scaling must occur using TensorFlow operators, as shown with `scaled_data_tensor` and the subsequent inference to the dense layer.

Finally, a less frequent but equally important source of the problem involves custom layers and how they integrate with the TensorFlow ecosystem. If you define a custom layer and its `call` method does not explicitly operate on or output tensors, it can also introduce this non-tensor input error. This frequently occurs when new layers are created without careful consideration of the framework’s conventions. For instance, my team once implemented a custom layer with a conditional statement that mistakenly returned a Python list rather than a tensor under specific circumstances, causing the downstream layers to break with this same error. The crucial point is that the `call` method needs to generate a Tensor result for each possible execution path.

```python
import tensorflow as tf
from tensorflow import keras

class CustomLayer(keras.layers.Layer):
    def __init__(self, units, condition_value, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units
        self.condition_value = condition_value

    def build(self, input_shape):
      self.dense_layer = keras.layers.Dense(units=self.units)

    def call(self, inputs):
      # 1. Incorrect operation, will return a list, not tensor if condition met.
      # if tf.math.reduce_mean(inputs) < self.condition_value:
      #     return [inputs, inputs]  # Returning a Python list

      # 2. Correct operation
      if tf.math.reduce_mean(inputs) < self.condition_value:
        return tf.concat([inputs, inputs], axis=-1)  # Returning a Tensor
      else:
        return self.dense_layer(inputs)

input_tensor = keras.Input(shape=(10,))
custom_layer = CustomLayer(units=5, condition_value=0.5)(input_tensor)
model = keras.Model(inputs=input_tensor, outputs=custom_layer)


dummy_data = tf.random.normal((1,10))

# This will throw a non-tensor input exception
# model(dummy_data)

# By ensuring tensor output, this works
model(dummy_data)
```
Here the `call` method of `CustomLayer` demonstrates the issue: the initial implementation, when the mean of the input is less than `condition_value`, the method returns a list containing two instances of `inputs`. This bypasses the TensorFlow graph, making it non-tensor. The second implementation, concatenates both the inputs to create a single Tensor of twice the length. Thus, the `call` method, in the `CustomLayer`, must *always* return a tensor to ensure a correct flow within the framework.

In summary, a 'non-tensor input' error in the Keras Functional API almost always stems from the improper handling of data flow within the symbolic computational graph. To mitigate such errors, one should consistently ensure that input data to each layer is a TensorFlow tensor, derived from `keras.Input` or `tf.convert_to_tensor`. Any preprocessing operations within the model architecture should also be performed using TensorFlow operations, and all custom layers must consistently output tensors.

For further study, I recommend exploring the TensorFlow official documentation, particularly the sections detailing the Functional API and the usage of tensors. The Keras documentation also provides numerous examples. Other helpful resources include guides on custom layer creation and the concept of symbolic graphs within deep learning frameworks. Additionally, examining community-maintained repositories showcasing complex models and inspecting their input and data flow strategies can provide valuable practical knowledge.
