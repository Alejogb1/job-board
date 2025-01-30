---
title: "How to fix 'input_shape' not being subscriptable in a Keras custom layer?"
date: "2025-01-30"
id: "how-to-fix-inputshape-not-being-subscriptable-in"
---
The error `TypeError: 'NoneType' object is not subscriptable` when dealing with `input_shape` in a Keras custom layer usually indicates that the layer’s `build` method is being called with a `None` value for the input shape instead of a tuple or a tensor shape. This scenario typically arises during the initial construction or when Keras attempts to infer the input shape automatically, especially when the layer is the first layer in a sequential model, or if you're using functional API layers without explicitly providing input shapes. I've encountered this myself while developing several custom layers for generative adversarial networks and learned to approach this problem with a focus on shape handling during the layer setup phase.

The core issue is that the `build` method in Keras custom layers receives the shape of the input tensor *after* the layer has been connected to the network. This shape is usually derived through a process of inference and propagation across the network. However, the initial shape could be `None` if it can’t be determined early in model construction. Subsequently, when you try to access specific dimensions using array indexing notation (e.g., `input_shape[1]`), you cause this error because `None` has no subscriptable elements. The solution lies in handling the `None` case within the `build` method and ensuring that input shapes are explicitly defined when appropriate. The `build` method is where you typically initialize weights based on the size of the input features, hence an accurate shape is needed.

The underlying mechanism is related to Keras’s lazy initialization of layers. Weights and other layer-specific variables are generally created at the `build` step, and this can only occur once a specific input shape has been established. Keras relies on this pattern for flexible model creation; you can construct a model without defining every dimension ahead of time, as long as Keras can eventually infer them through network structure.

To mitigate this error, I’ve adopted a structured approach consisting of the following:

1.  **Conditional checks:** Implement an `if` statement in the `build` method to verify that `input_shape` is not `None` before using it.
2.  **Early shape definition:** Explicitly specify input shapes when creating models, especially when using the functional API or when a custom layer is the first layer of a sequential model.
3.  **Alternative shape handling:** If the layer doesn’t immediately require shape information, such as when using a simple non-parameterized activation function, the error can sometimes be ignored. However, this requires careful consideration.

Below are several examples illustrating how to correct this error in different scenarios:

**Example 1: Custom Layer with explicit input shape:**

In this scenario, a custom dense-like layer needs to access the input dimension and creates trainable weights during `build`. If the input shape is not explicitly defined while creating the model, then `input_shape` could be None. This can be avoided by explicitly specifying the input shape during the layer creation process.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomDense(Layer):
    def __init__(self, units, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        if input_shape is None:
            raise ValueError("Input shape cannot be None.")

        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )
        super(CustomDense, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Correct Usage: Explicit input shape defined when using sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(10,)), #Explicitly defined input shape
    CustomDense(units=5),
])

input_tensor = tf.random.normal(shape=(3, 10))
output_tensor = model(input_tensor)

print("Output Shape:", output_tensor.shape)
```

In this example, by specifying `tf.keras.layers.Input(shape=(10,))` prior to `CustomDense`, I’m ensuring that the Keras infrastructure can resolve the input shape and it’s no longer `None` when `build` is called. The check inside the `build` method adds an extra layer of robustness.

**Example 2: Conditional Shape Access**

Sometimes, the input shape is only used conditionally. If the layer can proceed even when the exact input shape is not yet known and it relies on the shape during subsequent `call` operations, the error can be handled by deferring dimension specific operations until call operation or using dynamic shape handling using `tf.shape` during the `call` method. The following illustrates that pattern.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class DynamicLayer(Layer):
    def __init__(self, multiplier=2, **kwargs):
        super(DynamicLayer, self).__init__(**kwargs)
        self.multiplier = multiplier

    def build(self, input_shape):
        super(DynamicLayer, self).build(input_shape)


    def call(self, inputs):
        input_dim = tf.shape(inputs)[-1] # Access the shape dynamically
        output_dim = input_dim * self.multiplier
        reshaped = tf.reshape(inputs, [-1, output_dim])
        return reshaped

# Correct Usage: Shape is dynamic and accessed during call operation
model_dynamic = tf.keras.models.Sequential([
    DynamicLayer(multiplier = 2)
])
input_tensor = tf.random.normal(shape=(3, 5))
output_tensor = model_dynamic(input_tensor)

print("Output Shape:", output_tensor.shape)
```

In the example above, shape is not used during `build` method. If `input_shape` is `None`, the `build` method is skipped; Instead the shape is accessed dynamically during the `call` method using `tf.shape(inputs)`. This approach is suitable when precise shapes are not needed for weight initialization or where the shape can be handled directly within the `call` method.

**Example 3: Functional API with input shape**

The functional API is an alternative approach in Keras, where layers are defined and connected by explicitly passing tensors. If a custom layer that needs the shape information is the first layer, using the functional API, it's crucial to explicitly define an input layer and its shape to avoid similar problems.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

class CustomActivation(Layer):
     def __init__(self, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)

     def build(self, input_shape):
        if input_shape is None:
            raise ValueError("Input shape cannot be None.")
        super(CustomActivation, self).build(input_shape)


     def call(self, inputs):
        return tf.nn.relu(inputs) #Simple activation function

# Correct Usage with Functional API
input_layer = Input(shape=(10,))
custom_layer_out = CustomActivation()(input_layer)
model = Model(inputs = input_layer, outputs=custom_layer_out)
input_tensor = tf.random.normal(shape=(3, 10))
output_tensor = model(input_tensor)

print("Output Shape:", output_tensor.shape)

```
Here, by defining input using `Input(shape=(10,))` we explicitly defined the input shape before passing it to the `CustomActivation` layer. The explicit input definition is important because the Functional API does not implicitly propagate the shape information and the problem can occur when the layer needs the shape in `build` method.

For further understanding, I recommend consulting the official TensorFlow documentation on Keras layers and custom layers, as well as exploring code examples using sequential and functional API models. Studying deep learning courses that cover practical aspects of implementing deep learning models can also enhance one's understanding. Furthermore, research papers focused on the design of custom neural network layers can offer different implementation strategies. These resources have provided invaluable insights into resolving problems similar to the one described. Finally, I've found that regular practice with layer implementations, focusing specifically on the `build` and `call` methods, significantly reduces the likelihood of encountering shape-related errors.
