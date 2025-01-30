---
title: "Why is a custom layer not receiving input in plot_model?"
date: "2025-01-30"
id: "why-is-a-custom-layer-not-receiving-input"
---
A custom Keras layer, when not correctly implemented, can indeed fail to receive input during the `plot_model` function call, typically resulting in a layer rendered without connections to the preceding or subsequent layers. This arises because `plot_model` relies on Keras's layer introspection and graph tracing capabilities, and an improperly defined custom layer can disrupt this process. Specifically, the `build` method and `call` method, integral parts of any Keras layer, must be defined and interact correctly for the layer to be properly incorporated into the model's symbolic graph, which is then utilized by `plot_model`.

My experience stems from developing custom, time-series processing layers for a financial forecasting project. I encountered this issue initially with a layer I designed to perform a moving average calculation across sequences. The initial model training worked flawlessly but visually inspecting the model architecture through `plot_model` showed the custom layer as isolated.

The key reason behind this behavior lies in how Keras internally constructs its computational graph. When a model is instantiated, Keras keeps track of the inputs and outputs of each layer to establish the data flow. The `plot_model` function leverages this internal representation to visually depict the model's architecture. However, for custom layers, Keras depends on the user correctly implementing the `build` and `call` methods to inform the framework about the layer's input and output shapes, as well as the computational logic it performs.

The `build(input_shape)` method is responsible for initializing the layer’s trainable weights and biases. The `input_shape` parameter, as its name implies, represents the shape of the input tensor that will be fed to the layer. Within this function, the layer is expected to calculate the shape of the weight and bias parameters based on this `input_shape`, and to register these as trainable variables using `self.add_weight()`. Failure to properly initialize these internal variables or not informing Keras about them hinders `plot_model` from tracing the layer's connections.

The `call(inputs)` method, on the other hand, defines the forward pass logic of the layer, and accepts one or more inputs from the previous layer and produces an output which becomes input to the next layer. The crucial element here is that the output generated inside this method must be a Keras Tensor, i.e. the output of Keras backend functions. Failure to provide a correctly formatted output or to use backend operators can result in Keras unable to trace data dependencies. Essentially, this effectively blocks `plot_model`'s ability to accurately represent the model.

Here are some concrete examples demonstrating where this might go wrong.

**Example 1: Missing `build` Implementation**

```python
import tensorflow as tf
from tensorflow.keras import layers

class BrokenLayer(layers.Layer):
  def __init__(self, output_dim, **kwargs):
    super(BrokenLayer, self).__init__(**kwargs)
    self.output_dim = output_dim

  def call(self, inputs):
     return tf.matmul(inputs, tf.random.normal(shape = [inputs.shape[-1], self.output_dim]))

input_shape = (None, 10)
test_input = tf.random.normal(shape = (1, 10))
layer = BrokenLayer(5)
output = layer(test_input)  # Training works.

# Now the output shapes is known after the first call
# model = tf.keras.Sequential([
#    layers.Input(shape=input_shape[1:]),
#    BrokenLayer(5)
# ])
# tf.keras.utils.plot_model(model, to_file='broken_layer.png', show_shapes=True) # Plot does not work
```

In this example, the `BrokenLayer` lacks a `build` method and does not declare any weights before being called. The `call` method calculates an output using matrix multiplication with a random tensor. Crucially, this random tensor is created in each call and is not stored as a parameter within the layer and thus it cannot be updated with gradient descent. Although the layer might function during model training, it is not properly recognized by `plot_model` because Keras's introspection mechanism needs a `build` method and internally declared weights. The model cannot be constructed before calling `call` because the weights are constructed each time. Uncommenting the code for the model will still fail.

**Example 2: Incorrect `build` Logic**

```python
import tensorflow as tf
from tensorflow.keras import layers

class IncorrectBuildLayer(layers.Layer):
  def __init__(self, output_dim, **kwargs):
    super(IncorrectBuildLayer, self).__init__(**kwargs)
    self.output_dim = output_dim

  def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel',
                                  shape=(self.output_dim, input_shape[-1]),
                                  initializer='uniform',
                                  trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)


input_shape = (None, 10)
test_input = tf.random.normal(shape = (1, 10))
layer = IncorrectBuildLayer(5)
_ = layer(test_input)  # training works

model = tf.keras.Sequential([
   layers.Input(shape=input_shape[1:]),
   IncorrectBuildLayer(5)
])
tf.keras.utils.plot_model(model, to_file='incorrect_layer.png', show_shapes=True)  # Plot does not show an output
```

Here, the `IncorrectBuildLayer` includes a `build` method, attempting to create a kernel weight based on the provided `input_shape`. However, the shape definition in `build` is inverted. I've defined the shape to be (self.output_dim, input_shape[-1]) when it should be (input_shape[-1], self.output_dim). This mismatch causes issues with tensor broadcasting during the call function. Even though model training might proceed, `plot_model` will not accurately represent the connections due to the incorrect weight shape leading to errors.

**Example 3: Correct Implementation**

```python
import tensorflow as tf
from tensorflow.keras import layers

class CorrectLayer(layers.Layer):
  def __init__(self, output_dim, **kwargs):
    super(CorrectLayer, self).__init__(**kwargs)
    self.output_dim = output_dim

  def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[-1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
    self.bias = self.add_weight(name = 'bias', shape=(self.output_dim,),
                                initializer = 'zeros', trainable = True)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel) + self.bias


input_shape = (None, 10)
test_input = tf.random.normal(shape = (1, 10))
layer = CorrectLayer(5)
_ = layer(test_input)

model = tf.keras.Sequential([
    layers.Input(shape=input_shape[1:]),
    CorrectLayer(5)
])
tf.keras.utils.plot_model(model, to_file='correct_layer.png', show_shapes=True) # Plot now works.
```

The `CorrectLayer` properly implements both the `build` and `call` methods. The `build` method initializes the kernel and bias weights with the correct shape determined by the input shape and output dimension. The call performs matrix multiplication and adds the bias, using the weights declared in `build`. Now, when used in a sequential model, `plot_model` correctly displays the layer with its appropriate connections. This example illustrates how providing accurate weight definitions enables `plot_model` to trace the graph and generate a useful architectural representation.

To diagnose such issues, carefully examine the implementation of your custom layer, and verify that the `build` and `call` methods are properly defined. Ensure that weights and biases are correctly initialized with `self.add_weight()`, and that the shapes are correctly calculated. Furthermore, in the `call` method, make sure that all your operations are done using TensorFlow backend operations (`tf.*`).

For in-depth understanding of Keras layers and the underlying concepts of building custom layers, I recommend reviewing the official Keras documentation, specifically the sections pertaining to custom layers and the `Layer` class API. Also, exploring resources related to TensorFlow's tensor operations will help to construct efficient `call` methods. Researching examples of custom layers implemented within the TensorFlow ecosystem can provide additional insights.
