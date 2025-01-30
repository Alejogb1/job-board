---
title: "Why are matrix multiplication variables missing from Keras (TensorFlow 2.0) model plots?"
date: "2025-01-30"
id: "why-are-matrix-multiplication-variables-missing-from-keras"
---
A common point of confusion for those transitioning to TensorFlow 2.0's Keras API, particularly when dealing with custom layers and matrix operations, is the absence of specific matrix multiplication variables within model plot visualizations. This stems not from an oversight in the library, but from the fundamental way Keras handles trainable parameters in relation to the underlying mathematical operations. Essentially, Keras plots represent the computational *graph* of operations at a high level, not the low-level details of every variable involved in each operation.

Specifically, when you implement a custom layer involving, say, a learned weight matrix (W) and a bias vector (b) that you use with matrix multiplication and addition in the forward pass, these parameters (W, b) are *separate trainable variables* distinct from the matrix multiplication operation itself. Keras tracks these variables because they are part of the model's trainable weights, and thus contribute to loss function calculation and gradient updates. The model plot visualizes the *layer*, which encapsulates the *operation* (matrix multiplication, addition, activation etc.), but it doesn't delve into the internal mechanics of how that operation is implemented or what variables are involved during the forward pass using low-level tensor primitives. The underlying matrix multiplication itself, `tf.matmul()`, is a functional operation; it doesn't inherently carry model trainable variables that can be shown in plots. Instead, the plots show the high-level structure and flow of data through the *layers* that define that computational graph.

Let me illustrate this with some code examples. Assume we’re building a relatively straightforward deep learning model that includes a custom dense layer.

**Example 1: Basic Custom Dense Layer**

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

class CustomDenseLayer(layers.Layer):
  def __init__(self, units, activation=None, **kwargs):
    super(CustomDenseLayer, self).__init__(**kwargs)
    self.units = units
    self.activation = keras.activations.get(activation)
    self.w = None # will be initialized in build() method
    self.b = None # will be initialized in build() method

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer="random_normal",
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer="zeros",
                             trainable=True)

  def call(self, inputs):
      output = tf.matmul(inputs, self.w) + self.b
      if self.activation:
          output = self.activation(output)
      return output

# define the model
inputs = keras.Input(shape=(784,))
dense1 = CustomDenseLayer(128, activation="relu")(inputs)
output = CustomDenseLayer(10, activation="softmax")(dense1)
model = keras.Model(inputs=inputs, outputs=output)

keras.utils.plot_model(model, show_shapes=True, show_dtype=True)
```

This code defines a `CustomDenseLayer` that performs the familiar `output = matmul(inputs, W) + b` calculation. When we plot the model using `keras.utils.plot_model()`, the plot will correctly show the two `CustomDenseLayer` nodes along with their output shapes. Critically, *the matrix `w` and vector `b`  themselves are not explicitly represented* as separate nodes, even though they are crucial for computation inside each `CustomDenseLayer` instance.  The model plot depicts the computational *flow* through the layers, not the variables used within those operations in layers. The `CustomDenseLayer` nodes represent the entire operation defined by `tf.matmul(inputs, self.w) + self.b`, not just the variable `w` or the function `tf.matmul`.  `w` and `b` are indeed stored in `model.trainable_variables` and used during backpropagation, but they're managed by the `layers.Layer` infrastructure and represented implicitly in the plot.

**Example 2: Using the `Dense` Layer Directly**

To solidify this point, consider the equivalent using Keras’ built-in `Dense` layer:

```python
inputs = keras.Input(shape=(784,))
dense1 = layers.Dense(128, activation="relu")(inputs)
output = layers.Dense(10, activation="softmax")(dense1)
model = keras.Model(inputs=inputs, outputs=output)

keras.utils.plot_model(model, show_shapes=True, show_dtype=True)
```

The resulting plot is very similar, it shows the `dense` layers, shapes and datatypes. You will find that the matrix W and bias b are implicitly part of each `Dense` layer instance. There are not variable specific nodes in the plot for the weight matrix and bias vector. The internal implementation, which would involve similar matrix multiplications with internal weight matrix and bias vector, is abstracted away to maintain clarity and focus on the high-level network structure. The model plot still represents the computational *flow* through layers, not low-level details about operations, weight matrices, bias vectors and underlying functions.

**Example 3: A Layer with Additional Matrix Operations**

Let's introduce a slightly more complex scenario where we might be tempted to look for individual variables related to matrix ops:

```python
class CustomLayerWithMatrixOps(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomLayerWithMatrixOps, self).__init__(**kwargs)
        self.units = units
        self.w1 = None
        self.w2 = None
        self.b = None


    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(input_shape[-1], self.units),
                            initializer='random_normal', trainable=True)
        self.w2 = self.add_weight(shape=(self.units, self.units),
                            initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                             initializer='zeros', trainable=True)

    def call(self, inputs):
        output = tf.matmul(inputs, self.w1)
        output = tf.matmul(output, self.w2)
        output = output + self.b
        return output


inputs = keras.Input(shape=(784,))
custom_layer = CustomLayerWithMatrixOps(128)(inputs)
output = layers.Dense(10, activation="softmax")(custom_layer)
model = keras.Model(inputs=inputs, outputs=output)
keras.utils.plot_model(model, show_shapes=True, show_dtype=True)
```
Here, `CustomLayerWithMatrixOps` involves two learned weight matrices (`w1`, `w2`) and one bias `b`. Again, though three separate matrix operations are performed using these variables inside the layer during the `call` method and used during back propagation, the model plot shows only the single custom layer node and the dense layer output. The plot shows the layer, its input and output tensors, not explicit nodes for the operation `tf.matmul` or variables `w1` `w2`, and `b`. This clearly demonstrates that `keras.utils.plot_model` focuses on the high-level graph structure, *not* the low-level tensor manipulations and the parameters within those manipulations.

Therefore, the absence of explicit matrix multiplication variable nodes (like `W` or `b`) is not an error; it is an intentional design choice of Keras model plots. The purpose of these plots is to provide a high-level overview of the computational graph, highlighting the flow of data through layers. Details of operations, along with the underlying trainable variables used in those operations in the layers, are managed internally by Keras and its layers’ infrastructure and aren't directly visualized in the model plot.

**Resource Recommendations:**

To further understand this behavior and for practical Keras development:

*   Refer to the official TensorFlow Keras documentation, specifically the sections on custom layers, model subclassing and the usage of the `tf.matmul` function. These will reinforce the concepts covered above.
*   Study code examples from the Keras tutorials which demonstrate different layer types, custom layers and their related trainable weights. Observe how these weight matrices and bias vectors are defined and utilized within each layer instance.
*   Investigate debugging tools in TensorFlow, such as `tf.print` or the TensorFlow debugger, to examine the shapes and values of tensors during computation. This allows for a deeper insight into the actual variables being manipulated within the model graph, going beyond the high-level Keras model plots.

By focusing on the layer structure and operation flow, rather than the low-level tensor primitives involved in each individual operation, Keras plots maintain a manageable and informative view of the model’s overall architecture. Understanding this abstraction is crucial for leveraging the full power of Keras for building complex deep learning models.
