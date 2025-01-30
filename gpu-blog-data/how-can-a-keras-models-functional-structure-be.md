---
title: "How can a Keras model's functional structure be inspected?"
date: "2025-01-30"
id: "how-can-a-keras-models-functional-structure-be"
---
The precise structure of a Keras model, particularly those built using the functional API, can be opaque without dedicated inspection methods. Understanding the connections between layers, their shapes, and the overall data flow is critical for debugging, optimization, and advanced manipulation. I've found that while `model.summary()` provides a broad overview, it often lacks the granularity needed for complex architectures.

The primary method for inspecting the functional structure lies in examining the `layers` attribute of the Keras `Model` object, coupled with accessing individual layer attributes and their input/output tensors. Keras, in its functional API, explicitly maintains the relationships between layers through tensor connections, which are represented as Keras Tensor objects. The `Model` object itself stores a list of `Layer` instances in the order they were added (though this isn’t always the execution order when backpropagation comes in). I often begin by iterating through these layers and accessing attributes. It's a more hands-on method than simply relying on summaries.

This is what I’ll look at when inspecting the functional structure of a Keras model:

**1. Layer Objects:** Each element in `model.layers` is a `Layer` object. These objects possess various attributes, including:
    *   `name`: A string representing the unique name of the layer.
    *   `input`: A single Tensor or a list of Tensors, representing the input(s) to the layer.
    *   `output`: A Tensor representing the output of the layer.
    *   `input_shape`: Shape of input as a tuple, or list of tuples for multiple inputs
    *   `output_shape`: Shape of output as a tuple.
    *   `trainable`: A boolean indicating whether layer weights are trained during learning.
    *   `weights`: a list of tensors containing the trainable weights in the model.

**2. Tensor Objects:** The `input` and `output` attributes of a layer are Keras Tensor objects. These objects hold additional information including:
    *   `shape`:  The shape of the tensor.
    *   `dtype`: The datatype of the tensor.
    *   `_keras_history`: This attribute holds metadata about the layer that produced the tensor, including the layer itself and other tensors of which this tensor is an input. This provides the necessary link to map forward through the computational graph.

By accessing these attributes, I can reconstruct the model's computational graph and understand the data flow from input to output. This is especially beneficial when dealing with models containing shared layers, multiple inputs, or multiple outputs, where the standard `summary` function may not provide sufficient detail. The `_keras_history` attribute specifically links the output of one layer to the inputs of another.

Here are some illustrative code examples:

**Example 1: Simple Sequential Model (for contrast)**

This example uses the Sequential API to create a basic model for easier contrast with the more complex functional API inspection.  While Sequential models also have the `layers` attribute, they are much more limited in scope.  I would not use inspection as deeply as with functional API models.

```python
import tensorflow as tf
from tensorflow.keras import layers

# Create a Sequential Model
model_seq = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])


# Inspect layers
for layer in model_seq.layers:
    print(f"Layer Name: {layer.name}")
    print(f"  Input Shape: {layer.input_shape}")
    print(f"  Output Shape: {layer.output_shape}")
    print(f"  Trainable: {layer.trainable}")

    if hasattr(layer, 'weights') and layer.weights:
        print(f" Number of weights: {len(layer.weights)}")
    print("\n")
```
**Commentary:**

This snippet iterates through the `layers` of the sequential model, printing each layer's name, shape information, and trainability. While useful, the sequential model implicitly handles connectivity.  There's nothing to link the output of the first layer with the input of the second layer.  The shape information is useful, but the lack of detailed history is a crucial limitation. This demonstrates a contrast with how we need to inspect functional models.

**Example 2: Functional Model with Shared Layer and Multiple Inputs**

This example showcases a more realistic scenario where inspection is vital. Here, a shared embedding layer is used with two input branches.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Define input layers
input_a = layers.Input(shape=(10,), name='input_a')
input_b = layers.Input(shape=(15,), name='input_b')

# Shared embedding layer
shared_embedding = layers.Embedding(input_dim=50, output_dim=8, name='shared_embedding')
embedded_a = shared_embedding(input_a)
embedded_b = shared_embedding(input_b)

# Concatenate and process
merged = layers.concatenate([embedded_a, embedded_b], name='merged_layer')
dense_out = layers.Dense(1, activation='sigmoid', name='output_layer')(merged)

# Create the model
model_func = Model(inputs=[input_a, input_b], outputs=dense_out)

# Inspect layers
for layer in model_func.layers:
    print(f"Layer Name: {layer.name}")
    if hasattr(layer, 'input'):
      if isinstance(layer.input, list):
        for inp in layer.input:
          if hasattr(inp, '_keras_history'):
            print(f"   - Input From: {inp._keras_history.layer.name}")
        print(f"  Input Shapes: {[tensor.shape for tensor in layer.input]}")
      else:
        if hasattr(layer.input, '_keras_history'):
          print(f"  - Input From: {layer.input._keras_history.layer.name}")
        print(f"  Input Shape: {layer.input.shape}")

    if hasattr(layer, 'output'):
        if hasattr(layer.output, '_keras_history'):
          print(f"  - Output To: {layer.output._keras_history.layer.name}")
        print(f"  Output Shape: {layer.output.shape}")
    print("\n")
```

**Commentary:**

Here, I specifically show how `_keras_history` can be accessed.  The code iterates through each layer in `model_func`. For layers with inputs, it displays the producing layer's name using the `_keras_history` attribute.  This functionality is vital for understanding the flow of tensors and which layers contribute to which tensor output. The input and output shapes are also printed.  Notice how the shared embedding layer outputs a tensor that is then an input to both the "merged_layer," demonstrating how multiple input connections can be inspected. This kind of insight is crucial to understand complex architectures.

**Example 3: Accessing Weights and their Shapes**

It is also useful to examine the weight tensors of specific layers, particularly for debugging.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Create an example model
input_layer = layers.Input(shape=(20,))
dense_layer_1 = layers.Dense(10, activation='relu', name='dense_1')(input_layer)
dense_layer_2 = layers.Dense(5, activation='softmax', name='dense_2')(dense_layer_1)

model_weight = Model(inputs=input_layer, outputs=dense_layer_2)

#Inspect a particular layer's weight
for layer in model_weight.layers:
    if 'dense' in layer.name:
        print(f"Layer: {layer.name}")
        if hasattr(layer, 'weights') and layer.weights:
          for idx, weight in enumerate(layer.weights):
            print(f"  - Weight {idx} Name: {weight.name}")
            print(f"     Shape: {weight.shape.as_list()}")
            print(f"     Values : {weight.numpy()[0][0:3]}")
            print("\n")
```

**Commentary:**

This code snippet explicitly selects the dense layers from the model, iterates through each weight tensor, and then prints shape and first 3 values using `weight.numpy()`.  While printing all weight values is generally unhelpful for large models, I do find it valuable to inspect the shapes, data types, and small subsets for unexpected behaviour. This is particularly useful when initializing custom layers or debugging potential weight-related errors. I would also commonly use this to verify weight initialization is performing as expected.

For further learning and development, I recommend consulting the official Keras documentation on the `Model` class and the `Layer` class, as they contain detailed descriptions of the methods and attributes discussed above. Additionally, the 'Advanced Keras' documentation offers insights into the functional API and its inner workings. Specific tutorials focused on model debugging can also be valuable. For a deeper understanding of tensors and the underlying graph structures, reviewing the TensorFlow documentation is essential.
