---
title: "Can a TensorFlow/Keras model be expressed as a pure function of its weights and inputs?"
date: "2025-01-30"
id: "can-a-tensorflowkeras-model-be-expressed-as-a"
---
A neural network, at its core, is a complex composition of mathematical operations. Given a fixed architecture and specific weights, the process of propagating an input through the network and arriving at an output is, in essence, a deterministic computation. Therefore, a TensorFlow/Keras model *can* be conceptually represented as a pure function of its weights and inputs, albeit with certain practical caveats.

The "purity" of a function, in the context of functional programming, implies that for the same input, the function will always produce the same output, and it has no side effects. Neural networks, while appearing procedural in their typical training and inference phases, fundamentally adhere to this principle during *inference*. Consider a model trained with a fixed set of weights; during inference, these weights remain unchanged. Provided the same input data, the forward pass of the model, which comprises matrix multiplications, activation functions, and possibly other deterministic operations, will consistently produce the same output. The absence of any non-deterministic elements, such as random weight initialization or random augmentations during inference, solidifies this viewpoint.

However, it's crucial to distinguish between the *abstract* functional representation and the practical implementation within TensorFlow/Keras. The training process inherently *violates* the pure function principle due to the constant modification of weights through backpropagation. However, this is a training aspect; once the training is complete, and weights become frozen, the inference phase can be construed as a pure function. The weights serve as the constant, and the input to the model the argument of this function. This highlights the fundamental difference between the *state* of a neural network during training, which is variable, and its state during inference, which is fixed.

Consider this conceptualization: let `f` be our neural network model, `w` be the weights vector and `x` be the input vector, then:
`output = f(w, x)`

This function `f`, during inference, can be considered a pure function, because `f`, with its `w` being fixed, will always yield the same output with same input `x`.

Now, let's examine this concept through code examples. These examples are illustrative, focusing on clarity of the concept rather than optimized TensorFlow/Keras practices.

**Example 1: A Simple Sequential Model**

```python
import tensorflow as tf
import numpy as np

def forward_pass(weights, input_data):
    """
    Emulates the forward pass of a simple sequential model.
    Weights are flattened and separated into layers for demonstration purposes.
    """

    w1_size = 2 * 3  # Shape of the first layer weights: (2, 3)
    w1 = weights[:w1_size].reshape((2, 3))
    b1 = weights[w1_size: w1_size + 2] # Shape of first layer bias (2,)

    w2_size = 3 * 1  # Shape of the second layer weights: (3, 1)
    w2 = weights[w1_size + 2: w1_size + 2 + w2_size].reshape((3, 1))
    b2 = weights[w1_size + 2 + w2_size:] # Shape of second layer bias (1,)


    layer1_output = tf.nn.relu(tf.matmul(input_data, w1) + b1)
    layer2_output = tf.matmul(layer1_output, w2) + b2

    return layer2_output


# Generate some random weights (for demonstration). In practice, these would be trained.
np.random.seed(42)
weights_initial = np.random.rand(w1_size + 2 + w2_size + 1)


input_tensor = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
output1 = forward_pass(weights_initial, input_tensor)

# Demonstrate purity. Identical inputs produce the same outputs given the same weights.
output2 = forward_pass(weights_initial, input_tensor)


print(f"Output 1: {output1.numpy().flatten()}")
print(f"Output 2: {output2.numpy().flatten()}")

assert np.array_equal(output1,output2)
```

This example illustrates how, given a set of weights, the forward pass can be represented as a pure function.  The `forward_pass` function takes weights and input data and performs calculations to produce an output.  With the same weights and same input, the `output1` and `output2` are equal, which underscores the functional, deterministic nature during inference. The crucial part of understanding this is that these `weights_initial` are frozen and fixed throughout execution of our defined `forward_pass`.

**Example 2: Extracting Weights and Biases from an Existing Keras Model**

```python
import tensorflow as tf
import numpy as np

# Define a simple Keras model.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])

# Obtain the weights of a model
weights_list = model.get_weights()

# Flatten the weights into a single numpy array
flattened_weights = np.concatenate([w.flatten() for w in weights_list])

# input data
input_tensor = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)


# Re-implement the forward_pass function, but now we will retrieve shapes
# of weights and biases from our Keras model to ensure we do it correctly.
def forward_pass_from_model(weights, input_data, weights_list_shapes):

    w1_shape = weights_list_shapes[0]
    b1_shape = weights_list_shapes[1]
    w2_shape = weights_list_shapes[2]
    b2_shape = weights_list_shapes[3]

    w1_size = np.prod(w1_shape)
    w1 = weights[:w1_size].reshape(w1_shape)
    b1 = weights[w1_size: w1_size + np.prod(b1_shape)].reshape(b1_shape)

    w2_size = np.prod(w2_shape)
    w2 = weights[w1_size + np.prod(b1_shape): w1_size + np.prod(b1_shape) + w2_size].reshape(w2_shape)
    b2 = weights[w1_size + np.prod(b1_shape) + w2_size:].reshape(b2_shape)

    layer1_output = tf.nn.relu(tf.matmul(input_data, w1) + b1)
    layer2_output = tf.matmul(layer1_output, w2) + b2

    return layer2_output


model_output = model(input_tensor)
forward_pass_output = forward_pass_from_model(flattened_weights, input_tensor, [w.shape for w in weights_list])


print(f"Keras Model Output: {model_output.numpy().flatten()}")
print(f"Forward Pass Output: {forward_pass_output.numpy().flatten()}")

assert np.allclose(model_output.numpy(),forward_pass_output.numpy())
```

In this example, we extract the weights and biases from a Keras model, flatten them into a single vector and then use this vector along with information on shapes of weights and biases, in conjunction with the forward pass function to compute the output. This effectively constructs a purely functional representation that matches the original model’s inference output which reinforces the idea of the network’s forward pass being a pure function if the weights are fixed.

**Example 3: Handling More Complex Layers (Illustrative)**

```python
import tensorflow as tf
import numpy as np


# Simplified representation, a more comprehensive implementation would be significantly more complex.
# Let's assume we have a single convolution and a dense layer
def forward_pass_complex(weights, input_data, params):
    """
    Illustrative representation of how convolution layer weights could be handled.
    Note: Handling different strides, padding, etc would require additional logic.
    """
    conv_weights, conv_biases = weights[0], weights[1]
    dense_weights, dense_biases = weights[2], weights[3]

    conv_layer_output = tf.nn.conv2d(input_data, conv_weights, strides=params["strides"], padding=params["padding"]) + conv_biases
    conv_layer_output = tf.nn.relu(conv_layer_output)
    
    # Flatten output to match the dense layer
    flattened_output = tf.reshape(conv_layer_output, [input_data.shape[0], -1])
    dense_output = tf.matmul(flattened_output, dense_weights) + dense_biases
    return dense_output


# Mock weights and parameters for illustration purposes
# (Note: These wouldn't be arbitrary in a practical model)
input_data = tf.random.normal((1, 28, 28, 3))  # Batch size 1, image size 28x28, 3 channels
params = {"strides": 1, "padding": "SAME"}
conv_weights = tf.random.normal((3, 3, 3, 16)) # 16 filters of 3x3 size
conv_biases = tf.random.normal((16,))
dense_weights = tf.random.normal((28*28*16, 10)) # 10 outputs for the dense layer
dense_biases = tf.random.normal((10,))

weights = [conv_weights, conv_biases, dense_weights, dense_biases]
output1 = forward_pass_complex(weights, input_data, params)

# Demonstrate purity. Identical inputs produce the same outputs given the same weights.
output2 = forward_pass_complex(weights, input_data, params)

assert np.allclose(output1.numpy(), output2.numpy())


print(f"Output 1: {output1.numpy().flatten()[:5]}...")
print(f"Output 2: {output2.numpy().flatten()[:5]}...")


```

This example illustrates a conceptual extension to models with convolution layers.  The complexities of convolutions, pooling, and other layers are omitted for conciseness.  The code provides a conceptual idea that shows even with these layers the forward pass, with weights frozen, is a pure function. The key idea here is that given the same weights and input, the output of `forward_pass_complex` is identical.

For more thorough information, several resources can be consulted. Texts covering functional programming concepts, specifically in the context of mathematical functions, would be beneficial in establishing the theoretical basis. Standard textbooks or online materials that describe neural networks can provide a rigorous background on the underlying mathematics of the forward pass and the role of weights. Additionally, research papers that discuss functional perspectives on machine learning models often appear in various ML-related conference publications, and these would serve well. A comprehensive understanding of the TensorFlow and Keras API documentation is beneficial for practical application. These would provide a well-rounded understanding of the underlying principles.
