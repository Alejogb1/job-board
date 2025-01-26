---
title: "How can I resolve a shape mismatch error between tensors of size (None, None) and (None, 28, 28, 10)?"
date: "2025-01-26"
id: "how-can-i-resolve-a-shape-mismatch-error-between-tensors-of-size-none-none-and-none-28-28-10"
---

A shape mismatch error between tensors of size `(None, None)` and `(None, 28, 28, 10)` during tensor operations in a neural network usually indicates a fundamental misunderstanding of input and output shapes within your model’s data flow. This discrepancy often arises when a layer expecting a specific tensor rank and shape receives a tensor with an undefined second dimension, often the result of a dynamically sized input placeholder or an incorrectly reshaped output from a previous operation. I've seen this exact scenario multiple times while working on image processing pipelines, specifically with convolutional neural networks. The key is to systematically trace the tensor shapes from the input to the error-generating layer and identify the point where the `(None, None)` tensor is created.

The `(None, None)` shape signifies that the tensor's first dimension (usually batch size) is dynamically determined, which is common. However, the second dimension being `None` points to a missing shape declaration, which is often a consequence of improper flattening or reshaping operations or attempting to pass an improperly structured input into the model. The expected tensor, `(None, 28, 28, 10)`, represents a batch of images with dimensions 28x28 and 10 channels (likely class probabilities). Therefore, the resolution generally involves ensuring the preceding operations output the correct multi-dimensional shape before it reaches the layer expecting this format. The core problem lies in the under-defined second dimension, so we must introduce operations that specify or compute that dimension before tensor usage. This could be a resizing operation on a feature map, or reshaping operation on an array of values.

To diagnose this, I systematically print the shapes of the tensors involved at each step of the model using the debugging facilities of the particular deep learning framework. This allows me to identify exactly when this undefined second dimension comes into being. Here are a few common approaches I've used to address this, along with illustrative code.

**Example 1: Incorrect Flattening or Reshaping**

Consider a scenario where a convolutional layer output, after processing an image, is flattened prior to connection with a fully connected layer. Suppose the desired output of the convolutional layers *before* flattening is (batch\_size, 28, 28, 64). We then want to feed this to a fully connected layer. However, if you naively flatten using an operation like this, you might see the error you are encountering. The code, while not intended to be runnable, highlights the problem area.

```python
# Incorrect Flattening Example
import tensorflow as tf
import numpy as np

# Assume conv_output has a shape of (batch_size, 28, 28, 64)
conv_output = tf.random.normal(shape=(32, 28, 28, 64))

# Incorrect flattening, resulting in (batch_size, None)
flattened_output = tf.reshape(conv_output, [tf.shape(conv_output)[0], -1])

# Dummy dense layer, expecting an input with a specific number of features
# Causes a mismatch error because we don't know -1
dense_layer = tf.keras.layers.Dense(units=10)
try:
    output = dense_layer(flattened_output)
except Exception as e:
  print(e)


# Correct Flattening
flattened_output = tf.reshape(conv_output, [-1, 28*28*64])
# This works because we are defining the input shape
output = dense_layer(flattened_output)

print(f"output shape: {output.shape}")

```

Here, the first flattening operation uses `-1` as the second dimension of the reshape. The framework can infer this value as the number of dimensions such that the reshaping operation can proceed. However, this causes the second dimension to be a `None` during graph construction, as it is not explicitly specified during training. When we later connect this tensor to a layer that requires a specific input size the `(None, None)` shape results in a mismatch. The solution is to explicitly compute the flattened dimension, for instance `28*28*64`, and specify that in the reshaping operation as demonstrated in the second part of the code. This ensures that all subsequent layers know the shape of their inputs.

**Example 2: Incorrect Input Data**

Another common cause is feeding incorrectly shaped input data directly to the model. This scenario occurs frequently when you pre-process data using custom scripts which make an error in image resizing or conversion.

```python
# Incorrect Input Example (using numpy)
import numpy as np
import tensorflow as tf

# Create an "image" with a wrong shape
incorrect_image_data = np.random.rand(32,1).astype(np.float32) # Batch of 32 images, but shape is just one dimension
incorrect_image_data = tf.constant(incorrect_image_data)

# Define the model's input layer
input_layer = tf.keras.Input(shape=(28, 28, 1))

# Attempt to use a convolutional layer (this will fail)
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)

# Pass incorrect data to model, cause a shape mismatch.
try:
  output = conv_layer(incorrect_image_data)
except Exception as e:
  print(e)

# Correct Image Input
correct_image_data = np.random.rand(32, 28, 28, 1).astype(np.float32)
correct_image_data = tf.constant(correct_image_data)
output = conv_layer(correct_image_data)
print(f"output shape: {output.shape}")
```

In this example, the model expects an image input with a shape of `(28, 28, 1)`. However, `incorrect_image_data` only has the shape of `(32, 1)` which can not be processed as a typical image. This leads to a shape mismatch at the first layer since it expects an input with three spatial dimensions (height, width and channel). The resolution here is simply to ensure the input data has the proper structure, as demonstrated in the second portion of the code. I have spent hours debugging errors caused by such issues, and can say from experience that meticulously ensuring that data meets the expectations of the layers it is going into is often the primary step in debugging this kind of error.

**Example 3: Missing Reshape After Lambda Layers or Custom Logic**

Sometimes the incorrect reshaping occurs not in a typical layer, but inside a Lambda or custom function. This often occurs when a more complex manipulation is needed within the network, and care was not taken to correctly specify output dimensions after some operation has been performed.

```python
import tensorflow as tf

# Assuming some layer output (e.g., a flattened layer) with a shape (None, 784)
intermediate_layer = tf.random.normal(shape=(32, 784))

# Custom Lambda layer that does some operation
def custom_logic(x):
  # Here we are supposed to output a (28, 28, 1) feature map
  # But due to an error, we output x as is, (1, 784)
  return x

lambda_layer = tf.keras.layers.Lambda(custom_logic)
# Incorrect shape after lambda
lambda_output = lambda_layer(intermediate_layer)
try:
  reshape_layer = tf.keras.layers.Reshape((28, 28, 1))(lambda_output)
except Exception as e:
  print(e)

# Correct Custom Reshape.
def correct_logic(x):
  # Here, we properly reshape
  reshaped = tf.reshape(x, (-1, 28, 28, 1))
  return reshaped

correct_lambda_layer = tf.keras.layers.Lambda(correct_logic)
reshaped_output = correct_lambda_layer(intermediate_layer)
print(f"output shape: {reshaped_output.shape}")

```

In this scenario, a custom `Lambda` layer fails to reshape the tensor to the expected shape, `(None, 28, 28, 1)`. This results in another shape mismatch at the next layer. In the original lambda, we simply returned the input with shape `(None, 784)`, not the expected `(None, 28, 28, 1)` shape, which fails at the `Reshape` layer. Correcting this error requires that the user of the `Lambda` layer take responsibility for ensuring that the output tensor matches the required dimensions, as demonstrated in the second portion of the code. Such issues can be difficult to debug, and careful review of custom logic is important in complex systems.

Debugging shape errors like these requires a systematic approach. I recommend using print statements or the debugger tools of your framework, such as TensorFlow’s Eager execution debugging functionalities. You will need to track the shapes of your tensors as they flow through your network. This involves printing the output shape of every layer in the system, starting from the input to the problematic layer to pin point the cause of error. The key is understanding the expected input shapes of each layer and systematically debugging when a mismatch occurs.

To improve overall understanding, I recommend referring to several resources. Firstly, the official documentation of the deep learning framework (e.g., TensorFlow, PyTorch) is invaluable. Specifically, sections on tensor operations, reshaping, and data input pipelines. I also found it beneficial to review tutorials on convolutional neural networks. Examining code examples for image processing tasks helps solidify an understanding of typical tensor flows and the shapes encountered within these models. Finally, gaining some degree of formal understanding of linear algebra will likely be useful for a fundamental understanding of tensor transformations. These materials help in ensuring that you are able to systematically identify and correct errors such as these.
