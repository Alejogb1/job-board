---
title: "What are the supported operations in TensorFlow Lite for Microcontrollers?"
date: "2025-01-30"
id: "what-are-the-supported-operations-in-tensorflow-lite"
---
TensorFlow Lite for Microcontrollers (TFLM) offers a constrained yet powerful set of operations, reflecting its design for resource-limited embedded systems. The primary goal is to enable machine learning inference on devices with minimal memory, processing, and power capabilities. This necessarily means a selection of commonly used operations, optimized for performance, rather than a comprehensive suite. My direct experience porting a gesture recognition model to an ARM Cortex-M4 underlines this trade-off; we had to meticulously choose operations that TFLM supported, frequently restructuring the initial model architecture.

Fundamentally, TFLM supports a core set of operations found in common neural network architectures. This core is geared towards forward inference – calculating the output of a network given an input, rather than training, which is impractical on microcontrollers. The supported operations are meticulously optimized for low-power execution using integer arithmetic. This is a critical deviation from server-side TensorFlow, which often relies on floating-point operations. We primarily see implementations using 8-bit integers to reduce memory footprint and computational load.

The supported operations fall into several categories. First, *convolutional operations*, like `CONV_2D`, are critical for image-based models and signal processing. However, it's crucial to note the specific limitations. Padding options are typically restricted to 'SAME' and 'VALID' padding. Depthwise convolutions, `DEPTHWISE_CONV_2D`, are also present, a crucial element for mobile-net-style architectures. These operations are tailored for edge deployment scenarios.

*Pooling operations* are supported, with `MAX_POOL_2D` and `AVERAGE_POOL_2D` available. They are primarily used for downsampling feature maps and reducing computational load. These operations are generally less complex and therefore well-suited for microcontrollers. I regularly found myself using pooling after convolutional layers to reduce spatial dimensions of the feature maps, speeding up inference.

Next, *activation functions* play a key role in non-linear transformations within the network. `RELU`, `RELU6`, and `TANH` are provided. These are generally implemented with lookup tables or optimized calculations, given their frequent occurrence in a network. `SIGMOID` can be available in specific builds. While other more complex activation functions exist in full TensorFlow, their computational costs often exceed the constraints of microcontrollers.

*Dense operations* or *fully-connected layers* are another essential element. These are implemented as matrix multiplications, generally supported via an optimized `FULLY_CONNECTED` operation. These, often present at the final layers of the models, are crucial for mapping feature representations to class probabilities. I experienced firsthand that these operations, though conceptually straightforward, tend to be computationally intensive if not properly optimized for the specific hardware.

Element-wise operations are also crucial. The most commonly supported operations include `ADD`, `SUB`, `MUL`, and `DIV`. They are implemented efficiently on most microcontrollers with basic arithmetic processing units. These are frequently used in the residual connections of neural networks. Additionally, operations for reshaping and concatenating tensors like `RESHAPE`, `CONCATENATION`, and `TRANSPOSE` are available for manipulating the flow of information through the network.

Finally, *special operations* such as `SOFTMAX`, frequently used for output layers, along with `LOGISTIC`, are included. Furthermore, basic statistical operations like `MAX`, `MIN`, and `SUM` are supported. I routinely utilized `SOFTMAX` in the output layer of classification tasks after the fully connected layers.

It’s essential to acknowledge the dynamic nature of TFLM. Specific operation availability can depend on the exact library version and build configurations. For instance, specialized operations or custom kernel implementations are not universally present and must be explicitly built in. A thorough review of the library documentation for the specific version in use is paramount for practical implementation.

The following code examples illustrate how TFLM operations are typically employed in a model, presented here using the TensorFlow API. Note this is not code meant for direct execution on a microcontroller but for outlining model design.

**Example 1: Convolutional Layer**

```python
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as _schema
def conv_layer(input_tensor, filters, kernel_size, stride, padding):
    conv = tf.nn.conv2d(input_tensor, filters, strides=stride, padding=padding)
    return conv

input_shape = (1, 28, 28, 3) # Example input shape: batch size, height, width, channels
input_tensor = tf.random.normal(input_shape)
filters_shape = (3, 3, 3, 32) #Example shape: kernel size, kernel size, input channels, filters
filters = tf.random.normal(filters_shape)
kernel_size = (1, 3, 3, 1)
stride = (1, 1, 1, 1)
padding = "SAME"
conv_result = conv_layer(input_tensor, filters, kernel_size, stride, padding)
print(conv_result.shape)
```
This code snippet uses TensorFlow to demonstrate a 2D convolution. In TFLM, such operations are supported via `CONV_2D`. This example illustrates creation of a layer, where the shapes (input, kernel) need to be aligned for TFLM. The specific strides, padding type, and the kernel weights are critical and need to be considered during the model conversion process.

**Example 2: Fully Connected and Activation Layer**

```python
def dense_layer(input_tensor, weights, biases, activation):
    output = tf.matmul(input_tensor, weights) + biases
    if activation == "relu":
         output = tf.nn.relu(output)
    elif activation =="relu6":
         output = tf.nn.relu6(output)
    return output

input_shape = (1, 100) #Example input shape
input_tensor = tf.random.normal(input_shape)
weights_shape = (100,10) # Example shape: input size, output size
biases_shape = (10)
weights = tf.random.normal(weights_shape)
biases = tf.random.normal(biases_shape)
activation = "relu"
dense_result = dense_layer(input_tensor, weights, biases, activation)
print(dense_result.shape)
```
Here, a fully connected (dense) layer is defined, along with a ReLU activation. TFLM uses the `FULLY_CONNECTED` operation for the matrix multiplication with support for `RELU` and `RELU6` as specific activation functions. This illustrates one common pattern, of transforming high dimensional feature maps to lower dimensional representation for classification. This was a common pattern when moving larger models to the resource-constrained systems.

**Example 3: Pooling Layer**

```python
def pooling_layer(input_tensor, pool_size, stride, padding, pooling_type):
    if pooling_type == "max":
        output = tf.nn.max_pool(input_tensor, pool_size, strides=stride, padding=padding)
    elif pooling_type == "average":
        output = tf.nn.avg_pool(input_tensor, pool_size, strides=stride, padding=padding)
    return output

input_shape = (1, 14, 14, 32) # Example input shape
input_tensor = tf.random.normal(input_shape)
pool_size = (1,2,2,1)
stride = (1, 2, 2, 1)
padding = "SAME"
pooling_type = "max"
pooling_result = pooling_layer(input_tensor, pool_size, stride, padding, pooling_type)
print(pooling_result.shape)
```
This shows a pooling operation. The code shows `MAX_POOL_2D` (the primary operation used for pooling) in action, which TFLM efficiently executes on microcontrollers. The parameters related to stride, window size are crucial and should be aligned with the requirements of the model. Pooling is critical for reducing dimensionality.

For detailed information on the supported operations in a specific version of TensorFlow Lite for Microcontrollers, consult the official TensorFlow documentation. Specifically, review the sections on “TensorFlow Lite for Microcontrollers” and the “TFLite Model Schema” documentation. In addition, consider examining the relevant header files in the TFLM source code. Moreover, exploring community forums where TFLM engineers discuss best practices can provide practical guidance on effective use of the available operations. Examining example projects from repositories like the TensorFlow examples can highlight the practical application of these operations. These resources provide information far more specific than any general text on this topic.
