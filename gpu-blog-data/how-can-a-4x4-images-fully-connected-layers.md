---
title: "How can a 4x4 image's fully connected layers be replaced with convolutional layers?"
date: "2025-01-30"
id: "how-can-a-4x4-images-fully-connected-layers"
---
The core challenge in transitioning a fully connected (FC) layer to a convolutional (Conv) layer lies in reinterpreting the matrix multiplication of the FC layer as a spatially aware filtering operation. Specifically, an FC layer treats the input as a flattened vector, losing spatial information. A Conv layer, conversely, preserves this information by applying filters across the input's spatial dimensions. The crucial aspect is manipulating the kernel size and stride to achieve the equivalent mapping.

The typical scenario where this arises involves adapting a classifier trained on small images, such as those used in early deep learning benchmarks like CIFAR-10, for more general inputs. These classifiers often feature FC layers immediately after the final convolutional layers to reduce the dimensionality of the feature maps to a class label space. However, using Conv layers instead allows for greater flexibility, such as applying the same trained network to inputs of different sizes without requiring retraining. Furthermore, convolution's parameter sharing and local receptive fields provide computational advantages and increased efficiency.

Let's assume we have a simple network: a series of convolutional layers followed by two fully connected layers. The input to the FC layers is a 4x4 feature map with, say, 256 channels. This flattened into a 4096-element vector, acting as the input for our first FC layer. This first FC layer then outputs a 1024-dimensional vector and is followed by a second FC layer which gives the final 10-dimensional output corresponding to our class predictions. The objective is to replace these two FC layers with Conv layers.

The first essential step is understanding the equivalent kernel and stride configurations. The flattening of the 4x4x256 feature map within the fully connected layer is equivalent to performing a 1x1 convolution with a kernel size of 4x4 and a stride of 1. Each position on the 4x4 input receives a unique 4x4 filter that covers the whole input space. The number of these filters dictates the output dimensionality. This differs from a convolutional layer in that all 4x4 filter applications happen within one spatial location for that single output. This operation preserves the "depth" of the filters. We can visualize this operation by noting that a fully connected layer with 4096 input neurons and 1024 output neurons is equivalent to a convolution with 1024 filters of size 4x4 and a single spatial unit. This means we'll have a single value at every "depth" or channel (output depth).

To replace the FC layer which had 1024 output neurons, a corresponding convolutional layer would require 1024 such filters, meaning 1024 output channels. The input to our first convolutional replacement will be 4x4x256, and the output will be 1x1x1024. Similarly, to get the output to 1x1x10, for the second FC layer, we would use a 1x1 convolution to map from the 1x1x1024 feature map to a 1x1x10 feature map. This is because the 1x1 convolution effectively blends the 1024 channels into 10, as the filter size is 1. This is different from the prior convolution in that it does not have a receptive field of more than 1 pixel.

Let's examine how this is done in practice, using Python with TensorFlow. I have included equivalent PyTorch code as well, for broader comprehension.

**Code Example 1: TensorFlow Implementation**

```python
import tensorflow as tf

# Assume input from a previous convolutional layer
input_tensor = tf.random.normal((1, 4, 4, 256))  # Batch size of 1

#Original FC layer 1 equivalent conv layer
conv1 = tf.keras.layers.Conv2D(1024, (4, 4), strides=(1, 1), padding='valid')(input_tensor)  # Output: (1, 1, 1, 1024)

#Original FC layer 2 equivalent conv layer
conv2 = tf.keras.layers.Conv2D(10, (1, 1), strides=(1, 1), padding='valid')(conv1)  # Output: (1, 1, 1, 10)
print(f"Shape of conv2 output: {conv2.shape}")
#The output shape represents a single batch with a 1x1 spatial map and 10 channels
```

In this TensorFlow example, `Conv2D` with a `(4,4)` kernel and a stride of `(1,1)` achieves the initial flattening effect of the original fully connected layer for the input 4x4. The padding set to 'valid' ensures no additional padding is applied to input feature map. The second convolution utilizes a 1x1 kernel which is equivalent to a matrix multiplication and allows us to map from 1024 channels to 10. The output demonstrates a 1x1 spatial size, as expected, with the number of output feature maps being 10. The shape of the output will be [1, 1, 1, 10].

**Code Example 2: PyTorch Implementation**

```python
import torch
import torch.nn as nn

# Assume input from a previous convolutional layer
input_tensor = torch.randn(1, 256, 4, 4)  # Batch size of 1

#Original FC layer 1 equivalent conv layer
conv1 = nn.Conv2d(256, 1024, kernel_size=(4, 4), stride=(1, 1), padding=0)(input_tensor)
# Output: (1, 1024, 1, 1)

#Original FC layer 2 equivalent conv layer
conv2 = nn.Conv2d(1024, 10, kernel_size=(1, 1), stride=(1, 1), padding=0)(conv1)
# Output: (1, 10, 1, 1)
print(f"Shape of conv2 output: {conv2.shape}")
#The output shape represents a single batch with 1x1 spatial map and 10 channels
```

The PyTorch version is analogous. The `nn.Conv2d` module is used to construct both convolutional layers, with equivalent parameters passed. It is essential to note the channel order differences between Tensorflow and PyTorch. TensorFlow typically uses channels-last order, where the channel dimension is the last dimension of the input tensor, whereas PyTorch typically uses channels-first order where the channel dimension is the second dimension of the input tensor. This is why the PyTorch input has 256 as the second dimension, but in Tensorflow, this was the last. This is reflected in the convolutional layer instantiation, specifically, the input channel argument is specified before the number of output filters. The kernel size, stride and padding are set according to the same principles as the Tensorflow example.

**Code Example 3: Application to Larger Input Sizes**

```python
import tensorflow as tf
# Consider a slightly larger input shape from a convolutional layer
input_tensor_large = tf.random.normal((1, 8, 8, 256))

# Replace FC with Conv, no need to retrain.
conv_large_input = tf.keras.layers.Conv2D(1024, (4, 4), strides=(1, 1), padding='valid')(input_tensor_large)

conv2_large_input = tf.keras.layers.Conv2D(10, (1, 1), strides=(1,1), padding='valid')(conv_large_input)

print(f"Shape of conv2 output with larger input: {conv2_large_input.shape}")
#The output shape is [1, 5, 5, 10] where the 5 is the output size after passing the larger input to the initial convolutional layer
```
Here, I have demonstrated how a larger input can be passed through the same equivalent convolutional network. By using equivalent convolutional layers, as opposed to fully connected layers, the spatial size is not forced to be one. This is because the initial convolutional layer maps from 8x8 to 5x5, using a 4x4 filter and stride of 1. It also maps the initial 256 channels to 1024. The second convolutional layer reduces this down to 10 channels, with the spatial size remaining at 5x5.

By applying these convolution-based replacements, the network gains several crucial advantages. First, the model can now process images of varying input sizes without retraining. Second, the use of convolutional operations is more memory efficient and amenable to parallel processing. Third, replacing fully connected layers with convolution enables leveraging techniques such as convolutional sliding window approach for object detection.

For more comprehensive resources, I would suggest delving into the following material. Firstly, consider reviewing introductory textbooks on Deep Learning, particularly those covering Convolutional Neural Networks. In addition to these, explore practical implementations of CNNs, for instance, those in the TensorFlow and PyTorch documentation. I would recommend specific attention to sections on layer instantiation and spatial resolution manipulation. Research papers concerning Fully Convolutional Networks (FCNs) would also offer valuable insights, especially those addressing the conversion of traditional classifier networks to segmentation architectures. These materials provide a strong understanding of both the theory and practice, and enable developers to be more effective in the creation of such networks.
