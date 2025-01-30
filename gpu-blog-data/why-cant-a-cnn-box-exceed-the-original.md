---
title: "Why can't a CNN box exceed the original image size?"
date: "2025-01-30"
id: "why-cant-a-cnn-box-exceed-the-original"
---
Convolutional Neural Networks (CNNs), in their standard implementation, often operate under a constraint: the spatial dimensions of feature maps—the intermediate representations of the image within the network—cannot surpass the dimensions of the input image. This limitation stems from the core operations that define CNNs, namely convolution and pooling. It’s not an inherent theoretical barrier, but rather a consequence of how these layers are typically structured and utilized to extract hierarchical features. My experience developing image processing pipelines has consistently reinforced this behavior, leading to practical understanding of the underlying mechanisms.

The primary reason why feature map size generally decreases or at most remains the same is the convolution operation itself. A convolutional layer applies a kernel—a small matrix of weights—across the input feature map. The kernel is multiplied element-wise with a portion of the input, summed, and the result becomes a single value in the output feature map. This process is repeated across the entire input. Crucially, the spatial extent of the output feature map is dictated by the input size, the kernel size, the stride (the distance the kernel moves in each step), and the padding (the addition of extra pixels around the edges).

With valid padding, no padding is added to the input image. Each convolution step shrinks the output feature map compared to the input feature map. The number of pixels to be discarded at each edge is directly determined by the kernel size, minus 1, divided by 2. As an example a 3x3 kernel removes 1 pixel on each edge, a 5x5 kernel removes two, and so on. This ensures that no values outside the original feature map are used in the convolution process, thus avoiding artificial features on the image boundary.

Same padding aims to keep the spatial resolution the same, as the name suggests. The amount of padding is dependent on the size of the kernel, usually it is calculated to ensure the input dimension is preserved. For example, a 3x3 kernel would have a padding of 1 on each side of the input feature map. The output is the same size as the input, and no values are lost.

Another common operation, pooling, specifically reduces spatial dimensions. Pooling layers, such as max pooling or average pooling, typically slide a small window across the feature map and output a single aggregated value (the maximum or average, respectively). By default, pooling operations, especially max pooling, often utilize a stride equal to the window size, which significantly reduces the spatial dimensions. A 2x2 max pooling layer with a stride of 2 will effectively halve the width and height of its input.

While convolution and pooling operations, when used in a typical way, ensure a reduced or unchanged feature map size, it is not impossible to increase the spatial dimensions within a CNN. Techniques such as transposed convolutions and upsampling methods are employed when increasing the resolution of a feature map is required. However, these are not standard convolutional layers, and they introduce parameters to learn in the process.

Let me illustrate this with a few examples, using a Python-based framework like TensorFlow or PyTorch. I've often employed such tools in my daily work with image data:

**Example 1: Basic Convolution with Valid Padding**

```python
import tensorflow as tf

# Input image of size 28x28 with 3 channels
input_image = tf.random.normal(shape=(1, 28, 28, 3))

# Convolutional layer with a 3x3 kernel and no padding
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid')

# Apply convolution
output_feature_map = conv_layer(input_image)

# Output feature map size
print(output_feature_map.shape) # Output: (1, 26, 26, 32)
```

Here, I’ve created a simple convolutional layer with no padding. Notice that the input image is 28x28, and the output feature map is 26x26, demonstrating the spatial dimension reduction characteristic of valid padding. The number of channels has increased to 32 through the convolution. The batch size of 1 remains unchanged.

**Example 2: Convolution with Same Padding**

```python
import tensorflow as tf

# Input image of size 28x28 with 3 channels
input_image = tf.random.normal(shape=(1, 28, 28, 3))

# Convolutional layer with a 3x3 kernel and same padding
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')

# Apply convolution
output_feature_map = conv_layer(input_image)

# Output feature map size
print(output_feature_map.shape) # Output: (1, 28, 28, 32)
```

In this example, the convolutional layer uses ‘same’ padding. The output feature map is the same size as the input, specifically 28x28. This is how one keeps the spatial dimensions constant between layers of a network.

**Example 3: Max Pooling Layer**

```python
import tensorflow as tf

# Input feature map of size 28x28 with 32 channels
input_feature_map = tf.random.normal(shape=(1, 28, 28, 32))

# Max pooling layer
max_pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

# Apply max pooling
output_feature_map = max_pool_layer(input_feature_map)

# Output feature map size
print(output_feature_map.shape) # Output: (1, 14, 14, 32)
```

The max pooling layer reduces the spatial dimensions of the input feature map by half (to 14x14), whilst keeping the channel dimension unchanged at 32. Max pooling also reduces the parameters for a network, increasing the speed of calculation. This pooling operation demonstrates the dimension reduction typical of these layers.

These examples highlight how convolution and pooling layers, in their standard configurations, either maintain or reduce the spatial size of feature maps. It is important to note that specialized layers, such as transposed convolutions, do allow for upsampling operations to increase spatial dimensions but those are not part of the standard convolution and pooling layers in a network.

To further solidify understanding, several resources are beneficial. A comprehensive book on deep learning, especially ones that dedicate sections to CNNs, provides a solid theoretical foundation. University courses on computer vision and deep learning often offer in-depth explanations of these operations and the math involved. Online blogs and tutorials dedicated to deep learning also provide an excellent resource for practical examples and applications of CNN architectures. However, hands-on experience with image processing code is ultimately what solidifies understanding of the operations and their limitations. The technical documentation provided by libraries like TensorFlow and PyTorch are essential references as well. These resources, combined with practical experimentation, have equipped me to handle a variety of computer vision tasks effectively. The key constraint of standard convolutional layers is a consequence of how they have been designed, not a fundamental theoretical limitation.
