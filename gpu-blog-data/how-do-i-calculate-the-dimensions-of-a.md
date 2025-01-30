---
title: "How do I calculate the dimensions of a CNN's first convolutional layer?"
date: "2025-01-30"
id: "how-do-i-calculate-the-dimensions-of-a"
---
The output dimensions of a convolutional layer, particularly the first layer of a Convolutional Neural Network (CNN), are not simply determined by the input size but also by several critical hyperparameters. Specifically, the kernel size (filter size), stride, padding, and the number of output channels (filters) must be considered to compute the resulting feature map's spatial dimensions. I've encountered this regularly while architecting image processing pipelines for medical imaging applications, where maintaining precise spatial correspondence between feature maps and original data is crucial.

The fundamental operation of a convolutional layer involves sliding a kernel, also known as a filter, across the input data. The kernel performs element-wise multiplication with the corresponding region of the input and then sums the results, producing a single value in the output feature map. The way the kernel moves across the input, specifically how many pixels it skips in horizontal and vertical directions (stride), and how the edges of the input are handled (padding) are crucial factors that determine the spatial dimensions of the output. It’s not simply a matter of subtracting kernel size from input size.

Let's break down the common formula for calculating the output dimensions of a convolutional layer. Assuming a square input, where the height and width are equal (`input_size`), a square kernel of size `kernel_size`, a stride of `stride` in both directions, and `padding` applied to each side, the output size `output_size` can be computed as:

`output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1`

The `floor` function is used to obtain the integer part because feature map dimensions must be whole numbers, discarding any fractional remainder that may result from division. This formula provides the spatial dimensions (width and height) of the feature map. The depth of the feature map, which represents the number of output channels, is determined directly by the number of filters in the convolutional layer and is independent of the input size and padding. If the input is not square, the formula can be applied separately to calculate the output height and width by using the corresponding input height, input width, and stride values for height and width.

Here are three code examples demonstrating different configurations and illustrating the influence of the hyperparameters on output dimensions. These examples use Python and the PyTorch library, which is frequently employed in deep learning research and development.

**Example 1: No Padding, Stride 1**

```python
import torch
import torch.nn as nn

# Input parameters
input_size = 32 # Square input 32x32
kernel_size = 3
stride = 1
padding = 0
out_channels = 16 # 16 output channels (filters)

# Define the convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding)

# Example Input
input_tensor = torch.randn(1, 3, input_size, input_size) # Batch size 1, 3 color channels

# Compute the output dimensions manually
output_size = int((input_size + 2 * padding - kernel_size) / stride) + 1

# Apply the convolutional layer and check output
output_tensor = conv_layer(input_tensor)
print(f"Input Shape: {input_tensor.shape}")
print(f"Output Shape: {output_tensor.shape}")
print(f"Calculated Output Size: {output_size}x{output_size} ")
assert output_tensor.shape[2] == output_size
assert output_tensor.shape[3] == output_size
```

In this example, the input is a 32x32 image with 3 color channels. The kernel size is 3, stride is 1, and no padding is used. The formula computes an output size of (32 - 3 + 1) = 30. We expect an output tensor with dimensions of 1 (batch size) x 16 (output channels) x 30 x 30. The assertions verify our computation against the tensor's actual dimensions. As seen, a zero padding reduces the output spatial dimensions of the feature map.

**Example 2: Padding and Stride greater than 1**

```python
import torch
import torch.nn as nn

# Input parameters
input_size = 64
kernel_size = 5
stride = 2
padding = 2
out_channels = 32

# Define the convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding)

# Example input
input_tensor = torch.randn(1, 3, input_size, input_size)

# Compute the output dimensions manually
output_size = int((input_size + 2 * padding - kernel_size) / stride) + 1

# Apply the convolutional layer and check output
output_tensor = conv_layer(input_tensor)
print(f"Input Shape: {input_tensor.shape}")
print(f"Output Shape: {output_tensor.shape}")
print(f"Calculated Output Size: {output_size}x{output_size} ")
assert output_tensor.shape[2] == output_size
assert output_tensor.shape[3] == output_size
```

This example uses a 64x64 input, a kernel size of 5, a stride of 2, and a padding of 2. Applying the formula yields (64 + 4 - 5) / 2 + 1 = 32.5 which after the floor function results to 32. Thus, the output shape is 1 x 32 x 32 x 32.  Stride greater than one results in spatial downsampling of the feature maps. The inclusion of padding keeps the feature map from shrinking too rapidly.

**Example 3: Handling Rectangular Inputs**
```python
import torch
import torch.nn as nn

# Input parameters
input_height = 48
input_width = 64
kernel_size = 3
stride_height = 2
stride_width = 1
padding_height = 1
padding_width = 0
out_channels = 64

# Define the convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=out_channels,
                     kernel_size=kernel_size, stride=(stride_height, stride_width), padding=(padding_height, padding_width))

# Example input
input_tensor = torch.randn(1, 3, input_height, input_width)

# Compute the output dimensions manually
output_height = int((input_height + 2 * padding_height - kernel_size) / stride_height) + 1
output_width = int((input_width + 2 * padding_width - kernel_size) / stride_width) + 1

# Apply the convolutional layer and check output
output_tensor = conv_layer(input_tensor)
print(f"Input Shape: {input_tensor.shape}")
print(f"Output Shape: {output_tensor.shape}")
print(f"Calculated Output Size: {output_height}x{output_width} ")
assert output_tensor.shape[2] == output_height
assert output_tensor.shape[3] == output_width
```

This example addresses non-square inputs. An input with height 48 and width 64 is used. Different strides (2 and 1) and paddings (1 and 0) are applied to height and width dimensions, respectively. Output height is (48 + 2 - 3) / 2 + 1 = 24, and output width is (64 + 0 - 3) / 1 + 1 = 62.  These calculations allow to handle a more complex convolutional operation. The separate stride values for horizontal and vertical dimension introduce further granularity to spatial sampling.

These examples illustrate how different hyperparameter choices impact the resulting feature map's dimensions. These calculations, while seemingly straightforward, require careful consideration during the design of convolutional networks to ensure that the feature maps remain meaningful and the network architecture aligns with the required task.

For those seeking a deeper understanding of CNNs and convolutional layers, I recommend resources focusing on the theoretical foundations of deep learning.  Specifically, books covering topics such as deep learning fundamentals, convolutional network architectures, and computer vision are valuable.  Additionally, reputable online courses can provide a structured learning path with practical exercises to solidify these concepts. Peer-reviewed research papers provide the most up-to-date and in-depth analysis of advancements in the field of deep learning and should be consulted if one intends to stay at the cutting edge.
