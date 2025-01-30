---
title: "How can I create a single patch using PyTorch's `unfold` method?"
date: "2025-01-30"
id: "how-can-i-create-a-single-patch-using"
---
The `torch.nn.functional.unfold` method in PyTorch is primarily designed for extracting multiple overlapping patches from an input tensor, rather than creating a single patch directly. Its utility lies in efficient batch processing of these patches for operations like convolution. However, if the explicit goal is to isolate a single, specific patch from an input, `unfold` can be leveraged, albeit with careful parameterization. This approach deviates from its typical use case, requiring a precise understanding of how `unfold` defines the output shape based on its input parameters.

The core challenge in extracting a single patch is to manipulate `unfold`'s parameters such that it generates only one segment within its output tensor. The output tensor of `unfold` has a shape of `[N, C x K_H x K_W, L]`, where:

*   `N` is the batch size (the same as the input batch).
*   `C` is the number of input channels.
*   `K_H` and `K_W` are the kernel height and width, respectively, defined by the `kernel_size` argument.
*   `L` is the number of unfolded patches, or spatial locations where a patch is extracted.

The `L` dimension, which determines the number of output patches, is controlled by the input tensor’s spatial dimensions, kernel dimensions, stride, and padding. To extract a single patch, we need to ensure `L` equals 1. This typically means that the selected location of the "top-left" corner of the extracted patch, as determined by the effective sliding window position within `unfold`, is the only position with the valid window, given the selected stride and padding.

To achieve this, I have consistently found that setting the stride to equal the kernel dimensions and padding to zero is essential for single patch extraction when the starting location is the top-left corner. Essentially, the kernel will "jump" directly to the targeted single patch with the stride, creating exactly one unfolded patch when applied to the input tensor.

Here are three code examples demonstrating this approach:

**Example 1: Extracting the Top-Left Patch**

This example extracts the single patch starting at the top-left corner, which is typically location `[0,0]` from a tensor.

```python
import torch
import torch.nn.functional as F

# Input Tensor: Batch, Channels, Height, Width
input_tensor = torch.arange(1, 26, dtype=torch.float).reshape(1, 1, 5, 5) # Single channel 5x5 tensor

kernel_size = (3, 3)
stride = kernel_size
padding = 0

unfolded_tensor = F.unfold(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding)

# Reshape to original patch shape: Batch, Channels, Height, Width
single_patch = unfolded_tensor.reshape(1, 1, kernel_size[0], kernel_size[1])

print("Original Input Tensor:\n", input_tensor)
print("\nExtracted Single Patch:\n", single_patch)
```

In this example, the `input_tensor` is a 5x5 matrix. I want to extract a 3x3 patch from the top-left. The stride is set to `(3, 3)`, effectively making the window jump straight to the single patch location. The output is then reshaped into the original patch dimensions, using the kernel size values, confirming we've extracted a single 3x3 patch.

**Example 2: Extracting a Patch from the Center of the Image**

Extracting a patch that is not located at `[0,0]` requires careful calculation of the input tensor size, kernel size, padding, and the stride. The most simple method for selecting a location other than the top-left is to select the required region of the input tensor and then apply the single patch extraction method to this slice. The input tensor will be effectively treated as if its top left corner is at the start of the slice. In this instance, I'm slicing the original tensor to create an effective "image" with an off-set start position before extracting a patch from the slice.

```python
import torch
import torch.nn.functional as F

# Input Tensor: Batch, Channels, Height, Width
input_tensor = torch.arange(1, 26, dtype=torch.float).reshape(1, 1, 5, 5)

kernel_size = (3, 3)
stride = kernel_size
padding = 0

# Extract the center location based on kernel_size and input shape
h_start = input_tensor.shape[2]//2-kernel_size[0]//2
w_start = input_tensor.shape[3]//2-kernel_size[1]//2

# Slice the input tensor with the start indices to effectively center the tensor around patch extraction location.
sliced_input = input_tensor[:,:,h_start:h_start+kernel_size[0],w_start:w_start+kernel_size[1]]

unfolded_tensor = F.unfold(sliced_input, kernel_size=kernel_size, stride=stride, padding=padding)

# Reshape to original patch shape: Batch, Channels, Height, Width
single_patch = unfolded_tensor.reshape(1, 1, kernel_size[0], kernel_size[1])

print("Original Input Tensor:\n", input_tensor)
print("\nExtracted Center Patch:\n", single_patch)
```

Here, before passing into `unfold`, I calculate start coordinates for the center of the input tensor based on its dimensions and the kernel size. I then slice the original tensor based on those calculated coordinates. The center patch can then be extracted by using a standard `unfold` command, which will extract a patch from the top left of this slice.

**Example 3: Extracting a Patch with Batch Size > 1**

The approach works with batch sizes greater than 1, the key thing being to keep the correct values for kernel size, padding, and stride.

```python
import torch
import torch.nn.functional as F

# Input Tensor: Batch, Channels, Height, Width
input_tensor = torch.arange(1, 76, dtype=torch.float).reshape(2, 1, 5, 5)  # Batch size 2

kernel_size = (3, 3)
stride = kernel_size
padding = 0

unfolded_tensor = F.unfold(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding)

# Reshape to original patch shape: Batch, Channels, Height, Width
single_patch = unfolded_tensor.reshape(2, 1, kernel_size[0], kernel_size[1])

print("Original Input Tensor:\n", input_tensor)
print("\nExtracted Single Patch (Batch 2):\n", single_patch)
```

This example expands the previous examples to a batch size of 2, demonstrating the method's consistency and validity across batch sizes. The single patch that is extracted is valid for both batched examples.

In each of these examples, the critical factors to achieve the desired result are the stride equaling the kernel size and the padding being zero. These parameters ensure that `unfold` extracts exactly one patch from each position in the batch. This patch will start at the top-left of the input tensor (or slice in example 2), given no additional positional offsets for the `unfold` parameter. The method is computationally efficient for extracting single patches by leveraging the inherent optimization of the `unfold` operation, despite it being designed for multi-patch extraction.

For further learning and development, I recommend exploring the PyTorch documentation for `torch.nn.functional.unfold`, and associated functions such as `torch.nn.functional.fold`. In addition, studying source code from established machine learning frameworks that utilize patch-based processing pipelines is valuable. Research papers that discuss convolution and signal processing operations can offer deeper insights into the mathematical foundations underpinning the behavior of the `unfold` operation. These resources can enhance understanding beyond this specific application, deepening fundamental knowledge of tensor manipulation and image processing.
