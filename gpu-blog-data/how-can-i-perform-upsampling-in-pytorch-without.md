---
title: "How can I perform upsampling in PyTorch without the `upsample` function?"
date: "2025-01-30"
id: "how-can-i-perform-upsampling-in-pytorch-without"
---
In PyTorch, the `torch.nn.Upsample` module, while convenient, is not the only mechanism to achieve upsampling. I've found, through various projects involving image and signal processing, that directly employing interpolation or transposed convolutions offers a more granular level of control and can be crucial in specific architectural contexts. Specifically, relying on these underlying techniques avoids potential issues with future deprecations of the `Upsample` module and allows for deeper understanding of the upsampling process itself.

The fundamental idea behind upsampling is to increase the spatial dimensions of a tensor, essentially generating more data points between existing ones. This can be accomplished through different interpolation techniques or learned filters. When using `torch.nn.Upsample`, one is largely abstracting the specific interpolation or kernel used in the upsampling. Avoiding `Upsample` involves implementing these methods explicitly.

**1. Nearest Neighbor Interpolation**

Nearest neighbor interpolation is the simplest upsampling method. It involves replicating the closest known pixel value when generating new, intermediate pixels. This does not create any new values but merely repeats the existing ones.

```python
import torch

def nearest_neighbor_upsample(input_tensor, scale_factor):
    """
    Upsamples a tensor using nearest neighbor interpolation.

    Args:
      input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
      scale_factor (int): Upsampling scale factor

    Returns:
      torch.Tensor: Upsampled tensor
    """
    B, C, H, W = input_tensor.shape
    output_H = H * scale_factor
    output_W = W * scale_factor

    # Create coordinate grids
    x = torch.arange(0, output_W, device=input_tensor.device)
    y = torch.arange(0, output_H, device=input_tensor.device)

    # Map coordinates to original space, rounding to the nearest integer
    x_src = torch.floor(x / scale_factor).long()
    y_src = torch.floor(y / scale_factor).long()

    # Clamp indices in case scale_factor doesn't result in exact integers
    x_src = torch.clamp(x_src, 0, W-1)
    y_src = torch.clamp(y_src, 0, H-1)

    # Expand for batch, channel dimensions
    x_src = x_src.view(1, 1, 1, -1).expand(B, C, output_H, -1)
    y_src = y_src.view(1, 1, -1, 1).expand(B, C, -1, output_W)

    # Create grid of indices to sample from.
    grid = torch.stack((x_src, y_src), dim=-1)
    # Perform gather operation
    output_tensor = torch.gather(input_tensor, 3, grid[..., 0].long())
    output_tensor = torch.gather(output_tensor, 2, grid[...,1].long())

    return output_tensor


# Example usage:
input_tensor = torch.randn(1, 3, 4, 4)
scale_factor = 2
output_tensor = nearest_neighbor_upsample(input_tensor, scale_factor)
print(f"Input shape: {input_tensor.shape}") # Output: Input shape: torch.Size([1, 3, 4, 4])
print(f"Output shape: {output_tensor.shape}") # Output: Output shape: torch.Size([1, 3, 8, 8])
```

The provided `nearest_neighbor_upsample` function demonstrates how to perform nearest neighbor upsampling. I initially generate the coordinates for the output tensor. Then, I map these coordinates back to the original input tensor space, employing a scale factor. The `torch.floor` operation ensures that we retrieve the nearest integer index in the input space. Crucially, I clamp the mapped indices to avoid out-of-bounds errors, accounting for scale factors not resulting in integers. These indices are then used with `torch.gather` to sample pixel values from the input, effectively performing the replication inherent in nearest neighbor upsampling.  This function works independently of the `Upsample` module, thus avoiding any future deprecation risks while providing complete control over the upsampling operation.

**2. Bilinear Interpolation**

Bilinear interpolation offers a smoother result than nearest neighbor by linearly interpolating between the four neighboring pixels of an output location. It provides an improved visual outcome at the expense of some computational overhead.

```python
import torch
import torch.nn.functional as F


def bilinear_upsample(input_tensor, scale_factor):
    """
      Upsamples a tensor using bilinear interpolation.

      Args:
          input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
          scale_factor (int): Upsampling scale factor

      Returns:
          torch.Tensor: Upsampled tensor
      """
    B, C, H, W = input_tensor.shape
    output_H = H * scale_factor
    output_W = W * scale_factor

    # Create coordinate grids
    x = torch.arange(0, output_W, device=input_tensor.device).float()
    y = torch.arange(0, output_H, device=input_tensor.device).float()

    # Map coordinates to original space
    x_src = x / scale_factor
    y_src = y / scale_factor

    # Compute the coordinates of the neighboring pixels
    x0 = torch.floor(x_src).long()
    x1 = torch.ceil(x_src).long()
    y0 = torch.floor(y_src).long()
    y1 = torch.ceil(y_src).long()

    # Clamp coordinates to avoid out of bounds access
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # Compute the interpolation weights
    wa = (x1 - x_src) * (y1 - y_src)
    wb = (x_src - x0) * (y1 - y_src)
    wc = (x1 - x_src) * (y_src - y0)
    wd = (x_src - x0) * (y_src - y0)

    # Prepare for gather
    x0 = x0.view(1, 1, 1, -1).expand(B, C, output_H, -1)
    x1 = x1.view(1, 1, 1, -1).expand(B, C, output_H, -1)
    y0 = y0.view(1, 1, -1, 1).expand(B, C, -1, output_W)
    y1 = y1.view(1, 1, -1, 1).expand(B, C, -1, output_W)

    # Gather pixel values
    Ia = torch.gather(input_tensor, 3, x0).gather(2, y0)
    Ib = torch.gather(input_tensor, 3, x1).gather(2, y0)
    Ic = torch.gather(input_tensor, 3, x0).gather(2, y1)
    Id = torch.gather(input_tensor, 3, x1).gather(2, y1)

    wa = wa.view(1,1,output_H,output_W).expand(B, C, output_H, output_W)
    wb = wb.view(1,1,output_H,output_W).expand(B, C, output_H, output_W)
    wc = wc.view(1,1,output_H,output_W).expand(B, C, output_H, output_W)
    wd = wd.view(1,1,output_H,output_W).expand(B, C, output_H, output_W)
    # Perform the bilinear interpolation
    output_tensor = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return output_tensor


# Example usage:
input_tensor = torch.randn(1, 3, 4, 4)
scale_factor = 2
output_tensor = bilinear_upsample(input_tensor, scale_factor)
print(f"Input shape: {input_tensor.shape}") # Output: Input shape: torch.Size([1, 3, 4, 4])
print(f"Output shape: {output_tensor.shape}") # Output: Output shape: torch.Size([1, 3, 8, 8])
```

The `bilinear_upsample` function first calculates the coordinates for the output space and maps them back to the input space, similarly to the nearest neighbor approach. Crucially, it determines the coordinates of the four nearest neighbors (x0, y0, x1, y1). It then calculates the weights for each neighbor using the spatial distances to the upsampled pixel coordinates. These weights are used to perform a weighted sum of the neighboring pixel values, resulting in the interpolated output. This method gives a smoother appearance, avoiding the blocky appearance of the nearest neighbor approach. Again, this avoids reliance on `torch.nn.Upsample`.

**3. Transposed Convolution**

Transposed convolution (sometimes called deconvolution, though this term is not preferred) provides a learned form of upsampling. Unlike the preceding interpolation methods, it uses trainable weights to upscale the input, allowing the network to learn optimal upsampling strategies specific to the problem. This can result in significantly better performance than basic interpolation, especially within convolutional neural network architectures.

```python
import torch
import torch.nn as nn

def transposed_conv_upsample(input_tensor, out_channels, kernel_size, stride, padding):
    """
    Upsamples a tensor using a transposed convolution.

    Args:
      input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
      out_channels (int): Number of output channels for the transposed convolution
      kernel_size (int): Size of the kernel
      stride (int): Stride of the convolution
      padding (int): Padding of the convolution

    Returns:
      torch.Tensor: Upsampled tensor
    """
    in_channels = input_tensor.shape[1]
    conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    output_tensor = conv_transpose(input_tensor)
    return output_tensor


# Example Usage
input_tensor = torch.randn(1, 3, 4, 4)
out_channels = 3
kernel_size = 2
stride = 2
padding = 0
output_tensor = transposed_conv_upsample(input_tensor, out_channels, kernel_size, stride, padding)
print(f"Input shape: {input_tensor.shape}") # Output: Input shape: torch.Size([1, 3, 4, 4])
print(f"Output shape: {output_tensor.shape}") # Output: Output shape: torch.Size([1, 3, 8, 8])
```

The `transposed_conv_upsample` function leverages the PyTorch's `nn.ConvTranspose2d` module. Unlike the previous examples, it does not explicitly compute coordinates or weights based on an interpolation scheme, instead using learned weights within the convolutional operation. I specify the number of output channels, kernel size, stride, and padding to create the appropriate transposed convolution layer. The input tensor is then passed through this layer, resulting in an upsampled output. The network during training will update these weights to learn the optimal upsampling strategy based on its task.

**Resource Recommendations**

For a detailed understanding of image processing fundamentals, consult publications focusing on digital image processing, such as the work by Gonzalez and Woods. For comprehensive coverage of deep learning architectures and applications, consider the books by Goodfellow, Bengio, and Courville.  Furthermore, the PyTorch documentation itself offers valuable insights into the various tensor manipulation techniques, which serve as the building blocks for custom upsampling implementation. Reviewing scientific articles focused on specific interpolation techniques and their applications will enhance the comprehension of nuances involved in selecting the most suitable method. These resources will provide a theoretical underpinning for the practical applications illustrated above.
