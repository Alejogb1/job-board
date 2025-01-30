---
title: "What's the difference between linear and bilinear interpolation in PyTorch's `interpolate` function?"
date: "2025-01-30"
id: "whats-the-difference-between-linear-and-bilinear-interpolation"
---
When resizing images or feature maps using `torch.nn.functional.interpolate`, the choice between linear and bilinear interpolation impacts how new pixel values are computed, and consequently, the visual outcome. My experience implementing custom upsampling layers in several computer vision projects has highlighted the specific nuances of each method. Linear interpolation, in this context, is typically applicable to one-dimensional data such as sequences or time series, whereas bilinear interpolation extends this concept to two-dimensional data such as images and feature maps. Crucially, the "linear" and "bilinear" terminology refers to the dimensionality of the interpolation *operation* itself, not necessarily the dimensionality of the input tensor. PyTorch's `interpolate` function handles these differences effectively based on the input tensor's rank and specified mode.

Linear interpolation, at its core, finds a value along a straight line between two known values. When applied to a 1D tensor undergoing resizing, `interpolate` with `mode='linear'` computes values for new locations by considering the two nearest input data points on either side.  Specifically, given a location *x* in the output tensor between two locations *x1* and *x2* in the input tensor, where the corresponding values are *y1* and *y2*, the interpolated value *y* is calculated as:

*y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)*

The division by *(x2 - x1)* normalizes the distance, ensuring the result is weighted proportionally to its proximity to *x1* and *x2*. This is done for each location across the resized dimension.

Bilinear interpolation, on the other hand, extrapolates this logic to two dimensions.  Instead of a straight line, imagine a plane. To compute a new pixel at location *(x, y)* in the output image, the algorithm needs to first determine the four nearest known pixels in the input image: *(x1, y1)*, *(x2, y1)*, *(x1, y2)*, and *(x2, y2)* with corresponding pixel values *z11*, *z21*, *z12*, and *z22* respectively. It then performs two linear interpolations along one axis (e.g. x-axis) and finally a single interpolation along the other axis (y-axis) using these results, this method is computationally more expensive than linear interpolation. First, we perform linear interpolation on the values associated with the x-axis:

*z_top = z11 + (z21 - z11) * (x - x1) / (x2 - x1)*

*z_bottom = z12 + (z22 - z12) * (x - x1) / (x2 - x1)*

Then we perform linear interpolation on the values z_top and z_bottom based on the y axis

*z = z_top + (z_bottom - z_top) * (y - y1) / (y2 - y1)*

This resulting value *z* represents the interpolated pixel value at position *(x, y)*.

The choice between linear and bilinear isn't just about dimensionality, though. It is also about how accurately detail is preserved during the resize operation. Linear interpolation, while faster, can introduce noticeable artifacts like "blockiness" if applied to images. Bilinear interpolation generally results in smoother and more visually pleasing results by considering data from more source pixels to compute each target pixel. This is why I consistently use bilinear interpolation when working with image data, unless the computational cost is absolutely prohibitive.

Now, let's illustrate these concepts with some code examples.

**Example 1: Linear Interpolation on a 1D Tensor**

```python
import torch
import torch.nn.functional as F

# Simulate a 1D signal
input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(0).unsqueeze(0) # shape (1, 1, 5)
print(f"Input: {input_tensor.squeeze()}")

# Upsample to double the length using linear interpolation
output_tensor = F.interpolate(input_tensor, size=(10,), mode='linear', align_corners=False)
print(f"Upsampled Linear: {output_tensor.squeeze().tolist()}")

#Downsample to half the length using linear interpolation
output_tensor = F.interpolate(input_tensor, size=(2,), mode='linear', align_corners=False)
print(f"Downsampled Linear: {output_tensor.squeeze().tolist()}")

```

This example demonstrates resizing a 1D tensor. I convert the single dimensional vector into a tensor of shape (1,1,5) because the interpolate function requires input with at least three dimensions. The `size` argument determines the target length. Linear interpolation calculates the interpolated values based on the neighboring data point which is evident in both the upsampling and downsampling results.  `align_corners=False` is the generally recommended behavior.

**Example 2: Bilinear Interpolation on a 2D Tensor (Image-like data)**

```python
import torch
import torch.nn.functional as F

# Simulate a small grayscale image (height, width)
input_image = torch.tensor([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]]).unsqueeze(0) # shape (1, 1, 3, 3)

print(f"Input:\n{input_image.squeeze()}")

# Upsample the image using bilinear interpolation
output_image = F.interpolate(input_image, size=(6, 6), mode='bilinear', align_corners=False)
print(f"Upsampled Bilinear:\n{output_image.squeeze().round()}")

#Downsample the image using bilinear interpolation
output_image = F.interpolate(input_image, size=(2, 2), mode='bilinear', align_corners=False)
print(f"Downsampled Bilinear:\n{output_image.squeeze().round()}")

```

This example utilizes a 2D tensor mimicking a grayscale image. The `interpolate` function with `mode='bilinear'` is used to both increase and decrease the image's spatial dimensions. Notice the resulting values in the upsampled tensor and how they smoothly transition between the original pixel values. Downsampling a tensor via bilinear interpolation results in an averaging of values from the source data. Again `align_corners=False` is the standard recommended behavior.

**Example 3: Impact of Align Corners**

```python
import torch
import torch.nn.functional as F

input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(0).unsqueeze(0)
print(f"Input: {input_tensor.squeeze()}")

# Linear Interpolation with align_corners=True
output_tensor_true = F.interpolate(input_tensor, size=(10,), mode='linear', align_corners=True)
print(f"Upsampled Linear (align_corners=True): {output_tensor_true.squeeze().tolist()}")


# Linear Interpolation with align_corners=False
output_tensor_false = F.interpolate(input_tensor, size=(10,), mode='linear', align_corners=False)
print(f"Upsampled Linear (align_corners=False): {output_tensor_false.squeeze().tolist()}")

```

This example demonstrates the effect of the `align_corners` parameter, in this case when applied to linear interpolation of a 1D tensor. With `align_corners=True`, the corner pixels (the first and last elements) in the input are *always* aligned with the corresponding corner pixels in the output during interpolation.  This can be helpful in certain scenarios, especially when dealing with coordinate systems. When `align_corners=False`, the corners of the input do not necessarily correspond to the corners of the output which may result in small offsets. The choice between `True` and `False` depends on the specific use case and desired behavior of the resize operation; however, `False` is often recommended because it produces slightly more natural results when upsampling.

For further study on resizing methods, consider researching common image processing texts that explain how sampling and anti-aliasing relate to interpolation. Additionally, detailed descriptions of resizing techniques are available in publications related to computer vision and graphics. Consulting mathematical resources that discuss linear algebra and calculus may prove beneficial when one is striving for a theoretical understanding of the underlying interpolation processes. These resources provide a rigorous and comprehensive view on the subject. Understanding the mathematical underpinnings deepens the knowledge on how to properly employ methods such as linear and bilinear interpolation, which leads to better results when upsampling or downsampling data.
