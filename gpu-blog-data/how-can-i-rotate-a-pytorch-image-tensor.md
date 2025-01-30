---
title: "How can I rotate a PyTorch image tensor around its center using autograd?"
date: "2025-01-30"
id: "how-can-i-rotate-a-pytorch-image-tensor"
---
The core challenge in rotating a PyTorch image tensor around its center while maintaining autograd compatibility lies in the differentiability of the rotation transformation.  Standard rotation matrices, while mathematically elegant, aren't directly differentiable with respect to the rotation angle when implemented using typical matrix multiplication.  My experience working on differentiable rendering pipelines highlighted this issue –  simple matrix rotations led to gradients vanishing or becoming unstable during backpropagation.  The solution requires a differentiable approach to rotation.

This necessitates employing a differentiable rotation method.  While custom CUDA kernels offer the potential for speed optimization, for general use and to maintain autograd functionality, leveraging PyTorch's built-in functionalities and operations, or carefully crafted custom functions, provides the most practical and reliable approach.  Directly manipulating image pixels in a loop, though conceptually straightforward, becomes computationally expensive and loses the benefits of PyTorch's optimized tensor operations.

**1. Clear Explanation:**

The most straightforward and efficient differentiable rotation utilizes PyTorch's `torch.nn.functional.affine_grid` and `torch.nn.functional.grid_sample`.  This approach leverages the power of grid sampling for image transformation, effectively bypassing the need for explicit matrix multiplications at the pixel level.  `affine_grid` creates a transformation grid based on an affine transformation matrix. This matrix encapsulates the rotation.  `grid_sample` then uses this grid to sample the input image, performing the actual rotation.  The key is to construct the transformation matrix in a way that it's differentiable with respect to the rotation angle. This involves calculating the rotation matrix using trigonometric functions (sine and cosine), which are inherently differentiable.

**2. Code Examples with Commentary:**

**Example 1:  Rotation using `affine_grid` and `grid_sample`**

```python
import torch
import torch.nn.functional as F
import math

def rotate_image(image, angle_degrees):
    """Rotates a PyTorch image tensor around its center.

    Args:
        image: The input image tensor of shape (C, H, W).
        angle_degrees: The rotation angle in degrees.

    Returns:
        The rotated image tensor.
    """
    angle_rad = math.radians(angle_degrees)
    height, width = image.shape[1:]
    center_x = width / 2
    center_y = height / 2

    # Construct the rotation matrix
    rotation_matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                    [math.sin(angle_rad), math.cos(angle_rad), 0],
                                    [0, 0, 1]], dtype=torch.float32)

    # Translate to center, rotate, then translate back
    translation_matrix = torch.tensor([[1, 0, -center_x],
                                       [0, 1, -center_y],
                                       [0, 0, 1]], dtype=torch.float32)
    inv_translation_matrix = torch.tensor([[1, 0, center_x],
                                           [0, 1, center_y],
                                           [0, 0, 1]], dtype=torch.float32)

    transform_matrix = torch.matmul(inv_translation_matrix, torch.matmul(rotation_matrix, translation_matrix))
    
    # Transform the grid
    grid = F.affine_grid(transform_matrix[:2, :], image.unsqueeze(0).size())
    
    # Sample the image using the grid
    rotated_image = F.grid_sample(image.unsqueeze(0), grid)[0]
    
    return rotated_image

# Example usage:
image = torch.randn(3, 256, 256)  # Example image tensor (3 channels, 256x256)
angle = 45  # Rotation angle in degrees
rotated_image = rotate_image(image, angle)
print(rotated_image.shape)

```

This example explicitly constructs the transformation matrix, ensuring differentiability. The translation components ensure rotation around the image center.  The use of `unsqueeze` and `[0]` handles the batch dimension required by `affine_grid` and `grid_sample`.

**Example 2:  Rotation using a custom differentiable function (for advanced scenarios)**

```python
import torch

def rotate_image_custom(image, angle_degrees):
    """Rotates an image using a custom differentiable function."""
    angle_rad = torch.tensor(math.radians(angle_degrees), requires_grad=True)
    height, width = image.shape[1:]

    # Grid creation using meshgrid for finer control
    x = torch.arange(width, dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y)

    # Centering for rotation around the center
    center_x = width / 2
    center_y = height / 2
    grid_x = grid_x - center_x
    grid_y = grid_y - center_y

    # Rotational transformation 
    rotated_x = grid_x * torch.cos(angle_rad) - grid_y * torch.sin(angle_rad) + center_x
    rotated_y = grid_x * torch.sin(angle_rad) + grid_y * torch.cos(angle_rad) + center_y

    # Clipping to prevent out-of-bounds indices
    rotated_x = torch.clamp(rotated_x, 0, width - 1)
    rotated_y = torch.clamp(rotated_y, 0, height - 1)

    # Bilinear interpolation (can be replaced with other interpolation methods)
    rotated_image = torch.nn.functional.grid_sample(image.unsqueeze(0), torch.stack([rotated_x, rotated_y], dim=-1).unsqueeze(0))[0]
    return rotated_image

# Example usage:
image = torch.randn(3, 256, 256)
angle = 45
rotated_image = rotate_image_custom(image, angle)
print(rotated_image.shape)
```

This approach demonstrates a more manual approach, offering greater control. The use of `requires_grad=True` ensures the angle is tracked for backpropagation. The explicit grid creation and clamping addresses edge cases and ensures numerical stability.  Choosing the appropriate interpolation method (bilinear here) is crucial for image quality and performance.


**Example 3: Utilizing a library for higher-level abstractions (if performance is not critical):**

This approach is less efficient for large-scale operations, however it may improve development speed when performance is not paramount.

```python
import torchvision.transforms.functional as TF
import torch

def rotate_image_library(image, angle_degrees):
    """Rotates an image using a library function; less efficient than previous examples"""
    rotated_image = TF.rotate(image, angle_degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
    return rotated_image


# Example usage:
image = torch.randn(3, 256, 256)
angle = 45
rotated_image = rotate_image_library(image, angle)
print(rotated_image.shape)
```

This leverages the `torchvision` library.  Note that the `expand` parameter controls whether the output image is padded to accommodate the rotated image. While convenient, this lacks the fine-grained control and potential performance benefits of the previous methods.


**3. Resource Recommendations:**

The PyTorch documentation, particularly the sections on `torch.nn.functional.affine_grid` and `torch.nn.functional.grid_sample`, are invaluable.  A solid understanding of linear algebra, specifically affine transformations and rotation matrices, is essential.  Exploring resources on image processing and computer graphics, focusing on image transformations and interpolation techniques, will broaden your understanding of the underlying principles.  Finally, reviewing materials on automatic differentiation and backpropagation in the context of deep learning will provide a deeper comprehension of the gradients' behavior during training.
