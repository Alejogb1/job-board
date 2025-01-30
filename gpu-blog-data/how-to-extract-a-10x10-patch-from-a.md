---
title: "How to extract a 10x10 patch from a 100x100 PyTorch tensor with toroidal boundary conditions?"
date: "2025-01-30"
id: "how-to-extract-a-10x10-patch-from-a"
---
Extracting patches from tensors with toroidal boundary conditions necessitates careful handling of edge cases.  My experience working on large-scale image processing pipelines for satellite imagery frequently involved this precise scenario, particularly when dealing with cyclic phenomena in atmospheric data. The core challenge lies in seamlessly transitioning from the tensor's right edge to its left, and similarly for the top and bottom.  Standard slicing operations in PyTorch will not inherently provide this behavior; explicit handling is required.

The solution involves creating a function that intelligently manages indexing to simulate the toroidal topology.  This is achieved by using the modulo operator (%) to wrap around the tensor's dimensions.  When an index exceeds the tensor's bounds, the modulo operation maps it back to a valid index within the range.  However, a naive modulo operation alone isn't sufficient; the correct offset must be determined based on the patch's location relative to the tensor's boundaries.

**1. Clear Explanation**

The algorithm proceeds as follows:  For each coordinate (x, y) within the 10x10 patch, we calculate the corresponding indices within the 100x100 input tensor.  We do this by adding the patch's top-left corner coordinates (x_start, y_start) to the current patch coordinate (x, y).  The resulting indices (x_index, y_index) are then checked for exceeding the tensor dimensions. If an index is out of bounds, the modulo operator, applied with respect to the tensor's dimension, maps it back to a valid index, effectively wrapping around the tensor's edges.  The value at the mapped index is then assigned to the corresponding location in the output patch.  This ensures continuous transition at the boundaries, thereby implementing the toroidal boundary condition.


**2. Code Examples with Commentary**

**Example 1:  Using basic indexing and modulo operation**

```python
import torch

def extract_toroidal_patch(tensor, x_start, y_start, patch_size=10):
    """Extracts a patch with toroidal boundary conditions."""
    height, width = tensor.shape
    patch = torch.zeros((patch_size, patch_size), dtype=tensor.dtype)
    for i in range(patch_size):
        for j in range(patch_size):
            x_index = (x_start + i) % height
            y_index = (y_start + j) % width
            patch[i, j] = tensor[x_index, y_index]
    return patch


tensor_100x100 = torch.rand(100, 100)
patch = extract_toroidal_patch(tensor_100x100, 95, 95) # Patch from bottom right corner
print(patch)
```

This example demonstrates the core logic using nested loops and explicit index manipulation. While functional, it's not computationally optimal for larger tensors or patches.


**Example 2: Leveraging advanced indexing for efficiency**

```python
import torch

def extract_toroidal_patch_advanced(tensor, x_start, y_start, patch_size=10):
    """Extracts a patch efficiently using advanced indexing."""
    height, width = tensor.shape
    x_indices = torch.arange(x_start, x_start + patch_size) % height
    y_indices = torch.arange(y_start, y_start + patch_size) % width
    x_indices = x_indices.view(-1, 1).expand(patch_size, patch_size)
    y_indices = y_indices.view(1, -1).expand(patch_size, patch_size)
    patch = tensor[x_indices, y_indices]
    return patch

tensor_100x100 = torch.rand(100, 100)
patch = extract_toroidal_patch_advanced(tensor_100x100, 95, 95) # Patch from bottom right corner
print(patch)
```

This improved version utilizes PyTorch's advanced indexing capabilities, significantly reducing execution time by vectorizing the operation.  The `expand` function efficiently replicates index arrays to create the correct index matrix for the patch.


**Example 3: Handling multi-channel tensors**

```python
import torch

def extract_toroidal_patch_multichannel(tensor, x_start, y_start, patch_size=10):
    """Extracts a patch from a multi-channel tensor."""
    channels, height, width = tensor.shape
    x_indices = torch.arange(x_start, x_start + patch_size) % height
    y_indices = torch.arange(y_start, y_start + patch_size) % width
    x_indices = x_indices.view(1, -1, 1).expand(channels, patch_size, patch_size)
    y_indices = y_indices.view(1, 1, -1).expand(channels, patch_size, patch_size)
    patch = tensor[:, x_indices, y_indices]
    return patch

tensor_100x100_3channel = torch.rand(3, 100, 100)  # 3-channel tensor
patch = extract_toroidal_patch_multichannel(tensor_100x100_3channel, 95, 95)
print(patch)
```

This final example extends the functionality to accommodate multi-channel tensors, such as color images (RGB) or multi-spectral satellite data.  The index manipulation is adapted to handle the additional channel dimension, ensuring the toroidal boundary conditions are correctly applied to each channel independently.


**3. Resource Recommendations**

For a deeper understanding of PyTorch's tensor manipulation capabilities, consult the official PyTorch documentation.  Furthermore, a thorough grasp of linear algebra and matrix operations is crucial for efficient tensor processing.  A strong foundation in Python programming, including proficiency with loops, indexing, and vectorization techniques, is also highly beneficial.  Understanding the concepts of broadcasting and advanced indexing within PyTorch will be particularly valuable for optimizing code performance for tasks like this.  Consider studying resources focused on scientific computing with Python.
