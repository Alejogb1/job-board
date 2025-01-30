---
title: "How to select specific elements from a 3D PyTorch tensor using a 1D index tensor?"
date: "2025-01-30"
id: "how-to-select-specific-elements-from-a-3d"
---
PyTorch tensor indexing, especially across higher dimensions, can initially appear daunting. The key lies in understanding how index tensors are interpreted and applied to the target tensor, specifically when you intend to use a 1D index tensor to select elements from a 3D tensor. Fundamentally, the 1D index tensor serves as a pointer into a particular dimension of the higher-dimensional tensor. It's not a single index that applies across all dimensions, but rather a set of indices that selects along one dimension for a set of fixed values in the other dimensions. In the case of using a 1D index tensor on a 3D tensor, you're implicitly selecting a slice of the tensor determined by a combination of the index tensor and fixed values for the other two dimensions.

Let me illustrate with a recent project involving volumetric medical imaging. I frequently dealt with 3D tensors representing volumes, and needed to efficiently extract specific elements based on dynamically generated indices. The challenge often centered on interpreting these 1D index tensors, especially when they needed to apply across the "depth" dimension, while maintaining the "height" and "width" of the spatial information. The goal was not to select a single element, but rather a subset of elements scattered across the depth dimension of each 2D "slice".

Specifically, imagine a 3D tensor with shape `(depth, height, width)`, like a CT scan volume. A 1D index tensor, say with a length equal to `height * width`, needs to select elements along the `depth` dimension. Each element in this 1D index tensor refers to the depth value to pick for the corresponding height and width position. This means we aren't selecting a line of voxels from the same plane, but rather a possibly complex shape sampled across depth.

To accomplish this, broadcasting the 1D index tensor is crucial. The index tensor needs to have its shape adapted so it can be applied to the correct dimension of the 3D tensor. PyTorch indexing uses the index tensor to dictate along *one* dimension; the other dimensions are implicitly traversed.

Consider the first example. I have a 3D tensor, `data`, and a 1D index tensor, `indices`, where `indices` specifies the depth indices:

```python
import torch

# Example tensor: (depth, height, width)
depth, height, width = 5, 3, 4
data = torch.arange(depth * height * width).reshape(depth, height, width)
print("Original Data:\n", data)

# Example 1D index tensor (should have a length = height*width):
indices = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
print("Indices:\n", indices)

# Reshape indices to (1, height, width) for broadcasting
indices_reshaped = indices.reshape(1, height, width)

# Use advanced indexing
selected_elements = data.gather(0, indices_reshaped.expand(depth,height,width)) # use of expand
print("Selected Elements:\n", selected_elements)

```

Here, `data` is our 3D tensor. `indices` is our 1D index tensor, and needs to be broadcast into a 3D shape of `(1, height, width)` by reshaping it with `indices.reshape(1, height, width)`. The `gather` function does the actual indexing based on the index tensor. It picks from the `depth` dimension (specified by the first argument to gather as 0), using the corresponding `indices` to make its selection. Note that I used `expand` function on indices. It replicates the provided tensor along the new dimension. This way the indices align correctly for every `depth`. The output `selected_elements` has the same shape as the original `data` tensor, but the elements in depth have been sampled according to the index tensor.

A different indexing approach is possible if you do not want a tensor of the same shape as the original, and instead, want a flattened array containing only the elements selected by the indices. In this next case, I'm creating a 3D tensor and a 1D index tensor. Note that the 1D index tensor is not dependent on any of the dimension of 3D tensor. Each value of the 1D index tensor will select a depth element for corresponding elements in height and width.

```python
import torch

# Example tensor: (depth, height, width)
depth, height, width = 5, 3, 4
data = torch.arange(depth * height * width).reshape(depth, height, width)
print("Original Data:\n", data)

# Example 1D index tensor (should have a length = height*width):
indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2])
print("Indices:\n", indices)

# Calculate indices for other dimensions
height_indices = torch.arange(height).repeat_interleave(width).unsqueeze(0)
width_indices = torch.arange(width).repeat(height).unsqueeze(0)

# Use advanced indexing
selected_elements = data[indices, height_indices.expand(1,height*width), width_indices.expand(1,height*width)].flatten()
print("Selected Elements:\n", selected_elements)

```

Here, I create `height_indices` and `width_indices` by creating a repeating tensor with the values along those dimensions. The actual indexing then picks the indexed values. Note that for every element in `indices`, there will be one corresponding output from indexing the 3D tensor with the given indices. For example if the `indices` array has a value of 2 at element 0, then output element 0 will be the data value at `data[2,0,0]`.

A third example can be useful to illustrate what happens when you need to index a subset of 2D slices with your indices. In this scenario, imagine a 4D tensor representing a batch of 3D images. We want to extract information based on 1D indices per batch and per depth slice. This time we will work with a 4D tensor of size `(batch, depth, height, width)` and a 2D index tensor of size `(batch, height*width)`.

```python
import torch

# Example tensor: (batch, depth, height, width)
batch_size = 2
depth, height, width = 5, 3, 4
data = torch.arange(batch_size * depth * height * width).reshape(batch_size, depth, height, width)
print("Original Data:\n", data)

# Example 2D index tensor (should have shape = batch, height*width):
indices = torch.randint(0, depth, (batch_size, height * width)) # random values of depth
print("Indices:\n", indices)


height_indices = torch.arange(height).repeat_interleave(width).unsqueeze(0).expand(batch_size,height*width)
width_indices = torch.arange(width).repeat(height).unsqueeze(0).expand(batch_size,height*width)

selected_elements = data[torch.arange(batch_size).unsqueeze(1), indices, height_indices, width_indices]

print("Selected Elements:\n", selected_elements)

```

In this example, `data` represents a batch of volumetric data. The `indices` tensor now has a batch dimension, allowing us to pick from each 3D tensor separately. Here we are creating indices tensors for batch, height, and width. `torch.arange(batch_size).unsqueeze(1)` gives us the correct values for indexing along the `batch_size` dimension. `selected_elements` has the shape `(batch_size, height*width)` containing elements along the `depth` dimension as specified by the index tensor. The use of `expand` function here is important as it provides the ability to index each batch correctly with height and width indices.

To fully master tensor manipulation in PyTorch, I recommend the following resources for further study. Review the official PyTorch documentation which contains a detailed explanation on indexing and broadcasting mechanisms, including `gather` function. Also, consider books and courses that cover tensor operations, especially focusing on advanced indexing techniques. Look for examples that work with different dimensional tensors and practical cases that can provide the necessary intuition. These will solidify the fundamental knowledge needed for complex tensor manipulation. Further, spend time practicing building the index tensors using the different indexing methods provided by PyTorch. This will help greatly in the understanding of how to use this indexing method.
