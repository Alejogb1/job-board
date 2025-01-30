---
title: "How can I vectorize `torch.unique()` for multiple dimensions in PyTorch?"
date: "2025-01-30"
id: "how-can-i-vectorize-torchunique-for-multiple-dimensions"
---
`torch.unique()` operates on a single dimension, presenting a challenge when identifying unique elements across multiple dimensions of a PyTorch tensor. I've encountered this limitation repeatedly in my work processing multi-channel sensor data, where unique combinations across channels often represent distinct states or events. Vectorizing this operation demands careful handling of the tensor's structure and a strategy to collapse the multiple dimensions into a single one amenable to `torch.unique()`.

The fundamental problem lies in `torch.unique()`'s design, which is intended for identifying unique elements within a single sequence. When presented with a multi-dimensional tensor, the inherent notion of 'uniqueness' is ambiguous. Do we seek uniqueness along a specific axis, or do we desire the unique *combinations* of elements across multiple axes? The latter is the scenario we address here. My approach involves reshaping the tensor to effectively treat these combinations as single elements in a vectorized manner.

The core mechanism for achieving this vectorization is threefold: First, we must reshape the multi-dimensional tensor into a two-dimensional matrix, where each row represents a unique combination of elements from the original tensor's chosen dimensions. Second, we apply `torch.unique()` to this two-dimensional matrix, focusing on rows now considered single 'elements.' Third, we need to reconstruct the original shape for the indices, making it compatible for use on the original tensor. The initial reshaping and reconstruction are pivotal for maintaining data context and usability after the uniqueness operation.

Consider a three-dimensional tensor representing sensor readings over time. The dimensions might be (time, sensor_type, reading_channel), where, for example, we desire unique combinations across sensor_type and reading_channel for each time point. The goal is not to find unique values *within* these dimensions, but rather to find all unique combinations of these values across dimensions. Reshaping transforms this into a matrix of (time, product(sensor_type * reading_channel)), where each row will be input to `torch.unique`.

**Example 1: Simple Two-Dimensional Vectorization**

Let's consider a simpler case, a tensor with dimensions (batch_size, feature_size). I've used this approach when processing batches of time-series data. Suppose the tensor has shape `(5, 3)`, representing 5 samples with 3 features each.

```python
import torch

# Example tensor with dimensions (batch_size, feature_size)
tensor = torch.tensor([[1, 2, 3],
                       [1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [4, 5, 6]])

# Reshape for unique operation
reshaped_tensor = tensor.view(tensor.size(0), -1)

# Get unique rows and indices
unique_rows, indices = torch.unique(reshaped_tensor, dim=0, return_inverse=True)

print("Original tensor:\n", tensor)
print("\nReshaped tensor:\n", reshaped_tensor)
print("\nUnique Rows:\n", unique_rows)
print("\nIndices:\n", indices)


# Recover original shape for indices for use later
indices = indices.view(tensor.shape[0])


# Get uniques from original tensor using the indices
unique_elements = tensor[indices]
print("\n Unique Elements from Original Tensor:\n", unique_elements)
```

In this example, the tensor is reshaped to (5, 3). `torch.unique(dim=0)` then finds unique rows and returns them as `unique_rows`, along with a list `indices` which indicates the original position of each unique row. We then can recover our original tensor shape for indices to select elements from the original tensor. This showcases the essential steps: reshape, `torch.unique`, and reconstruct. Note the return of `indices`, which maintains a link to the original tensor.

**Example 2: Three-Dimensional Vectorization**

Now, consider the three-dimensional tensor that I previously mentioned, with dimensions (time, sensor_type, reading_channel). We will find all unique combinations of sensor_type and reading_channel. This scenario was a frequent occurrence for me when building models for interpreting sensor data streams.

```python
import torch

# Example tensor with dimensions (time, sensor_type, reading_channel)
tensor = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                       [[1, 2], [7, 8], [5, 6]],
                       [[9, 10], [3, 4], [11, 12]]])

# Reshape to (time, product(sensor_type * reading_channel))
reshaped_tensor = tensor.view(tensor.size(0), -1)

# Get unique rows and indices
unique_rows, indices = torch.unique(reshaped_tensor, dim=0, return_inverse=True)

print("Original tensor:\n", tensor)
print("\nReshaped tensor:\n", reshaped_tensor)
print("\nUnique Rows:\n", unique_rows)
print("\nIndices:\n", indices)


# Recover original shape for indices for use later
indices = indices.view(tensor.shape[0])


# Get uniques from original tensor using the indices
unique_elements = tensor[indices]
print("\n Unique Elements from Original Tensor:\n", unique_elements)
```

This code extends the previous concept. The tensor is reshaped from (3, 3, 2) to (3, 6). `torch.unique(dim=0)` identifies and extracts unique combinations. The returned indices are again reshaped to align with the first dimension of the original tensor. Note how the `product` implicitly occurs using the `-1` dimension parameter in `.view`. This is crucial for avoiding an explicit `prod()` operation. Again, we can recover the uniques from the original tensor using the returned indices.

**Example 3: Masking based on unique indices**

Building upon previous examples, consider a situation where you need to construct masks based on the generated unique indices. This can be valuable in scenarios like selecting specific examples from a larger tensor. I used this type of operation frequently when filtering data based on combinations of feature values.

```python
import torch

# Example tensor with dimensions (batch_size, feature_size)
tensor = torch.tensor([[1, 2, 3],
                       [1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [4, 5, 6]])


reshaped_tensor = tensor.view(tensor.size(0), -1)

# Get unique rows and indices
unique_rows, indices = torch.unique(reshaped_tensor, dim=0, return_inverse=True)
indices = indices.view(tensor.shape[0])

# Create masks
num_unique = unique_rows.shape[0]
masks = []

for i in range(num_unique):
    mask = (indices == i)
    masks.append(mask)

print("Original tensor:\n", tensor)
print("\nUnique Rows:\n", unique_rows)
print("\nIndices:\n", indices)

print("\n Masks:\n", masks)
```

Here, we obtain the unique rows and corresponding indices, as previously. Then, we create a list of masks, each corresponding to a specific unique row. The mask is `True` where the `indices` match the given `i`. These masks can be used for filtering, subsetting, or other operations. This demonstrates how the output of `torch.unique` can be used to control subsequent tensor manipulations.

In terms of resources, I would suggest studying the PyTorch documentation thoroughly, paying particular attention to `torch.reshape`, `torch.view`, and `torch.unique`. Understanding tensor manipulation is crucial for performing these types of operations effectively. Additionally, exploring blog posts or forums that discuss advanced tensor operations in PyTorch can provide additional insights. For learning more about data manipulation techniques, tutorials or coursework on linear algebra are highly beneficial. A general understanding of vectorized operations will help with efficiency and understanding. Experimenting with different tensor shapes and dimensions is also a great way to solidify the practical aspects of this approach.
