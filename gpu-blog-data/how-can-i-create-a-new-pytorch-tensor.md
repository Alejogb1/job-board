---
title: "How can I create a new PyTorch tensor from an existing one using specific indices?"
date: "2025-01-30"
id: "how-can-i-create-a-new-pytorch-tensor"
---
PyTorch's indexing capabilities allow for efficient creation of new tensors by selecting specific elements from existing tensors, a fundamental operation in many deep learning workflows. Based on my experience developing custom layers for image segmentation models, I've found these methods critical for managing both spatial and channel-wise data manipulations. The core principle involves using various indexing techniques to specify which elements of the source tensor should be included in the newly formed tensor.

### Understanding Indexing in PyTorch

PyTorch tensors support a range of indexing methods, including integer indexing, slicing, boolean indexing, and advanced indexing using other tensors. Integer indexing selects specific elements based on their absolute position, akin to array access in other programming languages. Slicing, using `start:stop:step`, allows for selecting a range of elements within a dimension. Boolean indexing selects elements based on a boolean mask of the same shape as the source tensor, and advanced indexing uses integer tensors to define non-contiguous selection of source tensor elements. The choice of method depends primarily on how the desired indices are specified and the desired resulting tensor shape.

### Method 1: Integer Indexing and Slicing

Integer indexing and slicing are straightforward for selecting contiguous regions or single elements. Assume a source tensor `source_tensor` of shape (5, 10) and a need to create a new tensor containing the second row and the 3rd to 7th columns. This can be done directly using integer indexing for the row and slicing for the column.

```python
import torch

source_tensor = torch.arange(50).reshape(5, 10)
# Create a new tensor using indexing: second row (index 1), columns 3 through 7
new_tensor = source_tensor[1, 2:7]
print("Source Tensor:\n", source_tensor)
print("New Tensor (integer indexed):\n", new_tensor)
print("Shape of new tensor:", new_tensor.shape)
```

In this case, `source_tensor[1, 2:7]` selects the second row (index 1) and columns from index 2 up to (but not including) index 7. The resulting `new_tensor` is a 1D tensor with five elements. This technique is very useful for selecting patches from image tensors, such as for convolutional operations with specific strides. Note that indexing on one axis only reduces the dimensionality of the tensor. If one wants to keep the dimensionality intact, slices would need to be applied to all dimensions. As an example, `source_tensor[1:2, 2:7]` will result in a tensor of shape (1, 5) which is a two dimensional view of the tensor.

### Method 2: Boolean Indexing

Boolean indexing is invaluable when elements need to be selected based on a condition. If, for instance, we wanted to extract all elements in `source_tensor` that are greater than 30, we would first create a boolean mask indicating which elements satisfy the condition.

```python
import torch

source_tensor = torch.arange(50).reshape(5, 10)
#Create boolean tensor indicating where tensor is greater than 30
mask = source_tensor > 30
# create the new tensor with boolean indexing
new_tensor = source_tensor[mask]
print("Source Tensor:\n", source_tensor)
print("Mask:\n", mask)
print("New Tensor (boolean indexed):\n", new_tensor)
print("Shape of new tensor:", new_tensor.shape)
```

Here, `source_tensor > 30` generates a boolean tensor of the same shape as `source_tensor`, where `True` corresponds to elements greater than 30. The resulting `new_tensor` is a 1D tensor containing only the selected elements. The size of the resulting tensor is not deterministic beforehand, but it can be very efficient when elements need to be selected based on a condition.

### Method 3: Advanced Indexing with Integer Tensors

Advanced indexing, utilizing integer tensors, provides the most versatile selection mechanism. Suppose we desire a new tensor containing elements from `source_tensor` at specific, non-contiguous row and column indices. This can be achieved by specifying two integer tensors: one for row indices and another for column indices.

```python
import torch

source_tensor = torch.arange(50).reshape(5, 10)
# Define index tensors
row_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([1, 3, 6])

# Index the original tensor using the defined row and column indices
new_tensor = source_tensor[row_indices, col_indices]

print("Source Tensor:\n", source_tensor)
print("Row Indices:", row_indices)
print("Column Indices:", col_indices)
print("New Tensor (advanced indexed):\n", new_tensor)
print("Shape of new tensor:", new_tensor.shape)

```

In this example, `row_indices` specifies the rows (0, 2, and 4) and `col_indices` the columns (1, 3, and 6). This creates a new tensor whose elements are `source_tensor[0, 1]`, `source_tensor[2, 3]`, and `source_tensor[4, 6]`, respectively. Advanced indexing using integer tensors allows for highly flexible non-contiguous element selection, often used in specialized scenarios, such as creating sparse convolutional kernels. A crucial aspect of this form of indexing is that it doesn't return a *view* of the source tensor, as opposed to other forms of indexing. The resulting tensor will be a copy.

### Memory Considerations and Performance

It's essential to understand the distinction between creating a *view* versus a copy of the source tensor. Integer indexing and slicing generally result in views; modifying a view will modify the original tensor. However, advanced indexing and boolean indexing typically produce a copy. This consideration is paramount when working with large datasets because creating copies can have substantial memory overhead. To mitigate this, using views where possible can reduce memory usage and improve the efficiency of data manipulations. For the scenarios where copies cannot be avoided, careful resource management might be necessary.

Furthermore, when selecting across multiple dimensions, proper alignment of index tensors is essential. When indexing with integer tensors, if the index tensors have the same dimension `N`, the resulting tensor is of shape `N` and is created by combining index `i` in each dimension to select from the original tensor. If this behaviour is not desired, the tensors might need to be modified or reshaped, using methods like `unsqueeze()` or broadcasting the dimensions to select a higher dimensional set of elements.

### Further Learning

For a deeper understanding of tensor manipulations, exploring the PyTorch documentation is critical. Topics like `torch.gather` and `torch.scatter`, which facilitate collecting and scattering tensor elements based on indexes, can be beneficial. Books on deep learning frameworks also provide comprehensive examples of tensor operations within practical contexts. Exploring practical uses of tensor operations, within a real network, can bring this knowledge in a deeper level. These resources will broaden the understanding of PyTorch tensor indexing, allowing more efficient and nuanced tensor manipulations.
