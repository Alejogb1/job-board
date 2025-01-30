---
title: "How to index PyTorch tensors?"
date: "2025-01-30"
id: "how-to-index-pytorch-tensors"
---
Indexing PyTorch tensors is fundamental to efficient data manipulation, especially when working with large datasets or intricate neural network architectures. A thorough grasp of indexing principles prevents common bugs and unlocks optimized performance. Having spent considerable time debugging segmentation models where improper tensor slicing led to catastrophic training failures, I've developed a strong understanding of this operation. PyTorch's indexing mechanisms are derived primarily from NumPy, offering a blend of basic slicing and more sophisticated techniques such as advanced indexing, which can significantly impact both memory usage and execution speed.

**1. Explanation of PyTorch Tensor Indexing:**

Fundamentally, tensor indexing in PyTorch permits you to select specific elements or sub-regions from a multi-dimensional array, the tensor. The most basic method employs integer indices, analogous to accessing elements in a Python list. For a one-dimensional tensor, `tensor[i]` retrieves the element at index *i*. In higher dimensions, we extend this principle: `tensor[i, j]` retrieves the element at row *i* and column *j* in a two-dimensional tensor (matrix).

Slicing, using the colon operator (`:`), is another critical tool. A slice expression such as `tensor[start:end]` selects a range of elements, going from the start index (inclusive) up to the end index (exclusive). Omitting the start or end defaults to the beginning or end of that dimension, respectively. For instance, `tensor[:]` copies the entire dimension while `tensor[::2]` retrieves every second element along the dimension. In multi-dimensional tensors, slices can be combined, for example `tensor[start1:end1, start2:end2]` to select a region of rows and columns. The step size can be negative to iterate in reverse order.

Advanced indexing involves using non-consecutive indices and utilizing tensors to index other tensors. Here, the tensor used for indexing must be of type `torch.long` as these indices represent specific memory locations within the underlying tensor's data storage. Advanced indexing offers the ability to gather arbitrary tensor elements, potentially even duplicating them or creating views with specific arrangements. Advanced indexing however, can return a copy of the data, not a view and care must be taken to understand the underlying consequences of that.

Broadcasting is often implicitly involved in indexing operations when combining tensor and scalar values during indexing. If we attempt to assign a scalar value to a portion of a tensor using advanced indexing (e.g. setting values based on a mask), broadcasting allows the scalar to effectively expand in place to fill the indicated region. This saves memory space and simplifies code.

**2. Code Examples and Commentary:**

The first example demonstrates basic slicing on a two-dimensional tensor.

```python
import torch

# Create a 3x4 tensor
tensor = torch.arange(12).reshape(3, 4)
print("Original Tensor:\n", tensor)

# Basic slicing
slice1 = tensor[1:3, 1:3] # Extract rows 1 and 2, columns 1 and 2
print("\nSlice 1:\n", slice1)

# Access individual element
element = tensor[0, 2] # Access element at row 0, column 2
print("\nElement:", element)

# Access an entire row
row2 = tensor[2,:]
print("\nRow 2:\n",row2)
```

Here, we first create a simple two-dimensional tensor. Then, we use basic slicing to obtain a sub-tensor consisting of rows and columns as specified. We also extract an individual element and also access an entire row using the colon. This emphasizes the flexibility in accessing and manipulating parts of a tensor.

The second example demonstrates advanced indexing using a tensor.

```python
import torch

# Create a 1D tensor
tensor = torch.arange(10)
print("Original Tensor:\n", tensor)

# Index tensor using a long tensor
indices = torch.tensor([2, 5, 8], dtype=torch.long)
indexed_tensor = tensor[indices]
print("\nIndexed Tensor using indices:\n", indexed_tensor)

# Index a multi-dimensional tensor
tensor2 = torch.arange(12).reshape(3, 4)
row_indices = torch.tensor([0, 2], dtype=torch.long)
col_indices = torch.tensor([1, 3], dtype=torch.long)
indexed_tensor2 = tensor2[row_indices, col_indices]
print("\nMulti-dimensional indexed tensor:\n", indexed_tensor2)
```

This shows how an index tensor can pick out elements from both one-dimensional and multi-dimensional tensors. Critically, note the use of `torch.long` which is mandatory when advanced indexing. This method is particularly useful when gathering data according to a predefined set of indices generated in another process. Specifically, note the difference between this operation and slicing, where a sequence of elements is returned, and here, where elements located at the exact given coordinates are returned.

The final example presents an indexing operation using a boolean mask.

```python
import torch

# Create a 2x3 tensor
tensor = torch.tensor([[1, 2, 3],
                     [4, 5, 6]])
print("Original tensor:\n",tensor)

# Create a boolean mask
mask = tensor > 3
print("\nMask:\n",mask)

# Indexing with the mask
masked_tensor = tensor[mask]
print("\nMasked Tensor:\n", masked_tensor)

# Example of setting values with a mask
tensor[mask] = 0
print("\nModified tensor:\n",tensor)
```

The boolean mask selects all elements of the tensor where the condition `tensor > 3` evaluates to true.  This allows for selective modification or extraction of tensor values based on conditions. The example additionally showcases how using the mask we can modify the original tensor in-place by setting values to 0. This is extremely useful during operations like clipping, thresholding, and applying conditional changes to the tensor data without needing loops and temporary tensors.

**3. Resource Recommendations**

For further study of tensor manipulation in PyTorch, the official PyTorch documentation is invaluable; its comprehensive nature and numerous examples are highly effective for understanding core functionality. "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann presents a practical introduction to the concepts in PyTorch through practical examples. The documentation for Numpy can also offer clarity as many principles are derived from Numpy functionalities, especially in terms of slicing. Reviewing practical tutorials on PyTorch for various deep learning models provides practical, hands-on experience in the application of various indexing mechanisms, reinforcing the knowledge gained. Examining source code examples from well-known open-source deep learning projects often provides insight into how complex tensor manipulations are applied effectively in real-world applications. By focusing on a balance of theory and practical examples, one can achieve a strong command of PyTorch tensor indexing.
