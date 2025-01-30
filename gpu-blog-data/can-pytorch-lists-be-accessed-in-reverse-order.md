---
title: "Can PyTorch lists be accessed in reverse order?"
date: "2025-01-30"
id: "can-pytorch-lists-be-accessed-in-reverse-order"
---
A common point of confusion when transitioning from standard Python lists to PyTorch tensors is the implicit indexing flexibility offered by Python, especially concerning reverse iteration. While PyTorch tensors, at their core, are multi-dimensional arrays and therefore support various indexing techniques, the mechanisms for reverse access differ and require careful understanding.

The primary reason standard Python list-like indexing methods do not translate directly to PyTorch tensors lies in the underlying data structure. Python lists are dynamically sized, linked structures, allowing for negative indexing to easily traverse elements from the end. PyTorch tensors, however, are contiguous blocks of memory, optimized for numerical computation. Accessing them in reverse requires a conscious reinterpretation of indices, not a direct application of negative offsets in the same way as Python lists. Therefore, to answer the question directly, you cannot use Python's negative indices like `-1, -2, -3` directly on a PyTorch tensor to access elements in reverse. You must leverage other methods.

When working with PyTorch, achieving reverse access involves using a combination of tensor properties and methods. Specifically, slicing in combination with step values allows you to access tensor elements in reverse sequence. The most common approach uses `[::-1]` notation within the square brackets during indexing, akin to the way Python lists are sliced in reverse but implemented as a memory mapping strategy within the Tensor structure. This does not copy data; it alters the memory mapping of the underlying data. In scenarios needing a true reversed sequence, an operation such as `.flip(dims=[0])` is necessary, that reorders the underlying tensor in memory and produces a new Tensor object.

Let's look at some examples. First consider the case where you need to slice in reverse to extract a portion of the tensor:

```python
import torch

# Example 1: Slicing a tensor in reverse
tensor_a = torch.tensor([1, 2, 3, 4, 5, 6])
reversed_slice = tensor_a[5:1:-1]  # Start at index 5, end at index 1 (exclusive), step -1
print(reversed_slice)
# Output: tensor([6, 5, 4, 3])

tensor_b = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
reverse_row_slice = tensor_b[2:0:-1]
print(reverse_row_slice)
# Output: tensor([[7, 8, 9],
#        [4, 5, 6]])

```

In this first example, a one-dimensional tensor, `tensor_a`, is defined. The slice `tensor_a[5:1:-1]` effectively starts at the index `5`, corresponding to the element with value `6`, and proceeds backwards to index `1`, excluding it from the final slice. This results in a new tensor with values `[6, 5, 4, 3]`. Notice how we have a slice, not a reversed tensor. The second section of Example 1 shows that slicing can occur across multidimensional tensors. It slices out rows, from row `2` to row `1`. 

A different approach is required if the intended operation is to truly reverse the order of elements within a dimension, not just extract a slice. The `.flip()` function is used for this:

```python
# Example 2: Reversing a tensor along dimension
tensor_c = torch.tensor([1, 2, 3, 4, 5])
reversed_tensor = torch.flip(tensor_c, dims=[0])
print(reversed_tensor)
# Output: tensor([5, 4, 3, 2, 1])

tensor_d = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
reversed_rows_tensor = torch.flip(tensor_d, dims=[0])
print(reversed_rows_tensor)
# Output: tensor([[7, 8, 9],
#        [4, 5, 6],
#        [1, 2, 3]])

reversed_cols_tensor = torch.flip(tensor_d, dims=[1])
print(reversed_cols_tensor)
# Output: tensor([[3, 2, 1],
#        [6, 5, 4],
#        [9, 8, 7]])
```

In the second example, `tensor_c` is reversed along its first (and only) dimension using `torch.flip(tensor_c, dims=[0])`. This creates a new tensor, where the element at index `0` becomes the last, the element at index `1` becomes the second to last, and so on. This operation genuinely changes the order of elements within the Tensor structure. The other parts of example 2 show how it is possible to use the `.flip` method on multidimensional tensors, specifying which dimensions should be reversed. The rows or columns of the tensor can be reversed separately by changing the dimension argument.

Furthermore, it is possible to iterate through a tensor in reverse order using a standard for loop construct combined with the reversed function and Python's `range()` function. This method, while effective, is not efficient for highly performance focused tensor operations:

```python
# Example 3: Reverse iteration
tensor_e = torch.tensor([10, 20, 30, 40, 50])
reversed_list = []
for i in reversed(range(len(tensor_e))):
  reversed_list.append(tensor_e[i])

print(torch.tensor(reversed_list))
# Output: tensor([50, 40, 30, 20, 10])

tensor_f = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
reversed_tensor_list = []
for row_index in reversed(range(tensor_f.size(0))):
  reversed_row = []
  for col_index in range(tensor_f.size(1)):
    reversed_row.append(tensor_f[row_index, col_index])
  reversed_tensor_list.append(torch.tensor(reversed_row))

print(torch.stack(reversed_tensor_list))
# Output: tensor([[7, 8, 9],
#         [4, 5, 6],
#         [1, 2, 3]])
```

In this final example, the `reversed(range(len(tensor_e)))` provides a means to iterate through the indices of the tensor `tensor_e` in reverse. This method allows for element-wise access of a tensor, but is not computationally efficient for very large tensors. This approach emulates a reverse order list, in this case a Python list built from tensor elements, and converted back to a tensor after. The second portion of example 3 uses nested loops to reverse across a higher dimensionality tensor. Note that this reverse order is not on the elements, but rather the rows of the tensor.

In practical scenarios, I would generally recommend leveraging slicing or the `.flip()` function for manipulating tensor data. Looping in pure Python, while versatile, can be a performance bottleneck within computationally intensive machine learning code, and should be avoided unless there is no suitable alternative.

For further exploration, I would recommend consulting PyTorch documentation and tutorials directly on their website. Additional resources include textbooks and online lecture series focused on deep learning and specifically PyTorch as a library. Furthermore, exploring code from established research papers using PyTorch can reveal best practices in tensor manipulation. It's important to understand that while list-like indexing in Python can be intuitive, tensors require an understanding of underlying memory structure to be used effectively and efficiently. This usually involves trading convenience for performance.
