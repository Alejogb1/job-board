---
title: "How to index a PyTorch tensor?"
date: "2025-01-30"
id: "how-to-index-a-pytorch-tensor"
---
Indexing PyTorch tensors effectively is crucial for manipulating and extracting data within neural network operations, and performance is tightly coupled with how one approaches this task. Having spent considerable time optimizing various deep learning models, I've found that a nuanced understanding of tensor indexing mechanisms is fundamental. The PyTorch library offers powerful yet concise methods, which, when employed correctly, can significantly reduce code complexity and improve execution speeds.

At its core, tensor indexing in PyTorch operates similarly to NumPy arrays but with added capabilities tailored for GPU acceleration. Essentially, you can access specific tensor elements or subarrays using integer-based indexing, slicing, advanced indexing (using other tensors as indices), and combinations thereof. The basic principle remains accessing data at specific memory locations, although the underlying hardware can dramatically alter performance characteristics.

**1. Basic Integer and Slicing Indexing**

The most straightforward indexing involves specifying integer indices for each dimension of a tensor. For a two-dimensional tensor (a matrix), you need two indices: one for the row and one for the column. For a three-dimensional tensor, you need three indices, and so forth. Consider a 3x3 matrix as a tangible starting point.

```python
import torch

# Create a 3x3 tensor
matrix = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

# Access individual elements
element_1_1 = matrix[0, 0] # Access the element at row 0, column 0. Output: 1
element_2_3 = matrix[1, 2] # Access the element at row 1, column 2. Output: 6
print(f"Element (0,0): {element_1_1}")
print(f"Element (1,2): {element_2_3}")

# Indexing with negative numbers
element_last_row_last_col = matrix[-1, -1] # Access last row, last column. Output: 9
print(f"Element (-1,-1): {element_last_row_last_col}")
```

Here, `matrix[0, 0]` directly accesses the element at the 0th row and 0th column, which is 1. Similarly, `matrix[1, 2]` returns 6, located at the 1st row and 2nd column. PyTorch, like Python lists, supports negative indices, where -1 represents the last element in a dimension, -2 the second last, and so on. This is valuable when you need to extract data from the tail end of a tensor. Slicing involves specifying a start, end, and step. The syntax resembles standard Python list slicing: `tensor[start:end:step, start:end:step, ...]`. This allows for the extraction of subsections or subarrays.

```python
# Slicing to extract a row
first_row = matrix[0, :] # All columns of the first row. Output: tensor([1, 2, 3])
second_row = matrix[1:2, :]  # The second row. Output: tensor([[4, 5, 6]]) (note the 2D output)
print(f"First row: {first_row}")
print(f"Second row: {second_row}")

# Slicing to extract columns
first_column = matrix[:, 0] # All rows of the first column. Output: tensor([1, 4, 7])
last_two_columns = matrix[:, 1:] # The last two columns. Output: tensor([[2, 3], [5, 6], [8, 9]])
print(f"First column: {first_column}")
print(f"Last two columns:\n{last_two_columns}")

# Sub-section slicing
sub_section = matrix[0:2, 1:3] #Rows 0,1 columns 1,2. Output: tensor([[2, 3], [5, 6]])
print(f"Sub-section:\n{sub_section}")
```

`matrix[0, :]` accesses all elements in the first row.  The colon `:` by itself implies taking all the indices in that dimension.  `matrix[:, 0]` accesses all the rows in the first column. Slicing creates a view of the original tensor, modifying this view modifies the original tensor which can lead to memory efficiencies. `matrix[0:2, 1:3]` takes a subsection from the original matrix from the rows 0 to 2 exclusive and the columns 1 to 3 exclusive.

**2. Advanced Indexing with Integer Tensors**

Advanced indexing takes this further, using a tensor of indices rather than a simple slice. This allows for the extraction of elements in an arbitrary order. The most significant characteristic here is that it returns a copy, rather than a view. The output shape depends on the shape of the index tensor itself and not the base tensor.

```python
# Advanced indexing using a tensor of indices
indices = torch.tensor([0, 2, 1])
indexed_row = matrix[indices, :] # Selects rows 0, 2, and 1 in that order
print(f"Indexed rows: \n{indexed_row}")

indices_2d = torch.tensor([[0, 1], [2, 0]])
indexed_elements = matrix[indices_2d, indices_2d] #Note: This is a combination of indexing methods which does not work as expected and needs debugging
indexed_elements_actual = matrix[indices_2d[:,0],indices_2d[:,1]]
print(f"Indexed elements (incorrect):\n{indexed_elements}")
print(f"Indexed elements (correct):\n{indexed_elements_actual}")

row_indices = torch.tensor([0, 1, 2])
col_indices = torch.tensor([2, 0, 1])
indexed_custom_elements = matrix[row_indices,col_indices] #  Selects elements (0,2), (1,0), and (2,1)
print(f"Custom indexed elements:\n{indexed_custom_elements}")
```

In the first case, `matrix[indices, :]`  selects the rows at indices 0, 2, and 1, and keeps all the columns, resulting in a new tensor with rows re-ordered. Indexing with 2D integer tensors can be confusing. Here, `matrix[indices_2d, indices_2d]` does *not* take elements `(0,0)`, `(1,1)`, `(2,2)`, `(0,0)`, this is an *incorrect* indexing method. The correct method is `matrix[indices_2d[:,0],indices_2d[:,1]]`.  Finally `matrix[row_indices, col_indices]` fetches individual elements. `matrix[row_indices,col_indices]` returns a tensor of elements at the cross product of rows and columns. The index tensors do not need to be of the same shape. For instance, `matrix[row_indices, col_indices[0:1]]` would take the elements from row indices `[0,1,2]` and column indices `[2]`.

**3. Boolean Indexing (Masking)**

Boolean indexing allows for selective element extraction based on a mask, a tensor of booleans of the same shape as the base tensor or broadcastable shape. Each true value in the mask corresponds to a selected element. I've often used boolean masks to filter data based on criteria during preprocessing or to perform element-wise conditional operations.

```python
# Boolean indexing
mask = matrix > 5 # Create a boolean tensor where True indicates an element > 5
masked_values = matrix[mask] # Select all elements that are greater than 5
print(f"Mask: {mask}")
print(f"Masked values: {masked_values}")

# Setting values based on conditions.
matrix[matrix < 3] = 0 # Setting values of 0 where they were below 3.
print(f"Modified matrix:\n {matrix}")
```

`mask = matrix > 5` creates a mask containing boolean values based on whether each element is greater than 5. `masked_values = matrix[mask]` uses the mask to extract elements that meet that condition, creating a new 1D tensor. These extracted elements are a copy, and not a view. It also allows for in-place modifications to the tensor. `matrix[matrix < 3] = 0` assigns 0 to any element less than 3 *in place* altering the original tensor itself.

**Resource Recommendations**

For further understanding, I'd strongly advise focusing on the official PyTorch documentation, particularly sections on tensor manipulation and advanced indexing. Look into books or papers that specifically detail tensor operations within deep learning frameworks.  Additionally, reviewing open-source code repositories for projects utilizing PyTorch will expose you to a range of practical indexing strategies. Practice using a range of synthetic and real datasets. Experimenting in a coding environment is the best way to grasp the nuances of each indexing method.

In conclusion, mastering tensor indexing in PyTorch isn't just about syntax; it's about understanding how operations translate to memory access and affect performance. Each method has specific use cases and characteristics. Knowing when to use integer indexing, slicing, advanced indexing, and boolean masking allows me to craft efficient and expressive code. Through considered application and iterative experimentation, tensor manipulation with PyTorch becomes an integral and empowering tool within deep learning development.
