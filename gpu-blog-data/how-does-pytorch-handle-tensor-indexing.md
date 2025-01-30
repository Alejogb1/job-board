---
title: "How does PyTorch handle tensor indexing?"
date: "2025-01-30"
id: "how-does-pytorch-handle-tensor-indexing"
---
PyTorch's tensor indexing operates fundamentally differently from typical array indexing in languages like Python or C++.  Its efficiency stems from leveraging optimized underlying C++ code and its awareness of the tensor's underlying storage layout.  This distinction becomes crucial when dealing with large datasets and complex operations, influencing performance significantly. My experience optimizing neural network training routines heavily relies on this understanding.


**1. Clear Explanation:**

PyTorch tensors are multi-dimensional arrays.  Indexing allows access to specific elements or sub-arrays within these tensors.  Unlike NumPy, which primarily uses integer indices, PyTorch supports a richer indexing scheme leveraging boolean indexing, advanced slicing, and integer array indexing.  Understanding these variations is critical for efficient code.

**Integer Indexing:** This is the most straightforward method, analogous to standard array indexing.  Multiple indices are used to specify the element's location across each tensor dimension.  For a tensor `t` of shape (3, 4, 5), `t[1, 2, 3]` accesses the element at the second row, third column, and fourth depth.  Negative indices are supported, counting from the end of the dimension.  `t[-1, -1, -1]` retrieves the last element.

**Slicing:**  Slicing allows selecting a contiguous section of the tensor.  It uses the `:` operator, specifying start, stop, and step size.  `t[1:3, 0:2, :]` selects a sub-tensor comprising rows 1 and 2, columns 0 and 1, and all depth elements. Omitted values imply defaults: start defaults to 0, stop to the dimension size, and step to 1.

**Boolean Indexing:** This method employs a boolean tensor of the same shape as the target tensor, selecting only elements where the corresponding boolean value is `True`.  Consider a tensor `t` and a boolean mask `mask`. `t[mask]` returns a 1D tensor containing elements where `mask` is `True`.  This proves invaluable for filtering data based on conditions.

**Integer Array Indexing:** This form allows selecting elements based on index arrays.  Suppose we want to select the first and third elements from each row of a 2D tensor `t`. We can use integer arrays: `rows = torch.tensor([0, 2])`, and `t[:, rows]` achieves the selection.  The flexibility extends to multi-dimensional cases, providing powerful manipulation options.

**Advanced Indexing:** A combination of these methods is possible and often utilized. For example, `t[boolean_mask, integer_array]` combines boolean filtering with specific element selection within the filtered subset.  The resulting tensor's shape might be irregular, dependent on the indices.  Understanding these shape changes is crucial for avoiding unexpected behavior.


**2. Code Examples with Commentary:**

**Example 1: Integer and Slice Indexing**

```python
import torch

# Create a 3x4 tensor
tensor = torch.arange(12).reshape(3, 4)
print("Original Tensor:\n", tensor)

# Integer indexing: Access the element at row 1, column 2
element = tensor[1, 2]
print("\nElement at [1, 2]:", element)

# Slicing: Extract a sub-tensor
sub_tensor = tensor[0:2, 1:3]
print("\nSub-tensor [0:2, 1:3]:\n", sub_tensor)

# Negative indexing: Access the last row
last_row = tensor[-1, :]
print("\nLast row:", last_row)
```

This example demonstrates basic integer and slice indexing.  Note the clear distinction between accessing a single element and extracting a sub-tensor.  The output provides a visual demonstration of the indexing results.


**Example 2: Boolean Indexing**

```python
import torch

# Create a tensor
tensor = torch.arange(10)
print("Original Tensor:", tensor)

# Create a boolean mask
mask = tensor > 5
print("\nBoolean Mask:", mask)

# Apply boolean indexing
filtered_tensor = tensor[mask]
print("\nFiltered Tensor:", filtered_tensor)
```

This example showcases boolean indexing.  The boolean mask filters elements based on the condition, yielding a smaller tensor containing only those elements satisfying the condition. This is extremely useful in data filtering and pre-processing.


**Example 3: Advanced Indexing**

```python
import torch

# Create a tensor
tensor = torch.arange(24).reshape(4, 6)
print("Original Tensor:\n", tensor)

# Integer array indexing
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 3, 5])
selected_elements = tensor[rows[:, None], cols]  # Note the use of broadcasting here
print("\nSelected Elements:\n", selected_elements)


# Combining boolean and integer indexing:
mask = tensor > 10
selected_subset = tensor[mask, rows] #this will only select from rows 0,2 if the condition is True.

print("\nSelected Subset (boolean and integer):\n",selected_subset)

```

This example combines integer array indexing and broadcasting, showcasing more advanced techniques.  Note the careful handling of dimensions to ensure correct element selection.  The use of `[:, None]` reshapes the `rows` tensor for proper broadcasting with `cols`.  The addition of boolean indexing further demonstrates the versatility of PyTorch's indexing capabilities.   Understanding broadcasting is critical in this context.



**3. Resource Recommendations:**

The official PyTorch documentation;  a well-structured linear algebra textbook;  advanced Python tutorials focusing on NumPy and array manipulation.  Understanding the underlying concepts of linear algebra and array operations is essential for mastering PyTorch's tensor indexing mechanisms.  Furthermore, a comprehensive text on numerical computing and its optimization techniques will greatly improve performance-related understanding.  A book dedicated to deep learning fundamentals will further highlight the application of these techniques in neural network development and training.
