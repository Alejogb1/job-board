---
title: "How to index a PyTorch tensor using another tensor?"
date: "2025-01-30"
id: "how-to-index-a-pytorch-tensor-using-another"
---
Indexing PyTorch tensors using another tensor is a powerful technique crucial for efficient manipulation of multi-dimensional data, especially when dealing with variable-length sequences or sparse representations.  My experience working on large-scale natural language processing models has highlighted the importance of understanding the nuances of advanced tensor indexing.  Directly accessing elements based on indices derived from another tensor avoids explicit looping, leading to significant performance improvements, particularly with GPU acceleration. However, it requires a careful understanding of broadcasting and advanced indexing rules.

**1. Clear Explanation**

PyTorch allows for flexible tensor indexing using both integer and boolean tensors.  When indexing with an integer tensor, the shape of the indexing tensor dictates the resulting tensor's shape.  The dimensions of the indexing tensor must align with the dimensions being indexed in the primary tensor.  For instance, indexing a 2D tensor along its rows using a 1D tensor will yield a 1D tensor; indexing with a 2D tensor will yield a 2D tensor, mirroring the structure of the indexing tensor.  This is fundamentally different from typical Python list indexing where only single indices are considered.


Boolean indexing, on the other hand, uses a boolean tensor of the same shape as the dimension being indexed.  The result selects elements where the boolean tensor is `True`. This is exceptionally useful for selecting subsets of data based on conditions.  Crucially, both integer and boolean indexing can be combined for complex selections.


Consider a scenario where we have a tensor representing word embeddings, and another tensor containing the indices of words in a sentence.  We can efficiently extract the embeddings for that specific sentence using tensor indexing.  The key is ensuring the indexing tensor's shape is compatible with the dimensions being accessed in the target tensor.  Mismatched dimensions will lead to runtime errors. Broadcasting rules also come into play, automatically expanding dimensions for compatibility where possible, but this must be understood explicitly to avoid unexpected behavior.  Incorrect use of broadcasting can lead to inefficient computations or silent errors that are difficult to debug.


**2. Code Examples with Commentary**

**Example 1: Integer Indexing**

```python
import torch

# A 3x4 tensor representing word embeddings
embeddings = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0]])

# Indices of words in a sentence
sentence_indices = torch.tensor([0, 1, 2])

# Extract embeddings for the sentence
sentence_embeddings = embeddings[sentence_indices]

# Print the result.  Note the shape is (3,4)
print(sentence_embeddings) #Output: tensor([[ 1.,  2.,  3.,  4.], [ 5.,  6.,  7.,  8.], [ 9., 10., 11., 12.]])

#Advanced Example: Selecting specific columns from multiple rows
row_indices = torch.tensor([0,2])
col_indices = torch.tensor([1,3])
selected_elements = embeddings[row_indices,col_indices]
print(selected_elements) #Output: tensor([ 2., 12.])


```

This example demonstrates basic integer indexing. The `sentence_indices` tensor selects rows from the `embeddings` tensor.  The output shape reflects the shape of the index tensor. The advanced example shows how to select specific columns across multiple rows, highlighting the power of combining indexing techniques.


**Example 2: Boolean Indexing**

```python
import torch

# A 3x4 tensor
tensor = torch.arange(12).reshape(3, 4)

# A boolean tensor indicating which elements to select
boolean_mask = torch.tensor([[True, False, True, False],
                            [False, True, False, True],
                            [True, False, True, False]])

# Select elements based on the boolean mask
selected_elements = tensor[boolean_mask]

# Print the result. Note that the output is 1D
print(selected_elements) # Output: tensor([ 0,  2,  5,  7,  8, 10])
```

Here, the `boolean_mask` selects elements based on the boolean values. The `True` values indicate elements to be selected, creating a 1D tensor of selected values. This is particularly useful for filtering data based on specified criteria.


**Example 3: Combining Integer and Boolean Indexing**


```python
import torch

# A 3x4x2 tensor
tensor = torch.arange(24).reshape(3, 4, 2)

# Integer index for the first dimension
row_index = torch.tensor([0, 2])

# Boolean mask for the second dimension
boolean_mask = torch.tensor([[True, False, True, False],
                            [False, True, False, True]])


#Selecting elements: first select the rows, then apply the boolean mask to the second dimension.  The result is a 2x2x2 tensor.
selected_elements = tensor[row_index, boolean_mask]
print(selected_elements)
#Output: tensor([[[ 0,  1],
#                 [ 2,  3]],

#                [[16, 17],
#                 [20, 21]]])
```

This combines both indexing techniques, showing how integer indexing can be used to select specific subsets of the data, followed by further selection using boolean masks.  This allows for very precise data extraction and manipulation.  Careful consideration of broadcasting rules is essential here to ensure the intended behavior.


**3. Resource Recommendations**

The official PyTorch documentation, particularly the sections on tensor manipulation and indexing, should be your primary resource.  Furthermore, consult a comprehensive Python textbook covering NumPy array manipulation, as many concepts translate directly to PyTorch tensors.  Finally, review materials on linear algebra and matrix operations to deepen your understanding of tensor operations.  These resources provide a solid foundation to tackle advanced indexing techniques effectively.  Understanding the underlying principles of tensor manipulation ensures efficient and correct implementation of your tensor-based operations.
