---
title: "How to get indices of rows with a specific value in a PyTorch tensor?"
date: "2025-01-30"
id: "how-to-get-indices-of-rows-with-a"
---
The core challenge in efficiently retrieving row indices based on a specific value within a PyTorch tensor hinges on leveraging PyTorch's broadcasting capabilities and its optimized array operations, rather than resorting to explicit Python loops.  My experience optimizing large-scale machine learning models has shown that this seemingly simple task can become a bottleneck if not handled judiciously.  Inefficient approaches can easily introduce O(n²) complexity where n is the number of rows, drastically impacting performance for high-dimensional tensors.

The most straightforward, yet often overlooked, solution involves leveraging boolean indexing in conjunction with `torch.nonzero()`.  This approach combines the power of PyTorch's element-wise comparison with its efficient index retrieval capabilities.  The key is to recognize that the comparison operation itself produces a boolean tensor that directly maps to the desired row indices.

**1. Explanation:**

The process can be broken down into three fundamental steps:

a) **Comparison:**  We perform an element-wise comparison between the tensor and the target value. This generates a boolean tensor of the same shape as the input tensor, where `True` indicates a match and `False` indicates a mismatch.

b) **Reduction:** To obtain row indices, we need to collapse the boolean tensor along its column (or other relevant) dimensions. This can typically be achieved with `torch.any()`, which returns `True` if any element along a given dimension is `True`. This transforms the boolean tensor into a one-dimensional tensor where `True` indicates a row containing the target value.

c) **Index Extraction:**  Finally, `torch.nonzero()` extracts the indices of all `True` values within this reduced boolean tensor. These indices directly represent the row numbers of interest.

**2. Code Examples with Commentary:**

**Example 1: Simple 2D Tensor**

```python
import torch

tensor = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [1, 8, 9],
                      [10, 11, 12]])

target_value = 1

# Element-wise comparison
comparison_result = tensor == target_value

# Reduce along columns to check for presence in a row
row_presence = comparison_result.any(dim=1)

# Extract indices of rows containing the target value
row_indices = torch.nonzero(row_presence).squeeze()

print(f"Rows with value {target_value}: {row_indices}")
```

This example clearly demonstrates the three steps outlined above.  The `squeeze()` function removes unnecessary singleton dimensions from the output of `torch.nonzero()`.  Note that this approach handles cases where the target value might appear multiple times within a single row; `torch.any()` ensures that a row is flagged if the target value exists anywhere within it.


**Example 2: Handling Multi-Dimensional Tensors**

```python
import torch

tensor = torch.tensor([[[1, 2], [3, 4]],
                      [[5, 6], [7, 1]],
                      [[9, 10],[11,12]]])

target_value = 1

# Element-wise comparison
comparison_result = tensor == target_value

# Reduce along the last two dimensions to check for presence in the first dimension
row_presence = comparison_result.any(dim=(1,2))

# Extract indices of rows containing the target value
row_indices = torch.nonzero(row_presence).squeeze()

print(f"Rows with value {target_value}: {row_indices}")
```

This extends the previous example to a 3D tensor. The key modification is adjusting the `dim` argument in `torch.any()` to reflect the dimensions we want to collapse (here, the last two dimensions, representing columns and a potential inner dimension).  This demonstrates the flexibility of the method in adapting to tensors of arbitrary dimensions.


**Example 3:  Performance Optimization for Large Tensors**

For exceptionally large tensors, memory efficiency becomes crucial.  While the above method is generally efficient, consider using `torch.where()` for a potential performance boost in certain scenarios.  `torch.where()` can directly return indices based on a boolean condition, avoiding the intermediate step of `torch.any()`. However, it might be less readable for beginners.

```python
import torch

tensor = torch.randint(0, 20, (10000, 5)) #Large tensor for demonstration
target_value = 7

# Efficient index retrieval using torch.where()
row_indices = torch.where((tensor == target_value).any(dim=1))[0]

print(f"Number of rows containing {target_value}: {len(row_indices)}")
```

This example uses `torch.randint` to generate a large sample tensor to highlight the importance of efficiency for large datasets.  The `torch.where()` function directly provides the row indices where the condition (`(tensor == target_value).any(dim=1)`) is true, making it a concise and potentially faster alternative.


**3. Resource Recommendations:**

I recommend reviewing the official PyTorch documentation on tensor indexing, boolean indexing, and the specific functions used in these examples (`torch.nonzero()`, `torch.any()`, `torch.where()`). Further exploration of  advanced tensor manipulation techniques within the PyTorch documentation will prove beneficial.  Understanding broadcasting behavior is also essential for proficient PyTorch programming.  Finally, examining code examples from established machine learning libraries that utilize PyTorch could provide further insight into best practices and performance optimization strategies.  These resources provide a solid foundation for mastering advanced PyTorch operations.
