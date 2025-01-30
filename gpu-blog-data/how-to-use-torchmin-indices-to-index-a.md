---
title: "How to use torch.min indices to index a multi-dimensional tensor?"
date: "2025-01-30"
id: "how-to-use-torchmin-indices-to-index-a"
---
The core challenge when using `torch.min`'s returned indices to index a multi-dimensional tensor arises from the fact that `torch.min` operates along a specified dimension, collapsing it, and therefore providing indices relative to that dimension *only*. To accurately index the original tensor, one needs to expand these indices to account for the dimensionality of the original tensor. This requires a careful understanding of how PyTorch manages tensor layouts and indexing.

**Explanation**

When `torch.min` is applied to a tensor, it returns two tensors: the minimum values and the indices where those minimum values were found. The `indices` tensor matches the shape of the `values` tensor, which is the original tensor with the dimension along which the minimum was computed collapsed (removed). These indices are *relative* to the collapsed dimension, and therefore not suitable for direct indexing of the full original tensor without proper adaptation.

For instance, consider a tensor with the shape `(batch_size, height, width, channels)`. If we perform `torch.min` along the `channels` dimension, the resulting `indices` tensor will have the shape `(batch_size, height, width)`. Each element in this `indices` tensor represents the index of the minimum value within the `channels` dimension for a specific location defined by the batch, height, and width.

To use these indices for accessing the original tensor, we cannot simply use the `indices` tensor directly as the indexer. Instead, we need to build a new set of indices that reflect the shape of the original tensor. We achieve this by creating meshgrids and/or `arange` tensors, combined with the output indices. Essentially, we generate indices for all the dimensions that were not collapsed by `torch.min` and then combine these with indices that were returned. This involves careful reshaping and expansion.

The general approach involves:

1.  **Determining the collapsed dimension:** Identify the dimension along which `torch.min` was applied. Let this be `dim`.
2.  **Generating indices for non-collapsed dimensions:** Construct coordinate indices for all dimensions excluding the collapsed dimension. These are often obtained using `torch.arange` and reshaping them appropriately. These are the indexes that are constant across what was collapsed.
3.  **Combining indices:** Using the returned indices from `torch.min`, merge the indices for non-collapsed dimensions and the indices obtained from `torch.min`. The final indexed tensor will have the same shape as the `values` tensor returned by `torch.min`.

**Code Examples**

Here are three examples illustrating different scenarios and indexing methods:

**Example 1: Minimum Along the Channel Dimension (4D Tensor)**

Consider a 4D tensor and finding the minimum along the channel dimension:

```python
import torch

def index_with_min_4d(tensor):
    batch_size, height, width, channels = tensor.shape
    min_values, min_indices = torch.min(tensor, dim=-1) # Minimum along channels
    
    # Create the meshgrid for the dimensions that were NOT collapsed.
    b = torch.arange(batch_size).reshape(batch_size,1,1).expand(batch_size, height, width)
    h = torch.arange(height).reshape(1,height,1).expand(batch_size, height, width)
    w = torch.arange(width).reshape(1,1,width).expand(batch_size, height, width)

    indexed_tensor = tensor[b, h, w, min_indices]
    return indexed_tensor

# Example Usage
input_tensor = torch.randn(2, 3, 4, 5)
indexed_result = index_with_min_4d(input_tensor)
print("Input shape:", input_tensor.shape)
print("Indexed result shape:", indexed_result.shape)
assert indexed_result.shape == torch.Size([2,3,4]), "Shape mismatch"
```

*   **Commentary:** Here we use `torch.min` along the last dimension (-1, the channel dimension). The meshgrid is made by expanding `torch.arange` which creates index tensors the size of the 3 dimensions unaffected by `torch.min`.  The final indexed tensor is of shape `(batch_size, height, width)`, which corresponds to the shape of the `values` tensor returned by `torch.min`. We use `indexed_tensor = tensor[b, h, w, min_indices]`, which performs the selection from the original tensor at the specified indices.

**Example 2: Minimum Along the Height Dimension (3D Tensor)**

Now, let’s examine a 3D tensor and compute the minimum along the height dimension:

```python
import torch

def index_with_min_3d(tensor):
    batch_size, height, width = tensor.shape
    min_values, min_indices = torch.min(tensor, dim=1) # Min along height.

    # Create the meshgrid for the dimensions that were NOT collapsed.
    b = torch.arange(batch_size).reshape(batch_size,1).expand(batch_size, width)
    w = torch.arange(width).reshape(1,width).expand(batch_size, width)

    indexed_tensor = tensor[b, min_indices, w]
    return indexed_tensor

# Example Usage
input_tensor = torch.randn(4, 5, 6)
indexed_result = index_with_min_3d(input_tensor)
print("Input shape:", input_tensor.shape)
print("Indexed result shape:", indexed_result.shape)
assert indexed_result.shape == torch.Size([4,6]), "Shape mismatch"
```

*   **Commentary:** This example is very similar to the first example but demonstrates a different dimension. We use `torch.min` along dimension 1 (the height dimension).  Again the meshgrid is generated to represent the dimensions that are not collapsed. Notice, we omit creating the index dimension for the height, as this is handled by `min_indices`.  The final indexed tensor has the shape (batch\_size, width), which is the shape of the `min_values` output of `torch.min`.

**Example 3: Minimum Along a First Dimension (2D Tensor)**

Here, we'll compute the minimum along the first dimension of a 2D tensor and index accordingly.  This will demonstrate that you can collapse any dimension.

```python
import torch

def index_with_min_2d(tensor):
    height, width = tensor.shape
    min_values, min_indices = torch.min(tensor, dim=0) # Min along height
    
    # Create the meshgrid for the dimensions that were NOT collapsed.
    w = torch.arange(width).reshape(1, width).expand(1, width)

    indexed_tensor = tensor[min_indices, w]
    return indexed_tensor

# Example Usage
input_tensor = torch.randn(5, 7)
indexed_result = index_with_min_2d(input_tensor)
print("Input shape:", input_tensor.shape)
print("Indexed result shape:", indexed_result.shape)
assert indexed_result.shape == torch.Size([1, 7]), "Shape mismatch"
```

*   **Commentary:** In this case, we compute the minimum along the first dimension of a 2D tensor.  The index grid is made only from `w`.  The resulting tensor is the width, which is the collapsed result of `torch.min`

**Resource Recommendations**

For a deeper understanding of tensor manipulation and indexing in PyTorch, I recommend exploring the following:

1.  **The official PyTorch Documentation:**  The official documentation provides in-depth information on tensor operations, including `torch.min`, indexing and reshaping methods.
2.  **PyTorch Tutorials:** PyTorch provides comprehensive tutorials covering a wide range of topics, including tensor manipulation.  Look at tutorials related to the `torch.arange`, `.reshape` and `torch.Tensor` indexing.
3.  **Advanced PyTorch Books:**  Several advanced books delve into the intricacies of PyTorch tensor operations and indexing.  These texts provide further insights into managing multi-dimensional tensors.  Specifically focus on sections related to broadcasting, advanced indexing and multi-dimensional data manipulation.

These resources will give a more thorough understanding of the nuances and techniques for using `torch.min` indices effectively. They also provide a stronger foundation for handling complex tensor manipulations in other contexts.
