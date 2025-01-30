---
title: "How can I implement scatter_max for a 2D array using NumPy or PyTorch?"
date: "2025-01-30"
id: "how-can-i-implement-scattermax-for-a-2d"
---
My experience optimizing simulations has frequently required efficient operations on sparse data. `scatter_max`, in particular, is a crucial element when dealing with accumulations or finding maxima in a scattered manner, especially when traditional looping is computationally expensive. NumPy, while not offering a direct `scatter_max` function, provides tools that, when combined, can achieve the same effect. PyTorch, on the other hand, has native support for it, making it a more straightforward choice in many circumstances.

The core challenge with `scatter_max` in a 2D context lies in distributing the values from a source tensor based on corresponding indices, but instead of accumulating, we retain the maximum value encountered for each destination index. In this scenario, the input includes: a *source* tensor of values, an *index* tensor denoting where each value should be scattered, and a *destination* tensor where maxima will be stored. The dimensionality of the index tensor is particularly important. For a 2D array, the index tensor will have two columns, representing the row and column indices within the destination.

**NumPy Implementation**

While NumPy lacks a singular function for `scatter_max`, we can use `np.maximum.at` with judicious index construction to simulate the behavior. The approach involves two critical steps: creating a flattened view of the destination array and converting the 2D indices into corresponding 1D flattened indices. The flattened index enables using `np.maximum.at`, which efficiently performs in-place maximum comparisons. We then reshape the flattened result back into the original 2D shape.

```python
import numpy as np

def numpy_scatter_max(destination, indices, source):
    """
    Implements scatter_max for a 2D array using NumPy.

    Args:
        destination (np.ndarray): The 2D array where maxima will be scattered.
        indices (np.ndarray): A 2D array of indices, [row, col] for each source value.
        source (np.ndarray): The values to scatter.

    Returns:
        np.ndarray: The modified destination array with scattered maxima.
    """
    flat_destination = destination.flatten()
    flat_indices = np.ravel_multi_index(indices.T, destination.shape)
    np.maximum.at(flat_destination, flat_indices, source)
    return flat_destination.reshape(destination.shape)

# Example Usage:
destination = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
indices = np.array([[0, 1], [1, 2], [0, 1], [2, 0]])
source = np.array([5, 8, 2, 9])

updated_destination = numpy_scatter_max(destination, indices, source)
print(f"Updated Destination (NumPy):\n{updated_destination}")
```
In the above code, `np.ravel_multi_index` converts the 2D indices into 1D equivalents that `np.maximum.at` accepts.  This function performs an in-place comparison, updating the destination only if the incoming value is greater. This avoids the need for explicit iteration, resulting in performance improvements especially for large arrays. Note that the initial destination array should contain the minimum possible value because the operation only performs maximum comparison. In practice, initializing it to `np.iinfo(destination.dtype).min` is recommended.

Another alternative in NumPy, although less common due to being slightly more involved, uses `np.add.at` with masking. While `add.at` is for addition, by using initial destination filled with the minimum and carefully designed mask using boolean indexing, we can simulate the behaviour of finding the maximum value.

```python
import numpy as np

def numpy_scatter_max_masking(destination, indices, source):
    """
    Implements scatter_max for a 2D array using NumPy with masking.

    Args:
        destination (np.ndarray): The 2D array where maxima will be scattered.
        indices (np.ndarray): A 2D array of indices, [row, col] for each source value.
        source (np.ndarray): The values to scatter.

    Returns:
        np.ndarray: The modified destination array with scattered maxima.
    """

    for i in range(source.shape[0]):
        row_idx, col_idx = indices[i]
        if source[i] > destination[row_idx, col_idx]:
            destination[row_idx,col_idx] = source[i]
    return destination


# Example Usage:
destination = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
indices = np.array([[0, 1], [1, 2], [0, 1], [2, 0]])
source = np.array([5, 8, 2, 9])
updated_destination = numpy_scatter_max_masking(destination, indices, source)
print(f"Updated Destination (NumPy - Masking):\n{updated_destination}")
```

This approach iterates through the source, and only modifies the destination array if a new maximum is found at the destination index. This, however, can be less efficient compared to the former approach with large input data due to python-level iteration.

**PyTorch Implementation**

PyTorch provides a direct `scatter_max` method for tensors, making implementation significantly more succinct and often more performant, especially when operating on GPU. This is advantageous when dealing with large datasets where GPU acceleration is important. The method accepts the dimension along which to scatter (in our case dimension `0` for flattening the index when input as two columns `[row, col]` where each element corresponds to one scattering), index tensor, and the source tensor. The return includes not just the tensor containing scattered maximum values, but also an index tensor which shows from which original index each maximum came.

```python
import torch

def pytorch_scatter_max(destination, indices, source):
    """
    Implements scatter_max for a 2D array using PyTorch.

    Args:
        destination (torch.Tensor): The 2D tensor where maxima will be scattered.
        indices (torch.Tensor): A 2D tensor of indices, [row, col] for each source value.
        source (torch.Tensor): The values to scatter.

    Returns:
        torch.Tensor: The modified destination tensor with scattered maxima.
    """
    destination_flat = destination.flatten()
    indices = indices.long()
    flat_indices = torch.ravel_multi_index(indices.T, destination.shape)
    max_val, _ = destination_flat.scatter_max(0, flat_indices, source)
    return max_val.reshape(destination.shape)


# Example Usage:
destination = torch.zeros((3, 3))
indices = torch.tensor([[0, 1], [1, 2], [0, 1], [2, 0]])
source = torch.tensor([5, 8, 2, 9])

updated_destination = pytorch_scatter_max(destination, indices, source)
print(f"Updated Destination (PyTorch):\n{updated_destination}")
```
In the PyTorch example,  `torch.ravel_multi_index` flattens 2D indices to 1D for `scatter_max`, similar to NumPy's approach.  PyTorch directly implements the scatter operation, usually employing highly optimized routines for efficient computation. Notice that before performing the operation, we convert the indices tensor to long data type since that is required by the `scatter_max` operation.

**Resource Recommendations**

For a deeper dive into NumPy, consult its official documentation which offers detailed explanations of array operations, especially concerning broadcasting and advanced indexing. Books specializing in scientific computing with Python, like the "Python Data Science Handbook" offer invaluable guidance on effective NumPy usage.

For PyTorch, begin with the official tutorials and documentation that cover basic tensor manipulations, as well as advanced topics like efficient indexing and GPU utilization. Deep learning focused texts, such as "Deep Learning with PyTorch" often provide context for efficient tensor operations within the broader framework of neural network development.  It is also beneficial to study the underlying implementation details, which are often detailed in official GitHub repositories, which can enhance understanding of algorithm efficiency and optimal application.
