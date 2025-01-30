---
title: "How to find unique vectors in a PyTorch tensor?"
date: "2025-01-30"
id: "how-to-find-unique-vectors-in-a-pytorch"
---
Identifying unique vectors within a PyTorch tensor, especially in the context of high-dimensional data, is a common requirement in tasks such as clustering, anomaly detection, and data preprocessing. My work in developing a novel deep learning-based anomaly detection system for satellite imagery, where each image is represented as a feature vector, highlighted the necessity of efficient methods for identifying and filtering redundant feature vectors. A naive approach involving element-wise comparisons proves computationally expensive with increasing dimensionality and tensor size. The key to efficient unique vector identification lies in leveraging PyTorch's capacity for vectorized operations and applying techniques that reduce comparison space.

Fundamentally, determining uniqueness of vectors necessitates a mechanism to compare them. Direct element-wise comparisons, while conceptually straightforward, become highly inefficient when dealing with large tensors. Instead, we should transform the vectors into a comparable form, such as strings, and then utilize sets to determine uniqueness. A common technique involves treating each vector as a tuple of its elements; however, this can be slow due to tuple creation overhead and relies on Python's hashing for equality checks. Therefore, a more efficient approach involves converting the tensor into a NumPy array and then using its `view` method to treat each vector as a contiguous block of memory. This permits us to obtain a view of the tensor as contiguous rows which are then converted to bytes, and these byte sequences can be easily hashed within a Python set for uniqueness assessment. The primary advantage of this approach stems from the optimized low-level memory access and comparison routines of NumPy and Python's hashing functionality. This methodology leverages the performance of these underlying libraries, resulting in faster processing speeds.

Let's examine three code examples that demonstrate this approach, increasing in complexity and functionality.

**Example 1: Basic Unique Vector Identification**

```python
import torch
import numpy as np

def find_unique_vectors_basic(tensor):
  """
  Finds unique vectors in a PyTorch tensor using NumPy byte conversion and set.

  Args:
    tensor: A PyTorch tensor of shape (N, D), where N is the number of vectors
      and D is the dimensionality of each vector.

  Returns:
    A PyTorch tensor containing only the unique vectors.
  """
  if tensor.ndim != 2:
      raise ValueError("Input tensor must be 2D.")
  tensor_np = tensor.detach().cpu().numpy()
  unique_vectors = set()
  unique_vector_indices = []
  for i, row in enumerate(tensor_np):
    row_bytes = row.tobytes()
    if row_bytes not in unique_vectors:
      unique_vectors.add(row_bytes)
      unique_vector_indices.append(i)

  return tensor[unique_vector_indices]


# Example Usage:
tensor_example = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9], [4, 5, 6]])
unique_vectors = find_unique_vectors_basic(tensor_example)
print("Unique Vectors (Basic):\n", unique_vectors)
```

In this first example, the core mechanism is illustrated. The PyTorch tensor is first moved to the CPU and converted to a NumPy array using `tensor.detach().cpu().numpy()`. Then, each row of the array is converted to a sequence of bytes using `row.tobytes()`. These bytes are then added to a set; a set's properties guarantee uniqueness, so only the first occurrence of each unique vector's byte representation is added. Finally, the indices of the unique vectors are used to index the original PyTorch tensor, and this subset of the tensor containing only unique vectors is returned. The main limitation here is the explicit loop which is suitable for smaller tensors, but less optimal for very large tensors.

**Example 2: Improved Efficiency with NumPy's View**

```python
import torch
import numpy as np

def find_unique_vectors_numpy_view(tensor):
  """
  Finds unique vectors in a PyTorch tensor using NumPy's view and sets.

  Args:
    tensor: A PyTorch tensor of shape (N, D).

  Returns:
    A PyTorch tensor containing only the unique vectors.
  """
  if tensor.ndim != 2:
      raise ValueError("Input tensor must be 2D.")
  tensor_np = tensor.detach().cpu().numpy()
  rows = tensor_np.view(tensor_np.dtype).reshape(tensor_np.shape[0], -1).view('S'+str(tensor_np.shape[1] * tensor_np.dtype.itemsize))

  unique_rows, indices = np.unique(rows, return_index=True)

  return tensor[indices]

# Example Usage:
tensor_example = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9], [4, 5, 6]], dtype=torch.int64)
unique_vectors = find_unique_vectors_numpy_view(tensor_example)
print("Unique Vectors (NumPy View):\n", unique_vectors)
```

This example enhances the previous one by using NumPy's `view` method to avoid explicit looping. By employing `tensor_np.view(tensor_np.dtype).reshape(tensor_np.shape[0], -1)`, we create a view of the underlying memory of the NumPy array as a sequence of bytes. Specifically, the shape is transformed from `(N, D)` to `(N, D * size_of_element)`, so that the each row of the view represents the byte representation of the original row in the original tensor. Subsequently, this row view is shaped to represent each row as a single byte sequence which can be directly compared for uniqueness, effectively removing element-wise comparisons within the loop. NumPy's `np.unique()` function is then used on these byte sequences, with the argument `return_index=True` to get the indices of the first occurrence of each unique vector. This effectively replaces our prior loop and leverages the optimized implementation of NumPy's `unique()` function, significantly improving speed, especially with larger tensors. Note that the data type of the input tensor is important for the row byte representation; using `torch.int64` means each integer is represented with 64 bits.

**Example 3: Handling Floating-Point Inaccuracies with Tolerance**

```python
import torch
import numpy as np

def find_unique_vectors_tolerance(tensor, tolerance=1e-6):
    """
    Finds unique vectors within a given tolerance, accounting for floating-point inaccuracies.

    Args:
      tensor: A PyTorch tensor of shape (N, D), where N is the number of vectors
        and D is the dimensionality of each vector.
      tolerance: A float indicating the acceptable tolerance when comparing
       floating-point values.

    Returns:
      A PyTorch tensor containing only the unique vectors.
    """
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D.")
    if not torch.is_floating_point(tensor):
        return find_unique_vectors_numpy_view(tensor)

    tensor_np = tensor.detach().cpu().numpy()
    unique_vectors = []
    indices = []
    for i, current_vector in enumerate(tensor_np):
        is_unique = True
        for unique_vector in unique_vectors:
            if np.all(np.abs(current_vector - unique_vector) <= tolerance):
              is_unique = False
              break
        if is_unique:
           unique_vectors.append(current_vector)
           indices.append(i)
    return tensor[indices]

# Example Usage:
tensor_example = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.000001], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
unique_vectors = find_unique_vectors_tolerance(tensor_example)
print("Unique Vectors (Tolerance):\n", unique_vectors)

```

This third example addresses the issue of floating-point inaccuracies when determining uniqueness. If the input tensor is not a floating point tensor, we fall back to our view based solution. If it is, then it introduces a `tolerance` parameter. Floating point numbers can have slightly varying values due to numerical representation, rendering standard equality checks unreliable. In this method, we iterate through the vectors and compare each against previously discovered unique vectors by calculating the absolute difference. If the absolute difference of every element is less than or equal to the specified `tolerance`, we assume they are duplicates and do not add the current vector to unique vectors. This example is less performant than example two as it iterates over every vector comparing against all known unique vectors and therefore is best utilized when floating point imprecision is a primary concern. The main advantage is its ability to control what constitutes "unique" in the face of floating point imprecision.

In summary, identifying unique vectors within a PyTorch tensor demands consideration of performance, data type, and potential numerical inaccuracies. The use of NumPy's `view` coupled with its `unique` method provides the best performance for most use cases. When handling floating-point numbers, implementing a tolerance based uniqueness check is needed at the expense of some performance.

For further study, one can explore topics such as: “NumPy advanced indexing,” “hashing algorithms for data representation,” and "floating-point comparison methods." Additionally, research into specialized algorithms for duplicate detection in high-dimensional data, such as Locality Sensitive Hashing, might prove beneficial for very large-scale datasets. Investigating methods for performing similar operations on the GPU may also be beneficial in some contexts.
