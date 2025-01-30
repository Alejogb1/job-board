---
title: "How can batched Euclidean distance be calculated efficiently?"
date: "2025-01-30"
id: "how-can-batched-euclidean-distance-be-calculated-efficiently"
---
The efficiency of batched Euclidean distance calculation hinges significantly on leveraging vectorized operations and memory layout. In my experience developing real-time machine learning systems, I've encountered scenarios where naive implementations, especially those involving iterative loops, proved to be significant performance bottlenecks. Understanding the underlying mathematical formula and how it interacts with linear algebra libraries is crucial for achieving optimal performance.

Euclidean distance, between two vectors *a* and *b*, is defined as the square root of the sum of squared differences of their corresponding elements:  √∑(aᵢ - bᵢ)². When dealing with batches of vectors, we are essentially calculating this distance pairwise for every vector in one batch against every vector in another batch, or, more commonly, against itself. A naive approach involving nested loops can lead to O(n²m) complexity, where n is the number of vectors and m is their dimensionality.

The key to efficiency lies in rewriting the calculation to take advantage of matrix operations. The Euclidean distance between all pairs of vectors in a matrix can be expressed using matrix algebra without iterative loops. Specifically, let's assume we have a batch of vectors represented by a matrix *A* of shape (n, m) where n is the number of vectors and m is their dimensionality. If we want the distance between every vector and itself, the first step involves computing the squared magnitude of each vector. This is computed by multiplying the transpose of the matrix by itself (A.T @ A). We need to consider that A has rows which represent the vectors so we will need to transpose each vector. Next, we need the dot product matrix of our initial matrix A with itself (A @ A.T). The last step is to add the first squared magnitudes matrix to the second squared magnitudes matrix and subtracting twice the dot product matrix. This whole new matrix represents the square of the euclidean distance. Applying a square root function to the final matrix will produce the final result.

Let's see some code examples.

**Example 1: NumPy Implementation**

```python
import numpy as np

def batched_euclidean_distance_numpy(A):
  """Calculates the Euclidean distance between all vectors in a batch.

  Args:
      A: A numpy array representing the batch of vectors (n, m)

  Returns:
      A numpy array containing the distance between every pair of vectors (n, n).
  """
  n, m = A.shape
  A_squared_magnitudes = np.sum(A**2, axis=1, keepdims=True)
  dot_product = A @ A.T
  distances_squared = A_squared_magnitudes + A_squared_magnitudes.T - 2 * dot_product
  distances = np.sqrt(np.maximum(distances_squared, 0)) #To avoid NaN values
  return distances

# Example Usage
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
distances = batched_euclidean_distance_numpy(A)
print(distances)
```

In this example, we use `numpy`’s vectorized operations. The `np.sum(A**2, axis=1, keepdims=True)` computes the squared magnitude of each vector in A and keeps the output as a column vector, facilitating broadcasting later on. The dot product `A @ A.T` calculates the dot product between each vector pair and the subsequent steps implement the mathematical formulation described earlier. The `np.maximum` function is essential because some values from the subtraction of the matrices can be extremely close to zero but due to float precision they are negative, resulting in `NaN` values after the square root operation, so setting them to zero first helps to avoid that error.

**Example 2: PyTorch Implementation**

```python
import torch

def batched_euclidean_distance_torch(A):
    """Calculates the Euclidean distance between all vectors in a batch using PyTorch.

    Args:
        A: A torch tensor representing the batch of vectors (n, m)

    Returns:
        A torch tensor containing the distance between every pair of vectors (n, n).
    """
    n, m = A.shape
    A_squared_magnitudes = torch.sum(A**2, dim=1, keepdim=True)
    dot_product = torch.matmul(A, A.T)
    distances_squared = A_squared_magnitudes + A_squared_magnitudes.T - 2 * dot_product
    distances = torch.sqrt(torch.clamp(distances_squared, min=0))  # Using clamp to avoid NaNs
    return distances


# Example Usage
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
distances = batched_euclidean_distance_torch(A)
print(distances)
```

This example mirrors the NumPy implementation but leverages PyTorch tensors and operations. The function utilizes `torch.sum` with `dim=1` to calculate the squared magnitudes, `torch.matmul` for the dot product, and `torch.sqrt` and `torch.clamp` to complete the Euclidean distance calculation. The use of PyTorch allows the code to leverage GPUs if available to enhance processing speed. Like the previous example, the clamp function ensures that we avoid negative inputs to the sqrt function.

**Example 3: TensorFlow Implementation**

```python
import tensorflow as tf

def batched_euclidean_distance_tf(A):
    """Calculates the Euclidean distance between all vectors in a batch using TensorFlow.

    Args:
        A: A TensorFlow tensor representing the batch of vectors (n, m)

    Returns:
        A TensorFlow tensor containing the distance between every pair of vectors (n, n).
    """
    n, m = A.shape
    A_squared_magnitudes = tf.reduce_sum(tf.square(A), axis=1, keepdims=True)
    dot_product = tf.matmul(A, A, transpose_b=True)
    distances_squared = A_squared_magnitudes + tf.transpose(A_squared_magnitudes) - 2 * dot_product
    distances = tf.sqrt(tf.maximum(distances_squared, 0.0)) # Using tf.maximum to avoid NaNs
    return distances


# Example Usage
A = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
distances = batched_euclidean_distance_tf(A)
print(distances)
```

The TensorFlow version is structurally similar, demonstrating the fundamental principles of batched Euclidean distance calculation across different frameworks. The functions used are analogous: `tf.reduce_sum` for sum reduction, `tf.matmul` for dot product, `tf.sqrt` and `tf.maximum`. TensorFlow, like PyTorch, is optimized for use with hardware accelerators, making it suitable for computationally intensive tasks. This implementation will use the GPU if available.

**Resource Recommendations:**

For a deeper dive into optimizing these calculations, several resources are beneficial. Consult documentation for the specific linear algebra library you are using (NumPy, PyTorch, TensorFlow). This documentation will provide insights into low-level optimizations and efficient memory utilization, particularly when employing GPU acceleration. Furthermore, studying matrix calculus and numerical linear algebra will improve your understanding of the underlying concepts, which in turn will aid in implementing and optimizing any algorithm. Finally, performance profiling tools can help identify bottlenecks in your code, particularly in larger scale applications, allowing for more targeted optimization strategies.

In conclusion, calculating batched Euclidean distances efficiently requires a transition from iterative, element-wise computation to vectorized operations, leveraging the power of linear algebra libraries. This strategy substantially reduces runtime complexity and improves performance, particularly in scenarios involving large datasets or real-time applications. It's important to be aware of the different libraries and their specific function calls and characteristics to optimize for every use case.
