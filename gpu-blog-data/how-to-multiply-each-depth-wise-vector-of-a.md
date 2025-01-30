---
title: "How to multiply each depth-wise vector of a 3D tensor by a 2D matrix using NumPy or TensorFlow?"
date: "2025-01-30"
id: "how-to-multiply-each-depth-wise-vector-of-a"
---
In deep learning, I’ve often encountered the need to apply a transformation, represented by a matrix, to each depth-wise vector within a 3D tensor. This operation is not straightforward due to the dimensional mismatch between a 2D matrix and the 3D tensor directly. The key is to understand that we are performing a batch matrix multiplication where each "batch" is a depth-wise vector and the transformation matrix is consistent across all batches. NumPy's broadcasting and TensorFlow’s batch matrix multiplication capabilities provide efficient solutions to this problem, but the specific techniques vary slightly between the libraries. Here’s how I’ve approached this, and the associated nuances.

A 3D tensor, often denoted as a shape `(H, W, C)`, can be interpreted as a stack of `H * W` vectors each of length `C`, arranged in a 2D plane of height `H` and width `W`. The goal is to multiply a transformation matrix of size `(M, C)` with each of these vectors, resulting in a new stack of vectors each of length `M`. The resulting tensor will then have a shape of `(H, W, M)`. We need to leverage either NumPy's or TensorFlow's ability to perform matrix multiplication over the last two dimensions while treating other dimensions as batch dimensions to achieve this.

**1. NumPy Implementation**

NumPy, while it does not have an explicit batch matrix multiply function in the same way as TensorFlow, employs broadcasting and its `matmul` (`@` operator or `np.dot`) function. Broadcasting allows NumPy to stretch smaller arrays during arithmetic operations, provided certain dimension compatibility rules are met. For this scenario, we need to ensure that the transformation matrix has appropriate shape relative to the last dimension of the 3D tensor. The key is to use NumPy's ability to perform matrix multiplication along the appropriate axes. Specifically, we need the matrix to be on the left and the vectors on the right. To facilitate this, we must swap the order of the last two dimensions of the tensor, then perform the matrix multiplication, and finally, swap back.

```python
import numpy as np

def numpy_multiply_depth_vectors(tensor_3d, matrix_2d):
  """Multiplies each depth-wise vector of a 3D tensor by a 2D matrix using NumPy.

  Args:
    tensor_3d: A NumPy array of shape (H, W, C).
    matrix_2d: A NumPy array of shape (M, C).

  Returns:
    A NumPy array of shape (H, W, M) representing the result of the multiplication.
  """
  H, W, C = tensor_3d.shape
  M, _ = matrix_2d.shape # C is assumed to match with last dim of tensor_3d

  # Transpose the tensor to (H, W, C) -> (H, C, W) for correct multiplication axis
  transposed_tensor = np.transpose(tensor_3d, (0, 2, 1))

  # Perform matrix multiplication with the transposed tensor. Note the matrix first, then the tensor.
  multiplied_transposed = matrix_2d @ transposed_tensor

  # Transpose the result back to (H, W, M)
  result = np.transpose(multiplied_transposed, (0, 2, 1))

  return result

# Example Usage:
tensor = np.random.rand(3, 4, 5)  # (H=3, W=4, C=5)
matrix = np.random.rand(6, 5)      # (M=6, C=5)
result_numpy = numpy_multiply_depth_vectors(tensor, matrix)
print("NumPy Result Shape:", result_numpy.shape) # Output: (3, 4, 6)
```

In this code, the `transpose` function is used before and after matrix multiplication. First the tensor is transposed to `(H, C, W)`, effectively making each vector of length `C` available as the last dimension. The `matmul` operator, represented here by `@`, then applies the matrix multiplication to the vectors. The result of the matmul is then transposed back to `(H, W, M)` as intended. This implementation handles the batch-wise matrix multiplication in a clear, efficient way.

**2. TensorFlow Implementation**

TensorFlow provides a more direct method for batch matrix multiplication using `tf.matmul`. This method automatically handles batching, making it simpler and, generally, more efficient for larger tensors, especially when leveraging hardware acceleration. The key difference with TensorFlow is that we can directly use the 3D tensor with a matrix with the understanding that tf.matmul will apply the multiplication over the last two dimensions, keeping the initial dimensions as batch dimensions.

```python
import tensorflow as tf

def tensorflow_multiply_depth_vectors(tensor_3d, matrix_2d):
  """Multiplies each depth-wise vector of a 3D tensor by a 2D matrix using TensorFlow.

  Args:
    tensor_3d: A TensorFlow tensor of shape (H, W, C).
    matrix_2d: A TensorFlow tensor of shape (M, C).

  Returns:
    A TensorFlow tensor of shape (H, W, M) representing the result of the multiplication.
  """
  H, W, C = tensor_3d.shape.as_list()
  M, _ = matrix_2d.shape.as_list()

  # Ensure correct tensor type
  tensor_3d = tf.cast(tensor_3d, tf.float32)
  matrix_2d = tf.cast(matrix_2d, tf.float32)

  # Perform batch matrix multiplication using tf.matmul
  result = tf.matmul(tensor_3d, tf.transpose(matrix_2d)) # transpose is necessary here

  return result

# Example Usage:
tensor_tf = tf.random.normal((3, 4, 5))  # (H=3, W=4, C=5)
matrix_tf = tf.random.normal((6, 5))      # (M=6, C=5)
result_tensorflow = tensorflow_multiply_depth_vectors(tensor_tf, matrix_tf)
print("TensorFlow Result Shape:", result_tensorflow.shape) # Output: (3, 4, 6)
```

Here, the `tf.matmul` function applies the matrix multiplication across all depth-wise vectors defined in the 3D tensor, treating `H` and `W` as the batch dimensions. Importantly, `tf.matmul` requires that the matrices be arranged such that the inner dimensions match properly for matrix multiplication to work. If matrix_2d is `(M, C)`, we need to `tf.transpose` it to become `(C, M)` so that the multiplication works as `(..., C) @ (C, M)`. The output will therefore be `(..., M)`. Note that even if `tensor_3d` is `(H, W, C)`, the batch dimensions `(H, W)` will not change due to batch matrix multiplication. This approach avoids explicit transposition of the 3D tensor itself, making it more direct than the NumPy implementation when the matrix needs to be transposed rather than the tensor, and potentially more optimized when leveraging TensorFlow specific hardware acceleration.

**3. NumPy using einsum**

NumPy also provides the `einsum` function, which is a powerful tool for performing operations across multiple axes in a more flexible way. `einsum` specifies the exact dimensions to sum over, which, for matrix multiplication, allows us to describe the desired matrix multiplication.

```python
import numpy as np

def numpy_multiply_depth_vectors_einsum(tensor_3d, matrix_2d):
  """Multiplies each depth-wise vector of a 3D tensor by a 2D matrix using NumPy einsum.

  Args:
    tensor_3d: A NumPy array of shape (H, W, C).
    matrix_2d: A NumPy array of shape (M, C).

  Returns:
    A NumPy array of shape (H, W, M) representing the result of the multiplication.
  """
  # Use einsum to specify the matrix multiplication axis explicitly
  result = np.einsum('hwc,mc->hwm', tensor_3d, matrix_2d)
  return result

# Example Usage:
tensor_einsum = np.random.rand(3, 4, 5)  # (H=3, W=4, C=5)
matrix_einsum = np.random.rand(6, 5)      # (M=6, C=5)
result_numpy_einsum = numpy_multiply_depth_vectors_einsum(tensor_einsum, matrix_einsum)
print("NumPy einsum Result Shape:", result_numpy_einsum.shape) # Output: (3, 4, 6)
```

In this implementation, `np.einsum('hwc,mc->hwm', tensor_3d, matrix_2d)` states that `tensor_3d` has dimensions `h, w, c` and `matrix_2d` has dimensions `m, c`. We want to contract (multiply along and sum over) the `c` dimension, and keep the `h, w, m` dimensions. The result is then of shape `(H, W, M)`. This solution is arguably the most concise as it abstracts the transposition logic. While it may be slightly less performant than the explicit transposition approach for smaller matrices, its clarity and generality make it very useful.

**Resource Recommendations:**

For further learning, I would suggest consulting these resources. For a deep dive on NumPy operations, I recommend reviewing the official NumPy documentation, specifically the sections regarding array broadcasting and linear algebra operations. The TensorFlow documentation is equally critical for understanding `tf.matmul` and its application to batch processing. Finally, exploring tutorials covering einsum syntax and applications, whether for NumPy or a similar library, will prove invaluable for efficient array manipulation, as it has utility beyond this one specific problem.
