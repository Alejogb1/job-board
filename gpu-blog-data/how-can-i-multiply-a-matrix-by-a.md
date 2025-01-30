---
title: "How can I multiply a matrix by a vector in PyTorch?"
date: "2025-01-30"
id: "how-can-i-multiply-a-matrix-by-a"
---
Matrix-vector multiplication is a fundamental operation in linear algebra and a common task in deep learning, often serving as the core of neural network forward passes. Within PyTorch, this operation is efficiently handled using the `@` operator or the `torch.matmul()` function, both performing matrix multiplication but with nuanced behavior regarding broadcasting. My experience developing a custom convolutional neural network for image segmentation revealed these nuances and their implications for performance and correctness. This discussion will focus on how to correctly perform matrix-vector multiplication in PyTorch, highlighting the importance of tensor dimensions and avoiding common pitfalls.

The core concept centers on the appropriate dimensions of the tensors involved. A matrix, denoted as *A*, will typically have dimensions *m x n*, while a vector, denoted as *v*, will have a dimension *n x 1* when treated as a column vector, or *1 x n* when treated as a row vector. To perform a valid matrix-vector product *Av*, the number of columns in matrix *A* must match the number of rows in vector *v*. The result will be a new vector with dimensions *m x 1*, assuming *v* is a column vector. PyTorch, however, automatically handles the case where the vector is supplied as a one-dimensional tensor, effectively treating it as a column vector when performing matrix multiplication.

**Example 1: Standard Matrix-Vector Multiplication**

The most direct scenario involves multiplying a 2D matrix by a 1D vector. In this case, PyTorch automatically infers the vector's appropriate dimensionality for the operation.

```python
import torch

# Define a matrix (2x3)
matrix = torch.tensor([[1.0, 2.0, 3.0], 
                       [4.0, 5.0, 6.0]])

# Define a vector (1x3) effectively treated as (3x1)
vector = torch.tensor([7.0, 8.0, 9.0])

# Perform matrix-vector multiplication using the @ operator
result = matrix @ vector

print("Matrix:", matrix)
print("Vector:", vector)
print("Result (matrix @ vector):", result)


# Perform matrix-vector multiplication using torch.matmul()
result_matmul = torch.matmul(matrix, vector)

print("Result (torch.matmul(matrix, vector)):", result_matmul)
```

Here, `matrix` is a 2x3 tensor, and `vector` is a 1D tensor of length 3. Because the number of columns in the matrix matches the number of elements in the vector, the multiplication is valid. The `@` operator and `torch.matmul()` function both return the expected result: a vector of dimension 2. `torch.matmul()` provides the same functionality as `@` in this basic scenario.

**Example 2: Handling Batches of Vectors**

A more common use case, especially in deep learning, involves performing matrix-vector multiplication on batches of vectors. These are frequently seen when processing multiple data points simultaneously through a network layer. In this case, the vectors are often arranged as rows within a 2D tensor. When the vector tensor has a batch dimension, its shape often becomes `batch_size x vector_size`. With this tensor shape, `torch.matmul` or the `@` operator automatically handle the matrix-vector multiplication as it iterates through the batch.

```python
import torch

# Define a matrix (4x5)
matrix = torch.randn(4, 5)

# Define a batch of vectors (3x5)
batch_vectors = torch.randn(3, 5)

# Transpose the matrix to enable element-wise multiplication
matrix = matrix.T

# Perform matrix-batch vector multiplication using the @ operator
result_batch = batch_vectors @ matrix


print("Matrix (Transposed):", matrix)
print("Batch Vectors:", batch_vectors)
print("Result (batch_vectors @ matrix):", result_batch)

# Perform matrix-batch vector multiplication using torch.matmul()
result_batch_matmul = torch.matmul(batch_vectors, matrix)

print("Result (torch.matmul(batch_vectors, matrix)):", result_batch_matmul)
```
In this case, `matrix` is a 4x5 tensor and we transpose it to make it a 5x4 matrix. `batch_vectors` is a 3x5 tensor, representing three vectors each of length 5. The multiplication produces a tensor of shape 3x4, where each of the three vectors is multiplied against the matrix. This demonstrates automatic broadcasting; the same matrix is applied to each vector in the batch. It's crucial to note that `torch.matmul` and `@` behave identically for these batched operations.

**Example 3: Broadcasting and Dimension Mismatches**

It’s essential to handle dimension mismatches correctly. If the dimensions are not compatible, PyTorch will generate an error, or when broadcasting, lead to an unexpected result if not handled well.  While PyTorch uses implicit broadcasting in matrix multiplication, explicit re-shaping can improve code clarity.

```python
import torch

# Define a matrix (2x3)
matrix = torch.tensor([[1.0, 2.0, 3.0], 
                       [4.0, 5.0, 6.0]])

# Define a vector of the wrong size
vector_mismatched = torch.tensor([7.0, 8.0])

try:
    # Attempt matrix-vector multiplication with mismatched dimensions
    result_mismatch = matrix @ vector_mismatched
except RuntimeError as e:
    print("Error:", e)

# Define a vector of the correct shape
vector_correct = torch.tensor([7.0, 8.0, 9.0])

# Manually reshape the vector into a column vector for clarity
vector_correct_reshaped = vector_correct.reshape(-1, 1)

# Perform matrix-vector multiplication with a reshaped vector
result_reshaped = matrix @ vector_correct_reshaped.squeeze()

print("Matrix:", matrix)
print("Vector (Correct Shape):", vector_correct)
print("Result (matrix @ vector_correct_reshaped):", result_reshaped)

```
In this scenario, `vector_mismatched` has an incompatible dimension for matrix multiplication. The error message highlights the dimension mismatch.  In contrast, `vector_correct` has a compatible shape for use with `matrix` even if the operation automatically treats it as a column vector. However, reshaping the vector explicitly to a column matrix is often beneficial for maintaining code readability, and clarity for other developers.  The `.squeeze()` method removes the extra dimension after the multiplication is performed, leaving just the resultant vector.

**Resource Recommendations**

To further deepen your understanding of tensor operations and matrix manipulations in PyTorch, I recommend consulting the official PyTorch documentation, specifically the sections related to `torch.matmul`, tensor broadcasting rules, and tensor reshaping. Additionally, introductory linear algebra texts, especially those focusing on matrix multiplication, provide the fundamental mathematics necessary to understand these operations.  Online courses in deep learning or machine learning often include in-depth tutorials on performing matrix-vector operations.  Lastly, practical experience gained from working through various examples, even small ones, will solidify the understanding of tensor shapes and the nuances of PyTorch's tensor algebra.
