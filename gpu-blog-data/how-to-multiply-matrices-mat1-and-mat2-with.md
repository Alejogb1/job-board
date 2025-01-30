---
title: "How to multiply matrices mat1 and mat2 with incompatible shapes in PyTorch?"
date: "2025-01-30"
id: "how-to-multiply-matrices-mat1-and-mat2-with"
---
Direct multiplication of tensors representing matrices in PyTorch requires strict adherence to shape compatibility, specifically that the number of columns in the first matrix must equal the number of rows in the second matrix. When these conditions are violated, standard matrix multiplication, achieved using the `@` operator or `torch.matmul`, will raise a runtime error. However, situations often arise where we need to perform operations that *resemble* multiplication despite incompatible shapes, requiring alternative strategies that manipulate tensor dimensions or use broadcasting. Over my years working on deep learning projects, I’ve often encountered these scenarios, prompting a need to leverage PyTorch's flexible tensor manipulation capabilities.

The core issue isn't that PyTorch prevents operations on incompatible shapes absolutely, but rather that standard matrix multiplication’s mathematical definition does not extend to arbitrary pairings. Therefore, solutions typically revolve around either reshaping tensors to allow valid multiplication or employing broadcasting techniques that effectively replicate smaller tensors across dimensions of a larger one, enabling element-wise operations. In instances where true matrix multiplication's mathematical underpinnings cannot be met with shape transformation, we instead perform operation resembling it, but achieving fundamentally different results. Understanding the intended outcome of the "multiplication," and the underlying data semantics are paramount when making a decision.

Here are three common scenarios, each paired with a practical PyTorch solution and associated code:

**Scenario 1: Element-Wise Multiplication with Broadcasting**

Frequently, the desired operation is element-wise multiplication, akin to Hadamard product, where each element in one matrix is multiplied by a corresponding element in another. If tensors have compatible shapes, this is simply achieved via `torch.mul` or the `*` operator. However, broadcasting allows this element-wise operation even when shapes don't perfectly align, so long as certain dimensionality rules are met.

For example, assume I had a 2x3 matrix representing, say, network activations from the first layer, and I had to scale it by a learned scale factor for each output feature at the next layer, represented as a 1x3 vector. While a naive matrix multiplication is impossible, broadcasting allows me to implicitly align the single row of the scaling vector across all rows of the activation matrix.

```python
import torch

# Matrix 1: 2x3 shape
mat1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Matrix 2: 1x3 shape
mat2 = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

# Broadcasting occurs, resulting in a 2x3 matrix with scaled activations
result = mat1 * mat2

print("Matrix 1:", mat1)
print("Matrix 2:", mat2)
print("Result of broadcasting-based element-wise multiplication:\n", result)
```

*Code Commentary:*

The `torch.tensor` calls initialize two matrices with varying shapes. I’ve explicitly defined `dtype=torch.float32` for numerical precision. The `*` operator initiates element-wise multiplication. Because `mat2` is a 1x3 matrix, PyTorch automatically "broadcasts" it across rows of `mat1`, essentially duplicating its single row, enabling multiplication between the corresponding elements. The output result is a 2x3 matrix where every element of `mat1` has been scaled by the matching column element of `mat2`. This operation is distinctly different from matrix multiplication, serving to effectively apply a column-wise weighting.

**Scenario 2: Outer Product using Reshaping**

Another frequent requirement involves calculating the outer product of two vectors which, while not strictly matrix multiplication in the classic sense, uses related tensor multiplication concepts. The outer product generates a matrix where every combination of elements between the vectors is multiplied. In my work on multi-modal data, I have regularly used outer products in feature combination calculations.

Imagine we have two vectors, where one could represent the embedding for an image and the other the embedding for an text. To obtain an affinity map for the cross-product of those representations I would use an outer product.  Directly attempting to use `@` or `torch.matmul` will fail. However, we can explicitly reshape the vectors into a row and column matrix, respectively, permitting matrix multiplication.

```python
import torch

# Vector 1: 3 elements
vec1 = torch.tensor([1, 2, 3], dtype=torch.float32)

# Vector 2: 2 elements
vec2 = torch.tensor([4, 5], dtype=torch.float32)

# Reshape to column and row respectively
mat1_reshaped = vec1.view(-1, 1) # creates a 3x1 column vector
mat2_reshaped = vec2.view(1, -1) # creates a 1x2 row vector

# Perform matrix multiplication to obtain the outer product
outer_product = torch.matmul(mat1_reshaped, mat2_reshaped)

print("Vector 1:", vec1)
print("Vector 2:", vec2)
print("Outer product matrix:\n", outer_product)
```

*Code Commentary:*

I initialized `vec1` and `vec2` as vectors of different sizes. The critical part is `vec1.view(-1, 1)` and `vec2.view(1, -1)`. The `view` method is used to reshape the vectors, `-1` inference works out the correct shape parameter using the other given parameter. `vec1` becomes a column matrix (3x1), while `vec2` becomes a row matrix (1x2).  Now, these are compatible for standard matrix multiplication.  `torch.matmul` then produces a 3x2 outer product matrix, where each element represents the product of each combination of elements from `vec1` and `vec2`.

**Scenario 3: Batched Matrix Multiplication with Broadcasting**

In many deep learning applications, operations are performed on batches of data, which can introduce a third dimension to tensors. Say I had a collection of images to transform with a single rotation matrix. The images would have a batch size followed by width and height, where as the rotation matrix would have no batch dimension. This would usually lead to some sort of for loop. PyTorch allows the implicit batched matrix multiplication using broadcasting rules.

For example, we might have a batch of vectors (say, embeddings of shape 2x5) that need to be multiplied by a fixed matrix (5x3), to project to a new space. The most naive approach would loop across the batch dimension of the batch of vectors, multiplying one at a time, However PyTorch's multiplication allows for broadcasting of the fixed matrix to a batch level.

```python
import torch

# Batch of vectors: shape 2x5
batch_vecs = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float32)

# Projection matrix: shape 5x3
projection_matrix = torch.tensor([[0.1, 0.2, 0.3],
                                 [0.4, 0.5, 0.6],
                                 [0.7, 0.8, 0.9],
                                 [1.0, 1.1, 1.2],
                                 [1.3, 1.4, 1.5]], dtype=torch.float32)

# Batch matrix multiplication, broadcasting the projection matrix across the batch
result = torch.matmul(batch_vecs, projection_matrix)

print("Batch of vectors:\n", batch_vecs)
print("Projection matrix:\n", projection_matrix)
print("Result of batch matrix multiplication:\n", result)
```

*Code Commentary:*

`batch_vecs` represents a batch of two 1x5 vectors. `projection_matrix` is a 5x3 matrix.  `torch.matmul`, when used on a batched vector and 2D matrix, performs batched matrix multiplication implicitly, repeating the `projection_matrix` for each vector in `batch_vecs`, resulting in a batch of 2x3 vectors. This efficient calculation avoids manual looping, which would have been required to apply the `projection_matrix` on each vector. It’s essential to note this is still mathematically a series of matrix multiplications and is different from element-wise multiplication and reshaping.

In conclusion, handling "incompatible shapes" in PyTorch isn't about bypassing mathematical constraints, but rather about leveraging PyTorch's powerful tools to achieve desired operations that go beyond classic matrix multiplication definitions. Whether by employing element-wise broadcasting, intelligent reshaping, or batched calculations, proper tensor manipulation and semantics understanding is crucial.

For further exploration, the PyTorch documentation itself is an invaluable resource (specifically the sections on tensor operations and broadcasting semantics). "Deep Learning with PyTorch" by Eli Stevens et al. provides a good introduction to using PyTorch, covering fundamental tensor operations in considerable detail. Also, consulting the linear algebra resources such as "Linear Algebra and Its Applications" by Gilbert Strang, can lead to a deeper appreciation of matrix multiplication and its alternatives. These will build a stronger understanding of the mathematical foundations that PyTorch leverages, allowing for more creative and efficient problem-solving when dealing with tensors and shape incompatibilities.
