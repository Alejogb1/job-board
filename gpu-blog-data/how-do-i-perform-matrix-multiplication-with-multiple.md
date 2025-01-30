---
title: "How do I perform matrix multiplication with multiple dimensions in TensorFlow?"
date: "2025-01-30"
id: "how-do-i-perform-matrix-multiplication-with-multiple"
---
Matrix multiplication in TensorFlow, particularly when dealing with tensors of more than two dimensions, requires a nuanced understanding of the `tf.matmul` operation and its interaction with broadcasting rules. I’ve personally encountered the complexities of multi-dimensional matrix multiplication numerous times, especially when constructing intricate neural network architectures or processing batches of time-series data. The core challenge stems from the fact that `tf.matmul` is inherently designed for 2D matrices, necessitating careful reshaping or the utilization of batch matrix multiplication when higher-dimensional tensors are involved. The key insight is that the operation fundamentally performs multiplication over the innermost two dimensions of the input tensors, treating any leading dimensions as batch dimensions. This treatment allows for efficient processing of multiple matrix multiplications simultaneously.

To elaborate, let's consider the conceptual model. When you have two 2D matrices, say *A* of shape (m, n) and *B* of shape (n, p), `tf.matmul(A, B)` produces a matrix *C* of shape (m, p). The standard row-column multiplication is executed. Now, if *A* has the shape (b, m, n) and *B* has the shape (b, n, p), `tf.matmul(A, B)` results in *C* with a shape of (b, m, p). Effectively, the multiplication is performed element-wise across the first dimension, 'b'. This represents the simplest case of batch multiplication. The 'b' dimension is implicitly broadcasted. The critical thing to remember here is that for a successful multiplication, the second last dimensions of both input tensors must match (n in the previous example) and the output tensor maintains dimensions common to both inputs but with the dimensions of the result of matrix multiplication of the innermost two dimensions. If both dimensions are not compatible, `tf.matmul` throws an error.

Extending beyond simple batch scenarios, when the dimensions of the input tensors are not directly compatible, the broadcasting rules of TensorFlow come into play. Tensor dimensions are considered compatible according to the following rules: 1) dimensions are equal, 2) one of the dimensions is one. If two tensors are compatible, then the leading dimensions can be broadcasted such that the input tensors have same dimensions by appending leading dimension of size 1 to tensor with fewer dimensions, and then stretching (or copying) tensors with dimension of size 1 to that of another. The actual computation of multiplication takes place in the last two dimensions of the broadcasted tensors, with all leading dimensions simply acting as batch dimensions.

Let's examine some practical code examples.

**Example 1: Standard Batch Matrix Multiplication**

In this case, we deal with a stack of matrices, each representing a batch element. We will perform a standard matrix multiplication for each batch element simultaneously.

```python
import tensorflow as tf

# Define input tensors
A = tf.constant([[[1, 2],
                 [3, 4]],
                [[5, 6],
                 [7, 8]]], dtype=tf.float32) # Shape (2, 2, 2) - batch of 2 matrices

B = tf.constant([[[9, 10],
                  [11, 12]],
                 [[13, 14],
                  [15, 16]]], dtype=tf.float32) # Shape (2, 2, 2) - batch of 2 matrices

# Perform batch matrix multiplication
C = tf.matmul(A, B)

print("Shape of A:", A.shape)
print("Shape of B:", B.shape)
print("Shape of C:", C.shape) # Output shape will be (2, 2, 2)
print("Result:\n", C)
```

This first example demonstrates the core of batch matrix multiplication. Two tensors `A` and `B`, both with shape (2, 2, 2), are multiplied using `tf.matmul`. The result, `C`, has the shape (2, 2, 2). The operation essentially performs matrix multiplication on the last two dimensions, while preserving the batch dimension (dimension 0). Note that corresponding batch elements are multiplied together. A[0] is multiplied with B[0] and A[1] is multiplied with B[1]. The dimension of the last two elements are of same sizes for both A and B and hence the matrices are compatible for multiplication.

**Example 2: Broadcasting with Matrix and Vector**

This example demonstrates how broadcasting rules enable operations between matrices and vectors, or more generally, tensors with different but compatible number of dimensions.

```python
import tensorflow as tf

# Define input tensors
A = tf.constant([[[1, 2],
                 [3, 4]],
                [[5, 6],
                 [7, 8]]], dtype=tf.float32)  # Shape (2, 2, 2)

B = tf.constant([1, 2], dtype=tf.float32) # Shape (2)
B = tf.reshape(B,[1, 2, 1]) # Shape (1,2,1)

# Perform batch matrix multiplication
C = tf.matmul(A, B)

print("Shape of A:", A.shape)
print("Shape of B:", B.shape)
print("Shape of C:", C.shape)  # Output shape will be (2, 2, 1)
print("Result:\n", C)
```

Here, `A` is a 3D tensor with shape (2, 2, 2), while `B` is a 1D tensor with shape (2). To make the shapes compatible, we reshape `B` to (1, 2, 1) so that the multiplication of the last two dimensions are valid. Then broadcasting ensures that B is effectively treated as (2, 2, 1) for multiplication. This results in `C` having shape (2, 2, 1). The multiplication then acts on the innermost dimension of the A with the last dimension of B, resulting in A(2,2,2) X B(2,1) = C(2,2,1). This is a form of matrix-vector multiplication where each matrix element of the batch in `A` is multiplied by a single vector `B`.

**Example 3: Handling Arbitrary Dimensions**

This final example demonstrates that the same principle applies regardless of the number of dimensions, as long as the innermost two dimensions can be matrix-multiplied and the leading dimensions are compatible according to the broadcasting rules.

```python
import tensorflow as tf

# Define input tensors
A = tf.constant(tf.random.normal([2, 3, 4, 5]), dtype=tf.float32) # Shape (2, 3, 4, 5)
B = tf.constant(tf.random.normal([2, 3, 5, 6]), dtype=tf.float32) # Shape (2, 3, 5, 6)

# Perform matrix multiplication
C = tf.matmul(A, B)

print("Shape of A:", A.shape)
print("Shape of B:", B.shape)
print("Shape of C:", C.shape)  # Output shape will be (2, 3, 4, 6)

# Example with broadcasting in additional dimension
D = tf.constant(tf.random.normal([1, 3, 5, 6]), dtype=tf.float32)  #Shape(1,3,5,6)
E = tf.matmul(A,D)
print("Shape of D:", D.shape)
print("Shape of E:", E.shape) # Output shape (2,3,4,6)

#Example with batching in additional dimension.
F = tf.constant(tf.random.normal([4,2, 3, 4, 5]), dtype=tf.float32) # Shape (4,2,3,4,5)
G = tf.constant(tf.random.normal([4,2,3, 5, 6]), dtype=tf.float32) # Shape (4,2,3,5,6)

H = tf.matmul(F,G)
print("Shape of F:", F.shape)
print("Shape of G:", G.shape)
print("Shape of H:", H.shape) # Output Shape (4, 2, 3, 4, 6)
```

In this example, we have tensors `A` with shape (2, 3, 4, 5) and `B` with shape (2, 3, 5, 6). The `tf.matmul` operation produces a result `C` with shape (2, 3, 4, 6). The innermost matrix multiplication occurs between the last two dimensions of tensors of shapes (4,5) and (5,6) resulting in shape of (4,6). The rest of the dimensions act as batch dimensions and are passed over during multiplication. The second example is with broadcast in the first dimension of D. We can observe that D of shape (1,3,5,6) is broadcasted with A of shape(2,3,4,5) to give E of shape (2,3,4,6) since dimension 1 of D is treated as size 2 in operation with A. The final example is of batched matrix multiplication across additional dimension. The multiplication is done on matrices across the last two dimensions, with other dimensions acting as batch dimensions. The dimensions must either be equal or 1.

For further understanding, I recommend reviewing resources that explain tensor broadcasting rules in detail, paying close attention to examples that involve operations with varying numbers of dimensions. Examining implementations of transformer networks or complex convolutional architectures in TensorFlow can also provide practical insight into the way multi-dimensional matrix multiplications are used in real-world scenarios. Furthermore, exploring documentation related to tensor manipulation and advanced indexing can help you gain a strong handle on how the structure of tensors impacts the resulting shape after matrix multiplication. Consulting tutorials on linear algebra operations with TensorFlow is another valuable approach as they often go over the basic principles with practical examples.
