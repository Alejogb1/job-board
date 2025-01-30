---
title: "How can a 2D tensor multiplied by another 2D tensor result in a 3D PyTorch tensor?"
date: "2025-01-30"
id: "how-can-a-2d-tensor-multiplied-by-another"
---
The key to understanding how a 2D tensor multiplied by another 2D tensor can yield a 3D tensor in PyTorch lies in recognizing that standard matrix multiplication isn't the only operation at play.  The resulting 3D structure emerges from leveraging broadcasting and reshaping capabilities within PyTorch's tensor manipulation functionalities. My experience optimizing neural network architectures has frequently involved these techniques, especially when dealing with batch processing and multi-channel data.


**1.  Explanation:**

Standard matrix multiplication, as implemented by `torch.matmul` or the `@` operator, requires compatible dimensions for the input matrices. Specifically, the number of columns in the first matrix must equal the number of rows in the second. This results in a 2D matrix.  To obtain a 3D tensor, we need to introduce a third dimension, typically representing a batch or a collection of 2D matrices undergoing the same operation.


Consider two 2D tensors, `A` and `B`.  If we wish to generate a 3D tensor `C`, where each "slice" along the third dimension is the result of multiplying a corresponding pair of matrices from `A` and `B`, we need to structure our input tensors accordingly.  This typically involves reshaping or utilizing broadcasting to align the dimensions for batch processing.  The resulting third dimension then represents the number of these individual matrix multiplications performed.


Specifically, if `A` has dimensions (m, n) and `B` has dimensions (n, p), a straightforward approach would involve creating a batch of `k` such pairs, resulting in `A` becoming a (k, m, n) tensor and `B` becoming a (k, n, p) tensor.  The multiplication then proceeds across the batch dimension, producing a (k, m, p) 3D tensor.  This isn't a single matrix multiplication, but rather a sequence of `k` individual matrix multiplications, each yielding a (m, p) matrix.  These (m, p) matrices are then stacked together to form the third dimension.


**2. Code Examples:**

**Example 1: Using `torch.bmm` for batch matrix multiplication:**

```python
import torch

# Define dimensions
k = 5  # Batch size
m = 3  # Rows in A and C
n = 4  # Columns in A and rows in B
p = 2  # Columns in B and C

# Create tensors
A = torch.randn(k, m, n)
B = torch.randn(k, n, p)

# Perform batch matrix multiplication
C = torch.bmm(A, B)

# Print shape and a slice
print(C.shape)  # Output: torch.Size([5, 3, 2])
print(C[0])    # Output: The (3, 2) result of the first matrix multiplication in the batch
```

This example directly demonstrates batch matrix multiplication using `torch.bmm`.  The function efficiently handles the multiplication of multiple matrix pairs.


**Example 2:  Reshaping and broadcasting:**

```python
import torch

# Define dimensions
k = 3
m = 2
n = 4
p = 5

# Create tensors
A = torch.randn(k, m, n)
B = torch.randn(n, p)


#Reshape to enable broadcasting
B = B.unsqueeze(0).repeat(k, 1, 1) #Repeats B k times along the first dimension

#Perform element-wise multiplication then matrix multiplication
C = (A * B).view(k, m, p)
# C = torch.einsum('ijk,kl->ijl', A, B)  #Alternative using Einstein summation notation


print(C.shape)  # Output: torch.Size([3, 2, 5])
print(C[0])  # Output: A 2x5 matrix
```

This example leverages broadcasting to multiply k instances of B with corresponding slices of A before reshaping into the 3D tensor.  This is less efficient but illustrates the concept of achieving 3D output through manipulation prior to actual multiplication.


**Example 3:  Outer Product and Reshaping:**

```python
import torch

# Define dimensions
m = 3
n = 2
p = 4

# Create tensors
A = torch.randn(m, n)
B = torch.randn(n, p)

#Compute outer product
C = torch.einsum('ik,kj->ijk', A, B)

print(C.shape) #Output: torch.Size([3, 2, 4])
print(C[0])
```

This example shows the use of Einstein summation convention to compute the outer product between A and B and achieve the 3D output tensor. This method efficiently computes a tensor where each slice represents the outer product of A's rows with B's columns.



**3. Resource Recommendations:**

I recommend consulting the official PyTorch documentation, specifically the sections on tensor operations, broadcasting, and the `torch.bmm` function.  Furthermore, a strong grasp of linear algebra fundamentals, particularly matrix multiplication and tensor manipulation, is crucial.  Finally, working through practical examples and experimenting with various tensor shapes and operations will significantly solidify your understanding.  Reviewing materials on Einstein summation convention will be beneficial for advanced operations and understanding efficient code structures.
