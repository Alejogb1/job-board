---
title: "What is np.einsum?"
date: "2025-01-30"
id: "what-is-npeinsum"
---
The core functionality of `np.einsum` lies in its ability to perform arbitrary tensor contractions in a highly efficient and expressive manner, significantly surpassing the readability and often the performance of explicitly written nested loops or even optimized library functions for specific contraction patterns.  My experience working on large-scale physics simulations consistently highlighted the limitations of manual tensor manipulation, leading me to rely heavily on `np.einsum` for its flexibility and speed.  This response details its mechanics, benefits, and usage with illustrative examples.

**1.  Clear Explanation:**

`np.einsum` (Einstein summation convention) operates based on a concise string notation specifying the indices involved in the tensor contraction.  This notation directly reflects the mathematical expression of the desired operation, eliminating the need for explicit looping and significantly improving code clarity. The input consists of one or more arrays and a subscription string.  This string defines how the input array indices are contracted (summed over) and which remain in the output.

The subscription string has a specific structure.  It is composed of comma-separated subsequences, where each subsequence corresponds to an input array.  Each character in a subsequence represents an index of the corresponding array.  Repeated indices signify summation over those dimensions.  Indices appearing only once in the entire subscription string represent the output array's dimensions.

For instance, consider the matrix product C = AB, where A is an m x n matrix and B is an n x p matrix.  The standard matrix multiplication can be expressed element-wise as:

C<sub>ik</sub> = Σ<sub>j</sub> A<sub>ij</sub>B<sub>jk</sub>

In `np.einsum`, this would be represented as:

`np.einsum('ij,jk->ik', A, B)`

Here, 'ij' corresponds to the indices of A, 'jk' to the indices of B, and 'ik' denotes the indices of the resulting matrix C.  The repeated index 'j' indicates summation over that dimension.  This simple example already demonstrates the conciseness and direct mapping to the mathematical representation.

Furthermore, `np.einsum` efficiently handles higher-dimensional tensors and more complex contraction patterns. Its strength lies in its ability to elegantly express operations beyond simple matrix multiplication, including tensor transpositions, diagonal extraction, inner and outer products, and many others.  The internal implementation leverages optimized BLAS and LAPACK routines whenever possible, resulting in considerable performance gains compared to manually implemented equivalents.


**2. Code Examples with Commentary:**

**Example 1: Matrix Multiplication:**

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.einsum('ij,jk->ik', A, B)  # Standard matrix multiplication
print(C)  # Output: [[19 22] [43 50]]

#Verification using standard numpy method:
C_standard = np.matmul(A,B)
print(C_standard) # Output: [[19 22] [43 50]]
```

This example directly demonstrates the matrix multiplication from the explanation above.  The clarity of the subscription string 'ij,jk->ik' immediately conveys the operation's intent.


**Example 2:  Trace of a Matrix:**

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

trace = np.einsum('ii->', A)  # Trace calculation (sum of diagonal elements)
print(trace)  # Output: 15
```

This showcases the power of `np.einsum` to succinctly express operations beyond basic matrix algebra.  The repeated index 'ii' specifies summation along the diagonal, and '->' indicates a scalar output (a single value representing the trace).  This is far more concise than an explicit loop to sum diagonal elements.


**Example 3:  Tensor Contraction:**

```python
import numpy as np

A = np.random.rand(2, 3, 4)
B = np.random.rand(4, 5)

C = np.einsum('ijk,kl->ijl', A, B)  # Contraction along the 4th dimension
print(C.shape) #Output: (2,3,5)

#Verification through explicit loops (significantly less efficient):

C_loop = np.zeros((2,3,5))
for i in range(2):
    for j in range(3):
        for l in range(5):
            for k in range(4):
                C_loop[i,j,l] += A[i,j,k] * B[k,l]
print(np.allclose(C, C_loop)) #Output: True (verifies correctness)
```

This example illustrates a more complex tensor contraction involving a three-dimensional array A and a two-dimensional array B.  The contraction occurs over the last dimension of A and the first dimension of B, resulting in a new three-dimensional array C.  The equivalent nested loops, included for comparison, are significantly less readable and considerably less efficient, especially for larger arrays.  The `np.allclose` function verifies the numerical equivalence of the two approaches.


**3. Resource Recommendations:**

The official NumPy documentation; a linear algebra textbook covering tensor operations and the Einstein summation convention; a comprehensive guide to numerical computation in Python.  Careful study of these resources will solidify your understanding of `np.einsum` and its wide array of applications.  Practicing with diverse examples will build familiarity and confidence in employing this powerful tool.  Working through the examples in the documentation and experimenting with progressively complex operations will reinforce the understanding.
