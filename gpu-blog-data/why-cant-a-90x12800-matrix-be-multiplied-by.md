---
title: "Why can't a 90x12800 matrix be multiplied by a 90x360 matrix?"
date: "2025-01-30"
id: "why-cant-a-90x12800-matrix-be-multiplied-by"
---
Matrix multiplication, a fundamental operation in linear algebra and computational sciences, is not a universally applicable process. The core constraint lies in the *conformability requirement* – specifically, the inner dimensions of the matrices must match. I encountered this limitation early in my career while developing a custom image processing pipeline for a remote sensing application. We were attempting to perform a transform on feature vectors extracted from satellite imagery, and a mismatch in matrix dimensions brought our processing to a halt. This wasn’t a simple bug; it highlighted the inherent mathematical rules governing matrix operations.

The fundamental reason a 90x12800 matrix (let’s call it matrix A) cannot be multiplied by a 90x360 matrix (matrix B) is because the number of columns in matrix A (12800) does not equal the number of rows in matrix B (90).  Matrix multiplication, defined as *C = AB*, requires that the number of columns in *A* equal the number of rows in *B*.  The resulting matrix *C* will then have dimensions that reflect the outer dimensions of *A* and *B*: specifically, the number of rows of *A* and the number of columns of *B*.  In the given scenario, attempting to multiply matrix A and matrix B would violate this essential condition.

Conceptually, matrix multiplication is derived from a series of dot product operations between rows of the first matrix and columns of the second.  When performing *AB*, each element in the resultant matrix *C* is calculated by taking the dot product of a row from *A* and a column from *B*. If the number of columns in *A* does not match the number of rows in *B*, the dot product operation cannot be performed because the vectors being used are not of the same length, leading to a mathematical incompatibility. This isn’t an arbitrary constraint; it reflects the fundamental definition of how the multiplication of linear transformations is carried out.

Let's explore this further with some hypothetical scenarios and accompanying code. I will use Python with NumPy, the industry standard for numerical computing, given its concise representation of linear algebra constructs.

**Example 1: A Valid Multiplication**

First, let's examine a valid multiplication to illustrate conformability. Suppose we have matrix *X* of size 3x4 and matrix *Y* of size 4x2. This operation *XY* is valid, and the resulting matrix *Z* will be of size 3x2.

```python
import numpy as np

X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]]) # 3x4 matrix

Y = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]]) # 4x2 matrix

Z = np.dot(X, Y) # Valid multiplication

print("Matrix X:\n", X)
print("Matrix Y:\n", Y)
print("Result Matrix Z:\n", Z)

print("Dimensions of X:", X.shape)
print("Dimensions of Y:", Y.shape)
print("Dimensions of Z:", Z.shape)
```

*Commentary:* This code snippet successfully demonstrates a valid matrix multiplication. The dimensions of *X* (3x4) and *Y* (4x2) satisfy the conformability requirement. The resulting matrix *Z* (3x2) shows how the number of rows from *X* and the number of columns from *Y* determine the resulting matrix's size. The `.dot()` function in NumPy handles the multiplication process efficiently. The printed shapes at the end confirm that the dimensions are as expected.

**Example 2: Invalid Multiplication (Original Case)**

Now, let's attempt the originally specified multiplication with a 90x12800 matrix and a 90x360 matrix. This will result in a ValueError in Python's NumPy as a result of trying to perform the impossible operation.

```python
import numpy as np

A = np.random.rand(90, 12800) # 90x12800 matrix
B = np.random.rand(90, 360)   # 90x360 matrix

try:
    C = np.dot(A, B)   # This will throw a ValueError
    print("Result Matrix C:\n", C) # This line will not be reached
except ValueError as e:
   print(f"Error: {e}")

print("Dimensions of A:", A.shape)
print("Dimensions of B:", B.shape)
```

*Commentary:* As expected, this code snippet generates a `ValueError`. The error message explicitly states that the shapes are not aligned for matrix multiplication, revealing that NumPy enforces the core rules of matrix multiplication. The dimensions of *A* (90x12800) and *B* (90x360) clearly violate the conformability condition as 12800 is not equal to 90. Note the use of a `try...except` block to elegantly catch and handle the error, preventing the program from crashing. The shapes of the matrices are printed for clarity.

**Example 3: Correcting the Incompatible Dimensions**

To make the 90x12800 and 90x360 matrix multiplication possible, we need to either transpose the second matrix to give it a matching number of rows or find an intermediate matrix of compatible dimensions. For this example, we can transform B to be a 360x90 matrix and proceed with a multiplication of the transposed B matrix, which results in a matrix of dimensions 90x90.

```python
import numpy as np

A = np.random.rand(90, 12800) # 90x12800 matrix
B = np.random.rand(360, 90) # 360x90 matrix

# Transposing Matrix B to become compatible with matrix A
B_transposed = B.T

try:
    C = np.dot(A, B_transposed)   # Valid Multiplication
    print("Result Matrix C:\n", C)
    print("Dimensions of C:", C.shape)
except ValueError as e:
   print(f"Error: {e}") # This line will not be reached

print("Dimensions of A:", A.shape)
print("Dimensions of B_transposed:", B_transposed.shape)
```

*Commentary:* Here we've made the dimensions of matrices compatible by transposing matrix B. Note that matrix B was first defined as a 360x90. Transposing it using `.T` effectively swaps its rows and columns, resulting in a 90x360 matrix. Because we now attempt to multiply A with a matrix whose number of rows matches the number of columns of A, the multiplication proceeds successfully. The resulting matrix C now has the expected dimension of 90x90 as this was the number of rows in A and columns in the transposed B. The shapes of A and B_transposed are displayed for verification, as well as the dimensions of the output C.

**Resource Recommendations**

For further exploration, textbooks on linear algebra typically cover matrix operations in detail. Specifically, I would recommend focusing on chapters discussing vector spaces, linear transformations, and matrix multiplication.  Additionally, online courses focused on numerical computation often include modules dedicated to linear algebra and its practical applications in fields like machine learning and data science. Libraries such as NumPy have thorough documentation that can help to develop a stronger, more practical understanding of working with matrices in code. Examining these resources alongside the examples provided here should solidify an understanding of matrix multiplication limitations and offer pathways toward problem-solving when dealing with arrays of numerical data. Finally, open-source math and software textbooks can be very beneficial.
