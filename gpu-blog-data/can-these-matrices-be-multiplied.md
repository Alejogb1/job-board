---
title: "Can these matrices be multiplied?"
date: "2025-01-30"
id: "can-these-matrices-be-multiplied"
---
The core requirement for matrix multiplication lies in the compatibility of their dimensions. Specifically, the number of columns in the first matrix must equal the number of rows in the second matrix. This rule, often overlooked, dictates whether a product even exists and, consequently, the resulting matrix's dimensions. Having spent years implementing linear algebra routines in embedded systems, I've seen firsthand how strict adherence to this principle is paramount to avoid runtime errors and produce meaningful results. Let's explore this concept in detail.

**1. Dimensionality Compatibility Explained**

Consider two matrices, A and B. Matrix A has dimensions m x n, where m represents the number of rows and n represents the number of columns. Similarly, matrix B has dimensions p x q, where p represents rows and q represents columns. For the matrix product AB to be defined, n *must* equal p. The resulting matrix, denoted as C, will then have dimensions m x q.

It is crucial to understand that matrix multiplication is generally *not* commutative. This means that in most cases, AB does not equal BA. In fact, BA may not even be defined if the dimensions of the matrices don't allow it.

The mechanics of matrix multiplication involve taking the dot product of each row of matrix A with each column of matrix B. The resulting dot product then becomes the element of matrix C at the corresponding row and column index. This operation is not possible if the dimensions do not line up.

Furthermore, the order of multiplication is critically important. If we attempt to perform BA when the dimensions do not permit it, a program will typically throw an error or produce undefined results. The rule, therefore, is not simply a suggestion, but a fundamental requirement in linear algebra.

**2. Code Examples with Commentary**

Let’s analyze a few scenarios through code examples, employing Python with the NumPy library. This library is widely used for its numerical computation capabilities, and particularly its efficient implementation of matrix operations.

**Example 1: Compatible Matrices**

```python
import numpy as np

# Matrix A: 2x3
A = np.array([[1, 2, 3], [4, 5, 6]])

# Matrix B: 3x2
B = np.array([[7, 8], [9, 10], [11, 12]])

# Matrix multiplication
C = np.dot(A, B)

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Resulting Matrix C:\n", C)

# Output (C will be a 2x2 matrix)
# Matrix A:
#  [[1 2 3]
#  [4 5 6]]
# Matrix B:
#  [[ 7  8]
#  [ 9 10]
#  [11 12]]
# Resulting Matrix C:
#  [[ 58  64]
#  [139 154]]

```

In this example, matrix A is 2x3, and matrix B is 3x2. The inner dimensions (3 and 3) match, allowing for multiplication. The resulting matrix C is 2x2 as predicted. The code uses `np.dot` to perform matrix multiplication, a more performant operation compared to manual implementation via nested loops, which I've often used when optimizing for resource-constrained hardware.

**Example 2: Incompatible Matrices**

```python
import numpy as np

# Matrix A: 2x3
A = np.array([[1, 2, 3], [4, 5, 6]])

# Matrix B: 2x2
B = np.array([[7, 8], [9, 10]])

try:
    # Attempt matrix multiplication (will raise an error)
    C = np.dot(A, B)
    print("Resulting Matrix C:\n", C) #This will not print
except ValueError as e:
    print(f"Error: {e}")

#Output:
#Error: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)

```

Here, matrix A is 2x3, and matrix B is 2x2. The inner dimensions (3 and 2) do not match. Consequently, NumPy's `np.dot` function raises a `ValueError`, indicating that the matrix shapes are not aligned for multiplication. This illustrates the importance of pre-multiplication dimension checks, something I've built into many of my embedded systems libraries.

**Example 3: Transpose to Enable Multiplication**

```python
import numpy as np

# Matrix A: 2x3
A = np.array([[1, 2, 3], [4, 5, 6]])

# Matrix B: 2x3
B = np.array([[7, 8, 9], [10, 11, 12]])

# Transpose Matrix B (to be 3x2)
B_transposed = np.transpose(B)

# Matrix multiplication
C = np.dot(A, B_transposed)

print("Matrix A:\n", A)
print("Transposed Matrix B:\n", B_transposed)
print("Resulting Matrix C:\n", C)

#Output:
#Matrix A:
# [[1 2 3]
# [4 5 6]]
#Transposed Matrix B:
# [[ 7 10]
# [ 8 11]
# [ 9 12]]
#Resulting Matrix C:
# [[ 52  58]
# [121 136]]

```

In this example, A is 2x3 and B is also 2x3. Direct multiplication A * B is not possible due to dimension mismatch. However, by transposing B to B_transposed (3x2), we create compatible dimensions for matrix multiplication. The inner dimensions of A (2x3) and B_transposed (3x2) now align. The resulting matrix C is 2x2 as expected. The transposition operation, a key tool I often employ in signal processing, demonstrates a way to circumvent dimension mismatch problems, enabling the use of multiplication where it initially would have failed.

**3. Resource Recommendations**

For those seeking a more rigorous understanding of linear algebra, I suggest exploring resources beyond simple online explanations. While online tutorials provide immediate results, solid foundation in textbooks is often needed for advanced applications.

*   **"Linear Algebra and Its Applications" by Gilbert Strang:** This textbook provides a thorough introduction to the theory and application of linear algebra. It is suitable for both undergraduate and graduate studies and is written in a clear, understandable style. I often find myself returning to this source when I encounter challenging issues in matrix operations.
*   **"Matrix Computations" by Gene H. Golub and Charles F. Van Loan:** A more advanced resource, this book covers numerical methods for matrix computations. It delves into details about algorithms for solving linear systems, eigenvalue problems, and singular value decomposition. This is a key reference for developing numerical stability in algorithms.
*   **University Course Materials:** Many universities offer publicly accessible course materials on linear algebra. Exploring these resources, often from institutions like MIT, Stanford, or Berkeley, can offer diverse perspectives and exercises. I have personally used MIT OpenCourseWare frequently for reference on numerical analysis concepts.

In conclusion, matrix multiplication hinges on the compatibility of matrix dimensions. Adhering to the rule that the number of columns in the first matrix must equal the number of rows in the second matrix is not optional but essential. The examples and resources highlight the practical implications and provide paths for deeper exploration. My experience has shown that a robust understanding of this concept is paramount for developing reliable and performant algorithms involving matrix operations, whether in embedded systems or higher-level software applications.
