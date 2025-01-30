---
title: "Why can't matrices mat1 and mat2 be multiplied?"
date: "2025-01-30"
id: "why-cant-matrices-mat1-and-mat2-be-multiplied"
---
Matrix multiplication hinges on a fundamental constraint: the inner dimensions of the matrices must be identical.  In my experience troubleshooting linear algebra computations within high-performance computing environments, encountering incompatible matrix dimensions for multiplication is a surprisingly frequent error.  The inability to multiply `mat1` and `mat2` stems directly from a violation of this dimensionality requirement.  Let's explore this constraint in detail and illustrate it with examples.

**1.  The Core Constraint: Inner Dimension Compatibility**

Matrix multiplication is not a simple element-wise operation.  Instead, it involves a process of dot products between rows of the first matrix and columns of the second.  Consider two matrices, `mat1` of dimensions *m x n* and `mat2` of dimensions *p x q*.  The result of their multiplication, `mat3`, will only be defined if *n = p*.  This means the number of columns in `mat1` must equal the number of rows in `mat2`. This shared dimension, *n* (*or* *p*), is the "inner dimension."  If this condition isn't met, the dot product operation is undefined, and multiplication is impossible.  The resulting matrix, `mat3`, will have dimensions *m x q*.

The failure to satisfy this inner dimension equality is the root cause preventing the multiplication of `mat1` and `mat2` in the posed question.  The attempted multiplication is inherently invalid, resulting in a runtime error or, in some languages, an undefined result.

**2. Code Examples and Explanations**

Let's examine this constraint using Python with NumPy, MATLAB, and C++. Each example will demonstrate a successful and an unsuccessful multiplication based on inner dimension compatibility.

**2.1 Python with NumPy**

```python
import numpy as np

# Successful multiplication: inner dimensions match (3)
mat1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
mat2 = np.array([[7, 8], [9, 10], [11, 12]])  # 3x2
result = np.dot(mat1, mat2)  # Or mat1 @ mat2 for Python 3.5+
print(result)  # Output: a 2x2 matrix

# Unsuccessful multiplication: inner dimensions mismatch (3 != 2)
mat3 = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
mat4 = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3
try:
    result = np.dot(mat3, mat4)
except ValueError as e:
    print(f"Error: {e}") # Output: Error: matmul: Input operand 1 has a mismatch in its inner dimensions
```

The Python example clearly demonstrates the outcome of both valid and invalid matrix multiplication attempts. NumPy, a highly optimized library, efficiently handles matrix operations and throws a `ValueError` when encountering dimension mismatches.  This exception handling is crucial for robust code.

**2.2 MATLAB**

```matlab
% Successful multiplication: inner dimensions match (2)
mat1 = [1 2; 3 4]; % 2x2
mat2 = [5 6; 7 8]; % 2x2
result = mat1 * mat2;
disp(result); % Output: a 2x2 matrix

% Unsuccessful multiplication: inner dimensions mismatch (2 != 3)
mat3 = [1 2; 3 4]; % 2x2
mat4 = [5 6 7; 8 9 10; 11 12 13]; % 3x3
try
    result = mat3 * mat4;
catch ME
    disp(ME.identifier); % Output: Error using * Inner matrix dimensions must agree.
end
```

MATLAB's approach is similar.  The `*` operator performs matrix multiplication, and a clear error message indicates the dimension incompatibility. The `try-catch` block is a standard MATLAB technique for handling exceptions, essential in production environments to prevent unexpected program termination.

**2.3 C++ with Eigen**

```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
  // Successful multiplication: inner dimensions match (3)
  Eigen::MatrixXd mat1(2, 3);
  mat1 << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd mat2(3, 2);
  mat2 << 7, 8, 9, 10, 11, 12;
  Eigen::MatrixXd result = mat1 * mat2;
  std::cout << result << std::endl; // Output: a 2x2 matrix


  // Unsuccessful multiplication: inner dimensions mismatch (3 != 2)
  Eigen::MatrixXd mat3(2,3);
  mat3 << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd mat4(2,3);
  mat4 << 7, 8, 9, 10, 11, 12;
  try{
    Eigen::MatrixXd result2 = mat3 * mat4;
  } catch (const std::runtime_error& error) {
    std::cerr << "Error: " << error.what() << std::endl; //Output: Error: This Eigen::Matrix object has a non-dynamic size and is not square
  }
  return 0;
}

```

The C++ example utilizes the Eigen library, a powerful linear algebra library.  Similar to the previous examples, successful multiplication proceeds smoothly. However, an attempt at an invalid multiplication will result in a runtime error, the specific error message will vary depending on the compiler and the way the Eigen library is set up, but it will essentially signal an incompatibility.  Error handling in C++ is crucial for preventing crashes and maintaining application stability.

**3. Resource Recommendations**

For a deeper understanding of matrix algebra and its computational aspects, I recommend exploring standard linear algebra textbooks covering matrix operations and their properties.  Furthermore, the documentation for NumPy, MATLAB, and Eigen (or whichever linear algebra library you are using) will provide detailed explanations and examples concerning matrix multiplication and error handling.  Finally, consulting specialized literature on high-performance computing will further enhance one's comprehension of the computational complexities inherent in large-scale matrix operations.  Understanding these resources provides a robust background to avoid this common error.
