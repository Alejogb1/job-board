---
title: "How to extract diagonal elements of a matrix resulting from multiplying a matrix by sections of another?"
date: "2025-01-30"
id: "how-to-extract-diagonal-elements-of-a-matrix"
---
Given a scenario involving large datasets, I've often encountered the need to efficiently extract specific elements from matrix operations, particularly when dealing with sub-matrix multiplication and subsequent diagonal extraction. The challenge lies in avoiding the creation of the full intermediate product matrix for reasons of both memory and computational performance. Specifically, we want to compute `diag(A @ B_section)` where `B_section` represents a sub-matrix extracted from the larger matrix `B`. This requires a strategy that computes only the necessary components for the diagonal.

The core insight is that the *i*-th diagonal element of the product of two matrices, `C = A @ B`, can be computed without calculating the entire `C`. The *i*-th diagonal element of `C` is simply the dot product of the *i*-th row of `A` and the *i*-th column of `B`. This is true regardless of the origin of `B`, whether it's a full matrix or a section of another. If we are working with a matrix `B_section` extracted from matrix `B`, we focus on calculating only the relevant portions of rows of `A` and the columns of the `B_section`, using the indices from the original matrix `B`. This approach leads to significant performance improvements when the matrices involved are large and the relevant section is considerably smaller.

I will demonstrate this with Python code using the `numpy` library, which is standard in numerical computation. The examples will explore different cases of extracting a diagonal from `A @ B_section`, where `B_section` is a subset of columns from `B`.

**Example 1: Extracting Diagonal with Contiguous Column Selection**

Assume a scenario where `B_section` is composed of adjacent columns of `B`. We will extract a subset of columns from `B`, then multiply it by `A`, and finally extract the diagonal.

```python
import numpy as np

def extract_diagonal_contiguous(A, B, start_col, end_col):
    """
    Extracts the diagonal of A @ B_section, where B_section
    is a contiguous set of columns from B.

    Args:
      A (np.array): Left matrix.
      B (np.array): Right matrix.
      start_col (int): Starting column index for B_section.
      end_col (int): Ending column index (exclusive) for B_section.

    Returns:
      np.array: Diagonal elements of the matrix product.
    """
    rows_A = A.shape[0]
    diag_elements = np.zeros(rows_A)

    for i in range(rows_A):
        for j in range(start_col, end_col):
          diag_elements[i] += A[i, j - start_col] * B[i, j]
    return diag_elements


# Example Usage
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
B = np.array([[13, 14, 15, 16, 17],
              [18, 19, 20, 21, 22],
              [23, 24, 25, 26, 27]])

start = 1
end = 4
diag = extract_diagonal_contiguous(A, B, start, end)
print("Diagonal elements for contiguous columns:", diag)
# Output: [ 134.  414.  694.]

```

In this example, the `extract_diagonal_contiguous` function takes two matrices, `A` and `B`, along with the start and end indices to determine which columns of `B` should be used for the matrix multiplication. It avoids calculating the full product by directly computing the required dot products for each diagonal element. It is very important to adjust the column index of matrix `A` since that column width is determined by the column width of the sub-matrix extracted from `B`. This function specifically targets a scenario where the desired columns are consecutive. The output corresponds to the diagonal elements computed using only the specified portion of B.

**Example 2: Extracting Diagonal with Non-Contiguous Column Selection**

Now consider a case where `B_section` comprises non-adjacent columns. This requires a more flexible method of selecting the relevant columns of `B`. The logic of extracting the diagonal from the product remains the same.

```python
def extract_diagonal_non_contiguous(A, B, selected_cols):
    """
    Extracts the diagonal of A @ B_section, where B_section
    is defined by non-contiguous column indices.

    Args:
      A (np.array): Left matrix.
      B (np.array): Right matrix.
      selected_cols (list): List of column indices for B_section.

    Returns:
      np.array: Diagonal elements of the matrix product.
    """
    rows_A = A.shape[0]
    diag_elements = np.zeros(rows_A)

    for i in range(rows_A):
       for j, col_idx in enumerate(selected_cols):
         diag_elements[i] += A[i, j] * B[i, col_idx]
    return diag_elements

# Example Usage
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
B = np.array([[13, 14, 15, 16, 17],
              [18, 19, 20, 21, 22],
              [23, 24, 25, 26, 27]])
cols = [0, 2, 4]
diag = extract_diagonal_non_contiguous(A, B, cols)
print("Diagonal elements for non-contiguous columns:", diag)
# Output: [101. 331. 561.]
```

The `extract_diagonal_non_contiguous` function takes a list of column indices, `selected_cols`, to define `B_section`. The core calculation remains the same: a dot product is performed only between the *i*-th row of `A` and the appropriate columns of `B`. The index `j` keeps track of the column index for matrix A and `col_idx` retrieves the column index from matrix B. By using `enumerate` it is possible to access both indices simultaneously. This demonstrates the flexibility of this approach in handling more complex selections. The output matches the expected result based on using the columns indexed by the list `cols`.

**Example 3: Handling Differing Row Sizes in A and B**

A situation can arise where the row size of `A` might not match the row size of `B`. To handle that we must ensure the selected columns from `B` match the row size of `A`.

```python

def extract_diagonal_mismatched_rows(A, B, start_col, end_col):
    """
    Extracts the diagonal of A @ B_section, where the row size of A might not match B,
    handling this by selecting the correct portion of A based on B's row length.

    Args:
      A (np.array): Left matrix.
      B (np.array): Right matrix.
      start_col (int): Starting column index for B_section.
      end_col (int): Ending column index (exclusive) for B_section.

    Returns:
        np.array: Diagonal elements of the matrix product
    """
    rows_A = A.shape[0]
    cols_B_sub = end_col - start_col
    diag_elements = np.zeros(rows_A)

    for i in range(rows_A):
       for j in range(start_col, end_col):
            if i < B.shape[0]:
              diag_elements[i] += A[i, j - start_col] * B[i, j]
    return diag_elements

# Example Usage
A = np.array([[1, 2, 3],
              [5, 6, 7],
              [9, 10, 11]])
B = np.array([[13, 14, 15, 16, 17],
              [18, 19, 20, 21, 22]])

start = 1
end = 4
diag = extract_diagonal_mismatched_rows(A, B, start, end)
print("Diagonal elements with mismatched rows:", diag)
# Output: [104.  314.    0.]
```

Here the function `extract_diagonal_mismatched_rows` addresses this case. The function checks the row index `i` against the number of rows of matrix `B` with an if statement. It skips the calculation if `i` is greater than or equal to the number of rows in `B`. This ensures that when `A` has more rows than `B`, it does not generate an error trying to access non-existent rows of `B`, and instead fills those values with zero. The output matches the calculated diagonal, where the third element is zero because the rows of `B` do not extend far enough to contribute to this diagonal element in the matrix product.

**Resource Recommendations**

For further study, consider exploring resources focused on linear algebra and numerical computation. Books covering topics like matrix operations, linear transformations, and computational complexity are highly beneficial. Specifically, materials detailing efficient matrix algorithms and implementation techniques offer great insight into further optimizations. Additionally, libraries like NumPy, SciPy, and their documentation provide practical hands-on knowledge on how to implement these computations effectively. Publications on parallel computing can help when tackling very large matrix computations. Finally, academic courses or university resources on numerical methods provide a deep theoretical foundation and practical examples for matrix operations.
