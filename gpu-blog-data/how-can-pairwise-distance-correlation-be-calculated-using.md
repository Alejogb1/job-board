---
title: "How can pairwise distance correlation be calculated using np.einsum?"
date: "2025-01-30"
id: "how-can-pairwise-distance-correlation-be-calculated-using"
---
Pairwise distance correlation, a measure of nonlinear association between two random vectors, can be efficiently computed leveraging the optimized tensor operations of `numpy.einsum`. I've found this approach, while not immediately intuitive, outperforms traditional loop-based methods when dealing with larger datasets, significantly reducing computational time. The core principle lies in reformulating the distance correlation formula into a sequence of tensor contractions.

The standard distance correlation calculation involves a multi-step process. First, we compute the distance matrices for each variable. These matrices contain all pairwise Euclidean distances between observations. Then, from these distance matrices, we derive the centered distance matrices. Finally, the distance correlation is calculated using the elements of the centered distance matrices. Traditional looping makes this calculation O(n^2) in computational complexity, where 'n' represents the number of observations. Using `np.einsum`, we leverage optimized BLAS routines to potentially reduce this complexity and significantly improve performance.

To understand how this is achieved, consider the following: The distance correlation for two random vectors, X and Y, with sample sizes 'n', is defined as the square root of the ratio of the distance covariance to the product of the distance standard deviations. The distance covariance is calculated based on the double-centered distance matrices of X and Y. The single centering, denoted `d_cent`, which involves subtracting the row mean, column mean, and adding the overall mean, is the primary factor we streamline with `einsum`.

Let's begin with the initial step: computing the distance matrices. Given a data matrix `x` where rows represent observations and columns represent variables, I can compute the distance matrix `D` as `D[i,j] = ||x[i] - x[j]||`, the Euclidean distance between the ith and jth observations. While `np.linalg.norm` can be used within loops, we can employ `einsum` in combination with broadcasting for better performance.

```python
import numpy as np

def pairwise_distance_matrix(x):
    """Computes pairwise distance matrix using einsum."""
    x_sq = np.einsum('ij,ij->i', x, x)
    D = np.sqrt(x_sq[:, None] + x_sq - 2 * np.einsum('ij,kj->ik', x, x))
    return D

# Example usage
x_data = np.array([[1, 2], [3, 4], [5, 6], [7,8]])
distance_matrix = pairwise_distance_matrix(x_data)
print("Distance Matrix:\n", distance_matrix)
```
In this function, `x_sq` stores the squared norms of each observation. The key step is the line constructing `D`. Here, `np.einsum('ij,ij->i', x, x)` computes the squared Euclidean norms along the row axes, followed by `x_sq[:, None]` to make it a column vector, facilitating broadcasting in the subsequent steps. We subtract `2 * np.einsum('ij,kj->ik', x, x)` to complete the expansion of `||x[i] - x[j]||^2 = ||x[i]||^2 + ||x[j]||^2 - 2*x[i].dot(x[j])`, before taking the square root. This eliminates the need for explicit loops and is generally faster for large datasets.

Now, with the distance matrix computed, the next crucial step is double centering. Let's implement a function using `einsum` to perform this task.

```python
def double_center_distance_matrix(D):
    """Double centers distance matrix using einsum."""
    n = D.shape[0]
    row_mean = np.einsum('ij->i', D) / n
    col_mean = np.einsum('ij->j', D) / n
    grand_mean = np.einsum('ij->', D) / (n * n)
    D_cent = D - row_mean[:, None] - col_mean + grand_mean
    return D_cent

# Example usage (using previous distance_matrix)
centered_distance_matrix = double_center_distance_matrix(distance_matrix)
print("Centered Distance Matrix:\n", centered_distance_matrix)
```

This function employs `einsum` to compute the row means, column means, and grand mean of the distance matrix in a vectorized way. Instead of traditional loops or iterative approaches, we can directly calculate these means as tensor contractions, and then apply them to the original distance matrix to achieve the double centering. The term `row_mean[:, None]` turns the row mean vector into a column vector, which helps correctly align terms for the subtraction using broadcasting features of NumPy. This implementation is efficient, avoiding loops and capitalizing on NumPy’s optimized calculations.

Finally, to compute the distance correlation, we need to complete the formula by calculating the distance covariance and the distance standard deviations. For this, we also use `np.einsum` to simplify calculations. Here's a function that puts it all together, including the distance correlation calculation:

```python
def distance_correlation(x, y):
    """Calculates distance correlation between two data matrices using einsum."""
    D_x = pairwise_distance_matrix(x)
    D_y = pairwise_distance_matrix(y)
    D_x_cent = double_center_distance_matrix(D_x)
    D_y_cent = double_center_distance_matrix(D_y)

    dcov_xy = np.sqrt(np.einsum('ij,ij->', D_x_cent, D_y_cent) / D_x_cent.shape[0]**2)
    dvar_x = np.sqrt(np.einsum('ij,ij->', D_x_cent, D_x_cent) / D_x_cent.shape[0]**2)
    dvar_y = np.sqrt(np.einsum('ij,ij->', D_y_cent, D_y_cent) / D_y_cent.shape[0]**2)

    if dvar_x == 0 or dvar_y == 0:
        return 0
    dcor_xy = dcov_xy / np.sqrt(dvar_x * dvar_y)

    return dcor_xy


# Example Usage
y_data = np.array([[10, 12], [13, 14], [15, 16], [17,18]])
distance_corr = distance_correlation(x_data, y_data)
print("Distance Correlation:", distance_corr)
```

In this final function, after calculating the double centered distance matrices, the distance covariance, denoted by `dcov_xy`, is computed through another `einsum` operation which directly sums the element-wise product of `D_x_cent` and `D_y_cent`, without requiring intermediary steps. Likewise, the distance variance for `x` and `y` (`dvar_x` and `dvar_y`) are computed using the same approach, summing the squared elements of the centered distance matrices. We then take the square root and form the ratio, checking for division by zero to avoid errors.

This series of code examples demonstrates the complete process of calculating pairwise distance correlation using `np.einsum`, showcasing the advantages of using tensor operations over traditional looping, in terms of conciseness and potential for improved computational performance.

For further exploration of the concepts and methods employed, I recommend examining the NumPy documentation focusing on `einsum` and broadcasting rules. Books on multivariate statistical analysis and machine learning often contain theoretical underpinnings of distance correlation, particularly in the context of dependence metrics. Statistical computing textbooks that discuss vectorized operations will also illuminate the practical advantages and efficiency gains possible from using these methods in larger datasets.
