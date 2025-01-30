---
title: "How can I apply a vectorized function to the Cartesian product of two ranges in PyTorch?"
date: "2025-01-30"
id: "how-can-i-apply-a-vectorized-function-to"
---
The challenge lies in efficiently computing a function over all combinations of elements from two independent sequences without resorting to explicit loops, leveraging PyTorch’s computational graph and parallelization capabilities. I encountered this while developing a custom kernel for a novel similarity metric involving pairwise comparison of high-dimensional feature vectors. My initial attempts with Python for-loops severely bottlenecked the computation, underscoring the need for a vectorized solution.

The core idea is to avoid explicit iteration and instead utilize PyTorch’s tensor broadcasting and vectorized operations. Broadcasting automatically expands the dimensions of tensors during arithmetic operations to match, implicitly creating the Cartesian product without materializing the entire product in memory. Instead of constructing a large intermediate tensor representing all pairs, we manipulate the original input tensors to perform the desired function on all combinations simultaneously.

Let’s break this down into a practical example. Assume we have two ranges: `range_a` and `range_b`, represented as PyTorch tensors. We want to apply a function `f(x, y)` to every possible combination of `x` from `range_a` and `y` from `range_b`.  Instead of nested loops like:

```python
def looped_function(range_a, range_b):
    result = []
    for x in range_a:
        for y in range_b:
            result.append(f(x,y))
    return torch.tensor(result).reshape(len(range_a), len(range_b))

```

which are notoriously slow and unsuitable for GPUs,  we use PyTorch's tensor manipulation for vectorization.  The key is to reshape `range_a` into a column vector,  `range_b` into a row vector, and the broadcasting mechanism will implicitly create the Cartesian product for us when we operate on them together in the function `f`.

Here's the initial code example demonstrating how we might achieve a simple multiplication across the Cartesian product:

```python
import torch

def vectorized_multiply(range_a, range_b):
    """
    Performs element-wise multiplication of the Cartesian product of two ranges.

    Args:
        range_a (torch.Tensor): The first range as a 1D tensor.
        range_b (torch.Tensor): The second range as a 1D tensor.

    Returns:
        torch.Tensor: A 2D tensor containing the result of the multiplication over the Cartesian product.
    """
    a_col = range_a.reshape(-1, 1)  # Convert to column vector
    b_row = range_b.reshape(1, -1)  # Convert to row vector
    return a_col * b_row            # Perform element-wise multiplication

# Example usage:
range_a = torch.tensor([1, 2, 3])
range_b = torch.tensor([4, 5, 6])
result_mult = vectorized_multiply(range_a, range_b)
print("Multiplication Result:\n", result_mult)

```
In the `vectorized_multiply` function, we reshape `range_a` into a column vector using `range_a.reshape(-1, 1)`. `-1` instructs PyTorch to infer the size based on other dimensions, in this case, maintaining the original elements in a column-wise tensor. Similarly, `range_b` is converted into a row vector using `range_b.reshape(1, -1)`. PyTorch's broadcasting mechanism, when used with element-wise multiplication (`*`), replicates the column vector across the rows and the row vector down the columns, implicitly forming the Cartesian product and multiplying each pair efficiently.

This approach works for straightforward element-wise operations. However, what if we want to apply a more complex function? Here’s an example of applying a custom function, `squared_difference`, on the Cartesian product:

```python
def squared_difference(x, y):
    """
    Calculates the square of the difference between two numbers.

    Args:
        x (torch.Tensor): The first input tensor.
        y (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The square of the difference (x-y)^2.
    """
    return (x - y)**2

def vectorized_squared_diff(range_a, range_b):
    """
    Calculates the squared difference over the Cartesian product of two ranges.

    Args:
        range_a (torch.Tensor): The first range as a 1D tensor.
        range_b (torch.Tensor): The second range as a 1D tensor.

    Returns:
        torch.Tensor: A 2D tensor containing the squared difference.
    """
    a_col = range_a.reshape(-1, 1)
    b_row = range_b.reshape(1, -1)
    return squared_difference(a_col, b_row) # Apply the custom function

# Example usage:
range_a = torch.tensor([1, 2, 3])
range_b = torch.tensor([4, 5, 6])
result_squared_diff = vectorized_squared_diff(range_a, range_b)
print("\nSquared Difference Result:\n", result_squared_diff)
```

In `vectorized_squared_diff`, the principle is identical. We reshape our input tensors to leverage broadcasting and then, importantly, the custom `squared_difference` function is applied to the broadcasted tensors. Again, no explicit looping occurs within the function, meaning all calculations happen concurrently and efficiently. The function `squared_difference` is written to accept and handle tensors directly using the available element wise operations.  This is critical for the vectorization.

Finally, a more realistic scenario involves applying functions that operate on vectors. For example, consider calculating the cosine similarity between all possible pairs of vectors from two lists. This is critical in many machine learning applications including content recommendation and image retrieval. This uses the dot product and norm functions which PyTorch conveniently provides and are already vectorized:
```python
def cosine_similarity(x, y):
    """
    Calculates the cosine similarity between two vectors.

    Args:
        x (torch.Tensor): The first vector.
        y (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The cosine similarity between the vectors.
    """
    dot_product = torch.sum(x * y, dim=-1, keepdim = True)
    norm_x = torch.linalg.norm(x, dim = -1, keepdim = True)
    norm_y = torch.linalg.norm(y, dim = -1, keepdim = True)
    return dot_product / (norm_x * norm_y)


def vectorized_cosine_similarity(vectors_a, vectors_b):
    """
    Calculates the cosine similarity between all pairs of vectors from two lists.

    Args:
        vectors_a (torch.Tensor): A tensor of vectors, shape (N, D).
        vectors_b (torch.Tensor): A tensor of vectors, shape (M, D).

    Returns:
         torch.Tensor: A tensor of cosine similarities, shape (N, M).
    """

    vectors_a_col = vectors_a.reshape(vectors_a.shape[0], 1, -1)
    vectors_b_row = vectors_b.reshape(1, vectors_b.shape[0], -1)

    return cosine_similarity(vectors_a_col, vectors_b_row)

# Example usage:
vectors_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
vectors_b = torch.tensor([[2.0, 4.0, 1.0], [3.0, 5.0, 2.0]])

result_cosine_sim = vectorized_cosine_similarity(vectors_a, vectors_b)
print("\nCosine Similarity Result:\n", result_cosine_sim)
```
In this example, each 'vector' is a dimension in the tensor. The same reshaping logic is applied, but now we are working with tensors of shape `(N, D)` and `(M, D)`, where `N` and `M` represent the number of vectors and `D` represents the vector dimensionality.  The broadcasting handles the pairing of the N and M dimensions of the vectors during calculation.  Note that the cosine_similarity is defined to work on the last dimension of the tensors with the `dim = -1` argument.  The `keepdim = True` in the dot product and norm allows for broadcasting in the subsequent division.

When working with such problems, several resource are essential. First, the PyTorch documentation itself provides an exhaustive overview of tensor operations, broadcasting rules, and best practices. Secondly, studying example use cases within the documentation, particularly those related to tensor manipulation, can be very instructive. Lastly, reviewing research papers involving machine learning and specifically those that demonstrate efficient implementation of common mathematical operations can provide inspiration and implementation ideas. Furthermore, analyzing source code of popular PyTorch libraries,  particularly those that operate directly on tensors, is useful in understanding common patterns and techniques.
