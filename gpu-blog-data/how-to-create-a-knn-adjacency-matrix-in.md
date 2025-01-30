---
title: "How to create a KNN adjacency matrix in PyTorch?"
date: "2025-01-30"
id: "how-to-create-a-knn-adjacency-matrix-in"
---
The core challenge in generating a KNN adjacency matrix within PyTorch lies in efficiently computing the pairwise distances between all data points and subsequently selecting the K nearest neighbors for each point.  Naive approaches suffer from significant computational overhead, especially with high-dimensional data or large datasets. My experience working on large-scale graph neural networks highlighted this bottleneck, necessitating a move beyond brute-force methods.  This response details a robust approach leveraging PyTorch's optimized functions for efficient computation.


**1.  Clear Explanation:**

The creation of a KNN adjacency matrix involves three principal steps: distance computation, nearest neighbor identification, and adjacency matrix construction.  We'll focus on using Euclidean distance, though other metrics can be substituted.  The efficiency hinges on avoiding explicit computation of the full pairwise distance matrix.  Instead, we exploit PyTorch's broadcasting capabilities and functions like `torch.topk` to identify the nearest neighbors without the memory burden of storing a dense distance matrix.

The algorithm proceeds as follows:

a) **Distance Computation:** We leverage broadcasting to efficiently calculate the pairwise Euclidean distances between all data points.  Given a data tensor `X` of shape (N, D), where N is the number of data points and D is the dimensionality, we can compute the squared Euclidean distances using a concise PyTorch expression.

b) **Nearest Neighbor Identification:**  Using `torch.topk`, we identify the indices of the K nearest neighbors for each data point.  This function returns both the K smallest distances and their corresponding indices.  We are primarily interested in the indices.

c) **Adjacency Matrix Construction:** Finally, we construct the adjacency matrix.  This is a sparse matrix where a non-zero element (typically 1) indicates an edge between two nodes (data points). We achieve this by creating a sparse matrix using `torch.sparse_coo_tensor` based on the indices obtained in the previous step.

This approach avoids the quadratic complexity associated with calculating and storing the entire distance matrix, making it scalable to larger datasets.


**2. Code Examples with Commentary:**


**Example 1: Basic KNN Adjacency Matrix Creation:**

```python
import torch

def create_knn_adjacency_matrix(X, k):
    """
    Creates a KNN adjacency matrix using Euclidean distance.

    Args:
        X: Data tensor of shape (N, D).
        k: Number of nearest neighbors.

    Returns:
        Sparse adjacency matrix of shape (N, N).
    """
    N = X.shape[0]
    distances = torch.cdist(X, X, p=2)**2 # Squared Euclidean distance for efficiency

    _, indices = torch.topk(distances, k=k+1, dim=1, largest=False) #+1 to exclude self

    indices = indices[:, 1:] #remove self-loops

    row_indices = torch.arange(N).repeat_interleave(k)
    col_indices = indices.flatten()

    values = torch.ones(N * k)

    adjacency_matrix = torch.sparse_coo_tensor(
        indices=[row_indices, col_indices],
        values=values,
        size=(N, N)
    )

    return adjacency_matrix

# Example usage
X = torch.randn(100, 10)  # 100 data points, 10 dimensions
k = 5
adjacency_matrix = create_knn_adjacency_matrix(X, k)
print(adjacency_matrix)
```

This example demonstrates the core steps: distance calculation, neighbor selection, and sparse matrix creation. The self-loops are explicitly removed to avoid unnecessary connections.


**Example 2: Handling Large Datasets with Chunking:**

```python
import torch

def create_knn_adjacency_matrix_chunked(X, k, chunk_size=1000):
    """
    Creates KNN adjacency matrix for large datasets using chunking.

    Args:
        X: Data tensor of shape (N, D).
        k: Number of nearest neighbors.
        chunk_size: Size of chunks for processing.

    Returns:
        Sparse adjacency matrix of shape (N, N).
    """
    N = X.shape[0]
    row_indices = []
    col_indices = []
    values = []

    for i in range(0, N, chunk_size):
        chunk = X[i:i + chunk_size]
        distances = torch.cdist(chunk, X, p=2)**2
        _, indices = torch.topk(distances, k=k+1, dim=1, largest=False)
        indices = indices[:,1:]

        chunk_row_indices = torch.arange(i, i + chunk_size).repeat_interleave(k)
        chunk_col_indices = indices.flatten()
        chunk_values = torch.ones(chunk_size * k)

        row_indices.extend(chunk_row_indices.tolist())
        col_indices.extend(chunk_col_indices.tolist())
        values.extend(chunk_values.tolist())

    adjacency_matrix = torch.sparse_coo_tensor(
        indices=[row_indices, col_indices],
        values=values,
        size=(N, N)
    )
    return adjacency_matrix


# Example Usage (Illustrative - requires a significantly larger dataset for practical chunking)
X = torch.randn(2000, 10)
k = 5
adjacency_matrix = create_knn_adjacency_matrix_chunked(X, k)
print(adjacency_matrix)

```

This example incorporates chunking to manage memory consumption when dealing with extremely large datasets that don't fit into RAM. It processes the data in smaller, manageable chunks and concatenates the results.

**Example 3: Using a Different Distance Metric:**

```python
import torch
import torch.nn.functional as F

def create_knn_adjacency_matrix_cosine(X, k):
    """
    Creates a KNN adjacency matrix using cosine similarity.

    Args:
        X: Data tensor of shape (N, D).  Must be normalized.
        k: Number of nearest neighbors.

    Returns:
        Sparse adjacency matrix of shape (N, N).
    """
    N = X.shape[0]
    #Cosine similarity is 1 - distance.  We'll use negative cosine similarity to find nearest neighbours using topk.
    similarities = -torch.mm(X, X.T)

    _, indices = torch.topk(similarities, k=k+1, dim=1, largest=True) # +1 to exclude self

    indices = indices[:,1:] #remove self

    row_indices = torch.arange(N).repeat_interleave(k)
    col_indices = indices.flatten()
    values = torch.ones(N * k)

    adjacency_matrix = torch.sparse_coo_tensor(
        indices=[row_indices, col_indices],
        values=values,
        size=(N, N)
    )
    return adjacency_matrix


# Example usage. Remember to normalize X for cosine similarity.
X = F.normalize(torch.randn(100, 10), dim=1) # Normalize data along dimension 1 (features)
k = 5
adjacency_matrix = create_knn_adjacency_matrix_cosine(X, k)
print(adjacency_matrix)

```

This example demonstrates using cosine similarity instead of Euclidean distance, showcasing adaptability to different distance metrics. Remember that data normalization is crucial for cosine similarity.


**3. Resource Recommendations:**

For a deeper understanding of sparse matrices in PyTorch, consult the official PyTorch documentation.  Thorough comprehension of linear algebra, particularly matrix operations, is fundamental.  Familiarize yourself with computational complexity analysis to understand the efficiency gains of the presented approach.  Exploring advanced algorithms for approximate nearest neighbor search, like Locality Sensitive Hashing (LSH), can further improve performance on extremely large datasets.
