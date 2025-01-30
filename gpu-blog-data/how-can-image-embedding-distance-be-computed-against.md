---
title: "How can image embedding distance be computed against a group of embeddings?"
date: "2025-01-30"
id: "how-can-image-embedding-distance-be-computed-against"
---
Image embedding distance, often representing semantic similarity, is a core operation in many applications, from image retrieval to clustering. I've frequently encountered the need to efficiently compute the distance between a single image embedding and a potentially large group of other embeddings, and optimization at this stage can significantly improve overall application performance. My experience has predominantly involved working with convolutional neural network-derived embeddings, but the general principles apply across various embedding methods. The challenge usually stems from the sheer volume of computations required when dealing with high dimensional embeddings and large sets of comparison vectors. The straightforward approach, iterating through the group and calculating the distance to each element, can become prohibitively slow.

The central problem lies in calculating a *distance metric* between a *query embedding* and a *set of reference embeddings*. A distance metric quantifies how dissimilar two embeddings are, with a smaller distance indicating higher similarity. Several metrics are commonly used: Euclidean distance (L2 norm), cosine distance, and Manhattan distance (L1 norm). The choice of metric often depends on the specific characteristics of the embedding space and the desired outcome. For instance, cosine distance is favored when the magnitude of embeddings is less informative, focusing more on the angular relationship. Euclidean distance, by contrast, is sensitive to both the direction and magnitude of vectors. In my work, I've found that cosine distance generally performs better for embeddings from deep learning models due to its normalization properties.

The naive approach involves a linear search where the distance from the query embedding is calculated against each embedding in the target group. While simple to implement, this method has a computational complexity of O(N), where N is the number of embeddings in the group. For large sets, this becomes the performance bottleneck. Optimizations typically revolve around techniques to avoid calculating all the individual distances. Techniques like Vectorized Computations, indexing structures (like approximate nearest neighbor libraries) are important. Vectorization exploits the parallelism offered by modern CPUs and GPUs, and indexing structures attempt to provide sub-linear or log-linear time access.

Consider an image retrieval system where you have a user submitting a query image (its embedding) and you want to retrieve the *k* most similar images from your database. Let's demonstrate this with Python and NumPy, a foundational library for scientific computing.

```python
import numpy as np
from numpy.linalg import norm

def euclidean_distance(embedding1, embedding2):
    """Calculates the Euclidean distance between two embeddings."""
    return norm(embedding1 - embedding2)

def cosine_distance(embedding1, embedding2):
    """Calculates the cosine distance between two embeddings."""
    return 1 - np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def calculate_distances_naive(query_embedding, group_embeddings, distance_metric=euclidean_distance):
    """Calculates distances using a naive, iterative approach."""
    distances = []
    for embedding in group_embeddings:
        distances.append(distance_metric(query_embedding, embedding))
    return np.array(distances)

# Example Usage
query = np.random.rand(512)  # Query embedding, 512 dimensions
group = np.random.rand(1000, 512) # Group of 1000 embeddings

distances_euclidean = calculate_distances_naive(query, group, euclidean_distance)
distances_cosine = calculate_distances_naive(query, group, cosine_distance)

print(f"Euclidean distances (first 5): {distances_euclidean[:5]}")
print(f"Cosine distances (first 5): {distances_cosine[:5]}")
```

In the first example, `calculate_distances_naive` function implements the previously mentioned iterative method. The `distance_metric` parameter allows for choosing between Euclidean and Cosine distance. Each distance is calculated individually, demonstrating a clear O(N) behavior. The print statements demonstrate the resultant distances. This approach is easy to understand, but as I mentioned before, it's not scalable for large datasets.

Leveraging NumPy's broadcasting capabilities significantly enhances efficiency by removing the loop and performing calculations in a vectorized manner.

```python
import numpy as np
from numpy.linalg import norm

def euclidean_distance_vectorized(query_embedding, group_embeddings):
    """Calculates Euclidean distance in a vectorized manner."""
    return norm(group_embeddings - query_embedding, axis=1)


def cosine_distance_vectorized(query_embedding, group_embeddings):
    """Calculates cosine distances in a vectorized manner."""
    query_norm = norm(query_embedding)
    group_norms = norm(group_embeddings, axis=1)
    return 1 - np.dot(group_embeddings, query_embedding) / (query_norm * group_norms)


# Example Usage
query = np.random.rand(512)  # Query embedding, 512 dimensions
group = np.random.rand(1000, 512) # Group of 1000 embeddings

distances_euclidean_vec = euclidean_distance_vectorized(query, group)
distances_cosine_vec = cosine_distance_vectorized(query, group)


print(f"Vectorized Euclidean distances (first 5): {distances_euclidean_vec[:5]}")
print(f"Vectorized Cosine distances (first 5): {distances_cosine_vec[:5]}")

```

The second example demonstrates vectorization. Note how the `euclidean_distance_vectorized` and `cosine_distance_vectorized` functions achieve distance computation without explicit loops. The subtraction (or dot product) and normalization happen concurrently across all elements of `group_embeddings`. This eliminates the Python loop overhead, relying on NumPy's optimized underlying C implementation. This significantly improves runtime especially with large datasets. In my practical work, I've observed speed increases of an order of magnitude or greater when using vectorized operations versus the iterative approach, especially on larger datasets.

For large scale applications, where the set of reference embeddings itself becomes very large or querying occurs frequently, even vectorized methods might not be sufficient and we can look at indexing and approximate nearest neighbor search, which have sub linear retrieval times. A commonly used approximate nearest neighbor indexing approach relies on hierarchical navigable small world (HNSW) graphs.

```python
import numpy as np
import hnswlib

def calculate_distances_hnsw(query_embedding, group_embeddings, k=10):
    """Calculates top-k distances using HNSW index."""
    num_elements = group_embeddings.shape[0]
    dim = group_embeddings.shape[1]

    # Initialize index
    p = hnswlib.Index(space = 'cosine', dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 32)

    # Add embeddings
    p.add_items(group_embeddings)
    p.set_ef(50)


    # Retrieve top k
    labels, distances = p.knn_query(query_embedding, k=k)

    return distances


# Example Usage
query = np.random.rand(512)  # Query embedding, 512 dimensions
group = np.random.rand(10000, 512) # Group of 10000 embeddings

distances_hnsw = calculate_distances_hnsw(query, group)

print(f"HNSW distances (top 10): {distances_hnsw}")
```

The last code example demonstrates the usage of `hnswlib` to approximate nearest neighbor search. The `calculate_distances_hnsw` function builds an HNSW index from the `group_embeddings` and uses the index to query the k-nearest neighbors for the given `query_embedding`. The indexing process has an initial cost but allows for fast retrieval.  The parameters like `ef_construction`, `M`, and `ef` control the index quality and performance. I generally suggest starting with the default and fine-tuning based on specific use case and dataset size.

Choosing the right approach is crucial, and the best choice depends on the scale of the problem and real-time performance requirements. The naive iterative method is rarely suitable for practical applications. Vectorization provides significant gains for moderately sized datasets, and HNSW indices are preferred when dealing with massive scale datasets where precise nearest-neighbor computation is prohibitively expensive.

To expand on this further, explore libraries such as FAISS by Facebook AI research, which offers a large collection of indexing structures and algorithms. Libraries such as Annoy and NMSlib should also be investigated. These libraries, and the algorithms they implement, provide a range of options with different trade-offs between accuracy, memory usage, and speed. Further investigation into computational complexity analysis will enable greater understanding of the various options and inform better decision-making.
