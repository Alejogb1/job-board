---
title: "How can K-means clustering be implemented in TensorFlow?"
date: "2025-01-30"
id: "how-can-k-means-clustering-be-implemented-in-tensorflow"
---
Implementing K-means clustering in TensorFlow leverages the framework's computational graph capabilities, allowing for optimized execution, especially on accelerators like GPUs. While TensorFlow isn't solely a clustering library, it offers the fundamental operations necessary to construct the algorithm efficiently. My experience building large-scale data processing pipelines within TensorFlow has shown that understanding these operations provides significant flexibility in how clustering is deployed, particularly in scenarios involving complex preprocessing or integration with deep learning models.

The core idea behind K-means remains consistent: partition N data points into K clusters, where each data point belongs to the cluster with the nearest mean (centroid). This process involves: initialization of centroids, assignment of points to the closest centroid, updating centroid positions based on assigned points, and repetition until convergence. A naive implementation might rely on explicit loops in Python, but TensorFlow's strengths lie in vectorized operations, which are crucial for performance with large datasets.

Therefore, the TensorFlow implementation is structured as a computational graph consisting of tensor operations representing these key steps. Initialization involves either randomly selecting K data points as initial centroids or using a more principled approach like the k-means++ algorithm (which isn't directly implemented in core TensorFlow, but could be added with custom operations). Distance calculation between data points and centroids is achieved using the `tf.reduce_sum(tf.square(tf.subtract(x, c)), axis=1)` operation (Euclidean distance) repeated using broadcasting. The assignment phase utilizes `tf.argmin` to identify the centroid index closest to each point. Centroid update relies on `tf.unsorted_segment_mean` to compute the mean of all data points within each cluster. The loop structure is typically managed with `tf.while_loop` ensuring iterative refinements until convergence, generally determined by either a fixed number of iterations or a small change in centroid positions across iterations.

Let’s examine code examples to clarify these points. First, I will present a simple implementation demonstrating the core steps. I will assume the input data is already preprocessed and structured as a `tf.Tensor`.

```python
import tensorflow as tf

def kmeans_simple(data, k, max_iterations=100):
    """
    Simple K-means implementation using TensorFlow.

    Args:
        data: tf.Tensor, input data of shape [N, D] where N is the number of samples
              and D is the number of features.
        k: int, number of clusters.
        max_iterations: int, maximum number of iterations.

    Returns:
       centroids: tf.Tensor, final centroids of shape [k, D].
       assignments: tf.Tensor, cluster assignments of each sample, of shape [N]
    """
    N = tf.shape(data)[0]
    D = tf.shape(data)[1]

    # Initialize centroids randomly by picking data points
    centroids_indices = tf.random.shuffle(tf.range(N))[:k]
    centroids = tf.gather(data, centroids_indices)

    def cond(i, centroids, assignments):
        return tf.less(i, max_iterations)

    def body(i, centroids, assignments):
        # Expand dimensions for broadcasting with centroids
        expanded_data = tf.expand_dims(data, 1) # [N, 1, D]
        expanded_centroids = tf.expand_dims(centroids, 0) # [1, k, D]
        
        # Calculate distances (Euclidean)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_data, expanded_centroids)), axis=2) # [N, k]
        
        # Assign data points to closest centroid
        assignments = tf.argmin(distances, axis=1) # [N]
        
        # Calculate new centroid positions
        new_centroids = tf.unsorted_segment_mean(data, assignments, k)
        
        return i + 1, new_centroids, assignments
    
    _, final_centroids, final_assignments = tf.while_loop(
        cond,
        body,
        loop_vars = (tf.constant(0), centroids, tf.zeros(N, dtype=tf.int64))
    )

    return final_centroids, final_assignments

# Example Usage:
data = tf.constant([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]], dtype=tf.float32)
k = 2
final_centroids, final_assignments = kmeans_simple(data, k)
print(f"Final Centroids: {final_centroids}")
print(f"Final Assignments: {final_assignments}")
```

This `kmeans_simple` function illustrates the basic operation, directly using random initialization. However, for improved initial conditions, employing a more robust method like k-means++ is advisable. A second example will illustrate a simple implementation of k-means++, followed by k-means. Here, the `initialize_centroids_kmeans_plusplus` function selects initial centroids that are well spread out and reduces the likelihood of converging to local minima.

```python
import tensorflow as tf

def initialize_centroids_kmeans_plusplus(data, k, seed=None):
    """
    Initializes centroids using the k-means++ algorithm.

    Args:
        data: tf.Tensor, input data of shape [N, D]
        k: int, number of clusters
        seed: int, optional seed for random ops

    Returns:
        centroids: tf.Tensor, initial centroids of shape [k, D]
    """
    
    N = tf.shape(data)[0]

    if seed:
      tf.random.set_seed(seed)

    first_centroid_index = tf.random.uniform(shape=(), minval=0, maxval=N, dtype=tf.int32)
    centroids = tf.gather(data, [first_centroid_index])

    for _ in range(1, k):
        
        expanded_data = tf.expand_dims(data, 1) # [N, 1, D]
        expanded_centroids = tf.expand_dims(centroids, 0) # [1, k, D]
        
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_data, expanded_centroids)), axis=2) # [N, k]
        
        min_distances = tf.reduce_min(distances, axis=1) # [N]
        
        distribution = min_distances / tf.reduce_sum(min_distances) # [N]
        
        next_centroid_index = tf.random.categorical(tf.math.log(tf.expand_dims(distribution, 0)), 1)[0][0]

        next_centroid = tf.gather(data, [next_centroid_index])
        centroids = tf.concat([centroids, next_centroid], axis=0)

    return centroids

def kmeans_kmeans_plusplus(data, k, max_iterations=100, seed=None):
    """
    K-means using k-means++ initialization.

    Args:
        data: tf.Tensor, input data of shape [N, D]
        k: int, number of clusters
        max_iterations: int, maximum number of iterations.
        seed: int, optional seed for kmeans++ initialization.
        

    Returns:
        centroids: tf.Tensor, final centroids of shape [k, D].
        assignments: tf.Tensor, cluster assignments of each sample, of shape [N]
    """
    N = tf.shape(data)[0]
    D = tf.shape(data)[1]

    # Initialize centroids using k-means++
    centroids = initialize_centroids_kmeans_plusplus(data, k, seed)
    
    def cond(i, centroids, assignments):
        return tf.less(i, max_iterations)

    def body(i, centroids, assignments):
        # Expand dimensions for broadcasting with centroids
        expanded_data = tf.expand_dims(data, 1) # [N, 1, D]
        expanded_centroids = tf.expand_dims(centroids, 0) # [1, k, D]
        
        # Calculate distances (Euclidean)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_data, expanded_centroids)), axis=2) # [N, k]
        
        # Assign data points to closest centroid
        assignments = tf.argmin(distances, axis=1) # [N]
        
        # Calculate new centroid positions
        new_centroids = tf.unsorted_segment_mean(data, assignments, k)
        
        return i + 1, new_centroids, assignments
    
    _, final_centroids, final_assignments = tf.while_loop(
        cond,
        body,
        loop_vars = (tf.constant(0), centroids, tf.zeros(N, dtype=tf.int64))
    )

    return final_centroids, final_assignments

# Example Usage:
data = tf.constant([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]], dtype=tf.float32)
k = 2
final_centroids, final_assignments = kmeans_kmeans_plusplus(data, k, seed=42)
print(f"Final Centroids (kmeans++): {final_centroids}")
print(f"Final Assignments (kmeans++): {final_assignments}")
```

The addition of k-means++ initialization in the second example addresses the limitations of random centroid selection, potentially improving the quality of the final clusters. The code performs an initial pass, gradually selecting centroids based on distances to already chosen centroids.

Lastly, I'll show an implementation incorporating convergence checks, ensuring the algorithm stops when the centroid positions stabilize.

```python
import tensorflow as tf

def kmeans_with_convergence(data, k, tolerance=1e-4, max_iterations=100, seed=None):
    """
    K-means using convergence checks on centroid movement.

    Args:
        data: tf.Tensor, input data of shape [N, D]
        k: int, number of clusters
        tolerance: float, the convergence tolerance.
        max_iterations: int, maximum number of iterations
        seed: int, optional seed for kmeans++ initialization
    Returns:
        centroids: tf.Tensor, final centroids of shape [k, D].
        assignments: tf.Tensor, cluster assignments of each sample, of shape [N]
    """
    N = tf.shape(data)[0]
    D = tf.shape(data)[1]
    
    centroids = initialize_centroids_kmeans_plusplus(data, k, seed)

    def cond(i, centroids, old_centroids, assignments):
        return tf.logical_and(
            tf.less(i, max_iterations),
            tf.reduce_any(tf.greater(tf.reduce_sum(tf.abs(tf.subtract(centroids, old_centroids))), tolerance))
        )

    def body(i, centroids, old_centroids, assignments):
        # Expand dimensions for broadcasting with centroids
        expanded_data = tf.expand_dims(data, 1) # [N, 1, D]
        expanded_centroids = tf.expand_dims(centroids, 0) # [1, k, D]
        
        # Calculate distances (Euclidean)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_data, expanded_centroids)), axis=2) # [N, k]
        
        # Assign data points to closest centroid
        assignments = tf.argmin(distances, axis=1) # [N]
        
        # Calculate new centroid positions
        new_centroids = tf.unsorted_segment_mean(data, assignments, k)
        
        return i + 1, new_centroids, centroids, assignments
    
    _, final_centroids, _, final_assignments = tf.while_loop(
        cond,
        body,
        loop_vars=(tf.constant(0), centroids, tf.zeros_like(centroids), tf.zeros(N, dtype=tf.int64))
    )

    return final_centroids, final_assignments

# Example Usage:
data = tf.constant([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]], dtype=tf.float32)
k = 2
final_centroids, final_assignments = kmeans_with_convergence(data, k, tolerance=1e-5, seed=42)
print(f"Final Centroids (convergence): {final_centroids}")
print(f"Final Assignments (convergence): {final_assignments}")
```

The inclusion of the convergence check provides a practical approach, halting when centroids cease to move significantly, saving unnecessary computation cycles.

For advanced users, I would recommend exploring implementations that include custom loss functions based on the nature of the dataset and/or problem requirements (e.g. weighted K-means). For those needing scalability or integration with larger pipelines, TensorFlow Datasets and distributed training options should be examined. Additional resources include books focusing on TensorFlow’s computational graph and specific tutorials on implementing custom models or algorithms. The official TensorFlow documentation offers foundational guidance on tensors, operations, and control flow. Research papers relating to clustering algorithms are a solid choice for those seeking a deeper theoretical understanding.
