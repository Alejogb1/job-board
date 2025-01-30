---
title: "How do I get the indices of specific values in a TensorFlow tensor?"
date: "2025-01-30"
id: "how-do-i-get-the-indices-of-specific"
---
Given the high-performance demands of many machine learning models I've deployed, efficiently locating indices of specific tensor values is a critical operation that often arises when dealing with data processing or model analysis. Direct iteration over large tensors, while seemingly straightforward, introduces unacceptable overhead, making vectorized solutions the preferred approach. The TensorFlow library provides several tools to accomplish this efficiently, particularly leveraging boolean masking and the `tf.where` operation.

The core concept involves first generating a boolean mask representing the locations where the target value is present within the tensor. Then, `tf.where` translates this boolean mask into a set of indices. Critically, `tf.where` returns these indices as a tensor of shape `[N, D]`, where `N` represents the number of matches and `D` corresponds to the dimensionality of the original tensor. For example, in a 2D tensor, `D` is 2, with each row representing a coordinate `[row, column]`. This output format allows for efficient subsequent operations such as slicing or indexing into other tensors using these found coordinates.

Let's examine a few practical use cases with code examples.

**Example 1: Finding Indices of a Single Value in a 1D Tensor**

Suppose I have a 1D tensor representing, for instance, class labels, and I need to locate all occurrences of a particular label value.

```python
import tensorflow as tf

# Example 1D tensor representing class labels
labels = tf.constant([1, 0, 2, 1, 1, 3, 0, 1], dtype=tf.int32)
target_value = 1

# Create a boolean mask where elements equal target_value are True
mask = tf.equal(labels, target_value)

# Use tf.where to convert the boolean mask into indices
indices = tf.where(mask)

print(f"Original Tensor:\n{labels.numpy()}")
print(f"Indices of value {target_value}:\n{indices.numpy()}")
```

In this example, I utilize `tf.equal` to compare each element of the `labels` tensor to the `target_value`. This returns a boolean tensor `mask`. `tf.where` then converts this mask, returning a tensor where each row is a single index representing the position of the value `1` in `labels`. In the output, you will see `[[0], [3], [4], [7]]`, representing the four locations (at index 0, 3, 4 and 7) where `1` was found. It is important to observe that even in a 1D tensor, the shape of the output of `tf.where` is `[N, 1]`.

**Example 2: Finding Indices of a Single Value in a 2D Tensor**

Now, let's expand to a 2D case, representing, as a use case, features or an image with multiple channels.

```python
import tensorflow as tf

# Example 2D tensor
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 2],
                      [1, 6, 7],
                      [8, 9, 1]], dtype=tf.int32)
target_value = 1

# Generate the boolean mask
mask = tf.equal(matrix, target_value)

# Get the indices
indices = tf.where(mask)

print(f"Original Tensor:\n{matrix.numpy()}")
print(f"Indices of value {target_value}:\n{indices.numpy()}")
```

Here, `tf.equal` operates element-wise across the entire `matrix`, creating a 2D boolean mask where `True` values align with occurrences of the `target_value`.  The `tf.where` function translates this into row-column coordinates. The output in this case will be `[[0, 0], [2, 0], [3, 2]]`, indicating the value of 1 was found at row 0, column 0, row 2, column 0 and row 3, column 2 respectively. Note how the resulting tensor shape is `[N, 2]`, reflecting the 2D nature of the input tensor.

**Example 3: Finding Indices of Multiple Values**

Sometimes I encounter situations requiring the retrieval of indices for multiple target values, like detecting a combination of categories in a dataset. While `tf.where` operates effectively on a single comparison, its efficient use with multiple targets requires a slightly different approach leveraging `tf.logical_or` in conjunction with boolean masks.

```python
import tensorflow as tf

# Example 2D tensor
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 2],
                      [1, 6, 7],
                      [8, 9, 1]], dtype=tf.int32)

target_values = [1, 2]

# Build a combined mask for multiple values
combined_mask = tf.zeros_like(matrix, dtype=tf.bool)
for target_value in target_values:
    combined_mask = tf.logical_or(combined_mask, tf.equal(matrix, target_value))

# Retrieve the indices using the combined mask
indices = tf.where(combined_mask)

print(f"Original Tensor:\n{matrix.numpy()}")
print(f"Indices of values {target_values}:\n{indices.numpy()}")
```

In this example, I create an initial mask of all false values with `tf.zeros_like`. I then iterate over the `target_values` list, building a new boolean mask for each, using `tf.equal`, and then combining these individual boolean masks using a logical OR using `tf.logical_or`.  The resulting `combined_mask` is now `True` where *any* of the `target_values` were present. Finally, `tf.where` is used as before to obtain the indices.  The result will be `[[0, 0], [0, 1], [1, 2], [2, 0], [3, 2]]`, indicating the locations where either `1` or `2` were found. This approach scales to an arbitrary number of target values.

It is also worth noting that if I need to compare floating point values, the `tf.equal` function will become too sensitive due to minute discrepancies in their representation. I have often had to employ a custom function with a tolerance for those cases which can be applied in a similar manner to the above examples by substituting `tf.equal` with my comparison function.

In summary, while iterating directly over tensor elements seems initially intuitive, vectorized operations utilizing boolean masking and `tf.where` provide substantial performance gains when extracting indices, which is crucial when dealing with very large tensors, typical in modern machine learning. The flexibility offered by the tensor-based processing allows for further downstream tasks using these index locations efficiently within TensorFlow's computational graph.

For further study, I would recommend researching the TensorFlow documentation, especially the sections on tensor transformations, boolean masking, and the functions `tf.where`, `tf.equal` and `tf.logical_or`. Additionally, a deeper look into computational graph optimizations within TensorFlow, particularly how vectorized operations are processed, will offer additional insights on the performance characteristics of the discussed approach. A textbook or online course specifically dedicated to deep learning with TensorFlow would provide the necessary theoretical background and practical knowledge.
