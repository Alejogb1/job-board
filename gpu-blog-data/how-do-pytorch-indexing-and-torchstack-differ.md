---
title: "How do PyTorch indexing and `torch.stack` differ?"
date: "2025-01-30"
id: "how-do-pytorch-indexing-and-torchstack-differ"
---
PyTorch's indexing mechanisms and the `torch.stack` function, while both operating on tensors, serve fundamentally different purposes.  My experience optimizing deep learning models has highlighted this distinction repeatedly, particularly when dealing with batch processing and model output manipulation. Indexing accesses and manipulates existing tensor elements, while `torch.stack` constructs a new tensor by concatenating input tensors along a new dimension.  This core difference underpins their distinct applications and performance characteristics.

**1. Clear Explanation:**

PyTorch indexing allows for the selection and modification of specific elements within a tensor. This is achieved using slice notation (similar to Python lists) and advanced indexing techniques employing boolean masks or integer arrays.  The result of indexing is a view (or sometimes a copy, depending on the operation) of the original tensor; modifications to the indexed portion directly affect the original tensor unless a detached copy is explicitly created.

In contrast, `torch.stack` takes a sequence of tensors (typically a list or tuple) as input and concatenates them along a new dimension, creating a higher-dimensional tensor.  Crucially, the input tensors must possess identical shapes except for the dimension along which stacking occurs. The resultant tensor is a new object independent of the input tensors; changes to the stacked tensor do not affect the originals.

The key distinction lies in *in-place modification* versus *tensor creation*. Indexing primarily deals with in-place modification (though copies can be made), altering the original tensor.  `torch.stack`, on the other hand, always creates a completely new tensor. This difference has implications for memory management and computational efficiency.  In scenarios where memory is constrained, employing indexing judiciously can be more efficient than using `torch.stack` to repeatedly create new tensors.


**2. Code Examples with Commentary:**

**Example 1: Indexing for Data Selection**

```python
import torch

# Sample tensor representing image batches (batch_size, channels, height, width)
image_batch = torch.randn(32, 3, 224, 224)

# Select the first 10 images
selected_images = image_batch[:10]  # Slice notation

# Select images 5, 10, and 15
indices = torch.tensor([4, 9, 14])  # Advanced indexing
specific_images = image_batch[indices]

# Extract the red channel from all images
red_channel = image_batch[:, 0, :, :]  # Selecting a specific channel

# Modify a specific pixel value (in-place modification)
image_batch[0, 0, 100, 100] = 1.0

print(selected_images.shape)  # Output: torch.Size([10, 3, 224, 224])
print(specific_images.shape)  # Output: torch.Size([3, 3, 224, 224])
print(red_channel.shape)  # Output: torch.Size([32, 224, 224])
```

This example showcases different indexing techniques for selecting subsets of the `image_batch` tensor. The modifications made to `image_batch` directly impact the original tensor due to the nature of indexing.

**Example 2: `torch.stack` for Batch Creation**

```python
import torch

# Individual tensors representing processed image features
feature1 = torch.randn(10, 128)
feature2 = torch.randn(10, 128)
feature3 = torch.randn(10, 128)

# Stack along the batch dimension (dim=0)
stacked_features = torch.stack([feature1, feature2, feature3], dim=0)

print(stacked_features.shape) # Output: torch.Size([3, 10, 128])

# Access individual features (indexing after stacking)
retrieved_feature2 = stacked_features[1]

# Modifications to retrieved_feature2 will not affect original feature2
retrieved_feature2[0,0] = 0.0
print(feature2[0,0]) # Output: A random value (not 0.0)
```

Here, `torch.stack` combines three feature tensors along a new dimension (batch dimension). The resulting tensor is distinct from the original features, and changes to it do not alter the original tensors.

**Example 3: Combining Indexing and `torch.stack`**

```python
import torch

# Input tensor of shape (batch_size, sequence_length, embedding_dim)
input_tensor = torch.randn(32, 20, 768)

# Select specific sequences using indexing
sequence1 = input_tensor[0]  # Shape: (20, 768)
sequence2 = input_tensor[5]  # Shape: (20, 768)

# Stack selected sequences
stacked_sequences = torch.stack([sequence1, sequence2], dim=0) # Shape: (2, 20, 768)

# Perform operations on stacked sequences
mean_sequence = torch.mean(stacked_sequences, dim=0)

print(mean_sequence.shape)  # Output: torch.Size([20, 768])

```

This demonstrates how indexing and `torch.stack` can be used together.  Indexing first selects the required data, and then `torch.stack` combines them into a new tensor for further processing. Note that changing `mean_sequence` does not change `sequence1` or `sequence2` because `torch.stack` created a new tensor.


**3. Resource Recommendations:**

The official PyTorch documentation.  A comprehensive textbook on deep learning with a strong focus on PyTorch implementation details.  Advanced PyTorch tutorials focusing on tensor manipulation and performance optimization.  These resources offer in-depth explanations and practical examples beyond the scope of this response.  Carefully studying these materials will solidify your understanding of PyTorch's tensor operations.
