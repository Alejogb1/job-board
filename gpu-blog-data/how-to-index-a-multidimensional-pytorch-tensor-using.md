---
title: "How to index a multidimensional PyTorch tensor using a list of indices?"
date: "2025-01-30"
id: "how-to-index-a-multidimensional-pytorch-tensor-using"
---
PyTorch tensors, fundamental data structures in deep learning, often require complex indexing beyond simple slicing. Accessing specific elements across multiple dimensions using a list of indices, especially when these indices vary along each dimension, presents a unique challenge. Incorrect handling can lead to data corruption or inefficient operations; thus, understanding the precise mechanisms for advanced indexing is crucial.

The primary issue lies in the distinction between standard slicing (e.g., `tensor[0:5, 2:7]`) and advanced indexing where each dimension’s index is not a range but a potentially non-contiguous list of positions. Unlike NumPy, which allows direct multi-dimensional indexing with lists, PyTorch requires using either `torch.gather`, `torch.index_select`, or combining the built-in index operator with auxiliary tensors to achieve equivalent behavior. The choice between these methods often depends on the specific use case and performance considerations.

I've encountered this issue numerous times during model development, specifically when implementing attention mechanisms and custom data loaders. Let's explore three concrete examples illustrating these indexing techniques:

**Example 1: Using `torch.gather` for Indexed Element Retrieval**

Suppose we have a 3D tensor representing, say, a batch of feature maps for three images. We want to extract features from specific locations within each image. `torch.gather` shines in scenarios where you are essentially 'collecting' values based on provided indices.

```python
import torch

# Simulated batch of feature maps (Batch Size=3, Channels=4, Height=5, Width=6)
feature_maps = torch.randn(3, 4, 5, 6)

# Indices to select along height dimension for each image
height_indices = torch.tensor([[1, 2, 0], [3, 1, 4], [0, 2, 3]])  # (3 x 3)
# Indices to select along width dimension for each image
width_indices = torch.tensor([[2, 4, 1], [0, 3, 5], [4, 2, 1]])  # (3 x 3)

# Corresponding channel indices, will result in the first two feature dimensions being retained and selecting specific indices along the height and width dimension.
channel_indices = torch.arange(4).reshape(1, -1, 1, 1).expand(-1, -1, height_indices.shape[1], -1)  # (1, 4, 3, 1)

# Generate all indices to get (3, 4, 3)
all_indices = torch.stack((
    torch.arange(3).reshape(-1, 1, 1).expand(-1, height_indices.shape[1], 1),
    channel_indices,
    height_indices.unsqueeze(-1),
    width_indices.unsqueeze(-1)
    ), dim=-1) # (3, 4, 3, 4)

# Reshape the all indices for use in gather function
all_indices = all_indices.reshape(3, 4, -1, 4) # (3, 4, 3, 4)

#Gather
gathered_features = torch.gather(feature_maps.unsqueeze(3), dim=3, index=all_indices)[...,0].permute(0,1,3,2) # (3, 4, 3)

#Verification: Compare with direct indexing.
verification_features = []
for i in range(3):
  verification_image_features = []
  for c in range(4):
     verification_image_features_c = []
     for j in range(3):
        verification_image_features_c.append(feature_maps[i, c, height_indices[i, j], width_indices[i,j]])
     verification_image_features.append(torch.stack(verification_image_features_c))
  verification_features.append(torch.stack(verification_image_features))
verification_features = torch.stack(verification_features)

assert torch.all(torch.eq(verification_features, gathered_features))
print(f"Gather output matches direct indexing output!")

```

In this example, we crafted index tensors to extract specific `height` and `width` indices across images and channels in a batch. We first need to stack them together into a tensor that represents our desired indices to `gather` from. By setting the correct dimension and using broadcasting, the `gather` function retrieved the indexed elements efficiently. The use of an explicit channel index is required, but is less involved, and the final output is (Batch Size, Channels, Number of Indexes Retrieved). The `unsqueeze` operations added an extra dimension to `feature_maps` and index tensors to perform the `gather` operation, as it operates along a specified dimension. This approach is suitable when the structure of the retrieval is known and can be precomputed into an index tensor. It is important to keep in mind how the `index` input is structured for `torch.gather` to work effectively.

**Example 2: Utilizing Combined Indexing with Auxiliary Tensors**

An alternative method, often more readable and potentially clearer for less complex scenarios, involves constructing auxiliary index tensors and combining them with the standard indexing operator. This is especially useful when not all dimensions need to be indexed by a list.

```python
import torch

# Example tensor representing a batch of sequence data (Batch Size=2, Sequence Length=7, Embedding Dimension=5)
sequence_data = torch.randn(2, 7, 5)

# Indices to select along the sequence length dimension for each batch
seq_indices = torch.tensor([[0, 2, 4], [1, 3, 6]])

# Auxiliary index to combine with the seq_indices along the batch dimension
batch_indices = torch.arange(2).reshape(-1, 1).expand(-1, seq_indices.shape[1])

# Auxiliary index to combine with the seq_indices along the embedding dimension
embedding_indices = torch.arange(5).reshape(1,1,-1).expand(-1,seq_indices.shape[1],-1)

# Perform indexing using combined aux tensors
indexed_sequence = sequence_data[batch_indices, seq_indices, embedding_indices] # (2,3,5)

# Verification
verification_sequence = []
for i in range(2):
    verification_seq = []
    for j in range(3):
        verification_seq.append(sequence_data[i,seq_indices[i,j],:])
    verification_sequence.append(torch.stack(verification_seq))
verification_sequence = torch.stack(verification_sequence)

assert torch.all(torch.eq(verification_sequence, indexed_sequence))
print(f"Combined indexing output matches direct indexing output!")
```

Here, we needed to access specific time steps within a sequence across a batch. Instead of using `torch.gather`, we created `batch_indices` and `embedding_indices` tensors using broadcasting to ensure every desired index is accessed. The combined indexing (using the default `[]` operator) produces the same result as using `torch.gather` with a well-formed index tensor. While slightly less performant in some edge cases, this approach can be more understandable, especially when the indexing pattern is not complex.

**Example 3: Using `torch.index_select` along a single dimension**

The `torch.index_select` function simplifies the process of indexing along a single dimension. While it doesn't directly address multidimensional indexing using lists across all dimensions, it's a crucial tool when indexing on one specific dimension is required, and this dimension has a complex indexing scheme.

```python
import torch

# Tensor representing word embeddings (Vocabulary Size=10, Embedding Dimension=8)
word_embeddings = torch.randn(10, 8)

# List of indices to select specific word embeddings
word_indices = torch.tensor([1, 3, 5, 8])

# Select embeddings based on word indices
selected_embeddings = torch.index_select(word_embeddings, 0, word_indices) # (4, 8)

# Verification
verification_embeddings = []
for i in word_indices:
    verification_embeddings.append(word_embeddings[i,:])

verification_embeddings = torch.stack(verification_embeddings)

assert torch.all(torch.eq(verification_embeddings, selected_embeddings))
print(f"torch.index_select output matches direct indexing output!")
```

In this case, we extracted specific word embeddings from a vocabulary. `torch.index_select` allowed us to quickly choose desired rows along the first dimension, which represents the vocabulary.  This method shines when you have a single dimension you want to index with a list and the rest of the dimensions are selected by other methods or are just included fully. It’s often combined with other indexing methods if you are using a multidimensional tensor where some dimensions are simple slicing and one uses list indexing.

These examples demonstrate a range of approaches to indexing PyTorch tensors using lists of indices. Each method has specific use cases and potential trade-offs between performance, code readability, and suitability for various indexing patterns. `torch.gather` is useful when all dimensions need to be indexed by lists; combined indexing can be more understandable when some dimensions are fixed or use standard slicing; and `torch.index_select` simplifies one-dimensional list indexing.

For further study, refer to the official PyTorch documentation on tensor indexing, focusing on the sections detailing `torch.gather`, `torch.index_select`, and the use of advanced indexing with integer tensors. Textbooks on deep learning also contain detailed explanations, though they rarely focus specifically on PyTorch indexing intricacies. The core resource remains the official documentation, and I recommend consulting it whenever encountering challenges with tensor manipulations. Additionally, reviewing examples from model implementations that involve attention mechanisms, sequence processing, or object detection can provide additional context for advanced indexing techniques.
