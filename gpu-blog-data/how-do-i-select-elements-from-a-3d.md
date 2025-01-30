---
title: "How do I select elements from a 3D PyTorch tensor using an array of indices along the second dimension?"
date: "2025-01-30"
id: "how-do-i-select-elements-from-a-3d"
---
The challenge of selecting elements from a PyTorch tensor using an array of indices, specifically along the second dimension in a 3D tensor, is a common requirement in various deep learning scenarios, particularly those involving batch processing and sequence manipulation. I've encountered this frequently when working with recurrent neural network outputs and attention mechanisms. The crux of the issue lies in understanding how PyTorch's indexing mechanism interacts with multi-dimensional tensors and how to leverage `torch.gather` for efficient element retrieval.

In a 3D PyTorch tensor of shape `(batch_size, sequence_length, feature_size)`, the second dimension typically represents a sequence of vectors, often representing words in a sentence or time steps in a signal. The goal is to select specific elements from this dimension using a separate tensor containing the desired indices for each batch element. This operation cannot be achieved through simple indexing because the indexing along the second dimension needs to vary for each element in the first (batch) dimension.

The correct approach is to use the `torch.gather` function. This function performs an index-based lookup and allows us to select elements from a source tensor based on a set of indices. Crucially, it operates along a specified dimension. For this task, the dimension will be the second (index 1), as we need to select elements along the sequence length. However, the key to using `torch.gather` effectively in this scenario lies in preparing the `indices` tensor to match the expected format by `torch.gather`.

The `indices` tensor we need to create should have the same number of dimensions as the source tensor, with shapes identical in all dimensions except the `dim` dimension where we perform the gathering. The indices along the `dim` dimension will determine which elements are selected from the source tensor. In our case, this means that if the source tensor has shape `(batch_size, sequence_length, feature_size)`, the indices tensor will have the shape `(batch_size, 1, feature_size) ` or `(batch_size, index_length, feature_size)`, where `index_length` specifies the length of the selection for each batch.

Let me illustrate this concept with some code examples.

**Example 1: Selecting Single Elements from Each Sequence**

Suppose we have a 3D tensor representing batch embeddings and we want to select one element from the second dimension for each batch.

```python
import torch

batch_size = 3
sequence_length = 5
feature_size = 4

# Generate random tensor
source_tensor = torch.randn(batch_size, sequence_length, feature_size)

# Define indices (random)
indices = torch.randint(0, sequence_length, (batch_size, 1, 1)) # Indices for second dimension
indices = indices.expand(-1,-1,feature_size)

# Perform gather operation
output_tensor = torch.gather(source_tensor, 1, indices)

print("Source Tensor shape:", source_tensor.shape)
print("Indices Tensor shape:", indices.shape)
print("Output Tensor shape:", output_tensor.shape)
print("Output Tensor:", output_tensor)

```

In this code, we create a random 3D tensor. The `indices` tensor is crucial. It has the same batch size as the source tensor. It initially holds the index we want to select for each batch using `torch.randint`. We then expand it to have the same `feature_size` along the last dimension. The `torch.gather` function selects the elements based on the indices along the second dimension (dim=1) and returns a tensor of shape `(batch_size, 1, feature_size)`. Each batch element is now represented by the feature at the location described in `indices`.

**Example 2: Selecting Multiple Elements from Each Sequence**

Now, consider a scenario where we want to extract multiple elements from each sequence using an index tensor of length greater than 1.

```python
import torch

batch_size = 2
sequence_length = 7
feature_size = 3
index_length = 3

# Generate random tensor
source_tensor = torch.randn(batch_size, sequence_length, feature_size)

# Define indices (random)
indices = torch.randint(0, sequence_length, (batch_size, index_length, 1))
indices = indices.expand(-1,-1, feature_size)


# Perform gather operation
output_tensor = torch.gather(source_tensor, 1, indices)

print("Source Tensor shape:", source_tensor.shape)
print("Indices Tensor shape:", indices.shape)
print("Output Tensor shape:", output_tensor.shape)
print("Output Tensor:", output_tensor)
```

Here, we select `index_length` number of elements from the sequence. The change is in the `indices` tensor's shape, which is now `(batch_size, index_length, feature_size)`. Consequently, each batch element in the output tensor consists of `index_length` feature vectors, gathered along the sequence. Notice again the expansion of indices along the feature size dimension.

**Example 3: Selecting Elements with Different Length for Each Batch**

The flexibility of `torch.gather` allows for more intricate selection. Suppose we want to select different numbers of elements for each batch and these lengths are known. To perform this, we can create a mask that allows us to select elements only where the index is valid for that particular sequence length.

```python
import torch

batch_size = 3
sequence_length = 5
feature_size = 2

# Generate random tensor
source_tensor = torch.randn(batch_size, sequence_length, feature_size)

# Define different lengths for each sequence
seq_lengths = torch.tensor([2, 4, 3])

# Create a max_index_length for the indices tensor, taking the max value of seq_lengths
max_index_length = seq_lengths.max()

# Generate indices with a shape of (batch_size, max_index_length, feature_size)
indices = torch.zeros(batch_size, max_index_length, 1, dtype=torch.long)
for batch_idx, length in enumerate(seq_lengths):
    indices[batch_idx, :length, 0] = torch.arange(length)

indices = indices.expand(-1,-1, feature_size)


# Perform gather operation
output_tensor = torch.gather(source_tensor, 1, indices)

#Create a mask for the result based on seq_lengths to remove padding
mask = torch.arange(max_index_length).unsqueeze(0) < seq_lengths.unsqueeze(1)

#Apply mask to the output_tensor
output_tensor = output_tensor * mask.unsqueeze(-1)
print("Source Tensor shape:", source_tensor.shape)
print("Indices Tensor shape:", indices.shape)
print("Output Tensor shape:", output_tensor.shape)
print("Output Tensor with Mask:", output_tensor)


```

This example shows how `torch.gather` can work with different `index_length` values for different batches, by using a zero-mask with the sequence lengths and ensuring that the `indices` are valid by generating values using `torch.arange`, which represent the actual indices we want to select, up to the length specified for each batch.

In summary, `torch.gather` is the function you need for advanced selection of elements along a specific dimension of a tensor. The key to effective usage lies in understanding how to correctly format the index tensor to match the tensor you are gathering from. It's a core tool for any PyTorch user involved in sequence modeling or any task requiring flexible tensor indexing. For deeper understanding, I recommend exploring the official PyTorch documentation which explains `torch.gather` in detail. Also, looking at tutorials focusing on sequence modeling and attention mechanisms would provide real-world context for this operation. Reviewing research papers that discuss advanced indexing operations in deep learning could offer additional insights. Reading through blog posts and tutorials that show specific implementations of sequence processing with PyTorch can solidify your understanding.
