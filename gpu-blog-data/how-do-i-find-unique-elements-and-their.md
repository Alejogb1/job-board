---
title: "How do I find unique elements and their first indices in a PyTorch tensor?"
date: "2025-01-30"
id: "how-do-i-find-unique-elements-and-their"
---
Finding unique elements within a PyTorch tensor, along with their initial indices, requires a specific combination of tensor operations that deviate from naive iterative methods often seen in list-based processing. The key fact here is that PyTorch excels at vectorized computation, meaning we should leverage its capabilities for efficiency rather than attempting explicit loops. My experience working with large spectral datasets in research has repeatedly underscored the need for this optimization, as processing millions of data points necessitates efficient tensor manipulation.

To achieve this task effectively, I typically utilize a multi-step approach. First, I use `torch.unique` to extract the distinct values present in the tensor. Second, I employ a combination of `torch.eq` (or a similar equality check) with `torch.argmax` (or a similar index retrieval) to locate the first occurrence of each unique element. Finally, I usually collect this information into a structured format (often a dictionary) for easy lookup.

Let's break this down with code examples. Imagine a 1D tensor representing a time series where we want to know the unique amplitude values and the time steps when they first occurred:

**Example 1: Basic 1D Tensor**

```python
import torch

# Sample 1D tensor
data = torch.tensor([1, 2, 3, 2, 1, 4, 3, 5, 1])

# 1. Find unique elements
unique_elements = torch.unique(data)
print(f"Unique Elements: {unique_elements}")

# 2. Find indices of first occurrence for each unique element
first_indices = []
for element in unique_elements:
    indices = (data == element).nonzero()
    first_index = indices[0, 0] if indices.numel() > 0 else -1 #Handle case where element is not found (unlikely in this process, but good practice)
    first_indices.append(first_index)
print(f"First Indices: {first_indices}")

# 3. Structure into a dictionary for easy access
result = dict(zip(unique_elements.tolist(), first_indices))
print(f"Result Dictionary: {result}")
```
In this first example, the code begins by creating a sample 1D tensor with some repeating integers. `torch.unique` efficiently identifies the distinct values in the tensor and returns them as a new tensor. The core logic resides within the loop. It iterates over each unique element. Inside, `data == element` performs an element-wise equality check returning a Boolean tensor. The `nonzero()` method then returns the indices where this equality holds. We retrieve the first index from this resulting tensor using indexing. The result is a list of the first appearing indices which can then be zipped with the unique elements into a dictionary.

Now, consider a more complex use case involving a 2D tensor. In signal processing, for instance, you might have multiple channels and wish to find the unique activation levels and corresponding time steps in a given channel.

**Example 2: 2D Tensor, Specific Channel**

```python
import torch

# Sample 2D tensor (e.g., multiple channels over time)
data_2d = torch.tensor([
    [1, 2, 3, 2, 1],
    [4, 5, 6, 5, 4],
    [1, 3, 2, 3, 1]
])

channel_index = 1  # Process the second channel (index 1)

channel_data = data_2d[channel_index]
print(f"Channel Data: {channel_data}")

unique_elements = torch.unique(channel_data)
print(f"Unique Elements: {unique_elements}")

first_indices = []
for element in unique_elements:
    indices = (channel_data == element).nonzero()
    first_index = indices[0, 0] if indices.numel() > 0 else -1
    first_indices.append(first_index)
print(f"First Indices: {first_indices}")

result = dict(zip(unique_elements.tolist(), first_indices))
print(f"Result Dictionary: {result}")

```

This second example extends the concept to a 2D tensor. I first select a single channel (`channel_index = 1`) effectively creating a 1D tensor from the row and then repeat the exact same logic as in the first example, demonstrating that this method is adaptable to tensors of varying dimensionality, as long as the selection reduces to a 1D vector for analysis of indices, allowing for a consistent method of analysis. This highlights a key strength of PyTorch, the ability to perform the same fundamental operations on tensors of different dimensions.

Finally, what if we wanted to extend this to finding the first occurrence of unique values across the entire tensor, regardless of channel. The approach would require flattening the tensor to treat it as a single sequence. This can be useful when processing image data or other multi-dimensional tensors where element order is the main concern.

**Example 3: Flattened Tensor for Global Indices**

```python
import torch

# Sample 2D tensor
data_2d = torch.tensor([
    [1, 2, 3, 2, 1],
    [4, 5, 6, 5, 4],
    [1, 3, 2, 3, 1]
])

flattened_data = data_2d.flatten()
print(f"Flattened Data: {flattened_data}")

unique_elements = torch.unique(flattened_data)
print(f"Unique Elements: {unique_elements}")

first_indices = []
for element in unique_elements:
    indices = (flattened_data == element).nonzero()
    first_index = indices[0, 0] if indices.numel() > 0 else -1
    first_indices.append(first_index)
print(f"First Indices: {first_indices}")

result = dict(zip(unique_elements.tolist(), first_indices))
print(f"Result Dictionary: {result}")
```
In this third example, the 2D tensor is flattened into a 1D tensor via the `flatten()` method, merging all its elements into one continuous sequence. The subsequent steps follow the logic of the previous examples, working now on a single, linearized dimension of the original tensor. This shows how we can manipulate data structures for flexible analysis, treating multi-dimensional data as a flat series if required.

In terms of resources, I'd recommend focusing on mastering the foundational PyTorch functions. The documentation on tensor manipulation functions like `torch.unique`, `torch.eq`, `torch.nonzero` and `torch.argmax`, should be your primary source. Beyond that, a detailed understanding of how tensor indexing works is crucial. While specific books on PyTorch are helpful, a thorough understanding of fundamental tensor concepts often trumps the need for any single comprehensive text. Practice implementing and adapting these operations on diverse, self-created datasets. Experimentation, paired with in-depth analysis of the function documentation, is the most efficient way to enhance your skills with PyTorch. Consistent use of the tools and documentation ultimately leads to greater proficiency.
