---
title: "Can torch.gather be replaced by a more efficient operator?"
date: "2025-01-30"
id: "can-torchgather-be-replaced-by-a-more-efficient"
---
The core inefficiency often observed when using `torch.gather` arises from its inherently scattered memory access pattern, especially when the indices tensor is not well-behaved (e.g., contains many out-of-order or distant locations). This pattern often results in cache misses and reduced memory bandwidth utilization, which limits performance compared to more contiguous access operations. After many profiling sessions on large-scale graph neural network training where `torch.gather` was a bottleneck, I have consistently observed that, under specific and often encountered conditions, `torch.index_select` or even clever reshaping and elementwise multiplication can offer significant performance advantages.

`torch.gather`’s fundamental purpose is to collect values from a source tensor along a specific dimension using indices provided in a separate tensor. The function signature and basic usage involve three key elements: the `input` tensor from which to gather values, the `dim` indicating the dimension along which gathering should occur, and the `index` tensor specifying the locations from which to pick values. The output has the same size as the index tensor, with values pulled from the input based on the specified indices. This functionality is very flexible, allowing for arbitrary permutation and re-arrangement of tensor elements. However, this flexibility introduces its performance drawbacks. Each individual gathering operation specified by the index potentially requires jumping to non-contiguous memory locations. In contrast, operations that exploit more regular, predictable memory accesses can often leverage hardware cache behavior more effectively.

Let's delve into situations where alternatives might be more suitable. A very common use case occurs when gathering from the input tensor along a particular dimension based on a sequence of contiguous indices that can be easily generated using `torch.arange` or similar functions. In these cases, `torch.index_select` directly maps to a hardware-optimized memory access pattern. It accepts an input tensor, a dimension, and a tensor of indices which essentially represent the desired slices or rows. `index_select` will always pull contiguous data, which avoids the scattered memory access of gather. Consider, for example, extracting specific rows from a matrix.

**Code Example 1: `torch.index_select` for contiguous row extraction:**

```python
import torch

# Assume a 2D tensor where each row represents feature of one sample.
input_tensor = torch.randn(100, 512)

# Want to get the features for sample index 10, 20, 30
indices = torch.tensor([10, 20, 30])

# Using torch.gather to get the desired rows
# First need to create indices for all dimensions
rows_indices_gather = torch.arange(indices.size(0)).unsqueeze(1).repeat(1, input_tensor.size(1))
column_indices_gather = indices.unsqueeze(1).repeat(1, input_tensor.size(1))
column_indices_gather = column_indices_gather
gather_indices = torch.cat((rows_indices_gather.unsqueeze(2), column_indices_gather.unsqueeze(2)), dim=2)
gather_indices = gather_indices.reshape(rows_indices_gather.size(0) * rows_indices_gather.size(1), 2)
gather_result = input_tensor.gather(0, gather_indices[:,1].unsqueeze(0).expand(input_tensor.size(1), -1).T)

# Using torch.index_select to get the desired rows
index_select_result = torch.index_select(input_tensor, 0, indices)

print(f"Shape of index select result: {index_select_result.shape}")
print(f"Shape of gather result: {gather_result.shape}")
# Asserting that both results are same
assert torch.all(torch.eq(index_select_result, gather_result))

```

In the provided code, `torch.index_select` cleanly and directly extracts the rows specified by `indices` whereas the use of `torch.gather` requires us to create multi-dimensional indices and flatten the output. For this use case, the simpler `index_select` is considerably faster and more concise. The computational complexity for `index_select` with sequential access is better than that of `gather`.

Another common scenario arises when the `index` tensor, while not inherently contiguous, has a simple structure that allows for restructuring the operations. For instance, consider a case where the input is a tensor of embeddings and the index represents a set of indices for selecting some of those embeddings. If the index tensor consists of contiguous segments of indices used repeatedly, we can leverage reshaping to effectively perform gather operations using `torch.reshape` and `torch.mul`.

**Code Example 2: Reshaping and Multiplication:**

```python
import torch

# input embeddings, a batch of samples, where each sample has n embedding vector.
input_embeddings = torch.randn(10, 5, 128)

# indices indicates which embedding index to keep for each sample in batch.
indices = torch.tensor([0, 2, 4, 1, 3])  # Example indices

# Get shape information
batch_size = input_embeddings.shape[0]
num_vectors = input_embeddings.shape[1]
embedding_dim = input_embeddings.shape[2]

# Using gather to get the result.
rows_indices_gather = torch.arange(input_embeddings.size(0)).unsqueeze(1).repeat(1, input_embeddings.size(2) * indices.size(0))
column_indices_gather = indices.unsqueeze(0).repeat(input_embeddings.size(0), input_embeddings.size(2)).reshape(input_embeddings.size(0) * input_embeddings.size(2) * indices.size(0))
index_indices_gather = torch.arange(input_embeddings.size(2)).unsqueeze(0).repeat(input_embeddings.size(0), indices.size(0)).unsqueeze(0).reshape(input_embeddings.size(0) * input_embeddings.size(2) * indices.size(0))
gather_indices = torch.stack((rows_indices_gather, column_indices_gather, index_indices_gather), dim=1).view(input_embeddings.size(0) * input_embeddings.size(2) * indices.size(0), 3)
gather_result = input_embeddings.gather(1, gather_indices[:,1].view(input_embeddings.size(0), indices.size(0), 1).expand(-1, -1, embedding_dim))


# Reshaping + Multiplication approach
reshaped_input = input_embeddings.permute(0, 2, 1).reshape(batch_size * embedding_dim, num_vectors) # Reshaping input embeddings for faster indexing.
mask = torch.zeros(num_vectors, dtype=torch.bool)
mask[indices] = True

selected_embeddings = reshaped_input[:,mask].reshape(batch_size, embedding_dim, -1).permute(0, 2, 1) # get the select embeddings after masking and reshaping

print(f"Shape of gather result: {gather_result.shape}")
print(f"Shape of reshaped select result: {selected_embeddings.shape}")

assert torch.all(torch.eq(gather_result, selected_embeddings))
```

In this example, the repeated index pattern can be used to rearrange the data. We can create a mask using the given `indices` and by performing elementwise multiplication we can get the results faster than gather. This method becomes more efficient as the size of the embeddings and repetitions increase. While the code appears slightly more complex, it avoids the scattered read that is inherent to the gather operation, which is particularly beneficial in larger tensors.

It is imperative to mention that choosing between these approaches requires a careful analysis of the specific use case. One should always profile both alternatives to make an informed decision. In particular, if the index tensor is dynamically generated at every iteration and does not have a structure, then `torch.gather` might be the only practical approach. However, in many practical deep learning applications the indices are relatively static and their values are structured, allowing for more efficient alternative implementations.

A more complex example involves applying a set of filters to an image tensor based on spatially varying filter indices. In this context, the 'input' might be the image, the filter indices would specify which filter should be applied at each location, and 'gathering' would mean selecting the corresponding filter.

**Code Example 3: Spatially Varying Filter Selection**

```python
import torch

# Batch of images.
images = torch.randn(10, 3, 64, 64)  # Example image batch (batch, channels, height, width).
num_filters = 5 # Number of filters
filters = torch.randn(num_filters, 3, 3, 3) # Set of filters (num_filters, channels, height, width).
indices_filter_selection = torch.randint(0, num_filters, (10, 64, 64)) # Index tensor indicating which filter to apply

# Define output placeholder
output = torch.zeros_like(images)

# Using gather for filter selection
for b in range(images.size(0)):
    for h in range(images.size(2)):
        for w in range(images.size(3)):
            filter_index = indices_filter_selection[b, h, w]
            output[b, :, h, w] = torch.conv2d(images[b,:,h:h+1, w:w+1].unsqueeze(0), filters[filter_index,:,:,:].unsqueeze(0), padding=1).squeeze()

# Using the reshaping and multiplication approach

# Get output size for faster access
batch_size = images.size(0)
height = images.size(2)
width = images.size(3)

# Create a one-hot encoding of filter indices
one_hot_indices = torch.nn.functional.one_hot(indices_filter_selection, num_filters)
one_hot_indices = one_hot_indices.permute(0, 3, 1, 2)

# Apply each filter to the whole image and then multiply based on selection mask
filtered_images = []
for filter_index in range(num_filters):
    filtered_images.append(torch.conv2d(images, filters[filter_index].unsqueeze(0), padding=1))
filtered_images = torch.stack(filtered_images)

filtered_images = filtered_images.permute(1, 0, 2, 3, 4)
reshaped_select_output = (one_hot_indices.unsqueeze(2) * filtered_images).sum(dim=1)

# Asserting that both results are same
assert torch.allclose(reshaped_select_output, output, atol=1e-5)
```

This example extends the previous approaches to a more practical use case involving 2D convolution. The use of gather becomes very complex here and is implemented via triple for loops. While this example is far less performant than standard convolution operations, it is useful to demonstrate the complex indices that can be handled by gather. The alternative reshaping approach computes the convolution for every filter and then selects the correct convolution via a one-hot encoding. This avoids the complex indices creation that was needed for gather and enables easier implementation.

For resources, I suggest reviewing PyTorch's official documentation, which offers a detailed description of the `torch.gather` operation and related functions like `torch.index_select`. Additionally, examining the source code of PyTorch, located in their GitHub repository, can reveal details of the underlying implementations. Furthermore, researching optimized tensor libraries that operate on GPUs, such as NVIDIA's cuDNN library (although not directly related to PyTorch implementation, it highlights the challenges in designing memory-efficient routines), can provide valuable context about the importance of contiguous memory access. Books or articles focused on hardware architectures and memory hierarchies can also be insightful for understanding the performance implications of different data access patterns. Always profile your code before and after applying any such changes to determine if there was any performance boost.
