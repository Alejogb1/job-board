---
title: "How to get the top values and their indices from a PyTorch tensor?"
date: "2025-01-30"
id: "how-to-get-the-top-values-and-their"
---
Achieving efficient retrieval of top values and their corresponding indices from a PyTorch tensor is a common task in various machine learning workflows, ranging from identifying the most influential features to selecting the most probable class in a classification problem. The operation necessitates careful consideration of both performance and the specific requirements of the task. I’ve personally encountered this during my work developing a real-time anomaly detection system where performance was critical and selecting a specified number of highest-scoring events was a recurring requirement. PyTorch provides several mechanisms for accomplishing this, but understanding their nuances and suitability to different tensor shapes and data types is important.

The core functionality resides within the `torch.topk` function. It allows extracting the 'k' largest elements from a given tensor, and importantly, also provides their respective indices within the original tensor. This functionality extends beyond 1D tensors and works for multi-dimensional tensors as well, with a key parameter, `dim`, controlling the dimension along which the top values are selected.

The `topk` function's primary parameters are: the input tensor, `k` (number of top elements to retrieve), and `dim` (dimension along which to sort). Additionally, an `largest` parameter (boolean) dictates whether to extract the largest or smallest values and a `sorted` parameter (boolean) controls whether the result is sorted. While typically the largest elements are desired, it can be very helpful to select the smallest.  The function returns a tuple containing two tensors: the top values themselves, and their corresponding indices. The indices are relative to the dimension specified in the `dim` argument.

For example, if you have a tensor representing probabilities of different classes, a typical use case is to select the top-k classes with the highest probabilities. Consider the tensor below.

```python
import torch

# Example 1: 1D tensor
scores = torch.tensor([0.1, 0.8, 0.3, 0.9, 0.5])
k = 3
top_values, top_indices = torch.topk(scores, k)

print("Original Tensor:", scores)
print("Top Values:", top_values)
print("Top Indices:", top_indices)
```

In this first example, a simple 1D tensor `scores` is created. The `torch.topk` function then extracts the three largest values (`k=3`) along with their indices from the original tensor. The output will reveal that the values `[0.9, 0.8, 0.5]` are the top three, and their indices within the original tensor are `[3, 1, 4]`.  This illustrates the most basic usage of `torch.topk`.  Note that indices are 0-based.

The flexibility of `torch.topk` becomes more evident when dealing with multi-dimensional tensors. When working with batch processing, one might desire to get the top elements from each batch separately, along a specific dimension. Consider the 2D tensor in the example below which could represent scores across multiple instances (rows) and multiple categories (columns).

```python
# Example 2: 2D Tensor
scores_2d = torch.tensor([[0.1, 0.8, 0.3, 0.9],
                         [0.7, 0.2, 0.95, 0.4],
                         [0.5, 0.6, 0.3, 0.75]])
k = 2
dim = 1  # Sort along the columns (per row)
top_values_2d, top_indices_2d = torch.topk(scores_2d, k, dim=dim)

print("\nOriginal 2D Tensor:\n", scores_2d)
print("Top Values along dimension 1:\n", top_values_2d)
print("Top Indices along dimension 1:\n", top_indices_2d)
```

In this case, we apply `torch.topk` to the 2D tensor, using `dim=1`. This means we find the top two values in *each row* of the matrix. The resulting `top_values_2d` contains the top two values in each row: `[[0.9, 0.8], [0.95, 0.7], [0.75, 0.6]]` and `top_indices_2d` contains their corresponding indices: `[[3, 1], [2, 0], [3, 1]]`.  The dimension `dim` plays a critical role here. Choosing `dim=0` would have given the top values within each column, as opposed to each row.  This was incredibly important when working on batches of spectrogram data in my earlier research, where the dimension needed careful selection depending on whether I wanted to get the largest frequency components within a single time slice (along dim=0) or across time-slices for a single frequency bin (along dim=1).

The third example will extend this with the `largest` and `sorted` parameters.  Let's get the smallest values, rather than the largest, and leave the result unsorted.

```python
# Example 3: 2D Tensor, Smallest, Unsorted
scores_2d = torch.tensor([[0.1, 0.8, 0.3, 0.9],
                         [0.7, 0.2, 0.95, 0.4],
                         [0.5, 0.6, 0.3, 0.75]])
k = 2
dim = 1  # Sort along the columns (per row)
top_values_2d_smallest, top_indices_2d_smallest = torch.topk(scores_2d, k, dim=dim, largest=False, sorted=False)

print("\nOriginal 2D Tensor:\n", scores_2d)
print("Smallest Values along dimension 1, Unsorted:\n", top_values_2d_smallest)
print("Smallest Indices along dimension 1, Unsorted:\n", top_indices_2d_smallest)
```
Here we have extracted the smallest values using `largest=False`, and we also have made the result unsorted with `sorted=False`. The return will include the smallest two values of each row, but these will be returned in the original order, rather than sorted by value. Note, it is common to sort these values, but this shows that is a configuration option of `torch.topk`

While `torch.topk` provides the functionality to extract both the values and indices, there are alternative approaches. In some specific cases where only the indices are needed, `torch.argsort` can be used followed by slicing to get the top 'k' indices. However, this will require additional work if both the values and indices are needed, and it generally has inferior performance to using `torch.topk` directly. Using `torch.topk` is also preferable because it more clearly communicates the programmer's intention to get the largest elements in the code, and it also performs this operation in an optimized manner.

For further exploration, consult the official PyTorch documentation on `torch.topk`.  Also, the books "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann, as well as "Programming PyTorch for Deep Learning" by Ian Pointer offer good explanations and further examples of using PyTorch tensors and functions.  Additionally, the "PyTorch Recipes" documentation provides good example use cases of functions, including `torch.topk`. Finally, consider working through some practical examples from open-source projects on platforms such as GitHub. Experimenting with different tensor shapes and exploring the nuances of the `dim` parameter will reinforce understanding of the function's capabilities. These methods have helped me when debugging issues or trying to discover how to optimize my model's processing.
