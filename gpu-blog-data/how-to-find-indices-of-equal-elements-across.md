---
title: "How to find indices of equal elements across two PyTorch tensors of different sizes?"
date: "2025-01-30"
id: "how-to-find-indices-of-equal-elements-across"
---
The core challenge in finding indices of equal elements across two PyTorch tensors of differing sizes lies in efficiently managing the combinatorial search space.  Direct comparison using broadcasting isn't feasible due to the size mismatch; a more nuanced approach involving iterative comparisons or leveraging PyTorch's advanced indexing capabilities is required. My experience optimizing similar tasks in large-scale image analysis projects highlighted the importance of minimizing computational overhead, particularly when dealing with high-dimensional tensors.

**1.  Explanation:**

The problem necessitates a strategy that systematically compares each element of the smaller tensor against all elements of the larger tensor.  A brute-force approach, while conceptually simple, suffers from significant performance penalties for larger tensors.  Therefore, I advocate for a method leveraging `torch.where` combined with broadcasting for efficient element-wise comparison and index retrieval.  This avoids explicit looping, a crucial factor for optimized performance in PyTorch.

The process can be broken down into these steps:

a. **Broadcasting:**  The smaller tensor is broadcast against the larger tensor. This implicitly replicates the smaller tensor along the relevant dimension to match the larger tensor's size.  This step requires careful consideration of the tensor dimensions to ensure correct broadcasting behavior.  Errors here commonly result in unexpected shapes or incorrect comparisons.

b. **Element-wise Comparison:** A boolean tensor is generated using `torch.eq` (element-wise equality check) based on the broadcasted tensors.  This boolean tensor indicates where elements are equal.

c. **Index Extraction:**  `torch.where` is used to extract the indices where the boolean tensor is `True`.  This returns the indices corresponding to the matching elements within the larger tensor.  These indices directly reference the positions of the elements in the larger tensor that are equivalent to elements in the smaller tensor.

d. **Index Mapping (optional):**  If the indices within the smaller tensor are also required, additional processing is needed to map these indices from the broadcasted tensor back to their original positions in the smaller tensor. This is straightforward if the broadcasting was along a single dimension. However, for multi-dimensional broadcasting, the mapping may necessitate more complex index transformations.


**2. Code Examples with Commentary:**

**Example 1: Simple 1D Case**

```python
import torch

tensor1 = torch.tensor([1, 5, 2, 8])
tensor2 = torch.tensor([2, 1, 9, 5, 3, 1, 8])

# Broadcasting tensor1 against tensor2
broadcast_tensor1 = tensor1.unsqueeze(1).expand(-1, len(tensor2))

# Element-wise comparison
comparison_tensor = torch.eq(broadcast_tensor1, tensor2)

# Index extraction using torch.where
indices_in_tensor2 = torch.where(comparison_tensor)[1]

print(f"Indices of equal elements in tensor2: {indices_in_tensor2}")
```

This example demonstrates the basic principle for 1D tensors. The `unsqueeze` and `expand` operations perform broadcasting efficiently. `torch.where` returns a tuple; we extract the second element representing indices along the second dimension (tensor2's dimension).

**Example 2: 2D Case with a Single Matching Element**

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[4, 5, 6], [1, 7, 2]])

broadcast_tensor1 = tensor1.unsqueeze(2).expand(-1, -1, len(tensor2[0]))

comparison_tensor = torch.eq(broadcast_tensor1, tensor2)
indices = torch.where(comparison_tensor)

print(f"Indices of equal elements in tensor2: row indices - {indices[0]}, column indices - {indices[1]}")
```

In this 2D scenario, broadcasting is more complex. The indices are now represented by two arrays from the output of `torch.where`, corresponding to row and column indices within `tensor2`.


**Example 3: Handling Multiple Matches and Mapping to Original Indices**

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([1, 4, 2, 1, 3, 2])

broadcast_tensor1 = tensor1.unsqueeze(1).expand(-1, len(tensor2))
comparison_tensor = torch.eq(broadcast_tensor1, tensor2)
indices_tensor2 = torch.where(comparison_tensor)[1]
indices_tensor1 = torch.where(comparison_tensor)[0]

print(f"Indices in tensor2: {indices_tensor2}")
print(f"Corresponding indices in tensor1: {indices_tensor1}")

# Processing to show the mapping
for i in range(len(indices_tensor1)):
  print(f"Element {tensor1[indices_tensor1[i]]} at index {indices_tensor1[i]} in tensor1 matches element at index {indices_tensor2[i]} in tensor2")

```

This example introduces a crucial aspect – obtaining the matching indices within the smaller tensor (`tensor1`). The loop at the end clearly demonstrates the mapping between the indices of matching elements in both tensors.

**3. Resource Recommendations:**

I recommend consulting the official PyTorch documentation for detailed explanations of tensor manipulation functions like broadcasting, `torch.eq`, and `torch.where`.   Thoroughly reviewing documentation on tensor reshaping and advanced indexing techniques is invaluable for mastering efficient PyTorch programming.  Finally, I suggest studying materials focusing on performance optimization strategies within the PyTorch ecosystem, as this aspect becomes increasingly crucial when processing large datasets.  This will equip you to identify and eliminate performance bottlenecks effectively.
