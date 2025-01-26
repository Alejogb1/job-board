---
title: "Why am I getting a 'RuntimeError: expected scalar type Long but found Float' in PyTorch Geometric?"
date: "2025-01-26"
id: "why-am-i-getting-a-runtimeerror-expected-scalar-type-long-but-found-float-in-pytorch-geometric"
---

In PyTorch Geometric (PyG), encountering a `RuntimeError: expected scalar type Long but found Float` typically stems from an attempt to use floating-point data where integer values are explicitly required, particularly for graph indices. These indices, representing node connections within the graph structure, must be of type `torch.long`, as they serve as memory addresses or lookups into tensor dimensions. Failing to enforce this type constraint results in the runtime error you’re observing. My experience debugging similar issues on a large-scale social network graph processing project frequently reveals this to be a common point of confusion, particularly for developers transitioning from standard PyTorch.

The core issue centers around PyG’s graph representation. PyG models typically operate on data represented as graphs composed of nodes and edges. Edge information is maintained using a tensor referred to as `edge_index`. This tensor stores pairs of node indices, denoting source and destination nodes for each edge. For example, in a directed graph where node 0 is connected to node 1, the `edge_index` tensor would include the column `[0, 1]`. PyG internally uses these indices as pointers to access node features or compute messages along the graph structure. These lookups necessitate integer type values. Representing these indices as floating point types like `float` creates an incompatibility with the underlying memory addressing and tensor operations. It is critical that these indices always have the `torch.long` dtype.

The error commonly arises because data loaded from external sources, or preprocessed manually, may not explicitly enforce integer type for graph indices. This is frequently observed after preprocessing data with libraries such as Pandas or NumPy, where numerical values are often defaulted to floating point types. When these converted values are directly utilized in creating a `Data` object for processing with a PyG model, a type mismatch occurs. PyG does not automatically cast non-integer tensor values to `torch.long`. This is by design, requiring the user to explicitly manage data types to ensure efficient memory usage and expected model behavior.

Furthermore, this error is not specific to just the `edge_index` tensor. It can occur in other tensor fields if used in similar indexing contexts within PyG models. Some custom layer implementations might require specific input tensors to be of long type to correctly perform indexing or lookup, leading to similar type related runtime errors.

To address the error, the appropriate tensor must be typecasted to `torch.long`. This can be achieved using `tensor.long()` or `tensor.to(torch.long)` operations directly before passing the tensor into PyG. This explicitly casts the tensor's dtype before it’s consumed by PyG.

Let's look at a few examples demonstrating this:

**Example 1: Edge Index Creation From a List of Tuples**

Consider an example where edge indices are initially stored in a list of tuples, where the indices might inadvertently be interpreted as float type numbers.

```python
import torch
from torch_geometric.data import Data

# Incorrect: indices as floats
edge_list = [(0.0, 1.0), (1.0, 2.0), (2.0, 0.0)]
edge_index = torch.tensor(edge_list).t()

# Correct: Explicitly cast to long
edge_index_long = edge_index.long()

# Example node features
x = torch.randn(3, 16)
data = Data(x=x, edge_index=edge_index_long)
print(data) # Now the data object is created without issues.
```

In this scenario, the initial creation of `edge_index` results in a tensor with a `float` datatype, which is incompatible. By applying `edge_index.long()`, we cast the tensor to the correct integer type, resolving the `RuntimeError`. The `.t()` function transposes the tensor to be of correct shape for edge index.

**Example 2: Reading Indices From NumPy Array**

Data loaded from NumPy arrays might also inherit float type, leading to type inconsistencies.

```python
import torch
from torch_geometric.data import Data
import numpy as np

# Incorrect: NumPy array with float type
edge_array = np.array([[0, 1], [1, 2], [2, 0]], dtype=float)
edge_index = torch.from_numpy(edge_array).t()

# Correct: Explicitly cast to long
edge_index_long = edge_index.long()

# Example node features
x = torch.randn(3, 16)
data = Data(x=x, edge_index=edge_index_long)
print(data)
```

Similar to Example 1, the NumPy array `edge_array` creates a tensor with floating point numbers. The use of `.long()` converts these to the correct type.

**Example 3: Using `torch.to` Function**

The `.to` function can also cast to the correct long dtype with greater flexibility, particularly useful if needing to target a specific device like `cuda`.

```python
import torch
from torch_geometric.data import Data

# Incorrect: Implicit float type
edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.float).t()

# Correct: Explicitly cast to long using .to
edge_index_long = edge_index.to(torch.long)

# Example node features
x = torch.randn(3, 16)
data = Data(x=x, edge_index=edge_index_long)
print(data)
```

Here, we demonstrate explicit casting of the edge index to a tensor of type `torch.long`. This method offers better control over tensor conversion if GPU acceleration is being used.

In all three examples, explicitly casting to `torch.long` rectifies the `RuntimeError`, allowing correct instantiation of the PyG `Data` object.

For further learning on working with PyG data, I recommend exploring the official PyTorch Geometric documentation. Particular attention should be paid to the data handling section of the documentation, which contains an extensive guide to the various data structures and their usage. Additionally, reviewing code examples provided in the official PyG repository is invaluable for understanding correct data preprocessing methods. Lastly, a thorough understanding of PyTorch tensor datatypes and casting operations is beneficial in avoiding similar issues in the future. These resources can provide the necessary foundation to address common type related issues and develop robust graph learning applications with PyG.
