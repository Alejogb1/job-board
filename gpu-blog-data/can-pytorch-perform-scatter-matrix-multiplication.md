---
title: "Can PyTorch perform scatter matrix multiplication?"
date: "2025-01-30"
id: "can-pytorch-perform-scatter-matrix-multiplication"
---
PyTorch does not offer a dedicated, single operation specifically named "scatter matrix multiplication." However, the functionality can be effectively achieved using combinations of existing PyTorch operations, primarily `torch.scatter_add_` or `torch.scatter_reduce_`, in conjunction with broadcasting and tensor manipulation. I've frequently encountered this requirement when implementing custom graph neural network layers, where aggregating feature vectors based on node connectivity is essential and avoids the memory inefficiencies of dense matrix multiplication when a graph is sparse.

The core issue revolves around interpreting what "scatter matrix multiplication" conceptually means. Standard matrix multiplication computes a dot product between rows of the first matrix and columns of the second matrix. "Scatter matrix multiplication," by contrast, implies that the result is accumulated not at a fixed row-column location, but according to an index tensor guiding where contributions from the input matrices should be summed or otherwise combined. A naive, direct translation of matrix multiplication into a scatter operation quickly falls apart without the necessary indexing. Therefore, simulating this process requires careful management of tensor dimensions and indexing.

In essence, we're dealing with a form of generalized reduction or aggregation. Instead of standard matrix products, we’re taking products that are then routed to specific locations within an output tensor, potentially with repetition of the output location, leading to accumulation (addition, in the most common case). Consequently, we must explicitly create the correct indices and the intermediate results before using PyTorch’s scatter operations.

My usual approach begins with a mapping tensor, denoted often as the `index` tensor, which dictates the output locations for the results of the element-wise multiplication between the input tensors, or a suitable representation of the inputs. Let's consider a typical case. Assume we have input matrix `A` of shape `(N, F)` representing node features where `N` is the number of nodes and `F` is feature dimensionality. Another matrix, `B` of shape `(E, 2)`, the edge index tensor, contains indices of the edges within the graph, where each row of the tensor represents an edge `(source_node, target_node)`. Our goal is to compute results of `A[source_node] * A[target_node]` for all edges and sum/reduce those results onto each node. This effectively multiplies feature vectors of connected nodes, and aggregates them for each node. The resultant tensor would then have the shape `(N, F)` – the same as the input feature matrix `A`, but with aggregated values based on node connectivity.

Let's clarify with some practical code examples:

**Example 1: Summation based scatter with source and destination features**

```python
import torch

# Example setup
N = 5  # Number of nodes
F = 3  # Feature dimensionality
E = 7 # Number of edges

A = torch.randn(N, F)  # Node feature matrix
edge_index = torch.tensor([[0, 1], [1, 2], [2, 0], [3, 4], [0, 3], [1, 4], [2,3]], dtype=torch.long).t() # Edge indices (source, destination)

# Extract source and destination indices
source_nodes = edge_index[0] # shape (E)
target_nodes = edge_index[1] # shape (E)

# Gather node features for source and destination nodes
source_features = A[source_nodes] # shape (E, F)
target_features = A[target_nodes] # shape (E, F)

# Multiply the features together on an element-wise basis, producing aggregated values for each edge
edge_multiplied_features = source_features * target_features # shape (E, F)

# Initialize a tensor for the output accumulation
output = torch.zeros(N, F) # shape (N, F)

# scatter add the results, aggregating for each node
output = output.scatter_add_(0, target_nodes.unsqueeze(1).expand(-1,F), edge_multiplied_features)

print("Output:\n", output)
```

In this example, `edge_index` defines connections.  The core operation is element-wise multiplication of source and target features for each edge, producing `edge_multiplied_features`. Then `scatter_add_` routes these resulting multiplied features to the correct output rows (specified by `target_nodes`), summing the values at each node. The `.unsqueeze(1).expand(-1,F)` expands `target_nodes` to have dimensions identical to the `edge_multiplied_features`, which is needed as the scatter function works on a source tensor which has the same number of dimensions as the target tensor, and the `index` has dimensions of 1 less than the source tensor. This example demonstrates how a product is distributed to its respective location on the output tensor.

**Example 2: Scalar Multiplication Followed by Scatter**

```python
import torch

# Example setup
N = 4
F = 2
E = 6

A = torch.randn(N, F)
edge_index = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2], [0, 2], [1, 3]], dtype=torch.long).t()
scalars = torch.randn(E,1)  # Edge specific scalar multiplication factor

source_nodes = edge_index[0]
target_nodes = edge_index[1]


source_features = A[source_nodes]
target_features = A[target_nodes]

# Perform scalar multiplication with the target features
weighted_target_features = target_features * scalars

# Initialize an output tensor
output = torch.zeros(N, F)

# Scatter_add weighted features to the output
output = output.scatter_add_(0, source_nodes.unsqueeze(1).expand(-1,F), weighted_target_features)

print("Output:\n", output)
```

Here, instead of element-wise multiplication of *two* feature vectors, we have a scalar value for each edge which we are using as a weight on the *target* node features before scattering them onto the *source* node features.  This showcases that the operation before scatter does not need to be limited to feature multiplication, but can be any valid operation producing a tensor compatible with the scatter. The scalar weights allow for more complex weighting strategies over the edges.

**Example 3: Using `scatter_reduce_` for Mean Aggregation**

```python
import torch

# Example setup
N = 6
F = 4
E = 10

A = torch.randn(N, F)
edge_index = torch.randint(0, N, (2, E)) # Random edge indices

source_nodes = edge_index[0]
target_nodes = edge_index[1]

source_features = A[source_nodes]
target_features = A[target_nodes]

# Compute edge features as an aggregation of the source and target features
edge_features = source_features + target_features

# Initialize output tensor
output = torch.zeros(N, F)

# Use scatter_reduce_ with 'mean' to aggregate edge features
output = output.scatter_reduce_(0, target_nodes.unsqueeze(1).expand(-1,F), edge_features, reduce='mean')

print("Output:\n", output)
```

This example demonstrates the use of `torch.scatter_reduce_`, allowing for not just summation but also other aggregation techniques such as the mean or max. It also demonstrates how the scattered values need not be the direct product of input tensors but can involve complex calculations before being routed to the output via the scatter operation. Here, we take the mean of the features of connected nodes.

These examples highlight the flexibility of PyTorch's scatter operations for implementing "scatter matrix multiplication"-like behaviours. The key is to construct the appropriate index tensor (e.g. using `edge_index`), perform any necessary element-wise operations or transformations to be scattered, and use `torch.scatter_add_` or `torch.scatter_reduce_` to accumulate the results into a target tensor based on the specified indices. The flexibility in what operation occurs *before* the scatter operation gives considerable power to use cases which require custom aggregation strategies.

When performing scatter operations, I find it very useful to thoroughly understand the dimensionality of the source, index and target tensors. Errors often arise from incorrect dimensional alignment. Debugging usually involves printing tensor shapes at each step, visually inspecting the index tensor to ensure it points to the correct locations, and creating smaller test cases to isolate the problem. Careful examination of the `torch.scatter_*` function signatures is essential, paying close attention to how indices are interpreted along different dimensions.

For learning more about these operations, I would recommend exploring the official PyTorch documentation specifically on `torch.scatter_add_`, `torch.scatter_reduce_` and `torch.gather`. Experimenting with small example tensors, as shown here, is the most effective learning strategy. Additionally, examining how graph neural network libraries utilize these operations for message passing provides excellent insight into how to apply these techniques within larger model contexts. Several online tutorials and blog posts also offer detailed explanations of these often confusing operations. Exploring these resources will solidify understanding, while careful attention to the dimensionalities will assist in any implementation.
