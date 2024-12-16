---
title: "How to batch graphs in PyTorch Geometric?"
date: "2024-12-16"
id: "how-to-batch-graphs-in-pytorch-geometric"
---

Let's tackle this one; batching graphs in PyTorch Geometric can seem daunting initially, but it’s quite manageable once you understand the core concepts. It's something I’ve worked extensively with, particularly when developing large-scale graph neural networks for recommendation systems a few years back. We were dealing with millions of user-item interactions, represented as graphs, and efficient batching was crucial to avoid running into memory constraints and accelerate training.

At its heart, batching graphs isn't as straightforward as batching tensors because graphs have variable structure; they don't necessarily have a uniform number of nodes or edges. We can't just stack them along a new dimension. PyTorch Geometric, however, handles this elegantly by creating a large, disconnected "super-graph." Each original graph becomes a sub-graph within this larger structure. This approach allows for parallel computation across the sub-graphs using the same underlying message-passing framework.

The key is understanding how PyTorch Geometric's `Data` object and its associated functionalities enable this batching process. The `Data` object, you see, essentially bundles all information about a single graph: the node features (`x`), the edge indices (`edge_index`), the edge attributes (`edge_attr`, if any), node labels (`y`), and other custom fields.

When batching, we're actually creating a new `Data` object that represents the combined graphs. Crucially, the `edge_index` is modified by adding offset node indices to ensure that we are not unintentionally connecting nodes across different graphs. Similarly, the node features and the labels are concatenated. The batch index (`batch`) is also added, which is a tensor of the same length as the number of nodes. It indicates to which graph each node belongs.

Now let me illustrate this with a few concrete code examples.

**Example 1: Basic Batching with Randomly Generated Graphs**

This first example demonstrates how to batch a couple of randomly generated graph data using `torch_geometric.data.DataLoader`.

```python
import torch
from torch_geometric.data import Data, DataLoader

def generate_random_graph(num_nodes, num_edges):
    x = torch.randn(num_nodes, 16) # Example node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges)) # Example edge indices
    y = torch.randint(0, 2, (num_nodes,)) # Example node labels
    return Data(x=x, edge_index=edge_index, y=y)


graphs = [
    generate_random_graph(num_nodes=10, num_edges=20),
    generate_random_graph(num_nodes=15, num_edges=30),
    generate_random_graph(num_nodes=8, num_edges=15)
]

loader = DataLoader(graphs, batch_size=3) # Batch all three graphs
batch = next(iter(loader))

print(f"Batched Node Features Shape: {batch.x.shape}")
print(f"Batched Edge Index Shape: {batch.edge_index.shape}")
print(f"Batch Indicator Shape: {batch.batch.shape}")

# Output would be something along the lines of:
# Batched Node Features Shape: torch.Size([33, 16])
# Batched Edge Index Shape: torch.Size([2, 65])
# Batch Indicator Shape: torch.Size([33])
```

As you can see here, the `DataLoader` efficiently handles the batching process. The node features are concatenated, the edge indices are shifted, and the batch indicator, often referred to as `batch` is also generated. Each graph's nodes now exist within the same data structures but remain segregated. This is fundamental for processing with GNN models.

**Example 2: Handling Edge Attributes**

Let's now consider a scenario involving edge attributes. Edge attributes might capture aspects like the type or weight of a relation between nodes. The batching logic extends smoothly to include these.

```python
import torch
from torch_geometric.data import Data, DataLoader

def generate_random_graph_with_edge_attr(num_nodes, num_edges):
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4) # Example edge attributes
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

graphs = [
    generate_random_graph_with_edge_attr(num_nodes=10, num_edges=20),
    generate_random_graph_with_edge_attr(num_nodes=15, num_edges=30)
]

loader = DataLoader(graphs, batch_size=2)
batch = next(iter(loader))

print(f"Batched Node Features Shape: {batch.x.shape}")
print(f"Batched Edge Index Shape: {batch.edge_index.shape}")
print(f"Batched Edge Attributes Shape: {batch.edge_attr.shape}")
print(f"Batch Indicator Shape: {batch.batch.shape}")

# Output would be something along the lines of:
# Batched Node Features Shape: torch.Size([25, 16])
# Batched Edge Index Shape: torch.Size([2, 50])
# Batched Edge Attributes Shape: torch.Size([50, 4])
# Batch Indicator Shape: torch.Size([25])
```

Here, `edge_attr` tensors are concatenated just like the node features and their structure remains in parallel, similar to `x` and `edge_index`. PyTorch Geometric automatically manages alignment during the batching process for us, allowing for easy processing in graph neural network layers.

**Example 3: Batched Graph Processing in a GNN**

Finally, let’s see how batched data are typically used within a GNN, demonstrating that we can apply a message passing mechanism effectively using batching. Here we will create a dummy model but the main intention is to demonstrate the usage of the batch variable `batch`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def generate_random_graph(num_nodes, num_edges):
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


graphs = [
    generate_random_graph(num_nodes=10, num_edges=20),
    generate_random_graph(num_nodes=15, num_edges=30)
]

loader = DataLoader(graphs, batch_size=2)
batch = next(iter(loader))

model = SimpleGCN(in_channels=16, hidden_channels=32, out_channels=2)
output = model(batch.x, batch.edge_index, batch.batch)

print(f"Output Shape: {output.shape}") # Output Shape: torch.Size([25, 2])

# The output tensor is batch-aware, meaning you can now use the `batch` variable to perform tasks specific to each graph within the batch.
# You can, for example, compute graph-level features using the following code:

import torch_geometric.utils as utils
graph_level_features = utils.scatter(output, batch, reduce="mean")
print(f"Graph-Level Features Shape: {graph_level_features.shape}") # torch.Size([2, 2])
```

Here, we define a basic two-layer GCN which takes batched `x`, `edge_index`, and the `batch` indicator. Using the `batch` variable allows us to aggregate the node representations of each graph in the batch individually. You can observe that the model's output size is directly related to the overall number of nodes, not the number of graphs in the batch. This is the important property that allows you to implement GNNs on multiple graphs at the same time. Also note how we utilized the helper function `scatter` which automatically computes the mean per graph.

For diving deeper into the theoretical underpinnings of graph neural networks, I would recommend the book "Graph Representation Learning" by William L. Hamilton. Additionally, “Deep Learning on Graphs” edited by Yao Ma and Jiliang Tang provides excellent insights into specific applications and more advanced techniques. For a more practical, implementation-focused understanding of PyTorch Geometric, the official documentation itself is a very good resource, as well as the various research papers by the authors behind the library.

In summary, batching graphs in PyTorch Geometric relies on transforming separate graph structures into a single, disconnected graph with additional metadata to track each original sub-graph. This approach enables parallel computation and easy integration within existing GNN workflows. The `DataLoader` simplifies the batching of `Data` objects by automatically combining the graph information and generating batch indices. By understanding these core ideas and the accompanying utility functions, you can efficiently process graph data and build complex, scalable GNN models.
