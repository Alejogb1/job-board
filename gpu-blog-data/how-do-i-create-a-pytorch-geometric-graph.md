---
title: "How do I create a PyTorch Geometric graph neural network dataset?"
date: "2025-01-30"
id: "how-do-i-create-a-pytorch-geometric-graph"
---
PyTorch Geometric (PyG) datasets require a specific data structure, differing significantly from typical image or text datasets. Successfully constructing a PyG dataset hinges on understanding how graph data is represented and manipulated within the library. My experience creating several graph-based models, particularly for social network analysis and molecule property prediction, has highlighted the critical role of correctly formatted data, a step often overlooked initially. This involves encoding nodes, edges, and their associated attributes (features) into a `torch_geometric.data.Data` object, and then organizing these objects into a collection that PyG’s dataloaders can efficiently process.

The core concept revolves around representing graphs as a set of nodes and edges, where each node may have features and each edge may be associated with an attribute or simply represented by its source and target nodes. The `torch_geometric.data.Data` object is the foundational element for this. It typically stores the following key components as PyTorch tensors:

*   `x`: This tensor holds node features. Its shape is usually `[num_nodes, num_node_features]`. Each row represents the features of a single node.
*   `edge_index`: This tensor defines the connectivity of the graph. It has a shape of `[2, num_edges]` where the first row lists the source nodes of the edges and the second row lists the target nodes. It's a crucial component that allows PyG to perform message passing between connected nodes.
*   `edge_attr` (optional):  If edges have associated features, this tensor stores them. Its shape is typically `[num_edges, num_edge_features]`.
*   `y`: This tensor holds the target variable(s) for the graph. This could be node-level labels (shape `[num_nodes]`), graph-level labels (shape `[1]`), or edge-level labels (shape `[num_edges]`). The dimensionality depends on the specific problem.

Creating a PyG dataset involves two key steps: generating individual `Data` objects from your raw data and then creating a suitable way to manage the collection of these `Data` objects, typically using `torch_geometric.data.Dataset`. We can either subclass the `torch_geometric.data.Dataset` class to handle custom data loading logic, or use helper functions for pre-existing data structures.

**Code Example 1: Creating a simple graph with manual tensors**

This demonstrates the creation of a `Data` object for a simple undirected graph with three nodes and two edges, where each node has two features, and edges do not have associated features.

```python
import torch
from torch_geometric.data import Data

# Node features: three nodes, each with two features
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)

# Edge indices (undirected): 0->1, 1->2, and therefore implicitly 1->0 and 2->1.
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

# Node labels
y = torch.tensor([0, 1, 0], dtype=torch.long)

# Construct the Data object
data = Data(x=x, edge_index=edge_index, y=y)

print(data)
print(data.num_nodes)
print(data.num_edges)
```

This code first initializes the node features, then the edge indices, and node labels as tensors. The `edge_index` explicitly defines edge direction. Though the edge list only specifies 0->1 and 1->2, the resulting graph, by default, will have 0->1, 1->0, 1->2 and 2->1 because the `Data` object assumes an undirected graph and automatically creates the reverse connections. This would have to be handled differently if you were handling directed graph edge lists. Note that `num_edges` here reports on the number of logical edges (including reverse connections).

**Code Example 2: Creating a graph with edge features**

This builds upon the previous example, adding edge features. We now assign a 2-dimensional feature vector to each edge, to demonstrate the usage of `edge_attr`.

```python
import torch
from torch_geometric.data import Data

# Node features
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)

# Edge indices (directed to emphasize edge attributes)
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

# Edge features. Note, each edge in edge_index now has a respective row of features here.
edge_attr = torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float)

# Node labels
y = torch.tensor([0, 1, 0], dtype=torch.long)

# Construct the Data object including edge_attr
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

print(data)
print(data.edge_attr)
print(data.num_edges)

```

Here, we add a tensor called `edge_attr` with the same number of rows as the number of edges defined by `edge_index`. It's important that `edge_attr`'s dimensions match the shape of `edge_index` based on the directionality and number of edges represented. `num_edges` still reports on logical edges, as explained before.

**Code Example 3: Creating a custom Dataset class**

For real-world applications, it’s often necessary to handle more complex data loading. The `torch_geometric.data.Dataset` class provides the framework for this. The following implements a basic example:

```python
import torch
from torch_geometric.data import Dataset, Data
import os

class MyGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_files = [f for f in os.listdir(self.raw_dir) if f.endswith(".pt")]

    @property
    def raw_file_names(self):
      # Return a list of the files you expect to be in 'raw_dir', used internally.
      return self.data_files

    @property
    def processed_file_names(self):
        #Used to control if a new dataset needs to be processed, typically returns a list.
      return ['data.pt']

    def download(self):
       #This is the function where your data should be downloaded, if you're
       # not working with pre-downloaded data.
        pass


    def process(self):
       #This is where you load in your raw files, generate the data objects
       # and save it down as your single preprocessed file.
      data_list = []

      for data_file in self.raw_file_names:
        # Simulate data loading from individual .pt files
        data = torch.load(os.path.join(self.raw_dir, data_file))
        data_list.append(data)
      torch.save(self.collate(data_list),
                    os.path.join(self.processed_dir, 'data.pt'))

    def len(self):
        return 1

    def get(self, idx):
       # This is where you load a data object. We're returning the whole thing because the
       # 'process' function collates all data into a single output file, as part of this example.
      data_file = torch.load(os.path.join(self.processed_dir, 'data.pt'))
      return data_file


# Example Usage:
# Assumes a raw data directory with .pt files already present

root_dir = 'graph_data' # This is your root directory, inside you'd have 'raw' and 'processed'
raw_dir = os.path.join(root_dir, 'raw')
os.makedirs(raw_dir, exist_ok=True) #Makes the raw directory, if not already present

# Generate three simple sample data files:
for i in range(3):
  x = torch.rand(10, 3)
  edge_index = torch.randint(0, 10, (2, 20))
  y = torch.randint(0, 2, (1,))
  torch.save(Data(x=x, edge_index=edge_index, y=y), os.path.join(raw_dir, f'data_{i}.pt'))

# Instantiate dataset and load data
dataset = MyGraphDataset(root_dir)

# Print data example
print(dataset[0])
```

This example demonstrates how to create a custom dataset. The dataset is designed to load individual graphs from `.pt` files stored in the `raw_dir` and process them, saving a single `.pt` file in the `processed_dir`. Note that the example implements a process function that concatenates and saves the raw dataset as a single unit. This is a simplified example, as it's not always appropriate for large datasets, but the function names and method signatures shown are those expected by PyG datasets.  The `download()` function, usually used for fetching data from an external source is not used here. The `len()` function is also a minimal implementation; a production `len()` function would be responsible for telling the dataloader how many data points it can expect in total. Likewise, the `get()` method needs to properly handle indexing and returning individual `Data` object from potentially a list of Data objects stored as part of the class.  This example returns the single, large, loaded `Data` object instead.

For further learning, I suggest examining the following resources. The official PyTorch Geometric documentation provides an extensive guide to the available functions and the theory of graph neural networks. Also, consider exploring publicly available PyG datasets like Planetoid, QM7, and QM9, which provide practical examples of handling various graph data types. Reading tutorials on graph neural networks will also prove helpful, often these tutorials will have their own guides to using and implementing the correct data formats for different tasks. I found that exploring existing open-source GNN implementations for tasks similar to my own helped clarify many aspects of PyG dataset creation that were initially confusing. Finally, experimentation and iterative refinement remain crucial to mastering the correct use of PyG data handling.
