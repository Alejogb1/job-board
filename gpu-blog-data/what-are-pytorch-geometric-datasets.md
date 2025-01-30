---
title: "What are PyTorch Geometric datasets?"
date: "2025-01-30"
id: "what-are-pytorch-geometric-datasets"
---
PyTorch Geometric (PyG) datasets, fundamentally, are specialized PyTorch `Dataset` objects tailored for representing graph-structured data. My experience in developing graph neural network models for biological networks has highlighted the critical importance of understanding their specific structure. Unlike standard image or text datasets, PyG datasets do not simply provide data points; they provide interconnected *graphs*. These graphs are described using attributes like node features, edge connections, and edge features. This means that interacting with PyG datasets requires a shift from thinking about data as independent samples to thinking about data as nodes and edges within a relational structure.

At their core, PyG datasets inherit from PyTorch's abstract `Dataset` class. This inheritance allows them to be seamlessly integrated into PyTorch's training pipelines, such as through `DataLoader` for batched learning. What distinguishes them is their internal organization that is geared toward graph processing. Instead of storing individual images or sequences, a PyG dataset stores a collection of `Data` objects (and sometimes `HeteroData` objects for heterogeneous graphs). Each `Data` object encapsulates the representation of a single graph or a batch of graphs. A typical `Data` object contains attributes such as `x`, `edge_index`, `edge_attr`, and `y`. `x` is the node feature matrix (size: [num_nodes, num_node_features]), `edge_index` represents the graph connectivity in COO format (size: [2, num_edges]), `edge_attr` is the edge feature matrix (size: [num_edges, num_edge_features]), and `y` holds the target values (size: [num_nodes] for node classification or [1] for graph classification).

PyG also provides a multitude of pre-built datasets spanning different application domains, such as social network analysis, molecular biology, and knowledge graphs. These datasets handle the complexities of loading, pre-processing, and managing graph data. Instead of manually creating `Data` objects from raw graph files, utilizing these pre-built datasets significantly streamlines development. Furthermore, they often have associated benchmarks that enable researchers to directly compare model performance. One practical distinction is that PyG Datasets typically do not load all data into memory at initialization. Instead, they access data on an as-needed basis when called by the `DataLoader`. This lazy loading mechanism is essential when dealing with large graphs or large collections of graphs.

The abstract `Dataset` class in PyTorch defines two methods that every subclass must implement: `__len__` and `__getitem__`. For PyG datasets, `__len__` typically returns the number of graphs (or the number of data objects) contained in the dataset. `__getitem__(idx)` returns a single `Data` object at a given index. This consistent interface allows users to easily iterate through the graphs in the dataset during training and evaluation.

Now, let’s illustrate these concepts with some code examples.

**Example 1: Loading and Inspecting a Pre-Built Dataset**

Here, we will demonstrate how to load the Cora dataset, a standard citation network benchmark. The code then accesses a single graph and inspects its attributes.

```python
import torch
from torch_geometric.datasets import Planetoid

# Load the Cora dataset
dataset = Planetoid(root='./tmp/cora', name='Cora')

# The dataset contains a list of Data objects, in this case, just one.
data = dataset[0]

# Print the shape of node features, adjacency index and target labels
print("Shape of node features:", data.x.shape)
print("Shape of edge indices:", data.edge_index.shape)
print("Shape of target labels:", data.y.shape)
print("Number of graphs", len(dataset)) # Check the number of graphs.

# Print sample of edge_indices, and node features
print("First 5 edge indices", data.edge_index[:, :5])
print("First 5 node features", data.x[:5, :])

# Inspect the type of the data
print("Type of data:", type(data))

# Check if graph is undirected
print("Undirected graph:", data.is_undirected())

# Check if training, validation, and test masks are present:
print("Train mask is available:", 'train_mask' in data)
print("Validation mask is available:", 'val_mask' in data)
print("Test mask is available:", 'test_mask' in data)
```

In this example, `Planetoid` loads the preprocessed data into a `Dataset` object. The first (and only, in this case) graph within the dataset is accessed via indexing, and its essential attributes are printed to the console. The training, validation and test masks are used to indicate which nodes should be included in a given stage of a machine learning training loop.

**Example 2: Creating a Custom Dataset**

This example shows how to construct a custom dataset from manually defined graph data.

```python
import torch
from torch_geometric.data import Dataset, Data

class CustomGraphDataset(Dataset):
    def __init__(self, graphs, transform=None):
        super().__init__(None, transform)
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# Create data for graph 1
x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
y1 = torch.tensor([0, 1, 0], dtype=torch.long)
data1 = Data(x=x1, edge_index=edge_index1, y=y1)

# Create data for graph 2
x2 = torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float)
edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
y2 = torch.tensor([1, 0], dtype=torch.long)
data2 = Data(x=x2, edge_index=edge_index2, y=y2)

# Aggregate data objects into custom dataset
graphs = [data1, data2]
custom_dataset = CustomGraphDataset(graphs)

# Access and print data
for i, data in enumerate(custom_dataset):
  print(f"Graph {i+1}: Node Features {data.x}, Edge indices {data.edge_index}, Labels {data.y}")
```

Here, we define a custom class inheriting from `Dataset` that receives a list of `Data` objects. The `__len__` method returns the number of graphs, and `__getitem__` returns a single `Data` object. We instantiate this custom dataset using manually constructed `Data` objects.

**Example 3: Using a Dataset with DataLoader**

This example demonstrates how to use the previously created custom dataset with PyTorch's `DataLoader`. This enables batch processing for model training.

```python
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Dataset, Data

class CustomGraphDataset(Dataset):
    def __init__(self, graphs, transform=None):
        super().__init__(None, transform)
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# Create data for graph 1
x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
y1 = torch.tensor([0, 1, 0], dtype=torch.long)
data1 = Data(x=x1, edge_index=edge_index1, y=y1)

# Create data for graph 2
x2 = torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float)
edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
y2 = torch.tensor([1, 0], dtype=torch.long)
data2 = Data(x=x2, edge_index=edge_index2, y=y2)

# Aggregate data objects into custom dataset
graphs = [data1, data2]
custom_dataset = CustomGraphDataset(graphs)

# Initialize DataLoader with batch size 2 (all data in one batch here)
dataloader = DataLoader(custom_dataset, batch_size=2)

# Iterate through batches
for batch in dataloader:
    print("Batch:", batch)
    print("Batch Node features:", batch.x)
    print("Batch edge indices:", batch.edge_index)
    print("Batch labels:", batch.y)

    # Perform operations on the batch if needed, e.g., pass it to a GNN
    # This section omitted for brevity.
```

Here, `DataLoader` produces mini-batches from our custom dataset. Notably, the individual `Data` objects in the batch are automatically transformed into a `Batch` object by PyG. This batch object contains all node features stacked together into a single tensor, all edge indexes concatenated together and with the node indices shifted to avoid overlap between graphs, etc. This representation can then be directly used in graph neural network models.

For further exploration, I would recommend consulting the official PyTorch Geometric documentation, which provides a comprehensive overview of all available datasets and their specific attributes. “Graph Representation Learning” by Hamilton provides good theoretical background. A deep dive into the source code of the different `torch_geometric.datasets` modules can help understand internal mechanisms, and the tutorial section of the official repository, offers a more application-oriented perspective on how datasets fit into complete pipelines. These resources present both fundamental knowledge and practical advice for effectively utilizing PyG datasets in real-world projects.
