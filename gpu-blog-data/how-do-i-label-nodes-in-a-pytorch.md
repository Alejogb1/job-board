---
title: "How do I label nodes in a PyTorch Geometric graph?"
date: "2025-01-30"
id: "how-do-i-label-nodes-in-a-pytorch"
---
Nodes in a PyTorch Geometric graph are not directly labeled with strings or text; instead, they are represented by indices, typically integers, and any categorical or feature-based labeling is achieved through associated data tensors. I've encountered this issue frequently while working on graph neural networks for molecular structure analysis, where nodes might represent atoms, and their features encode various chemical properties or types. This often requires mapping symbolic labels to numerical representations before feeding data to the model.

The `torch_geometric.data.Data` object, which forms the fundamental representation of a graph in PyTorch Geometric, stores node features in the `x` attribute. This attribute is a tensor where each row corresponds to a node and each column represents a feature. Thus, "labeling" isn't a direct assignment to the node itself but rather encoding that label as a feature or using it to look up a corresponding numerical feature. The key here is understanding that PyTorch Geometric operates on numerical data and expects nodes to be identifiable through their position in the feature tensor or their numerical index.

Let's examine how to incorporate node labels. First, if the labels are categorical, a common practice is to use one-hot encoding. Suppose we have a graph where each node belongs to one of three categories (A, B, C). We need to transform these symbolic categories into a numerical format. We assign each category an integer index (e.g., A->0, B->1, C->2). Then, we create a one-hot encoded vector for each node using that integer index.

Here’s a Python code example using PyTorch Geometric. Assume we have three nodes, with labels 'A', 'B', 'A' respectively and want to create our data object.

```python
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Define edges (assuming undirected, for simplicity)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long).t().contiguous()

# Node labels (string format for demonstration)
node_labels = ['A', 'B', 'A']

# Mapping from string labels to integer indices
label_to_index = {'A': 0, 'B': 1}

# Convert string labels to integer indices
indexed_labels = [label_to_index[label] for label in node_labels]

# One-hot encoding using torch.nn.functional.one_hot
num_categories = len(label_to_index)
one_hot_encoded = torch.nn.functional.one_hot(torch.tensor(indexed_labels), num_classes=num_categories).float()

# Create PyTorch Geometric Data object
data = Data(x=one_hot_encoded, edge_index=edge_index)

print("Node features (one-hot encoded):")
print(data.x)
print("Edge indices:")
print(data.edge_index)
```
In this example, `indexed_labels` converts the strings to integers which is then used to create one-hot encoded node features. The output of data.x is a tensor where each row has a value of 1 for the index corresponding to the label and zero otherwise. This resulting one-hot encoded representation is then used as our `x` node features. The edges are defined in the edge index matrix and used to create a standard PyTorch Geometric graph structure.

Sometimes, our labeling might not be purely categorical; instead, it could correspond to feature vectors that we want to use to represent each node. This is common when dealing with pre-computed embeddings or node attributes that carry some meaning. For instance, each node might have an associated vector describing its properties or its location in an embedding space.

Let’s create a node feature vector, instead of categorical variables, and put them in the Data object.

```python
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# Edge indices (again undirected for simplicity)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long).t().contiguous()

# Assume these represent node features: (x,y) coordinate location
node_features = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]], dtype=torch.float)


# Create PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index)

print("Node features (numerical):")
print(data.x)
print("Edge indices:")
print(data.edge_index)
```

Here, `node_features` is a tensor of shape (number of nodes, feature dimensions). The important aspect is that there's a one-to-one correspondence between the rows of the feature tensor and the nodes, implicitly "labeling" each node with its corresponding feature vector. The `Data` object directly takes these features and associates them with node positions based on the tensor row indices.

Another crucial point is handling the conversion from a textual representation, or other symbolic label, to the numerical representation as shown in the first code example. If our labels come from a specific dataset, there likely is already a mapping implemented, especially if we are using a standard dataset from `torch_geometric.datasets`. We can look at the dataset's implementation to see how it handles labeling. However, if we are building our own dataset, this is a step that we must implement ourselves, and we must ensure there is an appropriate mapping to numerical values prior to creating our `Data` objects. If this labeling is time consuming, it is often useful to precompute and store the numerical data for a dataset, so it does not need to be recomputed at each training epoch.

Finally, if the node labels are dynamic, such that they change during the course of a training or evaluation run, we can adjust the `x` attribute on the `Data` object directly, as demonstrated in the code below. This allows us to create feature tensors on the fly, as long as the shape of the tensor matches the expectations, and to update those tensors with new or adjusted node labels as needed.

```python
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Edge indices (undirected)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long).t().contiguous()

# Initial node feature tensors - these will be updated
node_features = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]], dtype=torch.float)

# Create the Data object
data = Data(x=node_features, edge_index=edge_index)

# Function to update node features
def update_node_features(data, new_node_features):
    data.x = new_node_features
    return data

# Demonstrate updating the node features
new_node_features = torch.tensor([[0.1, 0.2],
                            [0.3, 0.4],
                            [0.5, 0.6]], dtype=torch.float)

updated_data = update_node_features(data,new_node_features)
print("Original node features:")
print(node_features)
print("Updated node features:")
print(updated_data.x)
```
This example demonstrates how you can dynamically update the node features within a `Data` object during training. The `update_node_features` function updates the `data.x` with new feature tensors. It's crucial to remember that any update must preserve the tensor's expected shape.

For further learning, I'd recommend exploring the official PyTorch Geometric documentation, particularly the sections on data handling and dataset creation. The tutorials on the PyTorch Geometric website offer great hands-on experience. Also, reading through some of the research papers that use graph neural networks, especially within your own domain of interest, can provide excellent examples of how different researchers have handled node labeling for their projects. Finally, studying the source code of existing graph datasets is a great way to understand established methods for labeling nodes, features, and labels.
