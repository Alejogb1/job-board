---
title: "What PyTorch Geometric datasets are available?"
date: "2024-12-23"
id: "what-pytorch-geometric-datasets-are-available"
---

Alright, let's dive into the world of PyTorch Geometric datasets. It's a topic I've spent a fair amount of time navigating, especially during a project a couple of years back where we were working on graph-based anomaly detection. We needed very specific datasets, and the standard ones weren’t cutting it, leading to quite a deep investigation into the available options.

So, first things first, PyTorch Geometric (PyG) provides a robust collection of datasets, neatly categorized for different types of graph learning tasks. These datasets aren't just raw data dumps; they’re thoughtfully structured `torch_geometric.data.Data` objects, facilitating easier model development and experimentation. When we consider categories, we can generally talk about node classification, graph classification, and link prediction datasets as the main players.

For **node classification**, you're often looking at datasets where each node in a graph has a class label, and your goal is to predict these labels. Datasets commonly used here include Cora, Citeseer, and Pubmed, which are citation networks where nodes represent publications and edges represent citations. These are relatively small but excellent for quick prototyping and benchmarking. Then there’s the planetoid datasets, and the `ogbn-arxiv` dataset, which is part of the Open Graph Benchmark. The `ogbn-arxiv` is substantially larger and showcases the scale you might face in real-world scenarios. They come pre-processed and are ready to be used directly after download. We used Cora for our initial experiments and found its size was great for rapid iteration. It’s worth noting that these datasets sometimes also contain feature information associated with each node.

Then, **graph classification** datasets differ slightly. Instead of classifying individual nodes, here, we classify entire graphs. Think of molecular properties prediction, where each molecule represents a graph, and the task involves predicting certain characteristics. Datasets like the `MUTAG`, `PROTEINS`, and `NCI1` from TUDataset are typical here, often involving smaller graphs. There is also a whole category for the social network graphs and some benchmark datasets like `REDDIT-BINARY`. These graphs often vary in size and structure and provide a diverse testbed for graph neural networks. The `ogbg-molhiv` dataset from the Open Graph Benchmark represents a larger and more complex graph classification problem. I remember struggling with the `MUTAG` dataset because some of the graph samples required complex padding techniques for batch processing in the initial versions of PyG we used.

Finally, in **link prediction** we want to predict missing links in a graph. Datasets here are typically structured as graphs with a specific set of observed edges, and the goal is to infer the unobserved edges. Often, the data is split into training edges (observed) and test edges (unobserved), and the models should infer the missing test edges. Examples of such datasets include `Cora`, `Citeseer` and other citation network variants, where we can mask edges to create the link prediction problem. Social network datasets like Facebook or similar can also be used. These are particularly useful if you're looking into applications like friend recommendation in social media. In our anomaly detection project, we tried both node and link-based approaches, finding that link prediction performed better for specific network types.

Now, it's not all rosy, and you might encounter a situation where the built-in datasets aren't enough. For instance, if you need specific graph structures or data formats, you have to either pre-process existing data or define custom datasets. We hit this wall with our project because we were using time-evolving graphs that weren’t explicitly supported initially. This meant we had to create custom `torch.utils.data.Dataset` classes that could ingest our custom graph formats. In doing this, PyG’s core functionalities and data handling functions made the process more manageable.

Now, let's get to some examples. Here are a few code snippets demonstrating how to load some of these datasets:

**Example 1: Loading Cora for Node Classification**

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/cora', name='Cora')
data = dataset[0]

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Edge index (example): {data.edge_index[:,:5]}')
```

This snippet will download the Cora dataset if it's not already in `/tmp/cora`, and then it will print some essential details about the graph. Notice how quickly we can access the graph structure through `data.edge_index`, which holds the adjacency lists, and node feature matrix. This streamlined access to preprocessed data is a key benefit of PyG.

**Example 2: Loading MUTAG for Graph Classification**

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/mutag', name='MUTAG')
print(f'Number of graphs: {len(dataset)}')
data = dataset[0]
print(f'Number of nodes in the first graph: {data.num_nodes}')
print(f'Number of edges in the first graph: {data.num_edges}')
print(f'Number of features in the first graph: {data.num_node_features}')
print(f'Target label of first graph : {data.y}')
```

Here, we're loading the `MUTAG` dataset and looking at the properties of a single graph. The key difference to note is that here, the dataset represents a collection of graphs rather than a single large graph, and we access each sample using bracket `dataset[0]` indexing. The y property gives us the class for that specific graph.

**Example 3: Custom Dataset for Link Prediction**

```python
import torch
from torch_geometric.data import Data, Dataset
import numpy as np

class CustomLinkDataset(Dataset):
    def __init__(self, root, data_list, transform=None):
       self.data_list = data_list
       super().__init__(root, transform)

    def len(self):
       return len(self.data_list)

    def get(self, idx):
       return self.data_list[idx]

# Generate random dummy edge and feature data
num_nodes = 100
num_features = 20
edge_index = torch.randint(0, num_nodes, (2, 500))
node_features = torch.randn(num_nodes, num_features)

# create dummy data for the link prediction problem
data_obj = Data(x=node_features, edge_index=edge_index)

# Here you should usually include splitting logic for your link prediction edges, but I will skip this here for simplicity.
# create train and test masks for links
data_list = [data_obj, data_obj] # make copy to pretend that we have more data samples
custom_dataset = CustomLinkDataset(root='/tmp/custom_link_dataset',data_list=data_list)
data = custom_dataset[0]

print(f'Number of nodes in the data: {data.num_nodes}')
print(f'Number of edges in the data: {data.num_edges}')
print(f'Number of features in the data: {data.num_node_features}')
print(f'Edge index (example): {data.edge_index[:,:5]}')
```

This last example provides a snippet to illustrate how one would go about creating a custom dataset. This becomes handy, as I mentioned, when you have specialized data formats or need to implement specific train/test splits.

For further reading and deeper understanding, I'd highly recommend the following: For a foundational understanding of graph neural networks, check out *Graph Representation Learning* by William Hamilton. For a more PyG-centric dive, exploring the official documentation and example notebooks on the project's GitHub repository is invaluable. Additionally, papers from the Open Graph Benchmark (OGB) are crucial for understanding larger and more complex benchmarks and datasets. Specifically, I’d recommend “Benchmarking Graph Neural Networks” published in JMLR, which describes some of these popular graph benchmark datasets in detail.

In summary, PyTorch Geometric comes with a diverse set of graph datasets, covering many common graph learning scenarios. However, as with many other situations in software engineering, sometimes you have to get your hands dirty and craft some custom solutions. Having a grasp on these datasets and the ability to extend them provides a solid base for your graph learning endeavors.
