---
title: "How can PyTorch Geometric load a series of graphs from a .csv file?"
date: "2025-01-30"
id: "how-can-pytorch-geometric-load-a-series-of"
---
PyTorch Geometric (PyG) requires data to be represented as `torch_geometric.data.Data` objects, each encapsulating the node features, adjacency information (edge indices), and optionally, edge features. When loading graph datasets from a CSV file, we often need to convert tabular data, representing individual graph properties, into this structured format. This process, while not directly supported by built-in PyG functions for CSVs, can be achieved through custom data loading logic.

My experience with large-scale graph analysis on biological networks involved frequent data loading from CSV exports. Initially, the challenge stemmed from the lack of a single standard for CSV-based graph representations. Different research groups encoded graph structures in various ways. Subsequently, I developed a reusable framework to handle several CSV variations and consistently load the data into `Data` objects. This response will address common scenarios encountered in this process.

**Core Process**

The primary challenge is to translate the information encoded within a CSV into the `edge_index`, `x`, and potentially `edge_attr` attributes of the `Data` object. Typically, a CSV file might represent a graph using:

1.  **Edge List:** Each row defines a single edge, with columns indicating source and target node identifiers. Node features might be included in separate columns (or a different CSV file) or embedded within the CSV.
2.  **Adjacency List (less common):** Each row represents a node, and its columns list adjacent nodes, potentially alongside feature data.
3. **Node-centric Features:** Each row describes a single node and its features, while a separate file (or set of files) details the graph topology.

The following describes a general procedure that can be adapted to different CSV formats:

1. **CSV Reading and Parsing:** Use the Python `csv` module or `pandas` to read and parse the CSV file. This transforms tabular data into a list of rows or a DataFrame.
2. **Node and Edge Extraction:** Identify the columns representing the source, target, and node features. If present, edge features are also identified.
3. **Node ID Mapping:** Convert node identifiers into a contiguous numerical range (starting from 0). This ensures they can be used as array indices. A mapping (dictionary) between node identifiers and numerical IDs should be created.
4. **Edge Index Creation:** Construct the `edge_index` tensor. It is a 2 x N_edges tensor, where the first row contains the source node IDs and the second row contains the target node IDs. Each column defines a single edge. These IDs must be the numerical node IDs from the prior mapping.
5. **Node Feature Extraction:** Extract the relevant columns to create a `x` tensor, which is an N_nodes x F_node tensor, where each row contains the features of one node. If the features are not present, a zero matrix or identity matrix may be used. The features should be converted into a PyTorch tensor (typically float).
6. **Edge Feature Extraction:** If the CSV contains edge features, create an `edge_attr` tensor, which is an N_edges x F_edge tensor. Convert them to a PyTorch tensor.
7. **Data Object Creation:** Create a `torch_geometric.data.Data` object using the `edge_index`, `x`, and `edge_attr`. This is the PyG compatible representation of the graph.

**Code Examples**

The following examples will cover three common scenarios encountered while loading graph data from CSVs.

**Example 1: Edge List CSV with Node Features Embedded**

This example assumes a CSV file `edges.csv` formatted as follows:
```
source,target,feature_1,feature_2
node_a,node_b,0.1,0.5
node_b,node_c,0.3,0.7
node_a,node_c,0.2,0.6
...
```
Here, the first two columns define an edge, and columns 3 and 4 define node features. The source and target ids are non-numeric.

```python
import torch
from torch_geometric.data import Data
import pandas as pd

def load_graph_from_edge_list_with_features(csv_path):
    df = pd.read_csv(csv_path)
    node_set = set()
    for _, row in df.iterrows():
        node_set.add(row['source'])
        node_set.add(row['target'])

    node_to_id = {node: idx for idx, node in enumerate(node_set)}
    num_nodes = len(node_to_id)

    edge_index = []
    node_features = {}

    for _, row in df.iterrows():
        source_id = node_to_id[row['source']]
        target_id = node_to_id[row['target']]
        edge_index.append([source_id, target_id])

        node_features[source_id] = [row['feature_1'], row['feature_2']]
        node_features[target_id] = [row['feature_1'], row['feature_2']] # assuming feature values are repeated


    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    x = torch.tensor([node_features[i] for i in range(num_nodes)], dtype=torch.float) # ensure that features are ordered
    
    data = Data(x=x, edge_index=edge_index)
    return data

# Example usage
if __name__ == '__main__':
    # create dummy csv
    import csv
    with open('edges.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source','target','feature_1', 'feature_2'])
        writer.writerow(['node_a','node_b',0.1,0.5])
        writer.writerow(['node_b','node_c',0.3,0.7])
        writer.writerow(['node_a','node_c',0.2,0.6])


    data = load_graph_from_edge_list_with_features('edges.csv')
    print(data)
```
This example uses `pandas` for CSV reading and iterates over the rows, generating the `edge_index` and extracting node features.  The `node_to_id` dictionary creates the required mapping and is used to generate the tensors with numerical ids for the `edge_index` and the node features.

**Example 2: Edge List with Separate Node Feature File**

Assume we have `edges.csv` and `nodes.csv`:

`edges.csv`:
```
source,target
node_a,node_b
node_b,node_c
node_a,node_c
...
```
`nodes.csv`:
```
node_id,feature_1,feature_2
node_a,0.1,0.5
node_b,0.3,0.7
node_c,0.2,0.6
...
```

```python
import torch
from torch_geometric.data import Data
import pandas as pd

def load_graph_from_edge_list_and_node_features(edge_csv_path, node_csv_path):
    edge_df = pd.read_csv(edge_csv_path)
    node_df = pd.read_csv(node_csv_path)
    
    node_to_id = {node: idx for idx, node in enumerate(node_df['node_id'])}
    num_nodes = len(node_to_id)
    
    edge_index = []
    for _, row in edge_df.iterrows():
        source_id = node_to_id[row['source']]
        target_id = node_to_id[row['target']]
        edge_index.append([source_id, target_id])

    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    node_features = node_df[['feature_1', 'feature_2']].values
    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data

# Example Usage
if __name__ == '__main__':
    # create dummy csv files
    import csv
    with open('edges.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source','target'])
        writer.writerow(['node_a','node_b'])
        writer.writerow(['node_b','node_c'])
        writer.writerow(['node_a','node_c'])

    with open('nodes.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['node_id','feature_1', 'feature_2'])
        writer.writerow(['node_a',0.1,0.5])
        writer.writerow(['node_b',0.3,0.7])
        writer.writerow(['node_c',0.2,0.6])
    

    data = load_graph_from_edge_list_and_node_features('edges.csv', 'nodes.csv')
    print(data)
```

Here, the code reads edges and node features from two separate files. The node mapping is taken from the node csv. The source and target ids from the edge file are translated into their numeric counterparts and then are converted to a `torch.tensor`. The node features are read directly and also converted to a `torch.tensor`.

**Example 3: Edge List with Edge Features**

Assume the `edges.csv` contains information for edge features:

`edges.csv`:
```
source,target,edge_feature_1,edge_feature_2
node_a,node_b,0.2,0.8
node_b,node_c,0.5,0.6
node_a,node_c,0.3,0.4
```

```python
import torch
from torch_geometric.data import Data
import pandas as pd

def load_graph_with_edge_features(csv_path):
    df = pd.read_csv(csv_path)
    node_set = set()
    for _, row in df.iterrows():
      node_set.add(row['source'])
      node_set.add(row['target'])
    node_to_id = {node: idx for idx, node in enumerate(node_set)}
    num_nodes = len(node_to_id)

    edge_index = []
    edge_features = []

    for _, row in df.iterrows():
        source_id = node_to_id[row['source']]
        target_id = node_to_id[row['target']]
        edge_index.append([source_id, target_id])
        edge_features.append([row['edge_feature_1'], row['edge_feature_2']])

    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    
    # creating a dummy node feature matrix. The real node features can be loaded separately.
    x = torch.eye(num_nodes, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_features)
    return data

# Example usage:
if __name__ == '__main__':
    # create dummy csv
    import csv
    with open('edges.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source','target','edge_feature_1', 'edge_feature_2'])
        writer.writerow(['node_a','node_b',0.2,0.8])
        writer.writerow(['node_b','node_c',0.5,0.6])
        writer.writerow(['node_a','node_c',0.3,0.4])


    data = load_graph_with_edge_features('edges.csv')
    print(data)
```

This example is similar to the first example, but now it extracts the edge features using the corresponding columns in the CSV, creating the `edge_attr` tensor. It also creates a dummy node feature matrix which is replaced with real data, if available.

**Resource Recommendations**

For further understanding of PyTorch Geometric, review the official documentation for `torch_geometric.data.Data` and related functions. For CSV processing, familiarize yourself with the Python `csv` module or `pandas` library documentation. Additionally, exploring tutorials and examples specific to graph neural networks and data preparation can provide practical insights. Further research into specific types of graph structures and CSV formats might be needed, based on the specific application.
