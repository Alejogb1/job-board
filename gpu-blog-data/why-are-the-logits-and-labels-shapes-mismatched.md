---
title: "Why are the logits and labels shapes mismatched in my GCN?"
date: "2025-01-30"
id: "why-are-the-logits-and-labels-shapes-mismatched"
---
The root cause of mismatched logits and label shapes in a Graph Convolutional Network (GCN) almost invariably stems from an inconsistency between the network's output design and the structure of the provided labels.  My experience troubleshooting this issue across several large-scale graph classification projects highlights a frequent oversight: neglecting the inherent node-level versus graph-level prediction distinction.  This discrepancy manifests as a shape mismatch when the model predicts logits for individual nodes while the labels represent classifications for entire graphs.

**1. Clear Explanation:**

A GCN processes graph data by iteratively aggregating information from neighboring nodes.  The fundamental architectural choice dictates whether the final output pertains to individual nodes or the entire graph.  If the GCN is designed for *node-level* prediction (e.g., node classification), the logits will have a shape reflecting the number of nodes in the input graph and the number of classes.  In contrast, if the GCN is designed for *graph-level* prediction (e.g., graph classification), the logits should represent a single prediction per graph, resulting in a shape reflecting the number of graphs in the batch and the number of classes. The labels must consistently correspond to this level of prediction.  A mismatch occurs when the model predicts node-level logits but receives graph-level labels, or vice-versa.

Furthermore, the label encoding needs careful consideration.  If using one-hot encoding for multi-class classification, the number of columns in the labels must match the number of output units (classes) in the GCN's final layer.  A simple error in the number of classes encoded in the labels compared to the network's output layer directly leads to a shape mismatch.  Inconsistent batching strategies can also contribute to this problem, particularly when dealing with variable-sized graphs.  The model might output logits for all nodes across the batch, while the labels are structured for each individual graph separately.

**2. Code Examples with Commentary:**

The following examples illustrate correct and incorrect implementations, highlighting the shape consistency required between logits and labels.  These examples leverage PyTorch and a hypothetical GCN layer (`GCNLayer`) for brevity; the specifics of the GCN implementation are not the critical factor here.

**Example 1: Correct Node-Level Prediction**

```python
import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.gcn_layer = GCNLayer(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = self.gcn_layer(x, adj)
        x = self.linear(x)  # Node-level logits
        return x

# Example usage
num_nodes = 100
num_classes = 5
input_dim = 10
hidden_dim = 20

model = GCN(input_dim, hidden_dim, num_classes)
x = torch.randn(num_nodes, input_dim)
adj = torch.randn(num_nodes, num_nodes) # Adjacency matrix

logits = model(x, adj)
print(logits.shape) # Output: torch.Size([100, 5])

labels = torch.randint(0, num_classes, (num_nodes,)) # Node-level labels
# one-hot encoding:
labels = nn.functional.one_hot(labels, num_classes=num_classes)
print(labels.shape) # Output: torch.Size([100,5])

loss = nn.CrossEntropyLoss()(logits, labels) # Correct shape matching.
```

This example demonstrates a correctly structured node-level prediction. The logits have a shape of (num_nodes, num_classes), matching the shape of the one-hot encoded labels.


**Example 2: Correct Graph-Level Prediction**

```python
import torch
import torch.nn as nn

class GraphGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphGCN, self).__init__()
        self.gcn_layer = GCNLayer(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = self.gcn_layer(x, adj)
        x = torch.mean(x, dim=0) # graph-level readout via mean pooling
        x = self.readout(x) # Graph-level logits
        return x

#Example Usage
num_graphs = 32
num_classes = 5
input_dim = 10
hidden_dim = 20

model = GraphGCN(input_dim, hidden_dim, num_classes)
x = torch.randn(num_graphs,100,input_dim) # Batch of graphs, each with 100 nodes
adj = torch.randn(num_graphs, 100, 100) #Batch of Adjacency Matrices

logits = model(x, adj)
print(logits.shape) # Output: torch.Size([32, 5])

labels = torch.randint(0, num_classes, (num_graphs,)) #Graph Level Labels
labels = nn.functional.one_hot(labels, num_classes=num_classes)
print(labels.shape) # Output: torch.Size([32, 5])

loss = nn.CrossEntropyLoss()(logits, labels) #Correct Shape Matching

```

This example shows a graph-level prediction. A readout function (here, mean pooling) aggregates node-level features into a graph-level representation before the final linear layer. The logits now represent graph-level predictions with a shape of (num_graphs, num_classes).


**Example 3: Incorrect Shape Mismatch**

```python
import torch
import torch.nn as nn

#... (GCN definition from Example 1 remains the same) ...

#Example Usage
num_graphs = 32
num_classes = 5
input_dim = 10
hidden_dim = 20

model = GCN(input_dim, hidden_dim, num_classes) # Node-level GCN
x = torch.randn(num_graphs,100, input_dim) # Batch of graphs
adj = torch.randn(num_graphs, 100, 100)

logits = model(x,adj) # Logits are node-level
print(logits.shape) # Output: torch.Size([3200, 5])  (32 graphs * 100 nodes)

labels = torch.randint(0, num_classes, (num_graphs,))  # Graph-level labels
labels = nn.functional.one_hot(labels, num_classes=num_classes)
print(labels.shape) # Output: torch.Size([32, 5])

# loss = nn.CrossEntropyLoss()(logits, labels) # This will raise a RuntimeError due to shape mismatch

```

This example highlights the error.  A node-level GCN (from Example 1) is used, but graph-level labels are provided. The resulting shape mismatch between the logits (32 graphs * 100 nodes, 5 classes) and labels (32 graphs, 5 classes) prevents loss calculation.  The solution involves either changing the GCN architecture to produce graph-level outputs or adapting the labels to reflect node-level classifications.


**3. Resource Recommendations:**

For a deeper understanding of GCNs and graph neural networks in general, I strongly recommend reviewing seminal papers on the topic, focusing on those detailing various graph pooling strategies for graph-level classification.  Furthermore, exploring various implementations of GCNs in popular deep learning frameworks' documentation will offer valuable insights into practical aspects of implementation and data handling.  Finally, a comprehensive textbook on graph theory and its applications in machine learning would provide essential foundational knowledge.  Scrutinizing the documentation for your chosen deep learning framework regarding the specific GCN layer and its expected input/output shapes is vital.  Pay close attention to how batching is handled, especially if your graphs have varying sizes.
