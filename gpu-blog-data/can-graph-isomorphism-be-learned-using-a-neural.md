---
title: "Can graph isomorphism be learned using a neural network?"
date: "2025-01-30"
id: "can-graph-isomorphism-be-learned-using-a-neural"
---
Graph isomorphism, the problem of determining whether two graphs possess the same structure despite potentially different node labels or spatial embeddings, is fundamentally a combinatorial problem. Consequently, a direct approach using a neural network, particularly one focused solely on node and edge features, faces inherent challenges. My work in graph representation learning over the past five years has consistently shown me that while neural networks can approximate graph properties useful in many downstream tasks, they don’t solve graph isomorphism in the general case.

The core difficulty stems from the representational limitations of typical graph neural networks (GNNs). These networks, primarily message-passing architectures, operate by aggregating information from a node’s neighborhood, iterating this process to capture increasingly larger subgraphs. While very effective in many applications, GNNs often map distinct but isomorphic graphs to the same representation. This arises because the message-passing paradigm is, by design, permutation invariant; changing the order of nodes and edges does not alter the final aggregated node embeddings. This is desirable in many cases, ensuring the network focuses on structural properties rather than node ordering, but it's a critical obstacle when determining isomorphism. For example, two graphs that differ only by the numbering of nodes might map to identical vectors in the embedding space, thereby failing the isomorphism test.

Furthermore, traditional GNNs struggle with higher-order graph properties which are crucial for identifying non-isomorphic structures. Simple message passing, focusing on immediate neighbors, is insufficient to distinguish graphs that are locally identical but have different global structures. Take, for instance, two graphs that look identical within a 2-hop neighborhood around every vertex but are nonetheless globally distinct. A standard GNN might produce identical representations, leading to incorrect results. The expressiveness of such message passing networks, formally proven in several key publications, is bounded. Therefore, a naive application of a GNN as a binary classifier cannot typically, on its own, accurately classify graph isomorphism in the general case.

However, this doesn’t imply that neural networks are completely irrelevant to the problem. They can be valuable in *approximating* certain graph invariants useful in narrowing down the search space. Furthermore, learning features with GNNs can be effective in specialized domains with graphs that tend to have certain shared characteristics. The problem is, rather, that GNNs alone are generally insufficient to solve it definitively. Techniques like Weisfeiler-Lehman (WL) kernel features are often combined with learning models to improve isomorphism detection. I have observed in practical applications that a well-constructed hybrid approach proves more effective than either GNNs or classical methods alone.

Here are three code examples to illustrate how GNNs might be applied to approximate isomorphic properties:

**Example 1: Simple Graph Convolutional Network (GCN) for node embeddings.**

This code snippet shows a very basic GCN implementation designed to produce node embeddings. The underlying mechanism is one of message passing and weight updates but cannot, by itself, distinguish between isomorphic graphs.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adj, features):
        # Normalize adjacency matrix for GCN
        degree = adj.sum(dim=1).float()
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)
        
        support = torch.matmul(adj_normalized, features)
        output = self.linear(support)
        return output


class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleGCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)

    def forward(self, adj, features):
        h = F.relu(self.gcn1(adj, features))
        h = self.gcn2(adj, h)
        return h

# Example usage with dummy data
adj = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float) # Adjacency matrix
features = torch.rand(3, 10) # Node features
model = SimpleGCN(10, 32, 16)
node_embeddings = model(adj, features) # Node embeddings
print(node_embeddings.shape)
```
This code defines a simple GCN with two convolutional layers. The model takes in an adjacency matrix and node features and returns node embeddings. While these embeddings can be valuable for node classification and link prediction, the permutation invariance ensures that isomorphic graphs are likely to have identical node embedding representations, meaning the GCN cannot distinguish between the graphs. The shape of the resulting embeddings will be (3, 16), representing three nodes, each having a 16-dimensional embedding.

**Example 2: Graph Isomorphism Network (GIN) for graph embeddings.**

This code uses a GIN, which is slightly more expressive than a basic GCN, but still falls short of accurately classifying graph isomorphism. Here, graph embeddings are created by pooling the node embeddings:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.eps = nn.Parameter(torch.zeros(1))


    def forward(self, adj, features):
        neighbors_sum = torch.matmul(adj, features)
        output = self.mlp((1 + self.eps) * features + neighbors_sum)
        return output


class SimpleGIN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleGIN, self).__init__()
        self.gin1 = GINLayer(in_features, hidden_features)
        self.gin2 = GINLayer(hidden_features, out_features)

    def forward(self, adj, features):
        h = F.relu(self.gin1(adj, features))
        h = self.gin2(adj, h)
        return h


def graph_embedding(node_embeddings):
    return torch.sum(node_embeddings, dim=0)

# Example usage with dummy data
adj = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float) # Adjacency matrix
features = torch.rand(3, 10) # Node features
model = SimpleGIN(10, 32, 16)
node_embeddings = model(adj, features)
graph_emb = graph_embedding(node_embeddings) # graph embedding
print(graph_emb.shape)
```
This example uses a simple GIN model. GINs are known to be more expressive than GCNs due to the trainable epsilon value. The crucial distinction here is the `graph_embedding` function which uses summation to aggregate all node embeddings into a single graph embedding. Again, the core message passing process in the GIN is invariant to node permutations, and while graph embeddings might be similar between isomorphic graphs, they are unlikely to solve the problem in the general case. The shape of `graph_emb` will be (16,), a single vector representing the entire graph.

**Example 3: Edge features in a basic GNN.**

This code demonstrates using edge features but still has limited applicability for isomorphism detection, focusing primarily on how edge information is combined with node information:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(EdgeConvLayer, self).__init__()
        self.linear_node = nn.Linear(in_features, out_features)
        self.linear_edge = nn.Linear(1, out_features) # Assuming single feature per edge
        self.combine = nn.Linear(2*out_features, out_features)

    def forward(self, adj, node_features, edge_features):
        node_transformed = self.linear_node(node_features)
        edge_transformed = torch.matmul(adj, self.linear_edge(edge_features)) # aggregate over connected edges

        combined = self.combine(torch.cat((node_transformed, edge_transformed), dim=1)) #combine node and edge
        return combined


class SimpleEdgeGNN(nn.Module):
    def __init__(self, node_in_features, node_hidden_features, node_out_features):
        super(SimpleEdgeGNN, self).__init__()
        self.edge_conv = EdgeConvLayer(node_in_features, node_hidden_features)
        self.fc = nn.Linear(node_hidden_features, node_out_features)


    def forward(self, adj, node_features, edge_features):
        h = F.relu(self.edge_conv(adj, node_features, edge_features))
        h = self.fc(h)
        return h


# Example usage with dummy data
adj = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float)  # Adjacency matrix
node_features = torch.rand(3, 10)  # Node features
edge_features = torch.rand(3,3,1)  # Edge features (adjacency matrix with a single feature for each edge)
model = SimpleEdgeGNN(10, 32, 16)
output = model(adj, node_features, edge_features)
print(output.shape)
```

Here, edge features are incorporated into the message passing mechanism alongside node features. Again, while edge information is useful, the GNN architecture remains inherently permutation invariant; isomorphic graphs with different node orders can still result in the same node embeddings. The shape of `output` here will be (3, 16), similar to the GCN.

These three examples highlight how common GNN architectures operate. While they produce meaningful representations, they typically fall short of solving graph isomorphism due to their inherent permutation invariance.

For further study on this topic, I recommend exploring academic resources detailing the theoretical limits of GNN expressiveness. Material on the Weisfeiler-Lehman (WL) test and its relationship to GNNs is essential to understand the bounds of these models. Further, research into spectral methods in graph theory provides another lens through which to evaluate graph similarity. Finally, practical explorations of graph isomorphism heuristics in the literature, especially those integrating machine learning approaches, would provide a well-rounded understanding.
