---
title: "How can graph autoencoders be implemented using PyTorch-Geometric?"
date: "2025-01-30"
id: "how-can-graph-autoencoders-be-implemented-using-pytorch-geometric"
---
Graph autoencoders (GAEs) leverage the inherent structure of graph data for unsupervised learning tasks like node embedding and link prediction. Their implementation using PyTorch Geometric (PyG) provides a streamlined approach due to PyG’s robust support for graph-based operations. My experience developing a large-scale knowledge graph embedding system revealed the crucial role GAEs played in generating compact and informative representations from complex relational data.

A GAE, fundamentally, is composed of two core components: an encoder and a decoder. The encoder, often implemented as a graph neural network (GNN), maps the input graph to a low-dimensional latent space, effectively compressing the information. The decoder then attempts to reconstruct the original graph structure from this compressed representation. The objective is to minimize the reconstruction loss, ensuring the latent space captures essential structural and node-specific features.

PyG offers several classes and functions that facilitate the straightforward construction of these components. A common approach is to utilize `torch_geometric.nn` for the GNN encoder, allowing flexible choices like `GCNConv`, `GraphConv`, or `SAGEConv`. The decoder can be implemented using a simple inner product of the latent node embeddings to predict the edges, a common practice for link prediction based tasks.

The encoding phase involves passing the input graph data through one or more GNN layers. Each layer aggregates information from the neighboring nodes using specific message-passing mechanisms determined by the GNN architecture used. The resulting node features at the final layer form the latent representation. Conversely, the decoder leverages these learned representations. For a basic GAE, reconstruction is achieved by calculating the dot product of the latent vectors for each node pair, then predicting the edges of the graph based on these scores. A threshold or a sigmoid function may be employed on the dot product results to obtain a probabilistic measure of edge existence.

Let's consider three implementation examples.

**Example 1: Basic GAE with GCN Encoder**

```python
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        adj_pred = torch.sigmoid(torch.matmul(z, z.transpose(0, 1)))
        return adj_pred

class GAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAE, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_pred = self.decoder(z)
        return adj_pred

# Example Usage
if __name__ == '__main__':
    num_nodes = 100
    num_features = 10
    edge_index = torch.randint(0, num_nodes, (2, 500))
    x = torch.randn(num_nodes, num_features)

    data = pyg_data.Data(x=x, edge_index=edge_index)

    in_channels = num_features
    hidden_channels = 32
    out_channels = 16

    model = GAE(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        adj_pred = model(data.x, data.edge_index)

        # Reconstruction Loss (Binary Cross Entropy)
        adj_true = torch.zeros(num_nodes, num_nodes)
        adj_true[data.edge_index[0], data.edge_index[1]] = 1
        loss = nn.functional.binary_cross_entropy(adj_pred, adj_true)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
```
This example showcases a simple GAE using a two-layer GCN as the encoder and an inner product decoder for edge reconstruction. The reconstruction loss is calculated as binary cross-entropy. We first define the encoder class `GCNEncoder`, which takes node features `x` and adjacency information `edge_index`. We then define the `InnerProductDecoder` that computes the adjacency matrix through an inner product of the encoded representation. The `GAE` model combines both.  During training, the model attempts to minimize the reconstruction loss, pushing the latent node representations to capture relevant relational information. The example demonstrates training for a defined amount of epochs and prints loss every tenth step. The input graph data is artificially created for demonstration.

**Example 2: GAE with SAGEConv Encoder and Sigmoid Reconstruction**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGEEncoder, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class SigmoidDecoder(nn.Module):
    def __init__(self):
        super(SigmoidDecoder, self).__init__()

    def forward(self, z, edge_index):
       src, dst = edge_index
       edge_scores = torch.sum(z[src] * z[dst], dim=-1)
       return torch.sigmoid(edge_scores)

class GAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAE, self).__init__()
        self.encoder = SAGEEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = SigmoidDecoder()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        edge_pred = self.decoder(z, edge_index)
        return edge_pred, z

# Example Usage
if __name__ == '__main__':
    num_nodes = 100
    num_features = 10
    edge_index = torch.randint(0, num_nodes, (2, 500))
    x = torch.randn(num_nodes, num_features)

    data = pyg_data.Data(x=x, edge_index=edge_index)

    in_channels = num_features
    hidden_channels = 32
    out_channels = 16

    model = GAE(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        edge_pred, embeddings = model(data.x, data.edge_index)

        # Reconstruction Loss (Binary Cross Entropy)
        edge_labels = torch.ones(edge_pred.shape[0])
        loss = nn.functional.binary_cross_entropy(edge_pred, edge_labels)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

```
This second example substitutes the `GCNConv` with `SAGEConv` in the encoder, and the decoder now outputs a prediction for each edge using the sigmoid function on a sum of the element-wise product of node embeddings. This allows prediction of edge existence based on embedding similarity, specifically, we extract the source and destination node embeddings from all edges and predict a probability.  This demonstrates a slightly different approach to decoding and can potentially be useful depending on the task and edge distributions. The reconstruction loss remains binary cross-entropy, now calculated for the predicted edges based on the latent representation.

**Example 3: GAE for Node Embedding Extraction**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class GraphConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvEncoder, self).__init__()
        self.conv1 = pyg_nn.GraphConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        adj_pred = torch.sigmoid(torch.matmul(z, z.transpose(0, 1)))
        return adj_pred

class GAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAE, self).__init__()
        self.encoder = GraphConvEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_pred = self.decoder(z)
        return adj_pred, z

# Example Usage for node embeddings
if __name__ == '__main__':
    num_nodes = 100
    num_features = 10
    edge_index = torch.randint(0, num_nodes, (2, 500))
    x = torch.randn(num_nodes, num_features)

    data = pyg_data.Data(x=x, edge_index=edge_index)

    in_channels = num_features
    hidden_channels = 32
    out_channels = 16

    model = GAE(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        adj_pred, node_embeddings = model(data.x, data.edge_index)

        adj_true = torch.zeros(num_nodes, num_nodes)
        adj_true[data.edge_index[0], data.edge_index[1]] = 1
        loss = nn.functional.binary_cross_entropy(adj_pred, adj_true)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

    # Embedding extraction after training.
    _, embeddings_final = model(data.x, data.edge_index)
    print("\nExample Node Embeddings:")
    print(embeddings_final[0:5,:])

```
This third example is structurally similar to the first, using a different encoder `GraphConv`  but emphasizes extraction of learned node embeddings. Here, we return both the predicted adjacency matrix and the learned embeddings from the model. After the training loop, we extract the final learned embeddings using a final forward pass, demonstrating that GAE models can be used for obtaining node embeddings. This demonstrates the dual nature of GAEs where you may use the decoder for edge prediction or use the learned node embeddings for downstream tasks.

To enhance one's understanding of implementing graph autoencoders, I recommend exploring the following resources: the official PyTorch documentation provides in-depth details about tensor operations and network definitions. Furthermore, a detailed look at the PyTorch Geometric documentation, specifically the sections on `torch_geometric.nn` and `torch_geometric.data`, is invaluable. Lastly, the published literature on graph neural networks and graph autoencoders provides comprehensive theoretical background and various practical applications.
