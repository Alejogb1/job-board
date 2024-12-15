---
title: "How to use GRU in a non sequential context?"
date: "2024-12-15"
id: "how-to-use-gru-in-a-non-sequential-context"
---

alright, so you're asking about using grus in a non-sequential context. that's a fun one, i've definitely been there. most folks, myself included initially, think of grus, or gated recurrent units, as these time-series powerhouses, perfect for handling sequences like text or audio. and they are, but limiting them to just that use-case is throwing away a lot of potential. i mean, the ‘recurrent’ part of their name makes everyone think they must have sequences. the thing is, what defines the sequence? the internal mechanics of a gru are all about the gates and cell states, and that’s totally orthogonal to a strict time dimension.

i remember when i first encountered this. it was maybe five, six years back. i was working on a project involving protein folding prediction. we had a bunch of spatial data, atom positions and such, and i was trying all sorts of graph neural networks. then, i stumbled upon a paper that used grus not on temporal sequences but rather based on relationships in the graph— like, they treated the neighbors of a node as a “sequence”. it blew my mind a bit. it was like using a hammer to screw things, but it worked… and it worked really well in some circumstances.

the key is to redefine what constitutes the “sequence” that a gru processes. it doesn't have to be time. it can be any ordered set of inputs where the order is relevant to the computation you want to perform. the "hidden state" of the gru stores information about the processed sequence, so the order in which we process the inputs can become important.

so, practically, how do we make this happen? let’s look at a few situations and code examples to make it clearer.

example 1: processing node features in a graph

instead of each node being an independent entity, what if we consider an order based on how they are connected? you could, for instance, order the neighbors of a node by their id, their degree, any arbitrary order really. you can then feed the node’s feature vectors of each neighbor in this order into the gru.

here's a simplified python snippet using pytorch:

```python
import torch
import torch.nn as nn

class GraphGRU(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(feature_size, hidden_size, batch_first=True)

    def forward(self, node_features, neighbor_indices):
       
        batch_size = node_features.size(0)
        hidden_states = []

        for i in range(batch_size):
           
            neighbors_features = node_features[i, neighbor_indices[i]]

            _, h_n = self.gru(neighbors_features.unsqueeze(0))  
            
            hidden_states.append(h_n.squeeze(0))
        
        return torch.stack(hidden_states, dim=0)

if __name__ == '__main__':
    
    feature_size = 5
    hidden_size = 10

    graph_gru = GraphGRU(feature_size, hidden_size)

    node_features = torch.randn(3, 5, feature_size) # 3 nodes, each with 5 neighbor features
    neighbor_indices = [
        torch.tensor([0, 1, 2, 3, 4]), # node 1 neighbours
        torch.tensor([1, 2, 0, 4, 3]), # node 2 neighbours
        torch.tensor([4, 3, 2, 1, 0])  # node 3 neighbours
    ]

    output = graph_gru(node_features, neighbor_indices)
    print(output.shape)  # output: torch.Size([3, 1, 10]) 3 nodes, 1 hidden state, 10 hidden dimensions
```

in this example, `node_features` contains the feature vectors for each node’s neighbors and `neighbor_indices` specifies the order to feed them to the gru. the gru’s hidden state captures information based on the order of neighbors of a node. this could be useful in a context where relationships between nodes matter and the ordering of neighbours contain relevant info. this is just one way of using the order of the gru, it can be arbitrary based on your context, but the concept is the same.

example 2: processing set data

imagine you have a set of features, but the order in which they appear influences your task. perhaps you’re dealing with sensor data where the relative arrangement of sensors on a device has some significance, it isn't necessarily related to time or a graph structure. you could make a “pseudo-sequence” based on sensor position or some other deterministic scheme.

```python
import torch
import torch.nn as nn

class SetGRU(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(feature_size, hidden_size, batch_first=True)

    def forward(self, features, order_indices):
        ordered_features = features[:, order_indices, :] # reorder features 
        _, h_n = self.gru(ordered_features)
        return h_n.squeeze(0)
        
if __name__ == '__main__':
    
    feature_size = 10
    hidden_size = 20
    num_sets = 5 # batch size
    num_features = 7 # each set has 7 features

    set_gru = SetGRU(feature_size, hidden_size)

    features = torch.randn(num_sets, num_features, feature_size) 
    order_indices = torch.randperm(num_features) 

    output = set_gru(features, order_indices)
    print(output.shape)  # output: torch.Size([5, 20])
```

here, `features` represents the set of features, and `order_indices` determines a custom ordering. we’re explicitly changing the order of the input features before we feed them into the gru. this can add value when, by chance, the order has information that is relevant to the solution to our problem.

example 3: processing feature interaction with positional encoding

another way is to use positional encoding and treat features as elements in a sequence. even if they don’t have an inherent order, we can impose one through positional encodings, sort of like transformer models but with a gru. this allows for richer interactions between feature vectors because the “position” is taken into consideration by the gru. imagine features extracted from an image, there is no defined sequence, but we can inject a "sequence" via positional encoding of the grid.
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class FeatureInteractionGRU(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.pos_encoder = PositionalEncoding(feature_size)
        self.gru = nn.GRU(feature_size, hidden_size, batch_first=True)

    def forward(self, features):
        encoded_features = self.pos_encoder(features)
        _, h_n = self.gru(encoded_features)
        return h_n.squeeze(0)
        
if __name__ == '__main__':
    
    feature_size = 16
    hidden_size = 32
    num_features = 10
    batch_size = 3

    interaction_gru = FeatureInteractionGRU(feature_size, hidden_size)
    features = torch.randn(batch_size, num_features, feature_size)
    output = interaction_gru(features)
    print(output.shape)  # output: torch.Size([3, 32])
```
here, positional encoding adds "order" to the features which allows the gru to better process their interdependencies when they don't have a sequential nature. it's a bit like saying "feature a is to the left of feature b and this changes the context." it works surprisingly well when you want to enrich feature interactions.

a word on resources: for a good theoretical grounding, i’d suggest “deep learning” by goodfellow, bengio and courville. it's a dense read, but it lays the foundation. also, search on arxiv.org for papers on graph neural networks, attention mechanisms and transformers, as these tend to explore different ways of processing non-sequential data and have practical implications for re-thinking how we apply grus. you will see a lot of researchers use it as a base in more innovative applications of the gru.

and, i guess, the obligatory joke: why did the gru cross the road? because it was a hidden state of affairs! (sorry, i couldn't resist).

in conclusion, grus are not just time-series tools, they are recurrent computation machines that can be adapted to different contexts depending on how you define the "sequence" of inputs. you might need to be creative about how you choose your sequence definition and order based on your particular situation, but the fundamental mechanics are still the same. the main difference is to define sequences where they might not exist explicitly, and use a gru to learn information based on these "fake" sequences. go experiment, try different orderings, and you might be surprised at what you can achieve.
