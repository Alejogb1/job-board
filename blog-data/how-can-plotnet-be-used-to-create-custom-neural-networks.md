---
title: "How can PlotNet be used to create custom neural networks?"
date: "2024-12-23"
id: "how-can-plotnet-be-used-to-create-custom-neural-networks"
---

Okay, let's talk PlotNet. I remember back in '18, working on a particularly challenging time series forecasting project, I stumbled upon the limitations of off-the-shelf neural network architectures. That's when I started experimenting more deeply with tools that allowed for greater flexibility in network design, and PlotNet was one of the more interesting ones. It's not your run-of-the-mill framework, and while it can be a bit more involved initially, the control it gives you over your network's structure is powerful.

PlotNet, at its core, isn't a single library but a methodology, a way of conceptualizing and implementing neural networks using computational graphs. It treats the network as a flow of operations, where each node represents a specific mathematical function, and edges define the dependencies and data flow. This approach allows you to visually map out your neural network and then execute it as a computational graph. The most notable advantage here is the ability to define custom operations and network architectures that aren't readily available in high-level libraries like TensorFlow or PyTorch. This is particularly handy when your problem domain pushes the boundaries of standard architectures. Think beyond convolutional and recurrent layers - we’re venturing into custom activation functions, unusual layer interconnections, or specialized normalization techniques, all easily integrated within PlotNet's graph-based framework.

Now, to illustrate, let’s dive into some practical examples. We'll look at a custom activation function, a somewhat unusual skip connection implementation, and a custom layer designed for specific input manipulation.

**Example 1: Custom Activation Function**

Let’s say we need an activation function that is essentially a modified ReLU, but with a slight smoothing component. Standard libraries might not have this directly. With PlotNet, I can define a custom node for this:

```python
import numpy as np
import networkx as nx

def custom_relu_smooth(x, alpha=0.1):
  """Custom smoothed ReLU activation."""
  return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def build_activation_node(graph, name, input_node, alpha=0.1):
    graph.add_node(name, operation=custom_relu_smooth, alpha=alpha)
    graph.add_edge(input_node, name)
    return name

# Example usage in a small graph
g = nx.DiGraph()
input_node = "input_layer"
g.add_node(input_node, data=np.array([-2, -1, 0, 1, 2]))

activation_node = build_activation_node(g, "custom_activation", input_node, alpha=0.2)

# Execute the graph
def execute_graph(graph):
  for node in nx.topological_sort(graph):
    if "data" in graph.nodes[node]: # Process the input
        continue
    incoming_edges = graph.in_edges(node, data=True)

    if not incoming_edges:  # Skip nodes without incoming edges, typically input nodes
      continue
    
    input_data = graph.nodes[list(incoming_edges)[0][0]]["data"]
    
    if "alpha" in graph.nodes[node]:
        graph.nodes[node]["data"] = graph.nodes[node]["operation"](input_data, graph.nodes[node]["alpha"])
    else:
        graph.nodes[node]["data"] = graph.nodes[node]["operation"](input_data)

execute_graph(g)
print(f"Result after custom activation: {g.nodes[activation_node]['data']}")
```

Here, we build a computational graph using `networkx`. The `build_activation_node` function adds our `custom_relu_smooth` function as a node in the graph. The `execute_graph` function performs a topological sort and then evaluates each node's operation in the correct order, storing results as node data. This illustrates how you create your own operation and use it within the PlotNet framework.

**Example 2: Custom Skip Connection**

Consider a scenario where you want a custom skip connection that adds a scaled version of the input to a deeper layer, rather than a simple addition.

```python
def scaled_add(x, y, scale=0.5):
    """Custom skip connection operation."""
    return x + scale * y

def build_skip_connection(graph, name, input_node, skip_node, scale=0.5):
    graph.add_node(name, operation=scaled_add, scale=scale)
    graph.add_edge(input_node, name)
    graph.add_edge(skip_node, name)
    return name


g2 = nx.DiGraph()
input_node_1 = "input_1"
input_node_2 = "input_2"
g2.add_node(input_node_1, data=np.array([1,2,3]))
g2.add_node("layer_1", data=np.array([4,5,6]))

skip_node = "layer_1"
skip_connection_node = build_skip_connection(g2, "skip_connection", input_node_1, skip_node, scale=0.3)

# Execute graph
def execute_graph_v2(graph):
    for node in nx.topological_sort(graph):
        if "data" in graph.nodes[node] and not any("operation" in graph.nodes[prev_node] for prev_node, _, _ in graph.in_edges(node, data=True)):
            continue

        incoming_edges = list(graph.in_edges(node, data=True))
    
        if not incoming_edges:  # Skip nodes without incoming edges, typically input nodes
            continue
        
        input_data = [graph.nodes[prev_node]["data"] for prev_node, _, _ in incoming_edges ]

        if "scale" in graph.nodes[node]:
            graph.nodes[node]["data"] = graph.nodes[node]["operation"](*input_data, graph.nodes[node]["scale"])
        else:
            graph.nodes[node]["data"] = graph.nodes[node]["operation"](*input_data)


execute_graph_v2(g2)
print(f"Result after custom skip connection: {g2.nodes[skip_connection_node]['data']}")
```

Here, we have a `scaled_add` function and a `build_skip_connection` function to establish connections in the graph. Crucially, the `execute_graph_v2` function is adapted to handle multiple inputs for nodes, accessing them through the incoming edges.

**Example 3: Custom Input Manipulation Layer**

Imagine your neural network needs to preprocess its input in a specific, non-standard way. You could define a custom layer using PlotNet for this purpose. In this example, it's a layer that adds a constant to only even indices of the input.

```python
def custom_input_processor(x, constant=2):
    """Custom input processing."""
    output = x.copy()
    for i in range(0, len(output), 2):
        output[i] += constant
    return output

def build_input_layer(graph, name, input_node, constant=2):
    graph.add_node(name, operation=custom_input_processor, constant=constant)
    graph.add_edge(input_node, name)
    return name

g3 = nx.DiGraph()
input_node_3 = "input_layer"
g3.add_node(input_node_3, data=np.array([1, 2, 3, 4, 5, 6]))
processing_layer = build_input_layer(g3, "processing_layer", input_node_3, constant=1)
execute_graph(g3)
print(f"Result after custom input layer: {g3.nodes[processing_layer]['data']}")
```

Here, we have a `custom_input_processor` that adds a constant only to specific indices and use `build_input_layer` to set the layer in our graph. We reuse the `execute_graph` here, showing it can work with different nodes and processes.

These three examples showcase how PlotNet's graph-based approach allows for highly customized neural networks. You're essentially defining your own building blocks and arranging them as you need.

For those interested in diving deeper into the theoretical aspects of computational graphs, and neural network architectures more broadly, I'd recommend starting with “Deep Learning” by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. It's a comprehensive resource that lays a robust foundation. Also, research papers that describe the underlying concepts of data flow and computational graphs, for example, from conferences like NIPS (NeurIPS) and ICML are very useful. Furthermore, the book "Pattern Recognition and Machine Learning" by Christopher Bishop is a classic for a comprehensive view of the statistical learning aspects which will provide an understanding on why you are actually designing and implementing a neural network.

While the initial learning curve with PlotNet is steeper than using pre-built libraries, the flexibility and control it offers is invaluable when you are working on more specialized tasks that push the limits of existing architectures. The more intricate a problem becomes, the more apparent it is that fine-grained control is required, and that's where this approach truly excels. In a nutshell, PlotNet enables you to move beyond just stacking layers - it allows you to design and define the very operations and flows that comprise your neural network.
