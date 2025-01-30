---
title: "How can I obtain a PyTorch model summary?"
date: "2025-01-30"
id: "how-can-i-obtain-a-pytorch-model-summary"
---
Understanding a PyTorch model's architecture, parameters, and trainable state is crucial for debugging, optimizing, and sharing models. I've found, through numerous projects building neural networks, that simply relying on printed model definitions quickly becomes cumbersome for anything beyond the most basic architectures. PyTorch itself doesn't offer a built-in ‘summary’ method akin to Keras, but several effective techniques and external libraries provide the information I require in a clear, structured manner.

The most straightforward method to understand a PyTorch model is direct introspection via its `named_modules()` and `named_parameters()` methods. These allow you to iterate through the model's layers and parameters respectively, providing their names and associated tensors. While this can provide detailed information, it's not concise nor well-structured for complex models. Consider a simple feedforward network:

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()

for name, module in model.named_modules():
    print(f"Module: {name}")

for name, param in model.named_parameters():
  print(f"Parameter: {name}, Size: {param.size()}, Requires Gradient: {param.requires_grad}")
```

This code iterates through the model, printing both modules (layers) and parameters. I find this raw output useful for granular debugging, especially when investigating initialization issues or specific layer configurations, but it lacks a consolidated summary format, which is what’s typically needed for rapid assessment. You can also easily infer the number of parameters in each module if you access each module directly using `model.fc1.weight`, etc. However this is not convenient for larger models and is error prone.

For a more structured, human-readable summary, I often utilize the `torchinfo` library. This package, which requires a simple installation (`pip install torchinfo`), offers a dedicated `summary` function that accepts a model and an input size (or an input tensor) and provides a clean, table-based output with crucial information such as layer type, output shape, number of parameters, and if the parameters are trainable. This has significantly improved how I visualize my models. Here's an example, expanding on the previous model:

```python
import torch
import torch.nn as nn
from torchinfo import summary

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
input_size = (1,10) # Batch size is 1 for this example
summary(model, input_size=input_size)
```
This produces a well-formatted table. The table is formatted differently depending on whether you are using a text based console or an interactive notebook. The summary shows the type of layer, the output shape, and the number of parameters (both trainable and non-trainable). This is particularly useful because it helps quickly identify the layers that contribute most to the overall parameter count of the model. It also clearly shows the output shape for each layer which can be useful in confirming the model is doing what was intended.

I have frequently employed `torchinfo` when exploring different network architectures, especially when comparing models of varying complexity and confirming that the sizes of layers and tensors propagate in the way I intended. The library’s capability to infer the model's shape is invaluable. Note that if a layer does not produce an output, the summary will reflect this by reporting `<None>`. This is common when looking at the output of a recurrent neural network before the final linear layer. The flexibility to input a batch size, and the ability to specify device (CPU or GPU), is helpful when examining performance characteristics during summary generation. `torchinfo` is my go-to for quick model inspections.

Beyond tabular summaries, there are occasions where visual representation is advantageous. For visualizing model graphs, particularly for tracing the flow of data, `torchviz` and `netron` are incredibly helpful. These are not direct model summaries in the sense that `torchinfo` provides, but they do illuminate the model's structure in a graphical manner. Note that `torchviz` is not actively maintained. `netron` allows one to load a model via ONNX format which is a common standard for model exchange.

```python
import torch
import torch.nn as nn
from torchviz import make_dot

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
x = torch.randn(1, 10) # Create a dummy input tensor
y = model(x) # Pass it through model to get the graph
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_graph", view=True) # Save graph as pdf and open it
```

This `torchviz` example, although a short code snippet, requires installing additional tools on some operating systems (Graphviz). `make_dot` needs to have some output from the model fed into it so that the computations can be captured. The resulting graph, saved to a PDF file in this case, visually displays the model’s layers and data flow. The parameter names are displayed in the graph to help identify each part of the model. I use such graphs when dealing with custom or unconventional architectures, where tracing data pathways is crucial to understanding model mechanics. While it might seem less directly informative than a text-based summary, I find that these graphical representations often help me quickly pinpoint unexpected flow issues. Note that `torchviz` will create a relatively complex graph even for simple models because it maps all the operations and not just the layer names. The `render` command also requires a PDF viewer or image viewer for displaying the graph.

For a more portable and modern approach, exporting a PyTorch model to ONNX and then visualizing it with Netron has been invaluable to me. Netron is a free, open source model visualization software that works for multiple types of models including ONNX format models.

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
dummy_input = torch.randn(1, 10) #Create a dummy input
torch.onnx.export(model, dummy_input, "model.onnx")
```

The above code snippet will export the model to `model.onnx`. The exported file can then be loaded directly into `netron` (or any other ONNX viewer) for visualization. The exported graph is well organized, contains the shapes and types of all the tensors.

In summary, for text-based summaries, `torchinfo` is my primary choice due to its clarity, conciseness and ease of use. When a visual approach is more appropriate, especially for inspecting the data flow within a network, I use model export via ONNX and view it via `netron`. I also use the methods provided directly by PyTorch (such as `named_modules()` and `named_parameters()`) for detailed introspection and debugging.  

For anyone starting with PyTorch, I recommend exploring these options and selecting the ones that fit your workflow, and then becoming familiar with their usage. I would also suggest referring to the official PyTorch documentation for more advanced methods of debugging. Reading academic papers and books with a focus on neural networks and software architecture can also be very informative. Using online tutorials and examples will also greatly accelerate your learning. I have found these resources and tools critical throughout my development process, and they continue to provide a sound foundation for understanding model behavior.
