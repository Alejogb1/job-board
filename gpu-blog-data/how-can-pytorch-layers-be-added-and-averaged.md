---
title: "How can PyTorch layers be added and averaged?"
date: "2025-01-30"
id: "how-can-pytorch-layers-be-added-and-averaged"
---
The crux of averaging PyTorch layers lies not in direct, element-wise manipulation of layer objects themselves, but rather in averaging the weights (and biases, if present) contained within their parameter tensors. These tensors are the numerical backbone of learned information in a neural network. Direct averaging of layer instances would result in a loss of the trained state. My experience from years spent fine-tuning large-scale models illustrates this quite clearly.

The process hinges on extracting the `.weight` and `.bias` (if the layer has one) attributes of each layer instance. Once these parameters, stored as `torch.Tensor` objects, are collected, they can be element-wise averaged. After averaging, these new, averaged parameters are loaded back into a fresh layer instance of the same type as the originals. This new layer will then possess parameters representing the mean of the inputs. This approach is necessary because PyTorch layers are, at a fundamental level, complex objects managing forward and backward propagation functionality in addition to storing these parameter tensors.

Let's break down the methodology through examples.

**Example 1: Averaging Linear Layers**

This example demonstrates the core concept with simple linear layers. Assume we have three pre-trained linear layers, perhaps from different runs of the same training procedure or three different ensemble members, each with an input size of 10 and an output size of 5.

```python
import torch
import torch.nn as nn

# Create three example linear layers
layer1 = nn.Linear(10, 5)
layer2 = nn.Linear(10, 5)
layer3 = nn.Linear(10, 5)

# Placeholder to simulate the layers already containing weights
with torch.no_grad():
    layer1.weight.copy_(torch.randn(5, 10))
    layer1.bias.copy_(torch.randn(5))
    layer2.weight.copy_(torch.randn(5, 10))
    layer2.bias.copy_(torch.randn(5))
    layer3.weight.copy_(torch.randn(5, 10))
    layer3.bias.copy_(torch.randn(5))


# Collect all weights and biases
weights = [layer1.weight, layer2.weight, layer3.weight]
biases = [layer1.bias, layer2.bias, layer3.bias]

# Averaging the weights
avg_weight = torch.mean(torch.stack(weights), dim=0)

# Averaging the biases
avg_bias = torch.mean(torch.stack(biases), dim=0)


# Create a new linear layer and load the averaged parameters
avg_layer = nn.Linear(10, 5)

with torch.no_grad():
    avg_layer.weight.copy_(avg_weight)
    avg_layer.bias.copy_(avg_bias)

# Example usage
input_tensor = torch.randn(1, 10)
output_layer1 = layer1(input_tensor)
output_avg_layer = avg_layer(input_tensor)

print("Output from original Layer 1: ", output_layer1)
print("Output from averaged layer: ", output_avg_layer)

```

Here, `torch.stack` forms a new tensor by stacking the weight and bias tensors. The `torch.mean` operation calculates the mean across the first dimension, resulting in a single average weight and bias tensor. Critically, `torch.no_grad()` context manager is used during the parameter update step. This avoids tracking gradients during this process, which is not necessary when only loading the average. The new `avg_layer` contains the averaged weights and bias. The printed outputs demonstrate that both the original and averaged layers can be used in a forward pass, and produce different outputs.

**Example 2: Averaging Convolutional Layers**

The procedure is almost identical for convolutional layers, although convolutional layers have different dimensionalities compared to linear layers. The parameter extraction is the same process, but the shape of the parameter tensors will vary depending on the layer setup. Here we average 2D convolutional layers with a single input and output channel.

```python
import torch
import torch.nn as nn

# Create three example convolutional layers
conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
conv3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)


# Placeholder to simulate the layers already containing weights
with torch.no_grad():
    conv1.weight.copy_(torch.randn(1, 1, 3, 3))
    conv1.bias.copy_(torch.randn(1))
    conv2.weight.copy_(torch.randn(1, 1, 3, 3))
    conv2.bias.copy_(torch.randn(1))
    conv3.weight.copy_(torch.randn(1, 1, 3, 3))
    conv3.bias.copy_(torch.randn(1))

# Collect weights and biases
weights = [conv1.weight, conv2.weight, conv3.weight]
biases = [conv1.bias, conv2.bias, conv3.bias]


# Average the weights
avg_weight = torch.mean(torch.stack(weights), dim=0)

# Average the biases
avg_bias = torch.mean(torch.stack(biases), dim=0)

# Create a new convolutional layer and load the averaged parameters
avg_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

with torch.no_grad():
    avg_conv.weight.copy_(avg_weight)
    avg_conv.bias.copy_(avg_bias)

# Example Usage
input_tensor = torch.randn(1, 1, 10, 10)
output_conv1 = conv1(input_tensor)
output_avg_conv = avg_conv(input_tensor)

print("Output from original Convolution Layer 1: ", output_conv1)
print("Output from averaged convolutional layer: ", output_avg_conv)
```

The operation remains conceptually identical, even with the weights having four dimensions (`[out_channels, in_channels, kernel_height, kernel_width]`). Averaging happens correctly on tensors with these dimensionalities with the `torch.mean` method. The new convolutional layer, `avg_conv`, now holds parameters representing the arithmetic mean of the source convolutional layers. The printed outputs demonstrate that both original and average convolutional layers output different values based on their learned weights.

**Example 3: Averaging Layers from different modules**

It’s also possible to extract layers from different network modules and average them. Here we illustrate that averaging weights from equivalent layers across different modules is just an extraction and averaging operation on the parameters.

```python
import torch
import torch.nn as nn

# Define a simple module
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Create three example modules
module1 = MyModule()
module2 = MyModule()
module3 = MyModule()

# Placeholder to simulate the modules already containing weights
with torch.no_grad():
    module1.linear1.weight.copy_(torch.randn(5, 10))
    module1.linear1.bias.copy_(torch.randn(5))
    module1.linear2.weight.copy_(torch.randn(2,5))
    module1.linear2.bias.copy_(torch.randn(2))
    module2.linear1.weight.copy_(torch.randn(5, 10))
    module2.linear1.bias.copy_(torch.randn(5))
    module2.linear2.weight.copy_(torch.randn(2,5))
    module2.linear2.bias.copy_(torch.randn(2))
    module3.linear1.weight.copy_(torch.randn(5, 10))
    module3.linear1.bias.copy_(torch.randn(5))
    module3.linear2.weight.copy_(torch.randn(2,5))
    module3.linear2.bias.copy_(torch.randn(2))



# Collect all linear1 layers
linear1_layers = [module1.linear1, module2.linear1, module3.linear1]
# Collect all linear2 layers
linear2_layers = [module1.linear2, module2.linear2, module3.linear2]


# Function to average a list of layers
def average_layers(layers):
  weights = [layer.weight for layer in layers]
  biases = [layer.bias for layer in layers]
  avg_weight = torch.mean(torch.stack(weights), dim=0)
  avg_bias = torch.mean(torch.stack(biases), dim=0)
  return avg_weight, avg_bias

#Average layers linear1 and linear2
avg_lin1_weight, avg_lin1_bias = average_layers(linear1_layers)
avg_lin2_weight, avg_lin2_bias = average_layers(linear2_layers)

# Create a new model and populate with averaged weights
avg_module = MyModule()

with torch.no_grad():
    avg_module.linear1.weight.copy_(avg_lin1_weight)
    avg_module.linear1.bias.copy_(avg_lin1_bias)
    avg_module.linear2.weight.copy_(avg_lin2_weight)
    avg_module.linear2.bias.copy_(avg_lin2_bias)

# Example usage
input_tensor = torch.randn(1, 10)
output_module1 = module1(input_tensor)
output_avg_module = avg_module(input_tensor)


print("Output from original module 1: ", output_module1)
print("Output from the averaged module: ", output_avg_module)

```
In this example, the averaging is performed on the layers extracted from different modules. The code demonstrates that the average weights can then be used to create an averaged module with the same architecture as the original modules. Again we observe differences in the outputs. This highlights the versatility of this method.

**Resource Recommendations**

For a deeper understanding of PyTorch tensors, I recommend studying the core PyTorch documentation on tensor manipulation and operations. Specifically, explore the sections detailing tensor creation, indexing, and mathematical operations such as `torch.mean`, `torch.stack`.  To understand network parameters, examine the documentation regarding the `nn.Module` class, parameter initialization, and specifically how to access the `.weight` and `.bias` attributes. These resources are crucial to grasping the practical details of building networks and also manipulating them. Furthermore, studying code examples of neural networks that have been trained, such as in common open source repositories, can help illuminate the principles of working with PyTorch. These resources provide the essential background to understand and implement complex layer manipulation.
