---
title: "When should I use PyTorch's `nn.ModuleList` vs. `nn.Sequential`?"
date: "2025-01-30"
id: "when-should-i-use-pytorchs-nnmodulelist-vs-nnsequential"
---
The fundamental distinction between `nn.ModuleList` and `nn.Sequential` in PyTorch centers on their intended use for managing layers within a neural network: `nn.Sequential` enforces a linear, forward-pass execution of its constituent modules, while `nn.ModuleList` provides a flexible container for modules, permitting more intricate control over their processing. Having implemented numerous network architectures during my time developing image classification and sequence modeling algorithms, I've found that understanding this difference is critical for building scalable and adaptable models.

`nn.Sequential` shines when the data flow is inherently a series of transformations, one after the other. Each module's output serves as the input to the subsequent module. This structure is ideal for situations like a classic multi-layer perceptron (MLP), where you propagate data sequentially through linear and activation layers. The convenience offered by `nn.Sequential` stems from its ability to automatically handle the chaining of module calls during the forward pass. You define the modules in a specific order, and the forward method internally calls each in that same order, passing the output of the previous as input to the next. This drastically reduces boilerplate code and improves readability for simple, feed-forward architectures.

On the other hand, `nn.ModuleList` operates as a Python list specifically designed for modules. It registers the contained modules as parameters of the parent module, making them visible to PyTorch's parameter tracking and optimization machinery. However, it does not automatically call the modules sequentially, or enforce any forward propagation pattern on its own. This is left entirely to the developer. This level of freedom is necessary for architectures that have branches, skip connections, or where the application of layers is not strictly linear. Consider networks like residual networks (ResNets) with their skip connections or recurrent neural networks (RNNs) with iterative operations – these are cases where `nn.Sequential` is insufficient. `nn.ModuleList` gives us the flexibility to individually select which modules to activate at each step of the computation. We explicitly write code to control how the modules in the list will be utilized.

Let’s delve into concrete code examples.

**Example 1: Demonstrating `nn.Sequential` for a simple MLP**

```python
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Usage
model = SimpleMLP(input_size=10, hidden_size=50, output_size=2)
input_tensor = torch.randn(1, 10)
output_tensor = model(input_tensor)
print(output_tensor.shape)
```

In this example, `nn.Sequential` elegantly constructs a simple MLP. The `forward` method simply passes the input `x` through the sequential container `self.layers`. PyTorch then automatically applies each layer in the defined order: linear transformation, ReLU activation, and then the final linear transformation. This code is clean, concise, and self-documenting for a standard feed-forward operation.

**Example 2: Utilizing `nn.ModuleList` for a ResNet-like structure**

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Handle channel change for skip connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
          self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNetLike(nn.Module):
    def __init__(self, in_channels, num_blocks, hidden_channels):
        super(ResNetLike, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_channels,hidden_channels) for _ in range(num_blocks)])


        self.fc = nn.Linear(hidden_channels, 10) # Example output layers.

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        for block in self.residual_blocks:
          out = block(out)

        out = out.mean(dim=[2,3]) # Global Average Pooling for demonstration
        out = self.fc(out)

        return out

# Usage
model = ResNetLike(in_channels=3, num_blocks=3, hidden_channels=64)
input_tensor = torch.randn(1, 3, 32, 32)
output_tensor = model(input_tensor)
print(output_tensor.shape)
```

Here, `nn.ModuleList` facilitates a more sophisticated architecture similar to ResNets. We cannot use `nn.Sequential` here because the forward pass is not a simple linear chain of operations. Residual connections require explicitly adding the input to the output of a subset of layers.  We loop through the `self.residual_blocks`, calling each block in turn, managing the skipping connections and iterative application of blocks ourselves. This illustrates how `nn.ModuleList` provides the flexibility needed for non-trivial architectures. It should be emphasized that we are manually looping through the modules in the `forward` method, which makes the control flow more customizable at the cost of manual iteration.

**Example 3: Demonstrating conditional module application with `nn.ModuleList`**

```python
import torch
import torch.nn as nn

class ConditionalModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConditionalModel, self).__init__()
        self.module_list = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ])

    def forward(self, x, condition):
        out = x
        if condition:
            out = self.module_list[0](out) # Linear Layer
            out = self.module_list[1](out) # ReLU
        out = self.module_list[2](out) # Apply the second Linear layer
        return out

# Usage
model = ConditionalModel(input_size=10, hidden_size=50, output_size=2)
input_tensor = torch.randn(1, 10)
output_tensor_true = model(input_tensor, condition=True)
output_tensor_false = model(input_tensor, condition=False)

print(output_tensor_true.shape)
print(output_tensor_false.shape)
```

This example demonstrates the flexibility `nn.ModuleList` offers in applying modules conditionally. The `condition` argument in the `forward` method determines if the first two layers of `self.module_list` should be applied. This kind of conditional branching within the forward pass is not easily achieved with `nn.Sequential`. I've used this technique for adaptive computation, where the specific layers applied depend on input characteristics or other dynamic conditions. The capability to select or skip modules offers a greater level of design control for sophisticated deep learning models.

In summary, choose `nn.Sequential` for simple, linear stacks of layers where the forward flow is implicitly determined by the module's order. Opt for `nn.ModuleList` when your architecture requires more complex control, branching logic, conditional operations, or explicit looping through modules, as demonstrated in ResNet and other more dynamic network architectures.

For those seeking further understanding, I recommend reviewing the official PyTorch documentation, which provides the most authoritative guide. The examples in the documentation are very well written, and exploring these is a must for a deep dive. Study implementations of commonly used architectures like ResNet, VGG, and RNNs, paying close attention to how they organize modules, and experimenting with modifications is another highly effective learning technique. Finally, researching papers on recent network architectures will expose new and interesting use cases for `nn.ModuleList`. By understanding both the advantages and limitations of these two containers, you will be able to create more refined and expressive neural network models with PyTorch.
