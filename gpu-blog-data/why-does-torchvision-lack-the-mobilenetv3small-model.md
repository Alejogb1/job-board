---
title: "Why does torchvision lack the mobilenet_v3_small model?"
date: "2025-01-30"
id: "why-does-torchvision-lack-the-mobilenetv3small-model"
---
The absence of `mobilenet_v3_small` within `torchvision.models` stems primarily from a deliberate design choice prioritizing maintainability and resource allocation rather than a technical limitation preventing its inclusion.  My experience building custom models and deploying them in constrained environments has taught me firsthand the critical need for efficient model selection and the careful tradeoffs inherent in library maintenance.  `torchvision` focuses on providing readily usable, pre-trained models that are either widely adopted benchmarks or representative examples of significant architectures. The inclusion of a model like `mobilenet_v3_small` presents a specific context of diminishing returns given the resource allocation needed to support it.

The fundamental reason revolves around the concept of diminishing returns within a library of this scale.  `torchvision` aims to provide core, generally applicable models that serve as good starting points for a range of vision tasks. Adding each variant of an architecture, especially smaller ones like `mobilenet_v3_small`, increases the testing matrix and maintenance overhead exponentially. The core benefits of small model variants, like parameter reduction and quicker inference, are primarily valuable to end users with very specific computational constraints; they are less critical to the initial development or benchmarking workflows that `torchvision` caters to. Larger, more commonly used models like `mobilenet_v2` and `mobilenet_v3_large` provide a solid baseline for most development needs, and users can further modify those to suit their particular requirements, often creating custom small models that more precisely fit their problem space. This approach enables more efficient use of `torchvision` maintainers’ time and resources. The choice of focusing on core models ensures these fundamental building blocks get consistent attention and are well maintained across new PyTorch releases.

It is not an issue of technological infeasibility.  PyTorch and its underlying graph computational system are entirely capable of implementing `mobilenet_v3_small`. The structure of the network itself is well-defined and straightforward to program, as evidenced by its existence in other open-source repositories. The barrier is not the network architecture itself but rather the maintenance overhead.

To illustrate, let's examine how one would implement a simplified version of `mobilenet_v3_small` using standard PyTorch layers, mimicking the core structure and design.

**Code Example 1: Basic Block Construction**

This code snippet focuses on the basic building block of MobileNetV3 – the inverted residual block with a squeeze-and-excitation layer (SE). This demonstrates the core architectural elements that would be common across model variants.

```python
import torch
import torch.nn as nn

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, use_se, nl):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expansion))
        self.use_shortcut = stride == 1 and in_channels == out_channels
        layers = []
        # Expand conv
        if expansion != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nl)

        # Depthwise conv
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nl)

        if use_se:
            layers.append(SEModule(hidden_dim))

        # Linear Projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.layers(x)
        else:
             return self.layers(x)

```

This example highlights how straightforward it is to construct the basic building block of `mobilenet_v3_small`, utilizing standard PyTorch layers, demonstrating that the network structure is easily implementable.  The inverted residual block and the Squeeze-and-Excitation (SE) module are present.

**Code Example 2: Simplified Model Structure**

Here we showcase a simplified structure of the MobileNetV3 Small, indicating how the blocks could be combined to form a basic model. Note this is a significantly simplified model and doesn't represent the true `mobilenet_v3_small` in its entirety, but serves as an example of how a user could assemble such a model.

```python
class SimplifiedMobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.nl1 = nn.Hardswish(inplace=True)

        self.blocks = nn.Sequential(
            InvertedResidualBlock(16, 16, expansion=1, stride=1, use_se=True, nl=nn.ReLU(inplace=True)), # Simplified Example Block
            InvertedResidualBlock(16, 24, expansion=4.5, stride=2, use_se=False, nl=nn.ReLU(inplace=True)),  # Simplified Example Block
            InvertedResidualBlock(24, 24, expansion=3.6, stride=1, use_se=False, nl=nn.ReLU(inplace=True)), # Simplified Example Block
            InvertedResidualBlock(24, 40, expansion=4, stride=2, use_se=True, nl=nn.Hardswish(inplace=True)), # Simplified Example Block
            InvertedResidualBlock(40, 40, expansion=6, stride=1, use_se=True, nl=nn.Hardswish(inplace=True)), # Simplified Example Block
        )


        self.conv2 = nn.Conv2d(40, 96, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.nl2 = nn.Hardswish(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.nl1(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.nl2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```
This illustrates the ease with which layers can be assembled into a model like `mobilenet_v3_small`. The core building blocks are the inverted residual blocks, combined with other common layers. This is a vastly simplified structure; the actual model has many more layers and complex tuning of the parameters, yet the building blocks are fundamentally the same.

**Code Example 3: Model Instantiation and Usage**

Finally, this example demonstrates a basic usage pattern to show that a model constructed in such a way is usable without any modifications. This is also where the user would load custom pre-trained weights.

```python
if __name__ == '__main__':
    model = SimplifiedMobileNetV3Small()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print("Output size:", output.size()) # Output size: torch.Size([1, 1000])
```
This illustrates that our `SimplifiedMobileNetV3Small` class, like a standard `torchvision` model, can be instantiated and used readily, showcasing the direct path from constructing the model architecture to using it for inference.

Based on my experience, the absence of `mobilenet_v3_small` in `torchvision` should not be perceived as a limitation, but rather as a conscious decision to prioritize the resource allocation and long-term maintainability of the library. Users requiring this specific model variant can readily build it themselves using the readily available PyTorch primitives and the architecture definition found in the original paper or in other open-source libraries. The choice to omit it reflects a broader understanding of the tradeoffs in library development; maintaining a compact core set of highly used models is a far more efficient approach.

To further explore building custom models and related concepts, I suggest consulting resources that delve into convolutional neural network architectures, such as research papers on MobileNetV3, and in-depth tutorials from the PyTorch documentation.  Investigating resources focusing on model design for resource-constrained devices can also provide practical insights. Examining other high-quality open-source repositories can provide inspiration on how to implement this and similar models. Furthermore, tutorials that explore model construction, especially using PyTorch, would greatly aid in mastering this concept.
