---
title: "Can ResNet be implemented with only PyTorch's basic block?"
date: "2025-01-30"
id: "can-resnet-be-implemented-with-only-pytorchs-basic"
---
The foundational concept to grasp is that a ResNet architecture, while complex in its overall topology, is fundamentally built upon repeated applications of a relatively simple *residual block*. PyTorch provides a basic building block that can, with the right arrangement, indeed implement the core principles of a ResNet. The key is understanding how to use this block to create the necessary stacking and downsampling behavior. In my experience building image classification models, this approach, although seemingly basic, has proved both effective and flexible.

The core idea behind ResNet is the introduction of skip connections, also known as shortcut connections, that allow the network to learn residual mappings. Instead of learning an underlying function *H(x)* directly, the network learns a residual function *F(x) = H(x) - x*, where *x* is the input. This change in perspective eases the optimization of deeper networks, preventing the notorious vanishing gradient problem. PyTorch provides a `torch.nn.Conv2d`, `torch.nn.BatchNorm2d`, and `torch.nn.ReLU` as primitives, which can easily assemble a residual block.

We can implement a basic residual block using PyTorch, which does not inherently have a `BasicBlock` class as in some higher-level libraries, and use that block to construct a ResNet. This is achieved by defining our own block structure and iterating over this block in a structured way. We also need to deal with downsampling and feature map expansion which will be achieved through appropriate convolutions and shortcut connections. The standard ResNet uses layers with `stride=2` to reduce spatial dimensions, typically at the beginning of every "stage" of the network. We accomplish this within our building block through modified parameters.

Here's a breakdown of the building blocks, and how they can be used:

**1. Implementing the Basic Residual Block:**

This involves defining a class that contains two convolutional layers, batch normalization, ReLU activations and a shortcut connection if the feature maps change in dimensions. Critically, we must manage the skip connection. When the input and output feature map dimensions are the same, the skip connection is simply an identity mapping. When dimensions change, a 1x1 convolutional layer is used with appropriate stride.

```python
import torch
import torch.nn as nn

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual) #skip connection
        out = self.relu(out)
        return out
```

**Commentary:** This `BasicResidualBlock` implements the fundamental building block. `stride=1` keeps dimensions. We include batch normalization and the rectified linear unit activation to stabilize and non-linearize the network. The `shortcut` handles dimension changes when the input and output of a block have different number of channels, or when downsampling is required via stride.  This structure directly translates the mathematical idea of *H(x) = F(x) + x*. The `inplace=True` argument to the ReLU saves memory at the cost of slightly slower processing.

**2. Assembling the ResNet Architecture:**

With the block in place, we define our own ResNet class that instantiates these blocks, arranged in a sequential manner akin to a standard ResNet architecture. This approach allows us to control the structure by simply varying the number of blocks within a stage as well as the channels and strides.

```python
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
```

**Commentary:**  The `ResNet` class instantiates the initial convolution and max pooling operation, followed by several `_make_layer` operations. The `_make_layer` function creates several blocks and handles the increasing number of feature maps. It receives as input the output channels of the stage and number of blocks to create. We then use a global average pooling, flatten the output into a vector and use a final fully connected layer to output the prediction. In a real training setup, the initial parameters would be initialized correctly as well.

**3.  Instantiating the Network and Testing:**

Finally, we demonstrate how to instantiate a ResNet with specific architecture, and we generate a random input tensor to verify the code is correct.

```python
if __name__ == '__main__':
    # Example: ResNet-18 configuration
    num_blocks = [2, 2, 2, 2]
    model = ResNet(num_blocks=num_blocks, num_classes=1000)

    # Example input
    input_tensor = torch.randn(1, 3, 224, 224) # Example batch of one image, 3 channels, 224x224

    # Forward pass
    output_tensor = model(input_tensor)

    # Check output size
    print("Output shape:", output_tensor.shape)
```

**Commentary:**  This is an example of a ResNet18-like configuration. We can change num_blocks to create different sizes of resnets, such as ResNet-34 (i.e, `[3, 4, 6, 3]`). This example verifies that the code compiles and returns a tensor output that corresponds to the specified number of classes, which shows that the basic building blocks can be used to construct a ResNet architecture. In practice this model can be trained on a standard classification dataset.

**Resource Recommendations:**

To deepen the understanding of residual networks and their implementation, I would recommend exploring the original ResNet paper titled "Deep Residual Learning for Image Recognition", and any introductory material that explains PyTorch’s neural network building blocks. Studying codebases that implement similar models from scratch, or those that use frameworks such as PyTorch Lightning can also deepen the comprehension of deep learning modeling.
Specifically, understand the different ResNet variations (e.g., ResNet-18, ResNet-34, ResNet-50) and how the number of blocks per stage impacts overall performance. Explore how they address the vanishing gradients problem, and why such an approach works.
Also, focus on the principles of convolutional layers, batch normalization, and ReLU activation functions, which form the basis of almost all modern computer vision models. In-depth knowledge of these fundamentals is critical to successfully building custom neural networks. Lastly, study the process for training, validating, and testing image classifiers, particularly hyperparameter optimization.
