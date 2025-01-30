---
title: "How to freeze parameters except the classifier in PyTorch?"
date: "2025-01-30"
id: "how-to-freeze-parameters-except-the-classifier-in"
---
In my experience working with transfer learning in PyTorch, a common requirement is to freeze the pre-trained feature extraction layers while fine-tuning only the final classification head. This practice prevents the backpropagation algorithm from modifying the weights learned during the pre-training phase, allowing the model to adapt specifically to the target task's classification problem. The approach hinges on selectively setting the `requires_grad` attribute of model parameters. By default, PyTorch sets `requires_grad` to `True` for all parameters, meaning that gradients will be calculated for them during backpropagation. To freeze specific layers, we must disable gradient calculations for those parameters.

The core concept involves iterating through the model's named parameters, examining the names to identify which layers are intended to be frozen, and then setting the `requires_grad` attribute of those parameters to `False`. This has a crucial impact on the computational graph, as the backward pass will not compute gradients for the frozen parameters, saving computation time and preventing unintended weight updates within the pre-trained portion. The classification head, on the other hand, must have `requires_grad` set to `True` or, if it’s been cloned from a pre-existing classification head, already has this set by default. The optimizer will then only update the weights of the classifier.

Here’s how I have routinely implemented this technique, starting with an illustrative example using a fictional convolutional model. Assume that we have a model with a structure consisting of three convolutional blocks followed by a fully connected classifier:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ConvModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes) # Assume input size is adjusted here
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the model
model = ConvModel(num_classes=10)

# Freeze all parameters except the classifier
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False
    else:
        print(f"Classifier parameter {name} will be trained")

# Verify that all parameters except the classifier are frozen
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Frozen parameter: {name}")

# Now you can proceed with training only the fully connected layer.
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
```

In this example, I first define a simple convolutional model. Then, the crucial part is in the loop over `model.named_parameters()`. The names of the parameters allow for precise selection. Here I check to see if the parameter's name contains 'fc', indicating that it's part of the classifier. If it does not contain 'fc', I set `requires_grad` to `False`, thus freezing that layer. The parameters of the classifier remain trainable. After the loop, I confirm the parameters that have been frozen by again iterating through them and checking `requires_grad`. Lastly, I create an optimizer, making use of the filter function to only pass parameters whose attribute `requires_grad` evaluates to `True`.

A more complex model, like a pre-trained ResNet, requires a slightly different approach. Often pre-trained models are structured as a series of modules with complex naming conventions. Let’s consider a use case using a fictitious `PretrainedResNet`. Assume this ResNet is a class with a structure like `self.conv1`, `self.layer1`, `self.layer2`, `self.layer3`, `self.layer4` and `self.fc`. In general, the feature extraction part consists of all layers up to layer 4, and `self.fc` is the classifier.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PretrainedResNet(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
        							 nn.BatchNorm2d(64),
        							 nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(64))

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(128))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = PretrainedResNet(num_classes = 10)


for name, param in model.named_parameters():
   if 'fc' not in name:
      param.requires_grad = False

# Confirm the freezing
for name, param in model.named_parameters():
    if not param.requires_grad:
      print(f"Frozen parameter: {name}")

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
```

The principle here is the same as before. I loop through all parameters and check the parameter name. However, the `if 'fc' not in name:` conditional statement might be too simplistic if the model contained further classification layers that should also be included in the training. The condition would need to be adjusted to match the names of the layers you intend to freeze. We will examine such an instance in the third example.

The final example demonstrates how to freeze all layers *except* the classifier and some other very specific layers within the feature extraction, a situation I've encountered often while fine-tuning. Consider a variation of our `PretrainedResNet`. We want to freeze every layer *except* the classifier (`self.fc`) *and* the very last convolutional block (`self.layer4`).

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PretrainedResNetVariation(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedResNetVariation, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64))

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(128))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = PretrainedResNetVariation(num_classes = 10)
# Freeze all parameters except the classifier and the last convolutional block
for name, param in model.named_parameters():
    if 'fc' not in name and 'layer4' not in name:
        param.requires_grad = False


for name, param in model.named_parameters():
    if param.requires_grad:
      print(f"Trainable parameter: {name}")

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
```
This time, the condition is  `if 'fc' not in name and 'layer4' not in name:`. This statement checks to see if a parameter's name contains neither 'fc' nor 'layer4'. If it contains neither, its `requires_grad` attribute is set to `False`, freezing it. This is a nuanced form of parameter freezing that allows for a finer level of control.

In summary, freezing layers in PyTorch is fundamental when employing transfer learning. It's important to understand the specific naming conventions of the model's layers to accurately freeze the desired parts and to have the flexibility to adjust which parameters are trainable. Beyond the PyTorch documentation, further instruction can be found in books covering deep learning with Python. Online tutorials covering transfer learning often provide practical demonstrations. Finally, many university courses on deep learning, publicly available online, will cover these concepts.
