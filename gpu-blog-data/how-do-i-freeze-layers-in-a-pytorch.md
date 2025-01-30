---
title: "How do I freeze layers in a PyTorch neural network?"
date: "2025-01-30"
id: "how-do-i-freeze-layers-in-a-pytorch"
---
The persistent requirement to evaluate a neural network’s intermediate activations without impacting subsequent gradient computations necessitates freezing specific layers. This operation, frequently employed in transfer learning, fine-tuning, or feature extraction, involves preventing weight updates within designated parts of the network during the training process. In essence, frozen layers become static during backpropagation.

Freezing layers in PyTorch involves manipulating the `requires_grad` attribute of the parameters contained within those layers. By default, all parameters in a model have `requires_grad` set to `True`, which instructs PyTorch to track their gradients for optimization. To freeze a layer, this attribute must be set to `False` for all of its constituent parameters. This ensures that backpropagation computations will not flow through that layer, thereby preventing any parameter updates when an optimizer is called.

Several strategies facilitate layer freezing depending on the scope of the change required. When freezing a complete layer, such as a linear or convolutional layer, it’s straightforward to iterate through the layer's parameters and set their `requires_grad` flag. For instance, consider a pretrained ResNet model used for a downstream classification task:

```python
import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in resnet18.parameters():
    param.requires_grad = False

# Replace the final fully connected layer to match output classification
num_ftrs = resnet18.fc.in_features
resnet18.fc = torch.nn.Linear(num_ftrs, 10) #Assuming 10 classes
```

This code loads a pretrained ResNet18 model, iterates through each parameter, and sets `requires_grad` to `False`, effectively freezing the entire feature extraction backbone. The last fully connected layer is then replaced to align with the required output size of the classification problem. Consequently, during training, only the weights in the final fully connected layer will be adjusted through backpropagation. This approach is commonly deployed to leverage robust features learned by the pretrained model for a new, often smaller, dataset.

More nuanced freezing can be achieved by explicitly accessing specific layers within the model. For models with named layers, like `nn.Sequential` containers or custom networks with identifiable modules, targeted freezing becomes more accessible. Consider a scenario where only the initial convolutional layers of a custom network are to be frozen. Assume a network has named modules such as `conv1`, `conv2`, and `fc`. The following code demonstrates how to selectively freeze the `conv1` and `conv2` modules:

```python
import torch.nn as nn

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc = nn.Linear(32*10*10, 10) #Assuming 10 output classes and input image size of 32x32 with padding and stride to reduce input size to 10x10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32*10*10)
        x = self.fc(x)
        return x

model = CustomNetwork()

# Freeze conv1 and conv2 layers
for name, param in model.named_parameters():
    if 'conv1' in name or 'conv2' in name:
        param.requires_grad = False

#Example of using the model
input_tensor = torch.randn(1, 3, 32, 32)
output = model(input_tensor)
```
Here, `named_parameters()` yields parameter names and the corresponding parameter tensors. By checking if the layer name (`name`) contains `'conv1'` or `'conv2'`, the `requires_grad` attribute of the target parameters can be selectively set to `False`. Only the parameters in the fully connected layer (`fc`) and layers which are not directly contained in 'conv1' and 'conv2' are going to be optimized during training.

A third approach applies a filter at the module level, targeting entire modules rather than individual parameters. This strategy is particularly useful when the network architecture consists of named sub-modules. For instance, if a network has several named blocks, like `encoder`, `decoder`, and `classifier`, freezing the entire `encoder` can be directly achieved as follows:

```python
import torch.nn as nn

class NestedNetwork(nn.Module):
    def __init__(self):
        super(NestedNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(32,16, kernel_size=2, stride=2),
           nn.ReLU(),
          nn.ConvTranspose2d(16,3,kernel_size=2, stride=2)
        )

        self.classifier = nn.Linear(16 * 8 * 8 , 10)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        x = self.classifier(encoded.view(-1,16*8*8))
        return x

model = NestedNetwork()

#Freeze the entire encoder
for param in model.encoder.parameters():
  param.requires_grad=False

# Example of using the model
input_tensor = torch.randn(1, 3, 32, 32)
output = model(input_tensor)

```

In this example, the nested model contains encoder, decoder, and classifier blocks. The entire `encoder` module's parameters are frozen in place via module-level iteration. This means that during backpropagation, the parameters within the encoder will remain unchanged while the decoder and classifier parts are updated. This provides a way to target complex nested parts of the model architecture without excessive access and manipulation of parameters one by one.

These strategies provide a comprehensive framework for layer freezing in PyTorch, addressing common scenarios encountered in transfer learning and fine-tuning procedures. I have extensively used these methods in past projects involving pretraining and fine-tuning vision models, as well as in training complex architectures such as Autoencoders and GANs, where it is crucial to keep certain parameters fixed.

For further learning, resources such as the official PyTorch documentation provide detailed API specifications for parameter attributes and model manipulation. Tutorials focused on transfer learning strategies will typically include code snippets demonstrating layer freezing with varied levels of granularity. The deep learning literature also contains many papers where variations of these methods are employed, which can also provide insights. Open-source notebooks demonstrating fine-tuning and other model manipulation tasks are valuable starting points to understand the practical application of freezing. Studying such materials in conjunction with experimentation using your own datasets will quickly solidify your understanding and build expertise.
