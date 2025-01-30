---
title: "How can a Functional ResNet50 model be divided into multiple layers?"
date: "2025-01-30"
id: "how-can-a-functional-resnet50-model-be-divided"
---
The core challenge when dividing a Functional ResNet50 model into layers stems from its inherently interconnected, non-sequential structure involving skip connections and residual blocks, unlike a simple linear model. The traditional `torch.nn.Sequential` method isn't directly applicable. Instead, I've found success by leveraging the model's named modules and constructing custom wrapper functions or classes to extract specific sections while maintaining the correct data flow and preserving the functional paradigm.

The ResNet50, when instantiated using pre-trained weights in libraries like `torchvision`, is represented as a hierarchical collection of modules. These modules, accessed via their names, represent stages of the network, including convolution layers, batch normalization, and the crucial residual blocks. The key to splitting the model is to identify the boundaries between these stages based on the named modules and then craft code that selectively executes those modules in a desired sequence.

One common split I use frequently involves dividing a ResNet50 into three sections: an initial feature extraction (comprising the initial convolutional and pooling layers, and often the first residual block group), a mid-level processing stage (consisting of the subsequent residual block groups), and finally the classification section (including the average pooling and final fully connected layer). These aren't absolute boundaries, but they're logical for tasks like transfer learning where you might freeze early layers or experiment with different depths.

My approach uses function composition. I construct functions to perform each section individually, and then chain those function to create partial models. This method retains the Functional style because we are directly controlling how the modules operate on an input, rather than encapsulating all the operations within a monolithic module.

Here's a concrete example:

```python
import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet_features(x, model):
    """Extracts features up to the first block group."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)  # Layer 1 is the first block group.
    return x

def get_resnet_mid(x, model):
    """Processes features through mid-level residual groups."""
    x = model.layer2(x)
    x = model.layer3(x)
    return x

def get_resnet_classifier(x, model):
    """Extracts and processes classification features."""
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.fc(x)
    return x

if __name__ == "__main__":
    resnet = models.resnet50(pretrained=True)
    resnet.eval() # set to eval mode

    # sample input
    sample_input = torch.randn(1, 3, 224, 224)

    # Example of using the functions
    features = get_resnet_features(sample_input, resnet)
    mid_features = get_resnet_mid(features, resnet)
    output = get_resnet_classifier(mid_features, resnet)

    print("Output Shape:", output.shape)  # Expected shape will be [1, 1000] given the 1000 class ImageNet task.

    # Partial model 1: input to feature extraction layers
    partial_model_1 = lambda x: get_resnet_features(x, resnet)
    output_partial_1 = partial_model_1(sample_input)
    print("Partial Model 1 output:", output_partial_1.shape) # Shape expected [1, 256, 56, 56]

    # Partial model 2: input to final classifier layers
    partial_model_2 = lambda x: get_resnet_classifier(get_resnet_mid(x, resnet), resnet)
    output_partial_2 = partial_model_2(features)
    print("Partial Model 2 output:", output_partial_2.shape) # Expected shape [1, 1000]
```

In this example, I have defined three functions, `get_resnet_features`, `get_resnet_mid`, and `get_resnet_classifier`.  Each function takes an input tensor and the full ResNet model, and then performs a forward pass through the specified parts of the network. The `get_resnet_features` function covers the initial convolution and the first block group (`layer1`). The `get_resnet_mid` function then covers the next two residual block groups (`layer2` and `layer3`). Finally, `get_resnet_classifier` performs the final block group (`layer4`), average pooling, and fully connected layer.

I have also composed these functions to create partial models, and included assertions that demonstrate how these can be used in sequence or independently to perform partial forward passes.  These can be further composed if one requires more granular control. I also include the `eval()` call for the model, as it's best practice to set eval mode before passing data for inference as otherwise batchnorm and dropout layers will behave differently.

For slightly more complex use cases, you might want to use a custom class that encapsulates the splitting logic and provides more organized access. The following example shows how to accomplish that:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SplitResNet(nn.Module):
    def __init__(self, model):
        super(SplitResNet, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x

    def forward_mid(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward_classifier(self, x):
         x = self.layer4(x)
         x = self.avgpool(x)
         x = torch.flatten(x, 1)
         x = self.fc(x)
         return x

if __name__ == "__main__":
    resnet = models.resnet50(pretrained=True)
    resnet.eval()
    split_resnet = SplitResNet(resnet)

    sample_input = torch.randn(1, 3, 224, 224)

    # Call forward functions
    features = split_resnet.forward_features(sample_input)
    mid_features = split_resnet.forward_mid(features)
    output = split_resnet.forward_classifier(mid_features)

    print("Output Shape:", output.shape)  # Expected [1, 1000]

    # Partial Model example via composing methods
    partial_model_1 = lambda x: split_resnet.forward_features(x)
    output_partial_1 = partial_model_1(sample_input)
    print("Partial Model 1 output:", output_partial_1.shape) # Expected [1, 256, 56, 56]

    partial_model_2 = lambda x : split_resnet.forward_classifier(split_resnet.forward_mid(x))
    output_partial_2 = partial_model_2(features)
    print("Partial Model 2 output:", output_partial_2.shape) # Expected [1, 1000]
```

Here, I have inherited the `nn.Module` and created a `SplitResNet` class. In the constructor (`__init__`), we extract all the named layers, and then I have created individual functions for each section of the model to allow for the desired splitting of the operations. This is a cleaner and more organized method than the previous version, though it is slightly more verbose. Both of the methods provide functional access and do not change the structure or behavior of the base model, and can be adapted for other models with similarly nested layers.

Lastly, I've also employed a more fine-grained strategy when building models for distillation or specialized feature extraction. Instead of simply using layer groupings, you can create custom extraction functions that slice within specific `layer1`, `layer2`, etc. groups, giving precise control.

```python
import torch
import torch.nn as nn
import torchvision.models as models

def get_block_section(x, layer, block_start, block_end):
    """Extracts a subsection of blocks within a given residual layer."""
    out = x
    for i in range(block_start, block_end):
      out = layer[i](out)
    return out


if __name__ == "__main__":
  resnet = models.resnet50(pretrained=True)
  resnet.eval()
  sample_input = torch.randn(1, 3, 224, 224)
  x = resnet.conv1(sample_input)
  x = resnet.bn1(x)
  x = resnet.relu(x)
  x = resnet.maxpool(x)

  layer1 = resnet.layer1
  layer2 = resnet.layer2
  layer3 = resnet.layer3
  layer4 = resnet.layer4
  # Example use case for only first two blocks of resnet layer 2
  output_layer2_section = get_block_section(x, layer2, 0, 2)
  print("Output Layer2 Section: ", output_layer2_section.shape) # Shape expected [1, 512, 28, 28]

  #Example use case of all resnet layer 1 blocks
  output_layer1_full = get_block_section(x, layer1, 0, len(layer1))
  print("Output Layer 1 Full", output_layer1_full.shape) # Shape expected [1, 256, 56, 56]

  #Example composing partial layers: layer 1 followed by layer 2 blocks 0 to 2
  layer2_block_partial = lambda input: get_block_section(input, layer2, 0, 2)
  layer1_full = lambda input: get_block_section(input, layer1, 0, len(layer1))

  output_combined = layer2_block_partial(layer1_full(x))
  print("Output Combined: ", output_combined.shape) # Shape expected [1, 512, 28, 28]
```
In this more fine-grained example, the `get_block_section` allows you to extract subsections of individual layers (like `layer1`). This provides the most granular control of all the examples, and can be applied to arbitrary layers. The example also shows how this partial layer can be used in composition for further slicing of the model.

When delving further into this topic, I recommend exploring the following materials: the original ResNet paper, which offers a thorough look into its architecture; official PyTorch documentation for `torch.nn.Module`, where the named modules are explained in more detail; and tutorials on transfer learning, which often delve into extracting intermediate features from pre-trained models for downstream tasks.  Furthermore, examining the source code of the `torchvision.models` will reveal the precise structure and module naming of the layers, which I have found to be invaluable during my work.
