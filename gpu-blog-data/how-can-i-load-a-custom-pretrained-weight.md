---
title: "How can I load a custom pretrained weight in PyTorch Lightning?"
date: "2025-01-30"
id: "how-can-i-load-a-custom-pretrained-weight"
---
Pretrained weights, often crucial for achieving state-of-the-art results in deep learning, are not automatically loaded when using PyTorch Lightning's standard `Trainer` and `LightningModule` structure. A direct loading mechanism must be explicitly coded within the `LightningModule`. I've encountered this during several projects, particularly when porting models from research codebases lacking the structured approach offered by PyTorch Lightning. This often requires navigating checkpoint structures that differ from those created by Lightning itself.

The primary challenge stems from PyTorch Lightning's lifecycle management. While Lightning handles saving and restoring the entire model state (including optimizer and scheduler state), pre-trained weights often come in formats targeting only the model's parameters. Therefore, we must bypass the standard `load_from_checkpoint` method and interact directly with PyTorch’s state dictionary management. The solution revolves around manipulating the `state_dict` attribute of the model, which is a Python dictionary that maps parameter names to their values.

Loading custom pretrained weights, therefore, involves three key steps: obtaining the pretrained weights, aligning their keys with your model's parameters, and loading the weights into the model. I will illustrate each with code examples based on common scenarios.

**Example 1: Direct loading of a state dictionary with matching keys.**

The simplest case is when the pretrained weights have keys that directly match the keys in your `LightningModule`. This is relatively infrequent in my experience, but worth outlining. Suppose we have a `MyAwesomeModel` implemented as a `LightningModule`, and a pretrained weight file called `pretrained.pth`, which contains a state dictionary.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class MyAwesomeModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
         # Training logic here, simplified
        return {"loss": torch.tensor(0.1)}

    def configure_optimizers(self):
         # Optimizer logic here, simplified
        return torch.optim.Adam(self.parameters())


def load_pretrained_weights(model, pretrained_path):
    pretrained_state_dict = torch.load(pretrained_path)['state_dict']
    model.load_state_dict(pretrained_state_dict)
    return model


if __name__ == '__main__':
    # Example usage
    model = MyAwesomeModel(input_size=10, hidden_size=20, output_size=5)
    # Assume pretrained.pth exists with weights compatible with MyAwesomeModel
    # For this example, we'll create some dummy weights
    dummy_state_dict = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    torch.save({'state_dict': dummy_state_dict}, 'pretrained.pth')
    loaded_model = load_pretrained_weights(model, "pretrained.pth")

    # Check if the weights are actually loaded
    for name, param in loaded_model.named_parameters():
      print(f"Parameter: {name} - Mean weight value: {param.mean().item():.4f}")
```

In this straightforward example, `load_pretrained_weights` loads the state dictionary from the file and then calls `model.load_state_dict()`. This direct replacement is effective when key names perfectly align. Note that the training/optimization loop is simplified for brevity, which is common when focusing on initialization and weights loading. I prefer to have this loading mechanism inside a helper function.  This provides cleaner separation of concerns and increases code reuse, which is particularly beneficial in larger projects.

**Example 2: Partial Loading and Key Remapping**

In the real-world, pretrained weight keys seldom match exactly. A common issue is a prefix or suffix added to the keys, particularly in larger, more complex models. Assume the pretrained weight keys have a `module.` prefix:
```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict

class MyAwesomeModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
         # Training logic here, simplified
        return {"loss": torch.tensor(0.1)}

    def configure_optimizers(self):
         # Optimizer logic here, simplified
        return torch.optim.Adam(self.parameters())


def load_pretrained_weights_remap(model, pretrained_path):
    pretrained_state_dict = torch.load(pretrained_path)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in pretrained_state_dict.items():
        name = k.replace('module.', '') # Remove prefix 'module.'
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


if __name__ == '__main__':
    # Example Usage
    model = MyAwesomeModel(input_size=10, hidden_size=20, output_size=5)

    # Generate dummy weights, now with the 'module.' prefix
    dummy_state_dict = {f'module.{k}': torch.randn_like(v) for k, v in model.state_dict().items()}
    torch.save({'state_dict': dummy_state_dict}, 'pretrained_remap.pth')


    loaded_model = load_pretrained_weights_remap(model, "pretrained_remap.pth")
    # Check if the weights are actually loaded
    for name, param in loaded_model.named_parameters():
      print(f"Parameter: {name} - Mean weight value: {param.mean().item():.4f}")

```

Here, `load_pretrained_weights_remap` first loads the dictionary, then iterates through it, removing `module.` from each key before re-inserting it into the `new_state_dict`. This remapped dictionary is then loaded into the model. Using an `OrderedDict` is critical here to maintain the original ordering of the dictionary, which can sometimes be important. I always double-check this by inspecting the loaded parameters. Mismatched keys typically cause errors. Handling these prefix issues is a frequent task when working with pretrained models from various sources.

**Example 3: Selective Loading of Weights.**

Often, it isn't necessary (or desirable) to load all the weights. Suppose we want to use a pretrained ResNet backbone but only load its convolutional layers and ignore the final fully connected layer.
```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from torchvision.models import resnet18

class MyAwesomeModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=False) # initialize a resnet model
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        # Training logic here, simplified
        return {"loss": torch.tensor(0.1)}

    def configure_optimizers(self):
         # Optimizer logic here, simplified
        return torch.optim.Adam(self.parameters())


def load_pretrained_resnet_backbone(model, pretrained_path):
    pretrained_state_dict = torch.load(pretrained_path)['state_dict']
    model_state_dict = model.resnet.state_dict()

    loaded_weights = OrderedDict()
    for k, v in pretrained_state_dict.items():
      if k in model_state_dict and 'fc' not in k:
          loaded_weights[k] = v

    model.resnet.load_state_dict(loaded_weights, strict=False) # strict=False necessary to skip missing keys
    return model


if __name__ == '__main__':
    # Example Usage
    model = MyAwesomeModel(num_classes=5)

    # Create dummy ResNet-18 pretrained weights
    pretrained_resnet = resnet18(pretrained=True)
    torch.save({'state_dict': pretrained_resnet.state_dict()}, 'resnet18_pretrained.pth')

    loaded_model = load_pretrained_resnet_backbone(model, "resnet18_pretrained.pth")

    for name, param in loaded_model.named_parameters():
        print(f"Parameter: {name} - Mean weight value: {param.mean().item():.4f}")
```

In `load_pretrained_resnet_backbone`, only parameters whose keys appear in both the model's backbone (resnet) `state_dict` and the pretrained `state_dict`, while explicitly excluding the ‘fc’ layer weights, are loaded. The `strict=False` argument of `load_state_dict` prevents errors due to missing keys. This is very helpful for fine-tuning models where we only want to use part of the pretrained weights. Often I'll print out the keys in both dictionaries to precisely control this loading.  This avoids unexpected behavior due to inadvertently loading or ignoring weights.

**Resource Recommendations:**

For a more in-depth understanding, I would suggest consulting the following. The first is the official PyTorch documentation, which offers detailed explanations of `torch.nn.Module` and related functionalities including state dictionaries. Secondly, the PyTorch Lightning documentation is vital for understanding the LightningModule lifecycle. Finally, researching tutorials focusing specifically on fine-tuning large models with custom architectures provides a more broad perspective. These resources have proven invaluable to me over time when tackling this kind of issue.
