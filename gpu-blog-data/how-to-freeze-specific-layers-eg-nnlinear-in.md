---
title: "How to freeze specific layers (e.g., nn.Linear()) in a PyTorch model?"
date: "2025-01-30"
id: "how-to-freeze-specific-layers-eg-nnlinear-in"
---
Freezing specific layers in a PyTorch model often involves manipulating the `requires_grad` attribute of their parameters.  My experience optimizing large-scale image classification models has shown that strategically freezing layers can significantly improve training efficiency and prevent catastrophic forgetting when fine-tuning pre-trained networks.  Directly manipulating parameters is crucial; relying on solely model-level methods can be insufficient for granular control.


**1. Clear Explanation**

PyTorch's automatic differentiation relies heavily on the `requires_grad` attribute associated with each parameter within a module.  When set to `True` (the default), gradients are computed and used during backpropagation to update the parameter values. Setting it to `False` effectively freezes the parameter, preventing its update during training. This is particularly useful when fine-tuning pre-trained models.  You typically freeze layers in the earlier stages of a network, preserving their learned representations while allowing later layers to adapt to a new task.  Furthermore, this technique is effective in situations where computational resources are limited or where preventing overfitting on a smaller dataset is vital.

The process involves iterating through the model's layers and selectively setting the `requires_grad` attribute of the parameters within the target layers.  This can be achieved directly by accessing the layer's parameters using the `parameters()` method or more elegantly using a recursive function to traverse nested modules.  Careful consideration should be given to the specific layers chosen for freezing; improper selection may hinder performance.  For instance, freezing all convolutional layers in a CNN while leaving the fully connected layers unfrozen might lead to suboptimal results as the feature extractors are prevented from adapting to the new data.


**2. Code Examples with Commentary**

**Example 1: Freezing specific layers by name**

This method uses string matching to identify and freeze layers. While straightforward, it relies on consistent naming conventions within your model architecture.  In large or dynamically constructed models, a more robust method (like Example 2) is preferred.  I've used this approach extensively in projects involving transfer learning with pre-trained ResNet architectures.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

for name, param in model.named_parameters():
    if '0' in name: # Freeze parameters in the first layer
        param.requires_grad = False

# Verify the changes
for name, param in model.named_parameters():
    print(f"Layer: {name}, requires_grad: {param.requires_grad}")

# Continue with training...
```


**Example 2: Freezing layers using a recursive function**

This approach allows for flexible freezing based on layer type or even custom criteria. I found this particularly useful when dealing with complex architectures including residual blocks or attention mechanisms, where simple name-based freezing might be insufficient and lead to unintended consequences.  This recursive structure ensures consistent handling of nested modules within the model.


```python
import torch
import torch.nn as nn

def freeze_layers(model, layer_type):
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            for param in module.parameters():
                param.requires_grad = False

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
    nn.Linear(2, 1)
)

freeze_layers(model, nn.Linear) # Freeze all Linear layers

# Verify the changes
for name, param in model.named_parameters():
    print(f"Layer: {name}, requires_grad: {param.requires_grad}")

# Continue with training...
```


**Example 3: Freezing layers based on index**

This method allows freezing layers based on their positional index within the model's `Sequential` container.  While less descriptive than name-based approaches, it provides a simple and effective method when the model architecture is well-understood. I've employed this method during experimentation to identify the optimal number of layers to freeze for a particular task.


```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

for i in range(1): # Freeze the first layer (index 0)
    for param in model[i].parameters():
        param.requires_grad = False

# Verify the changes
for name, param in model.named_parameters():
    print(f"Layer: {name}, requires_grad: {param.requires_grad}")

# Continue with training...
```


**3. Resource Recommendations**

The official PyTorch documentation is indispensable.  A thorough understanding of the `torch.nn` module and the concepts of automatic differentiation is essential.  Exploring advanced topics like parameter groups within optimizers can further enhance your control over the training process.  Furthermore, studying published papers on transfer learning and fine-tuning provides valuable insights into effective strategies for freezing layers in different contexts.  Finally, working through tutorials focusing on convolutional neural networks and their applications is highly beneficial for developing a practical understanding of these concepts.
