---
title: "Why does accuracy drop to zero after setting the backbone trainable?"
date: "2025-01-30"
id: "why-does-accuracy-drop-to-zero-after-setting"
---
When fine-tuning pre-trained convolutional neural networks, observing a sudden drop in accuracy to zero after making the backbone (feature extractor) trainable is not uncommon and often indicative of catastrophic forgetting and gradient instability. This behavior stems from the interplay of several factors related to the learning dynamics and the initialization of network weights. It's a challenge I've repeatedly encountered, both during my research and when deploying models in production.

The core issue lies in the drastic change in the learning rate's impact on the pre-trained layers compared to the newly added classification layers. When a backbone is frozen, only the parameters of the classification head are updated during training. Typically, these new layers are initialized with random weights, and the training process seeks to optimize these weights to map the extracted features to the correct class. Because they are starting from scratch, these layers benefit from a relatively high learning rate. This initial training phase focuses exclusively on the new task, learning how to interpret the features already extracted by the frozen backbone.

However, making the backbone trainable fundamentally alters this dynamic. The pre-trained layers, although generally effective at extracting meaningful features, are not optimally aligned for the specific task at hand. They were trained on a different dataset and for a different classification problem. When these layers are suddenly exposed to gradient updates, especially with a learning rate that was previously appropriate for only the classification head, several problematic scenarios can emerge.

First, the initial weights in the backbone are optimized for their original task. These weights are often small and have a delicate balance. Directly applying a relatively large learning rate disrupts this balance, forcing significant updates early in training. This large update effectively rewrites the learned feature representations, rather than adapting them. This is known as catastrophic forgetting. The carefully learned weights, specific to the pre-trained domain, are abruptly overwritten with values that are initially useless for the target task. This means that the network, in essence, forgets the pre-trained features, leaving it without a strong feature extractor. Consequently, its classification performance plummets.

Second, large learning rate coupled with an improper optimizer can also introduce gradient instability. The gradients computed in the deeper layers of the network, especially when the backbone is large, are typically smaller compared to those calculated for the shallower layers in the classification head. This disparity can cause the gradient updates to the backbone to either vanish or explode. Vanishing gradients lead to minimal change in the backbone parameters, while exploding gradients disrupt the network significantly. This creates a situation where the backbone is essentially being trained randomly, leading to poor feature representations.

Third, pre-trained layers often possess a large number of parameters. Adjusting this vast parameter space simultaneously with a single learning rate is inherently problematic. The optimal learning rate for the classification head and the backbone is rarely identical. The classification layers require more aggressive adjustment early on, whereas the pre-trained layers generally need more conservative updates to avoid catastrophic forgetting.

Finally, batch normalization layers can exacerbate the problem. Batch norm statistics from the pre-training phase are crucial for stable training. If these are not updated appropriately, or if the batch size is too small after unfreezing, the statistical assumptions will break down and introduce further instability.

To mitigate this, several techniques can be employed. Gradual unfreezing, using differential learning rates, and careful tuning of the optimizer are crucial to preserve the benefits of the pre-trained model while adapting it for the new task.

Here are a few example scenarios with specific code and commentary:

**Example 1: Basic Unfreezing, Catastrophic Forgetting**

This scenario demonstrates the issue when unfreezing without further adjustment of the optimizer or learning rate.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze the backbone
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # Assuming 10 classes

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assume we do training now on the classification layer
# ... training code here ...

# Unfreeze the backbone
for param in model.parameters():
    param.requires_grad = True

# Keep optimizer the same, which now includes the full network
optimizer = optim.Adam(model.parameters(), lr=0.001) # Incorrect approach

# Assume we continue training
# ... catastrophic forgetting likely occurs here, leading to 0 accuracy
```

In this code, we initially freeze the backbone, train the classifier, and then unfreeze the entire model. Using the same learning rate as before will likely destroy the previously learned features.

**Example 2: Differential Learning Rates**

This example demonstrates a solution with differential learning rates using parameter groups.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze the backbone
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # Assuming 10 classes

# Define optimizer for classification head
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Assume we train just the classification layer
# ... training code ...

# Unfreeze backbone and assign params to backbone
params_to_optimize = []
params_to_optimize.append({'params': model.parameters(), 'lr': 1e-5}) # lower learning rate

# Create optimizer with differential learning rates
optimizer = optim.Adam(params_to_optimize, lr = 0.001) # higher learning rate

# Now train the entire model with the different learning rates
# ... this significantly less likely to cause catastrophic forgetting
```

Here, we have explicitly set a lower learning rate (1e-5) for the backbone layers compared to the new classification layer during the joint training, which helps to prevent the forgetting phenomenon.

**Example 3: Gradual Unfreezing**

This example combines gradual unfreezing and differential learning rates.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze the entire model initially
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # Assuming 10 classes

# Define optimizer for classification head
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Assume we train just the classification layer
# ... training code ...

# Unfreeze a few layers gradually
layers_to_unfreeze = [model.layer4, model.fc]

params_to_optimize = []

for layer in layers_to_unfreeze:
    for param in layer.parameters():
        param.requires_grad = True
    params_to_optimize.append({'params': layer.parameters(), 'lr': 1e-4})


for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

params_to_optimize.append({'params': [param for name, param in model.named_parameters() if 'layer4' not in name and 'fc' not in name], 'lr': 1e-5})


# Create optimizer with differential learning rates
optimizer = optim.Adam(params_to_optimize)

# Now train the entire model with the different learning rates
# ... this significantly less likely to cause catastrophic forgetting
```

This example provides an even more fine-grained approach. We unfreeze specific layers (layer4 and the fully connected layer) and apply different learning rates to those layers vs the rest of the network. This provides an additional level of control to ensure optimal finetuning.

**Resource Recommendations:**

For further exploration, I recommend exploring academic papers and tutorials on topics including transfer learning, fine-tuning convolutional neural networks, differential learning rates, and learning rate scheduling. I have found discussions on practical implementations across various deep learning frameworks insightful. Consult the documentation for frameworks like PyTorch and TensorFlow regarding their specific handling of parameter groups and optimizer configurations. Lastly, experiment with varying combinations of these techniques on a range of datasets to observe their specific effects on accuracy. Careful and systematic experimentation remains the cornerstone of deep learning success.
