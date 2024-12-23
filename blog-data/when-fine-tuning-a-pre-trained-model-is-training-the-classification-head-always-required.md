---
title: "When fine-tuning a pre-trained model, is training the classification head always required?"
date: "2024-12-23"
id: "when-fine-tuning-a-pre-trained-model-is-training-the-classification-head-always-required"
---

Alright, let's tackle this one. It's a question that resurfaces quite often, and for good reason. The seemingly simple process of fine-tuning a pre-trained model isn't always as straightforward as hitting ‘train.’ My experience, especially back when I was working with complex medical imaging systems, taught me that the decision of whether to train the classification head hinges quite a bit on the specifics of the problem and the dataset. It’s not a universal ‘always’ or ‘never,’ and failing to acknowledge this can lead to suboptimal, or even downright misleading, results.

Essentially, when you grab a pre-trained model, say one trained on ImageNet, you’re inheriting a rich feature extractor. This network has learned to identify hierarchical features ranging from edges and textures to more complex patterns. The classification head on such models, often one or more fully connected layers followed by a softmax, is then trained to map these extracted features to the specific classes present in the dataset it was trained on. Now, here's the crux: when adapting to a new task, *do* we really need to retrain that final mapping, or is it sufficient to just tweak the feature extractor layers?

The short answer is: it depends. If your target task is similar to the original task the pre-trained model was trained on, like simply classifying different breeds of dogs (which might still be present in the ImageNet dataset), you might be able to get away with freezing the feature extractor and *only* training the classification head. This is often the fastest path to a decent result, and is a common starting point, especially when your custom dataset is relatively small. The thinking here is that the already learned features are relevant and all we need is a new mapping to our specific classes. However, this is not always the optimal approach.

On the other hand, if your task deviates significantly, let’s say classifying microscopic images of different cell types (a scenario I’ve dealt with firsthand), the pre-trained model’s features, although powerful, may not be directly discriminative. In this case, attempting to just train a new head is likely going to yield poor results as the features the head operates on may not have the necessary variance or be tuned to what you need. Your pre-trained feature extractor may be identifying “edges” and “shapes” but not the subtle nuances between cancerous and healthy cells. You’ll likely find the model struggles to converge and generalizes poorly. In such scenarios, a more effective strategy is to fine-tune, meaning that you update the weights of at least some of the feature extraction layers along with, or after, training the classification head. There isn’t always a right answer, of course, and there are a range of valid strategies that sit in between these extremes. The best way to understand the answer for your particular situation is to experiment and explore.

Now, let's look at a few code examples using PyTorch to illustrate different scenarios.

**Scenario 1: Training Only the Classification Head**

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
# Replace the fully connected layer with a new one for our classification task
model.fc = nn.Linear(num_ftrs, 10) # 10 is the number of classes in the target task

# Optimizer (only optimizing parameters of the new linear layer)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Print the trainable parameters to verify
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}") # Should be relatively small

# Training loop (simplified)
# ... code for training the model using optimizer and loss function ...
```

In this snippet, we loaded a ResNet18 model and froze all the parameters apart from the last fully connected layer which we replaced with our own that maps features to 10 target classes. The optimizer is then setup to only update the parameters of the new `model.fc`. This corresponds to the 'only train classification head' approach.

**Scenario 2: Fine-tuning the entire model.**

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)


num_ftrs = model.fc.in_features
# Replace the fully connected layer with a new one for our classification task
model.fc = nn.Linear(num_ftrs, 10) # 10 is the number of classes in the target task

# Optimizer (will update all the parameters since we didn't freeze any)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Print the trainable parameters to verify
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")  # Should be much larger


# Training loop (simplified)
# ... code for training the model using optimizer and loss function ...
```

This second example does the opposite. All parameters, including the feature extractor's and the newly initialized classification head, are optimized in this training regime. This would lead to much longer training time but also greater performance flexibility, depending on the dataset. We've kept `learning_rate` lower than the previous example, it's common practice to reduce the learning rate for full model fine-tuning.

**Scenario 3: Fine-tuning specific layers (selective fine-tuning)**

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze all layers except for the last few convolutional blocks and the fully connected layer
# Get total parameters to process
total_parameters = list(model.parameters())

# We'll only unfreeze parameters in the last 2 blocks and fully connected
unfreeze_start_idx = len(total_parameters) - (14) # determined based on Resnet18 arch - last 2 blocks roughly correspond to the last 14 params
for i,param in enumerate(total_parameters):
    if i >= unfreeze_start_idx:
        param.requires_grad = True
    else:
        param.requires_grad = False

num_ftrs = model.fc.in_features
# Replace the fully connected layer with a new one for our classification task
model.fc = nn.Linear(num_ftrs, 10)

# Optimizer (will update the parameters we didn't freeze)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00005)

# Print the trainable parameters to verify
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}") # Will be smaller than full model fine-tuning but larger than just classification head.

# Training loop (simplified)
# ... code for training the model using optimizer and loss function ...
```

This third example illustrates a more nuanced approach. Here, we are unfreezing only specific layers towards the end of the feature extractor, in addition to the classification head. This strategy is useful when you suspect that high level, task-specific features may be encoded in specific layers, thus it’s best to tune them rather than use them as-is. We're also using the `filter()` with the optimizer to only select the params that have `requires_grad = True`. Note how the number of trainable parameters changes for each example.

The choice of which layers to fine-tune is rarely clear-cut, and often relies on experimentation, but one can be guided by these general principles. Generally, freezing more of the model and only training the head is faster and more resource efficient, but may not be optimal for significantly different target tasks, whereas fine-tuning lower layers as well as the head will likely lead to better performance on target tasks that are very different to the source task, but at the cost of greater time and training resources. It is important to note that often you do not even need to change the head and simply fine tuning the last few feature layers can also work well, in a similar approach to example 3.

For further exploration, I strongly recommend diving into literature about transfer learning, which directly covers this topic. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville provides a comprehensive theoretical understanding. Specifically, chapters on transfer learning and fine-tuning will be very informative. You should also explore the numerous research papers on adapting pre-trained models, specifically targeting papers exploring the impact of tuning different model depths. Papers such as “How transferable are features in deep neural networks?” by Yosinski et al., and related work, are very enlightening in these specific areas. The field is always evolving, but these will provide a solid foundation for further studies and practical application.
