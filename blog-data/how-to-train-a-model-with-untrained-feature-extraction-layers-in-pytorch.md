---
title: "How to train a model with untrained feature extraction layers in PyTorch?"
date: "2024-12-23"
id: "how-to-train-a-model-with-untrained-feature-extraction-layers-in-pytorch"
---

,  It’s a common scenario I've encountered several times, particularly when adapting pre-trained models for somewhat atypical tasks. Essentially, you've got a model, maybe a convolutional neural network (CNN) or transformer, where you want to leverage the high-level representation learned by its feature extraction layers. However, these layers haven’t been specifically trained for your current data, and you suspect—or know—they could benefit from targeted fine-tuning, and you want to achieve this alongside your head/classifier training, without breaking anything.

The primary issue is that directly using a pre-trained feature extractor often results in sub-optimal performance, especially when the input data distribution differs substantially from what the pre-trained model originally saw. Initially, many beginners assume freezing the feature layers is the best approach to prevent them from "unlearning" previous knowledge. Yet, we don't want that. We want those layers to become better *for our specific task*. Freezing, therefore, only helps with stability in the initial stages but limits how much we can adapt to new inputs and optimize performance.

So, how do we go about this? Well, the key here lies in understanding the PyTorch training loop and employing differential learning rates. In essence, we can allow the feature extraction layers to train, just at a significantly lower rate than the classifier. It's a way of nudging their learned patterns toward our goal without allowing them to overwrite all their previous capabilities too quickly. I've seen this work remarkably well in many projects. For instance, I remember a classification project involving medical imaging where pre-trained models on natural images performed poorly until we fine-tuned the feature extraction layers at a tenth of the rate of the classification head. That gave us the edge we needed.

Let’s look at some concrete examples. The first snippet shows a straightforward scenario:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# Load a pre-trained ResNet18 (or any model)
model = resnet18(pretrained=True)

# Replace the last layer (classifier head) with a new one
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # 10 classes for example

# Define a set of parameters for optimization and learning rates
feature_params = []
classifier_params = []

for name, param in model.named_parameters():
  if "fc" in name:
    classifier_params.append(param)
  else:
    feature_params.append(param)


optimizer = optim.Adam([
    {'params': feature_params, 'lr': 1e-5},  # Lower learning rate for feature layers
    {'params': classifier_params, 'lr': 1e-3}  # Higher learning rate for the classifier
], weight_decay=1e-5)

criterion = nn.CrossEntropyLoss()

# Dummy data for demonstration
inputs = torch.randn(64, 3, 224, 224)
labels = torch.randint(0, 10, (64,))

# Training step example
outputs = model(inputs)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Training step complete.")
```

In this code, we load a pre-trained ResNet18, and replace its fully connected classification layer with one suitable for 10 classes. We then separate all parameters into those that belong to the feature layers and those that are part of the classification head. Crucially, we define two parameter groups in the Adam optimizer, specifying distinct learning rates: a much lower one for the feature extraction part and a higher rate for the classifier. The rest of the process is standard: dummy data, a loss function, backpropagation and optimization.

Now, let's take this up a notch. Sometimes you might want to train specific layers within your feature extractor differently too. Maybe the first few convolutional layers do a general task, and the deeper ones, particularly, might benefit from fine-tuning. Here’s a snippet demonstrating that:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5) # Example with 5 classes

# Parameter groups with multiple learning rates
params = []
for name, param in model.named_parameters():
    if "layer1" in name:
       params.append({'params': param, 'lr': 1e-6}) # lowest rate - for general feature extraction
    elif "layer2" in name or "layer3" in name:
        params.append({'params': param, 'lr': 1e-5}) # intermediate learning rate - for deeper features
    elif "layer4" in name or "fc" in name:
        params.append({'params': param, 'lr': 1e-3}) # highest rate - classification
    else:
        params.append({'params': param, 'lr': 1e-7}) # low rate for base layers

optimizer = optim.Adam(params, weight_decay = 1e-5)

criterion = nn.CrossEntropyLoss()

# Dummy data
inputs = torch.randn(64, 3, 224, 224)
labels = torch.randint(0, 5, (64,))

# Training step
outputs = model(inputs)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Training step complete - multiple lr groups")
```

Here, I’m using ResNet50 instead, just to show that it applies to other architectures too. The key part is the different learning rates for different block within the network. We are assigning a learning rate of 1e-6 for the "layer1" params, 1e-5 for "layer2" and "layer3" params, and finally a much higher one, 1e-3, to "layer4" and "fc." The remaining initial convolutional layer parameters receive an even lower rate of 1e-7, demonstrating very granular control.

Finally, let's discuss a more flexible approach using a lambda function within the optimizer's parameter group setup, allowing dynamic control over learning rates at a more granular level using string matching:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0

model = efficientnet_b0(pretrained=True)

# Replace the classifier
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 5) #Example for 5 classes

def create_param_groups(model, base_lr):
  params = []
  for name, param in model.named_parameters():
    if 'features.1' in name: # example of a specific feature block
       params.append({'params':param, 'lr': base_lr*0.05}) # low lr for initial block
    elif 'features.2' in name or 'features.3' in name: # deeper blocks
        params.append({'params':param, 'lr': base_lr * 0.1}) # slightly higher for mid layers
    elif 'classifier' in name:
       params.append({'params':param, 'lr': base_lr})
    else:
      params.append({'params':param, 'lr': base_lr * 0.01}) # low rate for general features

  return params

base_lr = 1e-3
params = create_param_groups(model, base_lr)
optimizer = optim.Adam(params, weight_decay=1e-5)

criterion = nn.CrossEntropyLoss()

# Dummy data
inputs = torch.randn(64, 3, 224, 224)
labels = torch.randint(0, 5, (64,))

# Training step
outputs = model(inputs)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Training step complete using a function-based configuration.")
```

Here, we use an EfficientNet, and instead of manually defining each layer's learning rate, we use a function (`create_param_groups`). This method increases maintainability and scalability. The function allows fine-grained control of learning rates depending on a part of a layer's name. If 'features.1' appears, the rate is `base_lr * 0.05`; If it contains 'features.2' or 'features.3', the rate is `base_lr * 0.1`. If the layer is a part of classifier the learning rate is base_lr, otherwise it will use `base_lr * 0.01`. This lambda approach allows a more dynamic assignment of learning rates.

To enhance your understanding further, I highly recommend delving into papers on transfer learning, fine-tuning strategies, and multi-task learning. Specifically, reading "How transferable are features in deep neural networks?" by Yosinski et al. (2014) provides foundational knowledge on feature transferability. Additionally, explore the documentation of the ‘torch.optim’ package in the PyTorch official documentation for a deeper grasp of how parameter groups function. Also, "Deep Learning" by Goodfellow et al. (2016) is an excellent textbook which will cover many underlying concepts.

These examples give you a robust set of techniques to begin with. Remember that setting learning rates is an empirical process. Experiment, keep detailed records of your experiments, and you’ll find the right balance for your task. The key is not to be afraid to fine-tune even the feature extraction layers, but to do so intelligently.
