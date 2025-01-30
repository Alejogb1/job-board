---
title: "What is the correct shape of ResNet output logits?"
date: "2025-01-30"
id: "what-is-the-correct-shape-of-resnet-output"
---
Having spent a significant portion of the last five years fine-tuning deep learning models for image classification tasks, I've repeatedly encountered the nuances of ResNet architecture, particularly concerning the shape of its output logits.  The seemingly simple question of 'correct' shape belies underlying considerations regarding the model's design and intended application.

Fundamentally, ResNet output logits are not inherently constrained to a single, universal shape. Their final dimensions directly correlate with the number of classes the model is designed to predict. Unlike intermediate feature maps that represent extracted image features in a multi-dimensional space, logits are essentially unnormalized scores for each class. These scores are then typically passed through a softmax or sigmoid function to obtain probabilities. The shape of these logits must, therefore, match the desired number of predicted classes.

More specifically, a ResNet model intended for a typical N-class classification problem, where each input image belongs to one of N exclusive categories, will have output logits with a shape of `(batch_size, N)`. Here, `batch_size` refers to the number of input images processed in parallel and N is the number of distinct classes. This matrix organization directly maps each input image to N probability scores. For instance, in a binary classification problem (e.g., cat vs dog), N is 2, and the resulting output tensor for batch size of 32 will have the shape of (32, 2). Each row represents a single image and each column holds the raw score indicating how confident the model is for the particular class. It’s this raw score that, through activation and potentially loss function calculation, informs model improvement during backpropagation.

It's important to differentiate this from the intermediate feature maps that exist within ResNet's architecture. The output of convolutional layers, before reaching the fully-connected classifier, typically carries a shape like `(batch_size, channels, height, width)`. These are multi-channel feature representations and require subsequent processing, usually via global pooling to reduce the spatial dimensions before being fed into the final linear layer which produces the logits. This global pooling step is often either global average pooling or global max pooling which reduce the spatial height and width dimensions to 1 by effectively taking an average or max value within each feature map.

Let's now consider concrete code examples using Python and a popular deep learning framework.

**Code Example 1: Basic Image Classification (N-Class)**

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Assume a 10-class problem.
num_classes = 10
batch_size = 32

# Load a pre-trained ResNet18 model.
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer to match the desired number of classes.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Create a dummy input tensor with a batch of 32 images
input_tensor = torch.randn(batch_size, 3, 224, 224)

# Pass the input through the network
output_logits = model(input_tensor)

# Inspect the shape of the output logits.
print(f"Output Logits Shape: {output_logits.shape}")
#Expected Output: Output Logits Shape: torch.Size([32, 10])
```

In the above example, I load a pre-trained ResNet18 and modify its final layer to have an output dimension corresponding to the number of classes. This adaptation is crucial because the default ResNet configurations are usually trained on ImageNet, which has 1000 classes.  The core point here is the `nn.Linear(num_ftrs, num_classes)` command, which replaces the original ImageNet-specific fully-connected layer with one designed for 10 output classes. The shape of the `output_logits` is correctly `(batch_size, 10)`.

**Code Example 2: Binary Classification**

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Binary classification: two classes (e.g. 0 or 1).
num_classes = 2
batch_size = 64

# Load a pre-trained ResNet34 model.
model = models.resnet34(pretrained=True)

# Replace the final layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Create a dummy input tensor
input_tensor = torch.randn(batch_size, 3, 224, 224)

# Get the output logits.
output_logits = model(input_tensor)
print(f"Output Logits Shape: {output_logits.shape}")
#Expected Output: Output Logits Shape: torch.Size([64, 2])
```

This snippet shows how the output shape changes for binary classification. The only difference is setting the `num_classes` to 2, resulting in a shape of `(batch_size, 2)` for the logits.  Crucially, the rest of the ResNet model remains identical; it's the final linear layer that's adapted.

**Code Example 3: Multi-Label Classification (Different Shape)**

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Multi-label classification: multiple classes possible for each image
num_classes = 5
batch_size = 16

# Load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Replace the final layer for multi-label classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Create dummy input tensor
input_tensor = torch.randn(batch_size, 3, 224, 224)

# Get the output logits.
output_logits = model(input_tensor)
print(f"Output Logits Shape: {output_logits.shape}")

#Expected Output: Output Logits Shape: torch.Size([16, 5])
```

This final example presents a case of multi-label classification where multiple classes might be present in an individual image. The underlying principle regarding the output logits remains identical - the shape corresponds to the number of classes, regardless of the specific classification type. In this instance, it results in an output tensor of shape `(batch_size, 5)`. Notice that the output logits, in this context, are often passed into a sigmoid, instead of softmax. Each element in the final output vector is not a mutually exclusive probability, but an independent score indicating the probability of an image having that class.

Key considerations that should be kept in mind, particularly when debugging model output, are the following: the correct shape is determined by the downstream task, not by ResNet’s intrinsic architecture; ensure you are replacing the final linear layer in the network; always remember that the output is usually the raw scores before softmax or sigmoid and consider data batching for higher throughput.

In conclusion, the "correct" shape of ResNet output logits isn't a static property, but depends entirely on the task at hand. It is always a 2D tensor, of shape `(batch_size, N)` where `N` represents the number of classes the model is designed to distinguish. These logits are then transformed into probabilities to make predictions. It is important to be aware of the distinction between intermediate feature maps and these final logits. Further information can be gleaned from resources like documentation of different deep learning libraries, tutorials that elaborate classification scenarios and in-depth literature that discuss the details of convolutional neural networks.
