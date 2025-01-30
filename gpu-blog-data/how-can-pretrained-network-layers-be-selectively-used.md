---
title: "How can pretrained network layers be selectively used for transfer learning?"
date: "2025-01-30"
id: "how-can-pretrained-network-layers-be-selectively-used"
---
Transfer learning, specifically the selective reuse of pretrained network layers, hinges on understanding that not all layers within a deep neural network learn features equally relevant to different tasks. Early layers generally learn low-level features like edges and corners, while deeper layers capture more abstract and task-specific representations. Therefore, efficient transfer learning requires strategically choosing which layers to freeze (maintain pre-trained weights) and which to retrain (fine-tune for the new task). This approach can significantly reduce training time and data requirements, especially when the target dataset is small.

I've personally encountered this challenge multiple times when working with image classification projects. For example, when adapting a model pre-trained on ImageNet to classify medical scans, I quickly realized that re-training the entire network was both computationally expensive and prone to overfitting due to the significantly different feature distribution of medical images compared to natural ones. Consequently, selectively using pretrained layers became essential for achieving satisfactory performance.

The core principle behind this selective approach rests on the premise that the earlier layers of a pretrained network, trained on a massive dataset like ImageNet, already capture general-purpose visual features. These features are often relevant across many different image-related tasks. Therefore, instead of discarding these valuable representations, we freeze the weights of these layers. The trainable portion of our network then consists of the deeper layers and any custom layers added at the end, which are fine-tuned using our target dataset. This process allows our model to specialize to the specific nuances of the target task while retaining the robust general features learned from the pretrained model. This strategy of freezing some layers and training others is crucial to avoid catastrophic forgetting.

The selection of which layers to freeze or train depends on the specific tasks involved and the size of target dataset. A large target dataset similar to the pretraining dataset might allow for training more layers, whereas a smaller, disparate dataset calls for more conservative approach to freezing. Typically, the general practice is to freeze the early convolutional layers and progressively fine-tune deeper convolutional layers and fully connected layers. This practice often finds the best balance between preserving prior knowledge and adapting to the specific demands of the target task.

Consider the following hypothetical example using PyTorch: We'll start by loading a ResNet50 model pretrained on ImageNet, then customize it for a new classification task, freezing the initial layers and fine-tuning the later ones.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Freeze parameters in the early layers (e.g. layers before layer4)
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False # This freezes the weights

# Replace the final fully connected layer for our specific task (e.g. 10 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Example Usage (optional)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape) # This will output torch.Size([1, 10])
```
In this example, I freeze all layers up to, but not including, layer4. The key line is `param.requires_grad = False`, which prevents the gradient from being calculated for these layers during backpropagation. This effectively freezes the learned weights. I then replace the fully connected layer (model.fc) with a new linear layer that has an output size corresponding to the number of classes in our target task. We then train the model (excluding frozen layers) on the target dataset, adapting the later layers to the specific classification problem. The input tensor and output printing show the change to the final linear layer, that adapts to the new number of class predictions (10).

A slightly more flexible approach allows fine-tuning a specific portion of the network by defining layer groups that can be independently trained. This technique often yields better results when using smaller datasets that do not significantly deviate from the original dataset. Let us examine the same ResNet-50 model by training `layer3` and `layer4`

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)


# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layer groups (e.g., layer3 and layer4)
for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name:
        param.requires_grad = True

# Replace the final fully connected layer for our specific task
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Example Usage (optional)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape)
```
Here, after initially freezing all parameters, I then selectively unfreeze `layer3` and `layer4`. This allows for more focused fine-tuning, and the key aspect is the boolean check of the names. The unfreezing process allows the gradient to propagate through these particular layer groups, and updates their weights, while keeping earlier layers unchanged. This strategy is extremely useful when there is a need to modify a specific part of the model for better adaptation to new task. The input tensor and output printing show the same output shape modification as the previous example.

Another variation of this strategy, particularly useful when adapting pretrained models to domains that are significantly different, is to treat the pretrained network as a fixed feature extractor. In this approach, all layers are frozen, and then a new classifier is trained on the extracted features. This approach is less computationally demanding than fine-tuning the entire network, especially if the computational resources are constrained. It also prevents the pretrained weights from being damaged. Let us examine an example, where only a new linear layer is trained:

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Freeze all layers (use model.eval() when in inference)
for param in model.parameters():
    param.requires_grad = False

# Create a new classifier on top of the feature extractor
num_ftrs = model.fc.in_features
classifier = nn.Linear(num_ftrs, 10)


# Forward pass through feature extractor and classifier
def forward(x):
    with torch.no_grad():  # Disable gradient calculation for pretrained layers
      features = model(x)
    output = classifier(features)
    return output


# Example Usage
input_tensor = torch.randn(1, 3, 224, 224)
output = forward(input_tensor)
print(output.shape)
```

This final example illustrates the extreme case of freezing all the pretrained layers, including `model.fc`. A new classifier, `classifier` is created to fit the new output size requirements of the target task. Then a `forward` function is created to pass through the frozen feature extractor first, and then through the newly created trainable classifier. This is a good approach when the target dataset is very different to the training dataset. The key here is `torch.no_grad()` which ensures we are not accidentally backpropagating through the pretrained model layers, and we only train our new classifier.

In summary, selective use of pretrained network layers is a crucial technique for achieving effective transfer learning. The process involves strategically freezing early layers that learn general features and fine-tuning later layers that can adapt to the specific demands of the target task. The choice of layers to freeze or train depends on the similarity between the source and target tasks, as well as the available training data. While the examples provided were specifically in PyTorch, the general principles are applicable in different deep learning libraries like TensorFlow, Keras.

For those seeking more in-depth information, I recommend exploring resources on transfer learning, specifically focusing on fine-tuning strategies, available through online documentation and textbooks dedicated to deep learning. Look for explanations of concepts like ‘layer freezing,’ ‘feature extraction,’ and ‘fine-tuning,’ particularly within the context of convolutional neural networks. Additionally, many research papers delve into specific applications of transfer learning using pretrained models across various domains; reading these will provide a practical view on these methods. Exploring documented code in libraries like PyTorch and TensorFlow can also be beneficial for those who prefer learning by example.
