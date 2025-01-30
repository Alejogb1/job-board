---
title: "How can a PyTorch binary transfer learning model be extended to handle multiple image classes?"
date: "2025-01-30"
id: "how-can-a-pytorch-binary-transfer-learning-model"
---
The challenge in extending a binary PyTorch transfer learning model to multiclass classification lies in adapting the final output layer and the associated loss function to manage more than two classes, necessitating a refactoring of the model's decision-making process and evaluation metrics. Specifically, the final fully connected (linear) layer typically used for binary classification, which reduces features to a single output representing probability, must be modified to produce an output for each class.

My experience working on image classification pipelines for an autonomous driving research project, where we transitioned from vehicle/non-vehicle detection to classifying various road agents (pedestrians, cyclists, cars, etc.), highlighted this need for adaptation. We initially employed a pre-trained ResNet18 architecture fine-tuned for binary detection. However, the introduction of new agent classes required a fundamental shift from a single output neuron and the binary cross-entropy loss to a multi-output structure and categorical cross-entropy loss.

Let's detail the required adjustments. The core of any transfer learning approach starts with a pre-trained convolutional neural network (CNN), like those available through `torchvision.models`. These pre-trained models, often trained on massive datasets like ImageNet, provide an excellent feature extraction base. For binary classification, a common strategy is to replace the original classification layer of the pre-trained network with a single linear layer followed by a sigmoid activation function. This sigmoid then yields a probability between 0 and 1, which can be thresholded to classify into two categories.

Extending this to *n* classes requires replacing that single output neuron with *n* output neurons. Consequently, we do not use sigmoid activation but rather softmax activation. Softmax transforms the raw output scores (logits) into a probability distribution across the classes, with the probability for a given image belonging to each class summing to 1. Consequently, loss calculation also needs a switch from Binary Cross Entropy to Categorical Cross Entropy.

Consider a scenario where we wish to expand a binary classifier to classify images into three classes: "cat," "dog," and "bird." Our initial binary setup might be something akin to this (simplified for brevity):

```python
import torch
import torch.nn as nn
import torchvision.models as models

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)  # Output for 2 classes (binary)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x

# Binary model instance
binary_model = BinaryClassifier()
```

This model outputs a single value after the sigmoid. Now, let's adapt this model to classify three classes: cat, dog, and bird. We replace the single output linear layer with a three-output layer, and append a softmax layer:

```python
class MultiClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes) # output for n classes
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.resnet(x)
        x = self.softmax(x)
        return x

# Multiclass model instance for 3 classes
multi_model = MultiClassifier(num_classes=3)
```

Notice the replacement of `nn.Linear(num_ftrs, 1)` with `nn.Linear(num_ftrs, num_classes)`. The final output layer now consists of three neurons, each representing a class. Importantly, the final sigmoid activation has been replaced with a softmax activation applied across all class scores along `dim=1`. During forward propagation, the network now computes logits for each class; softmax normalizes them into probabilities, enabling multiclass classification.

The other essential modification concerns the loss function. Binary cross-entropy (BCE) loss calculates loss for binary classification based on log-probabilities from the sigmoid activation. Multiclass cross-entropy, implemented via `torch.nn.CrossEntropyLoss` in PyTorch, expects raw logits as its input and labels as indices representing the target class. This loss function internally handles the softmax calculation and compares predicted probabilities against the true class. In fact, `nn.CrossEntropyLoss` actually expects logits, meaning we don't need to apply the softmax as the last layer. The revised multi-class model would look like this:

```python
class MultiClassifierLogits(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassifierLogits, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes) # output for n classes


    def forward(self, x):
        x = self.resnet(x)
        return x

# Multiclass model instance for 3 classes
multi_model_logits = MultiClassifierLogits(num_classes=3)
```

Here, I've removed the softmax activation. If you now try to compute the loss with `nn.CrossEntropyLoss`, it will work correctly.

```python
loss_fn = nn.CrossEntropyLoss()
# Sample Data (batch size 4, each sample has 3 classes)
logits = torch.randn(4, 3)  # random logits
labels = torch.tensor([1, 0, 2, 1]) # random sample of label indexes
loss = loss_fn(logits, labels)
print (f"loss: {loss}")
```
This shows a complete end-to-end process of creating the class and showing that the loss function works as intended.

Beyond code modifications, transitioning from binary to multiclass classification also introduces considerations during the evaluation phase. While accuracy is a straightforward metric for binary classification, multiclass classification often benefits from metrics like precision, recall, F1-score, and confusion matrices for each class, providing a more nuanced understanding of model performance across categories. Furthermore, the `sklearn.metrics` library proves invaluable for calculating these metrics on your multi-class data.

In conclusion, adapting a PyTorch binary transfer learning model for multiclass classification requires three key adjustments: modifying the output layer to produce scores for each class, changing the activation from sigmoid to softmax (or removing it and using logits with `nn.CrossEntropyLoss`), and updating the loss function to cross-entropy. During evaluation, more advanced metrics than accuracy are beneficial. Proper understanding of these changes and the implications that each has on the model and its performance are required to successfully shift from binary to multiclass scenarios.

For further learning, I suggest studying:
- Deep Learning with PyTorch documentation and tutorials.
- Online courses on Convolutional Neural Networks and Transfer Learning.
- Research papers on multiclass image classification techniques.
- Books on Deep Learning concepts and practical implementations.
- Community forums, and online code repositories on GitHub.
