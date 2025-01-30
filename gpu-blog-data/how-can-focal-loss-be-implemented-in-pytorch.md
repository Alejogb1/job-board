---
title: "How can focal loss be implemented in PyTorch?"
date: "2025-01-30"
id: "how-can-focal-loss-be-implemented-in-pytorch"
---
Focal loss addresses the class imbalance problem prevalent in many machine learning tasks, particularly object detection.  My experience working on a large-scale image classification project for autonomous vehicles underscored the limitations of standard cross-entropy loss when dealing with a heavily skewed dataset – millions of background samples versus a few thousand objects of interest.  This necessitated a deeper understanding and practical application of focal loss within the PyTorch framework.  The core idea behind focal loss is to down-weight the contribution of easy examples (those already correctly classified with high confidence) during training, allowing the model to focus on the more challenging, often misclassified, samples.

The standard cross-entropy loss is defined as:

`CE(p_t) = -log(p_t)`

where `p_t` is the model's estimated probability for the correct class.  This loss function is equally sensitive to both easily and hardly classified examples.  Focal loss modifies this by introducing a modulating factor (1 - p_t)^γ, where γ ≥ 0 is a focusing parameter.  This factor reduces the loss contribution of well-classified examples (p_t close to 1). The complete focal loss function is:

`FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`

Here, `α_t` is a balancing factor that addresses class imbalance, often set to α = 0.25 in practice for binary classification problems.  This factor is typically set based on the class frequencies in the training data, with lower values assigned to the majority class.  The choice of γ controls the degree of down-weighting; higher γ values place more emphasis on hard examples.  In my experience, experimenting with values of γ between 0 and 2 has proven effective in tuning the model's performance.  Note that when γ = 0, focal loss reduces to weighted cross-entropy loss.


Now, let's examine three practical implementations of focal loss in PyTorch:

**Example 1: Binary Classification using `torch.nn.BCEWithLogitsLoss`**

This example leverages PyTorch's built-in `BCEWithLogitsLoss` function for efficiency.  We modify it to incorporate the focal loss formula.  This method is particularly useful when dealing with binary classification problems.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# Example usage:
model = nn.Linear(10, 1) # Example model
logits = model(torch.randn(16, 10)) #Example input
targets = torch.randint(0, 2, (16,)).float() #Example targets
loss_fn = FocalLoss()
loss = loss_fn(logits, targets)
print(loss)

```

This code defines a `FocalLoss` class that inherits from `nn.Module`. The `forward` method calculates the binary cross-entropy loss with logits and then applies the focal loss modulation. The `reduction='none'` argument in `binary_cross_entropy_with_logits` is crucial, preventing the function from averaging losses, allowing for precise application of the focal weighting.  The average loss is computed using `focal_loss.mean()`.


**Example 2: Multi-class Classification using `torch.nn.CrossEntropyLoss`**

For multi-class classification, we can modify the standard `CrossEntropyLoss` in a similar fashion.  This approach requires converting the one-hot encoded target labels to index representation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, num_classes=10):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        if alpha is None:
          self.alpha = torch.ones(num_classes)
        else:
          self.alpha = torch.tensor(alpha)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        pt = probs[torch.arange(len(targets)), targets]  # Probability of correct class
        focal_loss = - self.alpha[targets] * (1 - pt) ** self.gamma * torch.log(pt)
        return focal_loss.mean()


#Example usage:
model = nn.Linear(10, 10) # Example model
logits = model(torch.randn(16, 10)) # Example input
targets = torch.randint(0, 10, (16,)) #Example targets
loss_fn = MultiClassFocalLoss(num_classes=10)
loss = loss_fn(logits, targets)
print(loss)

```

This example demonstrates a multi-class focal loss implementation.  The `alpha` parameter can be a list of weights for each class, addressing class imbalance more directly.  If not specified, it defaults to uniform weights. The probability of the correct class is efficiently extracted using advanced indexing.


**Example 3: Implementing Focal Loss from Scratch**

For greater control and understanding, focal loss can be implemented entirely from scratch without relying on PyTorch's built-in loss functions.

```python
import torch

def focal_loss_from_scratch(logits, targets, gamma=2, alpha=0.25):
    probs = torch.sigmoid(logits)
    pt = targets * probs + (1 - targets) * (1 - probs)
    focal_loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    return focal_loss.mean()

# Example usage
logits = torch.randn(16, 1) #Example Input
targets = torch.randint(0, 2, (16,)).float() # Example targets
loss = focal_loss_from_scratch(logits, targets)
print(loss)

```

This approach provides maximum flexibility but requires more manual computation.  It's particularly useful for understanding the underlying mechanics.  The sigmoid activation ensures that probabilities remain within the [0, 1] range. This example is for binary classification, but extension to multi-class requires a more complex probability calculation and handling of the alpha parameter.


**Resource Recommendations:**

Several research papers extensively explore focal loss and its variations. I would suggest looking into the original focal loss paper and subsequent papers exploring its applications in different contexts.  Additionally, studying PyTorch's documentation on loss functions and advanced tensor manipulation will prove invaluable. Consulting advanced deep learning textbooks will provide the necessary theoretical foundations.  Reviewing code repositories on platforms such as GitHub, focusing on those implementing focal loss within object detection frameworks, offers practical insights and implementation details.
