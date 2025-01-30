---
title: "How can I design a custom loss function for a CNN?"
date: "2025-01-30"
id: "how-can-i-design-a-custom-loss-function"
---
Custom loss functions are crucial when standard metrics fail to accurately reflect the nuances of a specific problem, particularly in complex computer vision tasks. My experience building an automated medical image analysis pipeline, specifically for detecting subtle anomalies in retinal scans, quickly revealed the limitations of simple cross-entropy or mean squared error. The visual severity of these anomalies was not always linearly correlated to pixel differences, necessitating a custom approach.

At its core, a loss function quantifies the disparity between a model’s predictions and the actual ground truth values. It provides the objective that the optimization algorithm (e.g., gradient descent) attempts to minimize during training. Designing a custom loss function involves several key steps: understanding the problem’s intricacies, formulating a mathematical representation of the desired behavior, and implementing this representation within your chosen deep learning framework. The objective is to create a differentiable function that reflects the specific nuances of the application better than standard, off-the-shelf loss functions. This differential aspect is important because the loss function's gradient provides information to update the model's weights.

The first critical step is a thorough analysis of the problem. In the medical imaging case, subtle, clinically relevant anomalies might encompass only a small portion of the image, while large, but irrelevant artifacts could dominate pixel-wise calculations. Standard loss functions penalize equally across the entire image, which would create training that is not optimal. Similarly, in a situation where there might be a disproportionate class distribution (common in medical diagnostics), simple cross-entropy could lead to a model biased towards the majority class, potentially overlooking critical but rare anomalies. Therefore, the loss function needs to be specifically tailored to address these kinds of imbalances. Another aspect is incorporating specific domain knowledge. For example, the severity of a small anomaly might actually be more important than a larger one that is irrelevant.

Formulating a mathematical representation involves translating the qualitative understanding of the problem into a quantitative, differentiable equation. This representation can be a combination of standard loss functions or a unique expression that captures the specific requirements. For example, to emphasize the small, clinically relevant anomalies in the retinal scans, I incorporated a weighted cross-entropy loss combined with a boundary-aware loss that penalized errors along the edges of these anomalies more heavily. The weighting for the cross-entropy loss could compensate for class imbalance; by giving more weight to the underrepresented class, the model would be penalized more for misclassifying those images and would learn to classify them more effectively. This approach allowed the model to focus on what was important, rather than on simply minimizing global pixel differences.

After the mathematical representation is decided on, the function needs to be implemented. Most deep learning frameworks, such as TensorFlow and PyTorch, allow for the creation of custom loss functions using their underlying numerical computation libraries. The primary challenge here is ensuring that the function is differentiable concerning the model's outputs, enabling backpropagation during training. This usually involves implementing the function using tensor operations provided by the framework and not standard Python.

Let's examine three code examples using PyTorch, a framework I've employed frequently for these kinds of tasks.

**Example 1: Weighted Cross-Entropy for Class Imbalance:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_cross_entropy(outputs, targets, weights):
    """
    Computes weighted cross-entropy loss for binary or multi-class classification.

    Args:
        outputs (torch.Tensor): Model outputs (logits), of shape (batch_size, num_classes).
        targets (torch.Tensor): True labels, of shape (batch_size).
        weights (torch.Tensor): Class weights, of shape (num_classes).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    outputs = F.log_softmax(outputs, dim=1)
    loss = F.nll_loss(outputs, targets, weight=weights)
    return loss

# Example usage
# Assuming binary classification with class weights [0.1, 0.9]
weights = torch.tensor([0.1, 0.9])
outputs = torch.randn(5, 2)  # batch size of 5, two classes
targets = torch.randint(0, 2, (5,)) # random target tensor of class 0 or 1 for 5 images
loss = weighted_cross_entropy(outputs, targets, weights)
print(f"Weighted cross-entropy loss: {loss.item():.4f}")
```

This first example demonstrates the fundamental use of class weights to handle an imbalanced training set. It leverages PyTorch’s `log_softmax` to calculate log probabilities from model outputs, followed by `nll_loss` which computes negative log-likelihood. `nll_loss` includes an optional `weight` parameter that is used here to implement weighted cross entropy. By assigning higher weights to less frequent classes, I could make the model pay more attention to classifying these classes correctly. The sample tensors for outputs and targets are randomly generated, which will not match any actual problem.

**Example 2: Dice Loss for Segmentation Tasks:**

```python
import torch
import torch.nn as nn

def dice_loss(outputs, targets, smooth=1e-6):
    """
    Computes Dice loss for segmentation tasks.

    Args:
        outputs (torch.Tensor): Model output, typically probabilities of segmentation mask. Shape (batch_size, height, width).
        targets (torch.Tensor): Ground truth segmentation mask, shape (batch_size, height, width).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    outputs = outputs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (outputs * targets).sum(dim=1)
    denominator = outputs.sum(dim=1) + targets.sum(dim=1)
    dice_coefficient = (2 * intersection + smooth) / (denominator + smooth)
    dice_loss = 1 - dice_coefficient.mean()
    return dice_loss

# Example usage
outputs = torch.rand(5, 64, 64)  # Example probability outputs
targets = torch.randint(0, 2, (5, 64, 64)).float() # Example binary ground truth
loss = dice_loss(outputs, targets)
print(f"Dice loss: {loss.item():.4f}")
```

This example implements the Dice Loss, often beneficial in segmentation tasks, including the type of medical imaging I was working on. It calculates the Dice coefficient, which measures the overlap between predicted and true segmentation masks, and then subtracts it from 1 to transform it into a loss. The `smooth` parameter is incorporated to prevent division by zero, which is common in segmentation problems. Here I am flattening each image to calculate the dice coefficient in the batch dimension. Like the first example, I am using randomly generated tensors for outputs and targets, rather than actual images.

**Example 3: A Combination Loss: Weighted Cross-Entropy + Dice Loss:**

```python
import torch
import torch.nn as nn

def combined_loss(outputs, targets, weights, weight_dice = 0.5, smooth = 1e-6):
  """
    Combines weighted cross-entropy and Dice loss.

    Args:
        outputs (torch.Tensor): Model outputs (logits).
        targets (torch.Tensor): True labels/segmentation masks.
        weights (torch.Tensor): Class weights for cross-entropy.
        weight_dice (float):  Weight for Dice loss.
        smooth (float): Smoothing factor for Dice loss.

    Returns:
        torch.Tensor: Scalar combined loss value.
  """
  dice_component = dice_loss(outputs, targets, smooth=smooth)
  cross_entropy_component = weighted_cross_entropy(outputs, targets.long(), weights)

  return weight_dice*dice_component + (1-weight_dice)*cross_entropy_component

# Example usage
weights = torch.tensor([0.2, 0.8])  # Example weights for two classes
outputs = torch.rand(5, 2, 64, 64)
targets = torch.randint(0, 2, (5, 64, 64)).float()

loss = combined_loss(outputs, targets, weights)
print(f"Combined Loss: {loss.item():.4f}")
```

The final example showcases a combined loss function, blending weighted cross-entropy with Dice loss. I found this combination very effective for my medical image analysis task, as it incorporates both pixel-level classification and the degree of mask overlap, resulting in a model that was better suited to detecting clinically relevant anomalies. The `weight_dice` parameter controls the trade-off between the two loss components. The use of `.long()` to convert the target tensor from float to long (integer values for labels) before feeding it into the cross entropy is critical to avoid errors.

Implementing custom loss functions has a steep initial learning curve but can result in considerably better performance in complex tasks. In my experience, a strong understanding of both the underlying math and the limitations of common loss functions allows you to create very specific functions that greatly improve model performance. For those seeking deeper knowledge, I would suggest consulting material on optimization techniques in machine learning, information theory, and convex analysis. Additionally, numerous online courses and textbooks dedicated to deep learning go into detail regarding different loss function implementations. Specifically, resources that cover advanced image segmentation techniques will often discuss common loss functions like Dice Loss, and how they work. Deep learning textbooks on medical image analysis can also provide further useful information and applications.
