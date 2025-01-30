---
title: "How do I adjust input shapes for multi-class classification with TorchMetrics?"
date: "2025-01-30"
id: "how-do-i-adjust-input-shapes-for-multi-class"
---
TorchMetrics, while providing an elegant interface for evaluation, can present initial hurdles when dealing with multi-class classification and nuanced input shapes.  I've repeatedly encountered scenarios where discrepancies in predicted and target tensor dimensions lead to cryptic errors, usually stemming from implicit assumptions about the nature of the input. The core issue lies in understanding how TorchMetrics expects inputs for multi-class settings and subsequently how to transform your model's output and ground truth labels to comply. This requires a deep dive into the `compute` and `update` methods of the metric classes.

Fundamentally, for multi-class problems, TorchMetrics primarily interacts with tensors in one of two common configurations. The first, and most prevalent, expects predicted tensors to have a shape of `(N, C)` or `(N, C, …)` where *N* is the batch size, *C* represents the number of classes, and the trailing dimensions indicate if we have multi-dimensional data per sample (e.g., image data). The values at a specific `[i, c]` index represent the prediction probability/score/logit for sample *i* belonging to class *c*. Correspondingly, the target tensor has the shape `(N, )` or `(N, ...)`, where the values contain class indices (integers ranging from 0 to *C-1*) of the correct class for the i-th sample. The second case arises when the predicted tensor already encodes the most likely class through its index - often resulting from `torch.argmax`. Here, predicted and target have compatible dimensions i.e., both of the shape `(N, )` or `(N, ...)`. In this instance, both the predicted and target tensors directly represent the classified index.

The mismatch arises frequently when a model's output is not directly a probability tensor across all classes, and a user needs to handle post processing before handing it to a metric class like `Accuracy` or `F1Score`.  For instance, if your model performs regression or outputs a latent space, or your output is already `argmax`-ed before passing into the TorchMetrics. The crucial part, is adapting both the output and the ground truth to conform to these expectations. If the model outputs probabilities for only one class (e.g., a binary task) but you treat it as multi-class, this will also cause errors, given that the dimension will be `(N, 1)` instead of `(N, C)`.

Let's dissect this using three code examples to demonstrate how to adjust input shapes.

**Example 1:  Converting model logits to class probabilities**

Assume you have a model outputting logits with shape `(N, C)` and targets that are class indices `(N,)`, which is already well-suited for TorchMetrics usage when they are not passed through `torch.argmax`.

```python
import torch
from torchmetrics import Accuracy

# Simulate model outputs (logits) and targets
N = 100  # Batch size
C = 5   # Number of classes
logits = torch.randn(N, C)  # Shape: (N, C)
targets = torch.randint(0, C, (N,))  # Shape: (N,)

# Initialize the accuracy metric
accuracy = Accuracy(task="multiclass", num_classes=C)

# Update and compute metric
accuracy.update(logits, targets)
acc_value = accuracy.compute()
print(f"Accuracy: {acc_value}")

# Additional case of using softmax before computation.
probs = torch.softmax(logits, dim=-1)
accuracy.reset()
accuracy.update(probs, targets)
acc_value = accuracy.compute()
print(f"Accuracy (from Softmaxed logits): {acc_value}")

```

In this example, the predicted output (`logits`) are logits which are then directly passed to the metrics class. Torchmetrics performs the `torch.argmax` computation internally and computes the accuracy. It demonstrates a straightforward use case where the model's output is compatible with what the metric expects and demonstrates using a probability distribution as well. If your model does not output logits or probabilities, you will need to ensure that you are providing the right input shape to the metric.

**Example 2:  Handling model output as a class index**

Suppose your model outputs class indices directly after taking the argmax of the output, or you're using an encoder that outputs indices of the most likely class. In this case, both `predicted` and `targets` should have the same shape, `(N, )`. We are going to simulate the `torch.argmax` being used on the output before being used for metric computation.

```python
import torch
from torchmetrics import Accuracy

# Simulate model outputs (class indices) and targets
N = 100  # Batch size
C = 5   # Number of classes
logits = torch.randn(N, C)  # Shape: (N, C)
predicted_classes = torch.argmax(logits, dim=-1) # Shape (N,)
targets = torch.randint(0, C, (N,)) # Shape: (N,)

# Initialize the accuracy metric
accuracy = Accuracy(task="multiclass", num_classes=C)

# Update and compute metric
accuracy.update(predicted_classes, targets)
acc_value = accuracy.compute()
print(f"Accuracy: {acc_value}")

```

Here, the model's output `predicted_classes` is already the class index, we bypass the `torch.softmax` operation and then directly use this to compare to targets of the same shape. This represents an extremely common, slightly hidden, problem when a user is not aware that torchmetrics expects the logits/probabilities and performs an argmax internally. You will need to keep track of the dimensions to prevent errors.

**Example 3:  Dealing with image outputs and class indices**

This situation extends the previous two by adding spatial dimensions to our model's output, commonly seen in image classification where each pixel may have its own predicted class. Let us assume a 3D (Batch, Channel, Height, Width) Image for both the output of the model and the target. In order to compute the metrics correctly, we need to collapse the spatial dimensions and ensure we pass in tensors of the correct dimensions i.e., with the dimensions (Batch, Channel) for the model output and (Batch, Channel) for the target.

```python
import torch
from torchmetrics import Accuracy

# Simulate model outputs (logits) for multi-dimensional input and targets
N = 10  # Batch size
C = 5   # Number of classes
H = 32  # Height of image
W = 32  # Width of image
logits = torch.randn(N, C, H, W)  # Shape: (N, C, H, W)
predicted_classes = torch.argmax(logits, dim=1)  # Shape: (N, H, W)
targets = torch.randint(0, C, (N, H, W))  # Shape: (N, H, W)

# Initialize the accuracy metric
accuracy = Accuracy(task="multiclass", num_classes=C)

# Update and compute metric
accuracy.update(predicted_classes.view(N,-1), targets.view(N,-1)) # Reshape each to (N, H*W)
acc_value = accuracy.compute()
print(f"Accuracy: {acc_value}")

```

In the third example, the model produces an output with spatial dimensions, but after taking the `torch.argmax` operation, we reshape it into a single `N x (H*W)` matrix. Similarly, the targets are also flattened into the same `N x (H*W)` shape for correct comparison by the accuracy metric. This example is essential for models dealing with 2D or 3D data, such as image segmentation tasks. It explicitly shows that these spatial dimensions need to be flattened before they can be ingested into the metric class.

Understanding the expected shape of inputs for TorchMetrics in a multi-class setting is crucial for effective evaluation. Key considerations include: whether your model directly outputs logits/probabilities or class indices, and the presence of any spatial dimensions which need to be flattened. Always inspect your input and the expected dimensions by the metric before you start using TorchMetrics.

For further exploration, consider examining resources on:
*   **PyTorch documentation:** particularly the sections on tensor manipulation and the `torch.argmax`, `torch.softmax` functions.
*   **TorchMetrics documentation:** focus on the input requirements for specific metrics such as `Accuracy`, `F1Score`, `ConfusionMatrix` under the `torchmetrics.classification` package, and read the documentation carefully for each function you are planning to use. It is best to check what each specific metric requires for input.
*   **Textbooks on Deep Learning**: specifically those covering classification and evaluation.  Pay close attention to the sections describing standard evaluation metrics and the expected outputs from the model.
*   **Tutorials on Image Classification and Segmentation** on well known PyTorch documentation as well as on other sites. Tutorials will typically walk you through how to compute the metrics for specific problems.

By diligently adhering to these principles and taking a critical look at the dimensionality of your tensors, the integration of TorchMetrics into your multi-class classification workflow will become much more intuitive and error-free. These principles are the cornerstone of error-free evaluation in deep learning.
