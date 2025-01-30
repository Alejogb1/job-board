---
title: "How to apply weighted loss functions based on `train_dataloader` in PyTorch Lightning?"
date: "2025-01-30"
id: "how-to-apply-weighted-loss-functions-based-on"
---
Batch imbalance is a frequent challenge in deep learning, particularly when dealing with datasets exhibiting skewed class distributions. My experience training a model to classify satellite images of deforestation revealed that unweighted losses significantly favored the majority class (non-deforested areas), leading to poor generalization on minority classes (deforested regions). This necessitated the use of a weighted loss function, specifically tailored to the batch composition provided by the `train_dataloader` in PyTorch Lightning.

Implementing a dynamic weighted loss involves adapting the loss function to the prevalence of each class within a given batch. This contrasts with static weights, which are pre-defined based on the entire dataset's class distribution. The key advantage is a greater emphasis on underrepresented classes during each training iteration, rather than relying on potentially imbalanced entire-dataset statistics. This is essential because even with stratified sampling on a per-batch basis, the proportions can fluctuate from batch to batch.

The fundamental process comprises three steps. First, inside your PyTorch Lightning module, you calculate class-specific weights within the training step. Second, you use those dynamically calculated weights in the loss function of choice. Finally, ensure that the weighted loss is backpropagated appropriately. This involves avoiding manipulations of tensors that would detach them from the computational graph. Here, I will focus on three common use cases, demonstrating how this can be accomplished with both standard and more advanced methods.

**Example 1: Weighted Cross-Entropy for a Multi-Class Classification**

This is a relatively straightforward application, assuming your labels are integer-encoded. The example below uses `torch.nn.CrossEntropyLoss`, which accepts optional class weights as an input.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class WeightedClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Linear(10, num_classes) # Simplified model for demonstration
        self.loss_fn = nn.CrossEntropyLoss(reduction='none') # Reduction is managed manually

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 1. Calculate class counts per batch
        class_counts = torch.bincount(y, minlength=logits.shape[1])
        # Ensure no zero divisions by adding 1 to all class counts
        class_weights = 1.0 / (class_counts + 1.0)

        # 2. Compute loss
        batch_loss = self.loss_fn(logits, y)
        weighted_loss = (batch_loss * class_weights[y]).mean()

        self.log('train_loss', weighted_loss)
        return weighted_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
```

In this snippet, `torch.bincount` efficiently calculates the occurrences of each class within the batch. The weights are then computed as the inverse of these counts (plus 1 to avoid division by zero and to add a little smoothing), ensuring that rarer classes have higher weights. We use `reduction='none'` in `CrossEntropyLoss` to get the individual losses for each sample, then apply our dynamic weights before averaging with `mean()`. This is critical for a well-defined backpropagation graph.

**Example 2: Weighted Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for Multi-label Classification**

My experience with remote sensing data involved multiple overlapping classifications, which required treating each label independently. Here, I apply `BCEWithLogitsLoss` and adjust the weights for each label.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class WeightedMultiLabelClassifier(pl.LightningModule):
    def __init__(self, num_labels):
        super().__init__()
        self.model = nn.Linear(10, num_labels) # Simplified for demonstration
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # y is a multi-label tensor (batch_size, num_labels)
        logits = self(x)

        # 1. Calculate label-specific counts per batch.
        label_counts = y.sum(dim=0)
        # Ensure no zero divisions by adding 1 to all label counts
        label_weights = 1.0 / (label_counts + 1.0)


        # 2. Compute loss
        batch_loss = self.loss_fn(logits, y.float()) # BCEWithLogitsLoss expects float labels
        weighted_loss = (batch_loss * label_weights).mean()

        self.log('train_loss', weighted_loss)
        return weighted_loss


    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters())
```
This example showcases `BCEWithLogitsLoss` used with multi-label data. The critical distinction is that `y` represents a tensor of binary labels for each sample, for each class. The class-wise weights are then computed as the inverse number of positives (plus 1). The application of `y.float()` is important as `BCEWithLogitsLoss` works on float tensor labels. The core principle of dynamically calculating the weight based on counts in the batch remains the same.

**Example 3: Custom Loss Function with Dynamic Weights**

The previous examples have leveraged PyTorch's built-in loss functions. However, it might be required to employ a customized loss function. This example, using the same multi-label classification, demonstrates calculating weights and applying them within a custom loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CustomWeightedMultiLabelClassifier(pl.LightningModule):
    def __init__(self, num_labels):
        super().__init__()
        self.model = nn.Linear(10, num_labels) # Simplified model
        
    def forward(self, x):
        return self.model(x)

    def custom_loss(self, logits, target, weights):
          # Using the binary cross entropy function as an example
          bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
          # Here we apply weights
          weighted_bce = bce * weights
          return weighted_bce.mean()


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 1. Calculate label-specific counts per batch
        label_counts = y.sum(dim=0)
        # Ensure no zero divisions
        label_weights = 1.0 / (label_counts + 1.0)

        # 2. Compute loss with our custom function
        weighted_loss = self.custom_loss(logits, y.float(), label_weights)

        self.log('train_loss', weighted_loss)
        return weighted_loss

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters())

```
Here, I've created a `custom_loss` function that is based on the same `binary_cross_entropy_with_logits`, but enables weight assignment after the unreduced loss has been calculated. This offers flexibility, in scenarios where the integrated weight support provided by the native PyTorch loss functions is insufficient.

**Resource Recommendations**

For further exploration, I recommend reviewing the official PyTorch documentation for `torch.nn.CrossEntropyLoss`, `torch.nn.BCEWithLogitsLoss`, and other loss functions applicable to your problem. The documentation of `torch.bincount` is beneficial to understand the efficient counting of class occurrences. Understanding PyTorch's automatic differentiation mechanism is crucial for proper loss implementations. Furthermore, examining academic papers on imbalanced learning can provide insights into various techniques of handling class imbalance. Finally, delving into PyTorch Lightning’s core documentation helps understand how the library handles training steps, batch processing and logging, which are essential for debugging and effective training.
