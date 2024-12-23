---
title: "How can focal loss improve model performance on imbalanced datasets in PyTorch?"
date: "2024-12-23"
id: "how-can-focal-loss-improve-model-performance-on-imbalanced-datasets-in-pytorch"
---

 I remember a project back in 2018 involving medical image analysis where we had a severe class imbalance – tumors were significantly less frequent than healthy tissue. Accuracy was sky-high, but the model consistently missed the few critical instances of pathology. That’s when focal loss became a crucial part of our toolkit.

The problem with standard cross-entropy loss in imbalanced datasets is that the dominant class often overwhelms the learning process. The model becomes overly confident in predicting the majority class because it’s easy to achieve a low loss in that area. The minority class, despite its importance, doesn't get enough 'attention' during training. Focal loss addresses this by modulating the standard cross-entropy loss, reducing the loss contribution for well-classified examples and focusing on the hard, misclassified ones. This essentially shifts the training focus towards the minority classes, improving their overall classification performance.

Here's how it works mathematically: the standard cross-entropy loss for a binary classification problem is given by:

`ce(p_t) = -log(p_t)`

Where `p_t` is the model's estimated probability for the correct class.

Focal loss introduces two modulating factors: a focusing parameter, gamma (γ), and a modulating factor based on the predicted probability. The formula for focal loss is:

`fl(p_t) = -(1 - p_t)^γ * log(p_t)`

The `(1-p_t)^γ` term is crucial. When the model is confident and `p_t` is near 1 (a well-classified example), `(1-p_t)^γ` approaches 0, effectively diminishing the contribution of this sample to the overall loss. Conversely, when the model is unsure and `p_t` is close to 0, the modulating factor is closer to 1, giving the sample a higher weight. Gamma controls the strength of this modulation. A higher gamma value focuses more intensely on hard examples. Typical values for gamma range from 0 to 5, with 2 often being a good starting point. When gamma is 0, focal loss becomes equivalent to standard cross-entropy.

Now, let's translate that into some practical PyTorch code. Here's an initial implementation that works for binary classification:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt)**self.gamma * bce_loss

        if self.alpha is not None:
          focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
             return torch.sum(focal_loss)
        else:
            return focal_loss
```

This implementation incorporates a `gamma` parameter and optional alpha weighting, which can add another level of class balancing by explicitly weighting the loss associated with each class. `alpha` can be useful when the class imbalance is extreme. When `alpha` is not provided, we’re simply focusing on the harder examples in the dataset, regardless of the class.

For a multiclass problem (assuming `input` tensor is a batch of class probabilities and target is a batch of integer class labels), the adjustment needs to consider softmax activations and how to obtain `p_t`. Here's the modified loss for the multiclass scenario:

```python
class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss(log_prob, target, reduction='none')

        prob_target = torch.gather(prob, dim=-1, index=target.unsqueeze(-1))
        focal_loss = (1 - prob_target)**self.gamma * loss

        if self.alpha is not None:
          focal_loss = self.alpha[target] * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
             return torch.sum(focal_loss)
        else:
            return focal_loss
```

This version uses negative log-likelihood loss (`nll_loss`) after applying softmax to the logits. It's slightly more computationally efficient than implementing the log part in the class as we did in the binary case. Note the alpha weighting here is slightly different, as we need to use target indices to apply class specific weights via the `self.alpha[target]` indexing.

Finally, for a realistic application, consider a case with a custom dataset where a subset of samples are to be prioritized during the training phase. Here’s a snippet demonstrating that idea, incorporating a basic dataloader concept:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=2):
      self.num_samples = num_samples
      self.num_classes = num_classes
      self.data = torch.randn(num_samples, 10)
      self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


# Example use with custom dataset

dataset = CustomDataset(num_samples=1000, num_classes=2)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
focal_loss = MultiClassFocalLoss(gamma=2) # or FocalLoss for binary

num_epochs = 5

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = focal_loss(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

This snippet creates a dummy dataset and uses the `MultiClassFocalLoss` we defined earlier. It showcases the integration of the focal loss into a typical training loop. We could modify this example to include an alpha weighting tensor to give explicit importance to the minority classes, or define our `CustomDataset` such that the samples are imbalanced from the get-go.

Implementing these changes in my own past projects often led to a significant improvement in the performance metrics associated with the minority classes, usually at the cost of a marginal drop in the overall accuracy, which is usually acceptable when dealing with an imbalanced dataset.

For anyone interested in diving deeper into the theoretical aspects of focal loss and class imbalance techniques, I’d highly recommend reading the original “Focal Loss for Dense Object Detection” paper by Lin et al. from Facebook AI Research. Also, “Practical Recommendations for Imbalanced Learning” by He and Garcia provides very practical insights and techniques for practitioners. For more advanced loss function design, look into research papers on metric learning, like those that discuss contrastive loss and triplet loss, often used in settings where focusing on the 'hard' cases is required. Finally, any standard textbook on statistical learning and pattern recognition should discuss the underlying theory behind the importance of addressing imbalanced datasets. I hope that helps you get started, and perhaps even gives some concrete things to test out.
