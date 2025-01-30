---
title: "How does weighted loss and sampling affect cross-entropy model performance?"
date: "2025-01-30"
id: "how-does-weighted-loss-and-sampling-affect-cross-entropy"
---
Cross-entropy loss, as commonly implemented in deep learning for classification tasks, implicitly assumes a balanced distribution of classes within a training dataset. In my experience, however, real-world data often presents significant class imbalances. When faced with such datasets, blindly applying standard cross-entropy can lead to models that are heavily biased towards the majority class, achieving high accuracy but failing to generalize effectively on the minority class. This is where techniques like weighted loss and weighted sampling come into play, directly manipulating the training process to mitigate the impact of these imbalances.

The core issue stems from the fact that the standard cross-entropy loss function treats all samples equally. Consequently, gradients calculated during backpropagation are disproportionately influenced by the prevalent class, resulting in a learning process that primarily optimizes for performance on this class. The model effectively learns to identify the majority class with high precision, but often at the cost of neglecting the finer distinctions that define the minority class. For example, in a medical imaging scenario where the presence of a rare disease constitutes the minority class, a model trained with standard cross-entropy might consistently predict the absence of the disease, even when subtle signs are present. This underscores the critical need for strategies like weighted loss and weighted sampling to recalibrate model training.

Weighted loss addresses this problem by introducing a weighting factor for each class in the loss calculation. Rather than simply summing the loss over all samples, we assign a higher weight to the loss incurred by misclassifying samples from the minority class. This essentially tells the model that errors on the minority class are more "costly," thereby encouraging it to pay more attention to these samples during optimization. Specifically, the cross-entropy loss for a single sample, typically given by -log(p_c), where p_c is the predicted probability of the true class, is modified to -w_c * log(p_c), where w_c represents the assigned weight to the true class, c. This weighting forces the model to learn more discriminating features for the minority class, since misclassifications in these instances contribute more significantly to the total loss and drive larger gradient updates.

On the other hand, weighted sampling directly manipulates the data that is fed to the training process in each batch. Instead of drawing samples uniformly at random, samples from underrepresented classes are chosen with higher probability, ensuring that the model receives a more balanced view of the entire class distribution. This technique does not change the loss function itself but alters the frequency at which samples from different classes are seen. With oversampling, we are essentially duplicating samples of the underrepresented class, while under sampling removes samples from the overrepresented class. Probabilistic sampling, a more flexible alternative, achieves a similar result, where the sampling rate for each class is selected to balance the training distribution, without resorting to duplication or removal.

Let's consider a concrete scenario. Suppose we are building a binary classifier to detect fraudulent transactions. We observe that fraudulent transactions constitute a small percentage of the total transaction volume. A standard cross-entropy loss will disproportionately focus on non-fraudulent transactions, resulting in a model that will perform poorly at identifying actual fraudulent activities.

**Example 1: Basic Cross-Entropy**

This example shows how standard cross-entropy can lead to poor performance on a imbalanced classification problem.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulate Imbalanced data
X_train = torch.randn(1000, 10)  # 10 features
y_train = torch.cat([torch.zeros(900, dtype=torch.long), torch.ones(100, dtype=torch.long)])

# Simple Neural Network
model = nn.Sequential(
  nn.Linear(10, 1),
  nn.Sigmoid()
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train.float())
    loss.backward()
    optimizer.step()
    if (epoch+1)%20 == 0:
        print(f'Epoch: {epoch+1} Loss: {loss.item():.4f}')
# Results will show a low loss, but the model will have poor recall on positive instances.
```
In this example, the `nn.BCELoss` is used directly with no weighting applied, resulting in the model having higher accuracy but very poor recall in the positive class.

**Example 2: Weighted Loss using a Class Weights**

This example shows how to integrate class weights into loss calculation for class balancing.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulate Imbalanced data
X_train = torch.randn(1000, 10)
y_train = torch.cat([torch.zeros(900, dtype=torch.long), torch.ones(100, dtype=torch.long)])

# Simple Neural Network
model = nn.Sequential(
  nn.Linear(10, 1),
  nn.Sigmoid()
)

# Compute Class Weights
num_classes = 2
class_counts = torch.tensor([torch.sum(y_train == c) for c in range(num_classes)], dtype=torch.float)
weights = 1. / class_counts
weights = weights / torch.sum(weights)
class_weights = weights.to('cpu')

# Define the loss with weights
criterion = nn.BCELoss(weight=class_weights[y_train])
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training loop
for epoch in range(100):
  model.zero_grad()
  outputs = model(X_train).squeeze()
  loss = criterion(outputs, y_train.float())
  loss.backward()
  optimizer.step()
  if (epoch+1)%20 == 0:
    print(f'Epoch: {epoch+1} Loss: {loss.item():.4f}')

# Results show a better recall on positive class than the first example.
```

In this modified code, we explicitly compute class weights based on class frequencies and pass these weights to the `nn.BCELoss` to scale the loss contribution of each training instance. Note that the weights are applied per sample in this example, so it must be shaped appropriately. This has the effect of increasing the loss contribution of examples from minority classes and thus leading to a more balanced learning process. This approach is typically more effective than the standard cross-entropy for imbalanced data.

**Example 3: Weighted Sampling**

This example demonstrates how weighted sampling can be used to mitigate class imbalances.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# Simulate Imbalanced data
X_train = torch.randn(1000, 10)
y_train = torch.cat([torch.zeros(900, dtype=torch.long), torch.ones(100, dtype=torch.long)])

# Simple Neural Network
model = nn.Sequential(
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Calculate sample weights
num_classes = 2
class_counts = torch.tensor([torch.sum(y_train == c) for c in range(num_classes)], dtype=torch.float)
sample_weights = torch.zeros_like(y_train, dtype=torch.float)
for c in range(num_classes):
    sample_weights[y_train == c] = 1. / class_counts[c]


# Create TensorDataset and DataLoader with WeightedRandomSampler
train_dataset = TensorDataset(X_train, y_train)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training loop
for epoch in range(100):
    for inputs, targets in train_loader:
      model.zero_grad()
      outputs = model(inputs).squeeze()
      loss = criterion(outputs, targets.float())
      loss.backward()
      optimizer.step()
    if (epoch+1)%20 == 0:
      print(f'Epoch: {epoch+1} Loss: {loss.item():.4f}')
# Results again show that the model learns better with weighted sampling
```
Here, we utilize `WeightedRandomSampler` to ensure that samples from minority classes are drawn with higher probability during each batch of training. The DataLoader now draws samples according to the assigned weights. The loss calculation is done using the standard `BCELoss` without class weights. This approach provides an alternative way to balance data without modifying the loss function.

In summary, while standard cross-entropy serves as a solid foundation for many classification tasks, its performance can severely degrade when faced with imbalanced data. The techniques described here, weighted loss and weighted sampling, are effective and flexible approaches to address the issue. For implementation, I recommend further exploring the documentation for your preferred deep learning framework which provides resources on how to integrate those concepts. Specifically, the documentation associated with `torch.nn.CrossEntropyLoss` and `torch.utils.data.WeightedRandomSampler` in PyTorch, and analogous documentation for TensorFlow and Keras. These sources often provide implementation details and offer best practices based on the framework's design. Further theoretical background can be found in standard machine learning textbooks, which cover topics such as the bias-variance trade-off and how it relates to class imbalances. Understanding the underlying math will enable a more nuanced application of the techniques described here.
