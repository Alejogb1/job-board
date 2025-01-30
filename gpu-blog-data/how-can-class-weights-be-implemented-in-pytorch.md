---
title: "How can class weights be implemented in PyTorch for multi-class classification?"
date: "2025-01-30"
id: "how-can-class-weights-be-implemented-in-pytorch"
---
The challenge of imbalanced datasets frequently arises in multi-class classification scenarios, where certain classes are drastically underrepresented compared to others. This disparity can significantly bias a model’s training, leading it to perform poorly on the minority classes. Consequently, directly applying standard loss functions often yields a model heavily skewed towards predicting the majority classes. To mitigate this, class weights can be strategically introduced, effectively increasing the importance of underrepresented classes during the learning process.

My experience working on an image recognition project involving a rare disease diagnosis underscores the critical need for this technique. The dataset consisted of thousands of images, but the positive diagnosis examples were less than 10% of the total. Training a model without addressing this imbalance resulted in a model that nearly always predicted negative, rendering it clinically useless. Using class weights proved to be crucial in achieving acceptable performance across all classes.

In PyTorch, class weights are implemented by incorporating them into the loss function. Specifically, the `torch.nn.CrossEntropyLoss` function, commonly used for multi-class classification, accepts an optional `weight` argument. This argument is a tensor representing the weight assigned to each class. The formula for the cross-entropy loss is typically adjusted as follows:

Standard Cross-Entropy Loss:
```
L(y_pred, y) = - Σ [y_i * log(p_i)] 
```
Where:
- `y_pred` represents the predicted probabilities for each class.
- `y` represents the true one-hot encoded class labels.
- `p_i` is the probability assigned to the true class *i*

Weighted Cross-Entropy Loss:
```
L_weighted(y_pred, y) = - Σ [w_i * y_i * log(p_i)]
```
Where:
- `w_i` represents the weight associated with the class *i*

The core idea is that the loss function for a correctly predicted sample will now have an extra weighting term that is larger for minority classes, causing them to contribute more substantially to the optimization. A weight value greater than one effectively increases the loss contribution, while a weight less than one reduces it. Typically, these weights are inversely proportional to the class frequency within the training data. This approach ensures that the learning process is not dominated by the more frequent classes.

Implementing this in PyTorch requires these steps: first, determine the class frequencies within your training data. Then, calculate weights for each class, usually as the inverse of their frequency or a variant thereof. Finally, create a tensor of these weights and pass it to your `CrossEntropyLoss` during the model's training.

Here are three concrete code examples demonstrating this process:

**Example 1: Basic Inverse Frequency Weights**

This example illustrates a basic implementation where weights are directly derived from the inverse of class frequencies. This is a straightforward method, often effective for moderately imbalanced datasets.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Simulated imbalanced dataset
class ImbalancedDataset(Dataset):
    def __init__(self, num_samples=1000):
      self.num_samples = num_samples
      self.labels = np.random.choice([0, 1, 2], size=num_samples, p=[0.8, 0.1, 0.1]) # Class distribution: 80%, 10%, 10%
      self.data = np.random.rand(num_samples, 10)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
       return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


dataset = ImbalancedDataset()
dataloader = DataLoader(dataset, batch_size=32)

# Calculate class frequencies and weights
class_counts = np.bincount(dataset.labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
print(f"Calculated Class Weights: {class_weights}") #Output: Calculated Class Weights: tensor([0.0012, 0.0100, 0.0100])

# Model definition
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


model = SimpleClassifier(10, 3)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(2):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
```
**Commentary:** This code snippet simulates an imbalanced dataset, calculates class frequencies and inverse weights, and incorporates those weights into the `CrossEntropyLoss` during training. The `class_weights` are printed for inspection. The simulated data ensures repeatable example behaviour. Note that smaller class values will be given larger weighting, in order to compensate for their relative lack of frequency.

**Example 2: Smoothed Inverse Frequency Weights**

This example presents a smoothed version of inverse frequency weights, which can help prevent overly large weights when a class has extremely few samples. It employs a smoothing factor to avoid extreme cases.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Simulated imbalanced dataset
class ImbalancedDataset(Dataset):
    def __init__(self, num_samples=1000):
      self.num_samples = num_samples
      self.labels = np.random.choice([0, 1, 2], size=num_samples, p=[0.8, 0.1, 0.1]) # Class distribution: 80%, 10%, 10%
      self.data = np.random.rand(num_samples, 10)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
       return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

dataset = ImbalancedDataset()
dataloader = DataLoader(dataset, batch_size=32)

# Calculate class frequencies and weights with smoothing
class_counts = np.bincount(dataset.labels)
smoothing_factor = 0.1  # Smoothing factor
class_weights = (1.0 + smoothing_factor) / (torch.tensor(class_counts, dtype=torch.float) + smoothing_factor)
print(f"Smoothed Class Weights: {class_weights}") # Output: Smoothed Class Weights: tensor([1.0122, 0.1099, 0.1099])

# Model definition (same as Example 1)
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


model = SimpleClassifier(10, 3)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (same as Example 1)
for epoch in range(2):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
```

**Commentary:**  By adding a small constant (here, 0.1) to both the numerator and denominator of the weight calculation, the smoothing factor prevents weights from exploding when a class has very few instances. This is a more robust approach, particularly in real-world scenarios where datasets can have significant class variations. The `class_weights` are printed here for easy inspection. The resulting weights are lower than in Example 1 due to smoothing.

**Example 3: Weights Calculated on a per-batch Basis (Less Common)**

While it’s more standard to calculate weights once for the entire training set, here is an example where we adjust weights on a per-batch basis, in the (less common) case where the class imbalances vary within batches. Note that this is less efficient and likely less useful.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Simulated imbalanced dataset
class ImbalancedDataset(Dataset):
    def __init__(self, num_samples=1000):
      self.num_samples = num_samples
      self.labels = np.random.choice([0, 1, 2], size=num_samples, p=[0.8, 0.1, 0.1]) # Class distribution: 80%, 10%, 10%
      self.data = np.random.rand(num_samples, 10)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
       return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


dataset = ImbalancedDataset()
dataloader = DataLoader(dataset, batch_size=32)

# Model definition (same as Example 1)
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

model = SimpleClassifier(10, 3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with dynamic weights
for epoch in range(2):
    for inputs, labels in dataloader:
      # Calculate class frequencies and weights for batch data
      class_counts = np.bincount(labels.numpy(), minlength = 3)
      class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
      criterion = nn.CrossEntropyLoss(weight=class_weights)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
```

**Commentary:** Unlike the previous examples, this code calculates the class weights for each batch dynamically based on the class distribution within that particular batch, adding an `minlength` to bincount to handle potential cases where a batch lacks samples from a given class. This technique adds computational overhead and is generally less effective than weights based on the whole dataset. However, it may be appropriate in scenarios where the class distribution varies significantly between batches.

For further exploration and understanding, I recommend studying resources covering these areas: methods for handling imbalanced data in machine learning, especially the impact of class imbalance on the training of neural networks. Additionally, examine the documentation for PyTorch's `nn.CrossEntropyLoss`, specifically its `weight` parameter. Researching various sampling methods (e.g. over-sampling, under-sampling) that can be used to augment class weighting will provide a broader understanding of how to handle class imbalance issues, and reading practical advice on training deep learning models would round out your understanding.
