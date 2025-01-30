---
title: "How can I add custom weights to training data in PyTorch?"
date: "2025-01-30"
id: "how-can-i-add-custom-weights-to-training"
---
The core challenge in assigning custom weights to training data in PyTorch lies not in the framework's limitations, but in the precise understanding and application of the `sampler` object within the `DataLoader`.  Simply modifying loss functions won't suffice for nuanced control over sample selection frequency; instead, a weighted sampling strategy is crucial.  My experience building robust anomaly detection models, particularly those dealing with heavily imbalanced datasets in the financial sector, underscores this.  Ignoring weighted sampling can lead to models overwhelmingly biased towards the majority class, ultimately diminishing performance on the minority class – a critical issue in fraud detection, where the minority class (fraudulent transactions) is precisely the most valuable.

**1. Clear Explanation:**

PyTorch's `DataLoader` provides flexibility in how it iterates through your training data.  By default, it employs random sampling without replacement.  To introduce custom weights, we need to replace the default sampler with a `WeightedRandomSampler`. This sampler requires a list or array of weights, where each element corresponds to a single data point in your dataset.  The weight determines the probability of that data point being selected during each epoch.  Higher weights increase the probability of selection.  Critically, the weights must sum to one or the `WeightedRandomSampler` will fail.

The process involves three steps:

1. **Preparing the weights:**  This involves creating a list or array of weights reflecting the desired emphasis on each data point.  The weights must be non-negative. The strategy for assigning weights depends heavily on your specific needs, ranging from simple inverse class frequency to more sophisticated approaches involving cost-sensitive learning or uncertainty estimates.

2. **Creating a `WeightedRandomSampler`:** This object takes the weight list and the dataset length as input.

3. **Integrating into the `DataLoader`:**  The `sampler` argument in `DataLoader` is used to specify the custom sampler.


**2. Code Examples:**

**Example 1: Basic Weighted Sampling**

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

# Sample data (replace with your actual data)
features = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))  # Binary classification

# Calculate weights based on inverse class frequency
class_counts = torch.bincount(labels)
weights = 1.0 / class_counts[labels]

# Create dataset and sampler
dataset = TensorDataset(features, labels)
sampler = WeightedRandomSampler(weights, len(dataset))

# Create DataLoader
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Train your model using the dataloader
# ...
```

This example shows a common scenario: addressing class imbalance.  We calculate weights inversely proportional to class frequency, giving more weight to under-represented classes.  The `torch.bincount` function efficiently computes class frequencies.  This approach assumes a binary classification problem;  multi-class scenarios require a slight modification of the weight calculation.

**Example 2:  Weights from External Source**

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels, weights):
        self.data = data
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self):
        # ... your data loading logic ...
        return data_point, label, weight

# ... Load data and labels ...
# weights obtained from a pre-computed file or external model (e.g., uncertainty estimates)
weights = torch.load('weights.pt') # Replace 'weights.pt' with your file path

# Normalize weights to sum to 1
weights = weights / weights.sum()

dataset = MyDataset(data, labels, weights)
sampler = WeightedRandomSampler(weights, len(dataset))
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

This demonstrates how to incorporate weights calculated outside the main data loading process. This is particularly useful when weights are derived from pre-training or external knowledge sources.  Note the crucial normalization step to ensure the weights sum to one.  I've used a custom `Dataset` class for illustrative purposes.  This method offers increased flexibility for complex data loading scenarios.


**Example 3: Handling Weights within a Custom Dataset Class**

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # Calculate weights within the dataset class itself
        class_counts = torch.bincount(labels)
        self.weights = 1.0 / class_counts[labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.weights[idx]


# ... load your data ...
dataset = MyDataset(data, labels)
sampler = WeightedRandomSampler(dataset.weights, len(dataset))
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

```

This approach encapsulates weight calculation within the `Dataset` class. This keeps weight computation close to the data and can improve code organization, especially for large or complex datasets. The `__getitem__` method now returns weights along with the data and labels; while this is not directly required by `WeightedRandomSampler`, this structure is useful for scenarios where weights are needed during training alongside data and labels.


**3. Resource Recommendations:**

The official PyTorch documentation on `DataLoader` and `WeightedRandomSampler` is essential reading.  Beyond this, I recommend exploring resources on class imbalance handling techniques, including cost-sensitive learning and data augmentation methods.  Understanding different sampling strategies, such as stratified sampling, is also beneficial for mastering data handling in PyTorch.  A thorough grasp of probability and statistics is invaluable for designing effective weighted sampling schemes.
