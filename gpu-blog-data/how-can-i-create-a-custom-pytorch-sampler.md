---
title: "How can I create a custom PyTorch sampler for weighted class sampling with variable probabilities?"
date: "2025-01-30"
id: "how-can-i-create-a-custom-pytorch-sampler"
---
In my experience building classifiers for highly imbalanced datasets, a standard `WeightedRandomSampler` often falls short when the class probabilities need dynamic adjustment during training. I’ve frequently encountered scenarios where the distribution of classes evolves or where specific classes need boosting at different training stages. Therefore, a static weight assignment proves inadequate. A custom sampler in PyTorch, built upon `torch.utils.data.Sampler`, provides the necessary flexibility.

The foundational principle behind custom sampling is controlling the sequence in which data samples are provided to the dataloader. The `Sampler` class, a PyTorch abstract base class, achieves this through its primary method, `__iter__`, which yields indices. Instead of relying on pre-calculated weights, my approach centers on generating these probabilities on-the-fly, often influenced by current training metrics. To create this dynamic behavior, one needs to: (1) inherit from `torch.utils.data.Sampler`, (2) maintain a reference to the dataset's labels, and (3) implement a custom probability generator within the `__iter__` method.

Here’s an initial, bare-bones example that illustrates a basic random sampling mechanism that respects class identity but does not implement weighting:

```python
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np

class ToyDataset(Dataset):
    def __init__(self, num_samples, num_classes):
        self.labels = np.random.randint(0, num_classes, size=num_samples)
        self.data = np.random.randn(num_samples, 10)  # Placeholder data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CustomSamplerNoWeight(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

# Example Usage
num_classes = 3
num_samples = 100
toy_dataset = ToyDataset(num_samples, num_classes)
sampler = CustomSamplerNoWeight(toy_dataset)
dataloader = torch.utils.data.DataLoader(toy_dataset, batch_size=10, sampler=sampler)

for batch_data, batch_labels in dataloader:
    print(batch_labels)  # Print sample to confirm
    break

```

This initial example, `CustomSamplerNoWeight`, shows a basic implementation where the indices are simply shuffled randomly. While functional, it does not address the issue of weighted sampling. A crucial step towards dynamic weighting is to compute class probabilities based on some criterion. In many applications, class frequency is an important consideration. Below is an extended example that dynamically samples based on the class frequency within the dataset, but in a skewed fashion:

```python
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np


class ToyDataset(Dataset):
    def __init__(self, num_samples, num_classes):
        self.labels = np.random.randint(0, num_classes, size=num_samples)
        self.data = np.random.randn(num_samples, 10)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CustomSamplerWeightFrequency(Sampler):
    def __init__(self, dataset, skew_factor=2):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.num_samples = len(dataset)
        self.class_counts = np.bincount(self.labels)
        self.class_probs = self._calculate_probabilities(skew_factor)
        self.weights = self._calculate_sample_weights()

    def _calculate_probabilities(self, skew_factor):
        class_freq = self.class_counts / np.sum(self.class_counts)
        # Skew class probabilities, giving more weight to less frequent classes
        skewed_probs = np.power(class_freq, -skew_factor)
        return skewed_probs / np.sum(skewed_probs) # Normalize

    def _calculate_sample_weights(self):
        return np.array([self.class_probs[label] for label in self.labels])

    def __iter__(self):
        indices = np.random.choice(self.num_samples, size=self.num_samples, replace=True, p=self.weights)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

# Example Usage
num_classes = 3
num_samples = 1000
toy_dataset = ToyDataset(num_samples, num_classes)
sampler = CustomSamplerWeightFrequency(toy_dataset, skew_factor=2)
dataloader = torch.utils.data.DataLoader(toy_dataset, batch_size=10, sampler=sampler)


for batch_data, batch_labels in dataloader:
    print(batch_labels)
    break
```

In `CustomSamplerWeightFrequency`, I've introduced a class property to hold class probabilities and then used those to create weights for each sample in the dataset, prior to the `__iter__` method being called. `np.random.choice` is used within `__iter__` to generate sampled indices according to weights. The `skew_factor` allows adjustment of the probability skew toward less frequent classes. However, this approach is still static given that it recalculates probabilities only once in the constructor.

For dynamic adjustments, probabilities need to be recalculated within the `__iter__` method, ideally based on external factors, such as training epoch or validation performance. Below, I implement a sampler that can adapt by dynamically updating class probabilities before each epoch based on an external epoch number.

```python
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np

class ToyDataset(Dataset):
    def __init__(self, num_samples, num_classes):
        self.labels = np.random.randint(0, num_classes, size=num_samples)
        self.data = np.random.randn(num_samples, 10)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CustomSamplerDynamicWeight(Sampler):
    def __init__(self, dataset, initial_skew_factor=2):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.num_samples = len(dataset)
        self.class_counts = np.bincount(self.labels)
        self.skew_factor = initial_skew_factor
        self.weights = np.ones(self.num_samples)

    def _calculate_probabilities(self, skew_factor):
        class_freq = self.class_counts / np.sum(self.class_counts)
        skewed_probs = np.power(class_freq, -skew_factor)
        return skewed_probs / np.sum(skewed_probs)

    def _update_sample_weights(self, epoch):
       skew_factor = self.skew_factor + epoch * 0.5  # Dynamic adjustment based on the epoch
       class_probs = self._calculate_probabilities(skew_factor)
       self.weights = np.array([class_probs[label] for label in self.labels])

    def __iter__(self):
        self._update_sample_weights(self.current_epoch)  # Update before sampling
        indices = np.random.choice(self.num_samples, size=self.num_samples, replace=True, p=self.weights)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.current_epoch = epoch


# Example Usage
num_classes = 3
num_samples = 1000
toy_dataset = ToyDataset(num_samples, num_classes)
sampler = CustomSamplerDynamicWeight(toy_dataset, initial_skew_factor = 1)
dataloader = torch.utils.data.DataLoader(toy_dataset, batch_size=10, sampler=sampler)

for epoch in range(2):
    sampler.set_epoch(epoch)  # Adjust the skew_factor dynamically via epoch
    for batch_data, batch_labels in dataloader:
        print(f"Epoch: {epoch}, Labels:{batch_labels}")
        break  # Process only one batch to highlight epoch effect.
```

In `CustomSamplerDynamicWeight`, the `set_epoch` method allows external control over the skew factor, introducing dynamism via epoch number.  The `_update_sample_weights` now dynamically updates the weights before each epoch using the dynamic `skew_factor`. Here, skew increases as training progresses. This dynamic change highlights that class distribution of batches will not remain constant during training. This approach is an essential element for addressing changing class importance, a typical situation in complex machine learning scenarios.

For further study, I recommend examining PyTorch's official documentation for `torch.utils.data.Sampler` and `torch.utils.data.Dataset`.  Researching techniques for imbalanced learning beyond weighted sampling, such as focal loss, can offer additional tools for improving performance. Also, examining research papers on adversarial training might reveal alternative ways to dynamically adjust class importance that could further enhance these sampling techniques. Finally, studying sampling techniques that employ stratified sampling might prove useful for cases with strong class imbalances.
