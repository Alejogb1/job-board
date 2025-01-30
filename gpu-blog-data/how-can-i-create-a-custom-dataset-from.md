---
title: "How can I create a custom dataset from MNIST using PyTorch?"
date: "2025-01-30"
id: "how-can-i-create-a-custom-dataset-from"
---
Creating a custom dataset from MNIST using PyTorch necessitates a thorough understanding of PyTorch's `Dataset` and `DataLoader` classes, and a clear strategy for manipulating the MNIST data.  My experience working on large-scale image classification projects, specifically those involving handwritten digit recognition, has highlighted the importance of efficient data handling to optimize model training.  Directly accessing and modifying the raw MNIST data within PyTorch's framework provides granular control, surpassing the limitations of readily available pre-processed subsets.

**1. Clear Explanation:**

The MNIST dataset, readily available through PyTorch's `torchvision.datasets`, provides a convenient starting point. However, creating a custom dataset allows for targeted modifications, such as data augmentation, selective inclusion of specific digits, or the introduction of noise for robustness testing.  This is achieved by inheriting from PyTorch's `Dataset` class and overriding its core methods: `__init__`, `__len__`, and `__getitem__`.

The `__init__` method initializes the dataset, loading the raw MNIST data and performing any necessary preprocessing steps. This includes downloading the dataset if it isn't locally available, transforming the images (e.g., resizing, normalization), and potentially filtering the data based on specified criteria.

The `__len__` method returns the total number of samples in the dataset, essential for iterating through the data using PyTorch's `DataLoader`.

The `__getitem__` method is the heart of the custom dataset.  Given an index, it returns the corresponding image and label. This method allows for on-the-fly data augmentation or transformations before presenting the data to the model.

The `DataLoader` class then facilitates efficient batching and shuffling of the data, optimizing the training process.  Parameters like `batch_size`, `shuffle`, and `num_workers` significantly impact training speed and performance.  Careful consideration of these parameters is crucial for efficient model training.

**2. Code Examples with Commentary:**

**Example 1: Subset of MNIST Digits (0-4):**

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class MNISTSubset(Dataset):
    def __init__(self, root, transform=None, download=True, digits=(0, 1, 2, 3, 4)):
        self.mnist = datasets.MNIST(root, train=True, transform=transform, download=download)
        self.indices = [i for i, (img, label) in enumerate(self.mnist) if label in digits]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.mnist[self.indices[idx]]
        return img, label

# Example Usage
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
subset_dataset = MNISTSubset(root='./data', transform=transform)
subset_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)

for images, labels in subset_loader:
    # Training loop here
    pass
```

This example demonstrates creating a dataset containing only digits 0-4.  The `__init__` method filters the indices to include only these digits.  The `transform` argument applies standard MNIST normalization.


**Example 2:  Adding Gaussian Noise:**

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NoisyMNIST(Dataset):
    def __init__(self, root, transform=None, download=True, noise_std=0.1):
        self.mnist = datasets.MNIST(root, train=True, transform=transform, download=download)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        noise = torch.randn(img.shape) * self.noise_std
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1) #Ensure pixel values remain within [0,1]
        return noisy_img, label

# Example Usage
transform = transforms.ToTensor()
noisy_dataset = NoisyMNIST(root='./data', transform=transform, noise_std=0.2)
noisy_loader = DataLoader(noisy_dataset, batch_size=32, shuffle=True)

for images, labels in noisy_loader:
    #Training loop
    pass

```

Here, Gaussian noise is added to each image within the `__getitem__` method.  `torch.clamp` ensures that pixel values remain within the valid range [0,1].  The noise standard deviation (`noise_std`) controls the intensity of the noise.


**Example 3:  Data Augmentation with Random Cropping:**

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class AugmentedMNIST(Dataset):
    def __init__(self, root, transform=None, download=True):
        self.mnist = datasets.MNIST(root, train=True, transform=transform, download=download)
        self.transform_aug = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.ToTensor()])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        if self.transform_aug:
            img = self.transform_aug(img)
        return img, label

# Example Usage
transform = transforms.ToTensor()
augmented_dataset = AugmentedMNIST(root='./data', transform=transform)
augmented_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)

for images, labels in augmented_loader:
    #Training loop
    pass

```

This example demonstrates random cropping for data augmentation.  A separate `transform_aug` is defined, allowing for flexible application of augmentation techniques. The padding argument in `RandomCrop` prevents information loss at the edges.



**3. Resource Recommendations:**

For a deeper understanding of PyTorch's data handling capabilities, I highly recommend consulting the official PyTorch documentation.  The documentation provides comprehensive explanations of `Dataset`, `DataLoader`, and other relevant classes.  Exploring tutorials and examples specifically focusing on custom datasets will prove invaluable. Additionally, reviewing relevant chapters in introductory machine learning textbooks focusing on data preprocessing and handling will provide a strong foundation.  Understanding NumPy array manipulations will also be beneficial, given its frequent interaction with PyTorch tensors.
