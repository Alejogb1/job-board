---
title: "How can I obtain a smaller MNIST dataset in PyTorch?"
date: "2025-01-30"
id: "how-can-i-obtain-a-smaller-mnist-dataset"
---
The MNIST dataset, while widely used for introductory machine learning, often presents an unnecessary computational burden when experimenting with new architectures or debugging code. A smaller subset facilitates rapid iteration and reduces the resource overhead. Specifically, there are several methods to accomplish this within the PyTorch ecosystem, ranging from directly slicing the dataset to employing techniques for stratified sampling. I've frequently encountered this need during my development of custom training loops for generative models, where quickly testing ideas on smaller data chunks was critical.

**Explanation of Methods**

The primary strategy for obtaining a reduced MNIST dataset hinges on manipulating the `torchvision.datasets.MNIST` object after it's loaded. This class, upon instantiation, retrieves the entire dataset locally (or downloads it if necessary). It exposes attributes like `data` and `targets`, which are tensors representing the image pixels and corresponding labels respectively. These tensors can then be sliced, indexed, or randomly permuted to extract the desired subset.

Another approach involves using PyTorch's `torch.utils.data.Subset` class, which creates a subset view of an existing dataset using a specified index list. This method maintains the dataset's structure while only exposing a specific portion. This is particularly useful when preserving the original dataset is important. Finally, for more nuanced scenarios, one might resort to custom sampling techniques, perhaps to maintain class balance or replicate specific data distributions. These are usually implemented by creating custom dataset classes inheriting from `torch.utils.data.Dataset` with a custom `__len__` and `__getitem__` method, which includes the logic for sampling a smaller dataset.

**Code Examples and Commentary**

1. **Slicing:** This method is the most direct approach, allowing access to a contiguous subset based on index ranges.

```python
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

# Load the full MNIST dataset
mnist_full = datasets.MNIST(root='./data', train=True, download=True)

# Obtain a slice of the first 1000 images and labels
mnist_sliced_data = mnist_full.data[:1000]
mnist_sliced_targets = mnist_full.targets[:1000]


# Create a custom dataset from the sliced data and targets
class SlicedMNIST(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0).float() / 255.0, self.targets[idx]

mnist_sliced = SlicedMNIST(mnist_sliced_data, mnist_sliced_targets)


# Create a DataLoader
sliced_dataloader = DataLoader(mnist_sliced, batch_size=32, shuffle=True)

#Example Usage
for images, labels in sliced_dataloader:
    print(f"Batch Image shape: {images.shape}, Batch Labels Shape: {labels.shape}")
    break
```

*Commentary:* The example loads the standard MNIST training data. We take a slice of the `data` and `targets` tensors using standard Python indexing, selecting the first 1000 elements. A new dataset class, `SlicedMNIST`, is created to properly return single images, adding the necessary channel dimension, and normalizing the pixel values. A `DataLoader` is constructed for iterative access, which simplifies training loop creation.

2.  **`torch.utils.data.Subset`:**  This method is useful when we require a non-contiguous subset, defined by specific indices.

```python
import torch
from torchvision import datasets
from torch.utils.data import Subset, DataLoader
import random

# Load the full MNIST dataset
mnist_full = datasets.MNIST(root='./data', train=True, download=True)

# Create a list of random indices (e.g., 500 random indices)
num_subset = 500
random_indices = random.sample(range(len(mnist_full)), num_subset)

# Create the subset view
mnist_subset = Subset(mnist_full, random_indices)

# Create a DataLoader
subset_dataloader = DataLoader(mnist_subset, batch_size=32, shuffle=True)

#Example Usage
for images, labels in subset_dataloader:
    print(f"Batch Image shape: {images.shape}, Batch Labels Shape: {labels.shape}")
    break
```

*Commentary:* This method loads the standard MNIST dataset and then uses `random.sample` to select a set of 500 unique indices. Then,  `Subset` creates a new dataset view, referring back to the original MNIST dataset. This is memory-efficient since it does not copy the underlying data. Accessing it via the dataloader is the same as accessing any PyTorch dataset.

3. **Custom Dataset with Stratified Sampling:**  This example demonstrates a more complex sampling strategy, potentially maintaining class representation in the smaller set.  In real-world usage, this code would need to be augmented to return the images as tensors in the correct format.

```python
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Load the full MNIST dataset
mnist_full = datasets.MNIST(root='./data', train=True, download=True)

# Function to sample indices based on class
def stratified_sample_indices(targets, samples_per_class):
    classes = torch.unique(targets)
    indices = []
    for c in classes:
        class_indices = torch.where(targets == c)[0].numpy()
        sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(sampled_indices)
    return indices

# Number of samples per class required in the subset
samples_per_class = 20

# Get stratified sample indices
stratified_indices = stratified_sample_indices(mnist_full.targets, samples_per_class)

# Create custom dataset using sample indices
class StratifiedMNIST(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.dataset[index][0], self.dataset[index][1] # Returns the data, and target


# Create custom dataset instance
stratified_mnist = StratifiedMNIST(mnist_full, stratified_indices)

#Create DataLoader
stratified_dataloader = DataLoader(stratified_mnist, batch_size = 32, shuffle = True)

#Example Usage
for images, labels in stratified_dataloader:
    print(f"Batch Image shape: {images.shape}, Batch Labels Shape: {labels.shape}")
    break
```
*Commentary:* In this example, a custom `stratified_sample_indices` function is defined. This function iterates through unique classes in the dataset. For each class, it samples `samples_per_class` random indices. This ensures that each class is represented equally, unlike random sampling which can be skewed if certain classes are underrepresented in the larger dataset. A custom dataset class, `StratifiedMNIST`, is used to encapsulate the subset and present it in a way the dataloader can use.

**Resource Recommendations**

*   **PyTorch Official Documentation:** The official documentation for `torchvision.datasets`, `torch.utils.data.Subset`, and `torch.utils.data.Dataset` provides detailed information on their usage and implementation. Reading the documentation on `DataLoader` is also crucial for effectively using these datasets during training.
*   **PyTorch Tutorials:** The PyTorch website offers numerous tutorials that cover dataset handling and custom dataset creation. These tutorials often provide practical examples beyond simple class manipulation.  Specifically look at tutorials covering image dataset loading, custom dataset classes, and dataloaders.
*   **Online Machine Learning Courses:** Many introductory machine learning courses, particularly those using PyTorch, dedicate portions to dataset manipulation and loading. Exploring such materials will often reveal best practices and common pitfalls when creating smaller datasets.
* **Advanced Machine Learning Books:** Books covering specialized machine learning topics, such as generative modeling or deep learning for computer vision, often delve into sophisticated dataset handling, including stratified sampling and class-balanced datasets, for improved training performance.

In practice, the choice between slicing, using `Subset`, or a custom dataset depends heavily on the specific requirements of the task. Simple slicing is fast and effective for contiguous data, while `Subset` offers more flexibility for random subsets without copying the data, and custom datasets provides granular control over dataset construction and sampling. I personally find myself using a combination of these depending on the experiment and dataset size I am working with. When exploring many hyperparameter values on a new model, I usually start with a very small, sliced subset and then move to the other methods when moving on to more resource heavy experiments.
