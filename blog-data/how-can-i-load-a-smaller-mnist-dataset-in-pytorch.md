---
title: "How can I load a smaller MNIST dataset in PyTorch?"
date: "2024-12-23"
id: "how-can-i-load-a-smaller-mnist-dataset-in-pytorch"
---

,  I’ve often found myself needing a quick way to work with a reduced MNIST dataset, be it for rapid prototyping, debugging, or resource-constrained environments. Full-size datasets can be cumbersome when you're just trying to validate a new architectural idea, so efficient subsampling is a crucial skill. The good news is that it's surprisingly straightforward in PyTorch. Let’s break it down systematically.

The core challenge is not the *loading* itself, but the *selection* of a smaller subset from the standard MNIST dataset. PyTorch’s `torchvision.datasets.MNIST` module neatly handles the initial download and storage. The trick lies in how we manipulate the data after it’s loaded, or even during the loading process, to extract our desired reduced set.

My personal experience involved a project where I was experimenting with a novel learning rate scheduler. Training on the full MNIST felt wasteful during early testing, especially given how quickly things can go wrong in the early stages. I needed a way to quickly iterate, and waiting minutes for a single epoch wasn't conducive to that. I had initially tried slicing the tensors after loading the whole dataset, but this was inefficient in terms of memory usage. So, I refined my approach. I developed methods based on indexing and creating a custom subset. This saved both time and computational resources.

There are primarily three efficient methods I've found useful. Let's examine each:

**Method 1: Using `Subset` from `torch.utils.data`**

This is probably the most elegant and widely adopted way. It leverages PyTorch’s built-in `Subset` class, which allows you to create a new dataset object from an existing one using indices. Here's the breakdown:

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np


def load_smaller_mnist_subset(subset_size=1000):
    """Loads a smaller subset of the MNIST dataset."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = np.random.choice(len(full_dataset), subset_size, replace=False)
    subset = Subset(full_dataset, indices)
    return subset


if __name__ == '__main__':
    smaller_mnist = load_smaller_mnist_subset(500)  # 500 examples
    print(f"Number of samples in the subset: {len(smaller_mnist)}")
    print(f"Example sample shape: {smaller_mnist[0][0].shape}") # (1,28,28) - 1 channel, 28x28 image
    print(f"Example label: {smaller_mnist[0][1]}")

```

In this example, the `load_smaller_mnist_subset` function downloads the entire MNIST training dataset using `torchvision.datasets.MNIST`, but then uses numpy’s `random.choice` to randomly select `subset_size` indices without replacement. A `Subset` object is created using these random indices, which acts as a pointer to the original data and returns only the selected samples when accessed. This is significantly more memory-efficient than storing the entire dataset in memory when only a subset is required.  `replace=False` ensures we don't select the same index twice. The `transform` is the standard way to process the MNIST images to tensor and normalize them for better training.

**Method 2: Using Indices within a `DataLoader`**

Another straightforward method involves providing the indices directly to the `DataLoader`. This can be especially useful when you want to control the sample selection more dynamically, perhaps for a specific training regime or a cross-validation split.

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def load_mnist_with_indexed_dataloader(subset_size=1000, batch_size=32):
    """Loads a smaller MNIST dataset and returns a dataloader"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = np.random.choice(len(full_dataset), subset_size, replace=False)
    subset_sampler = torch.utils.data.SubsetRandomSampler(indices) # using a Sampler
    data_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=subset_sampler)
    return data_loader

if __name__ == '__main__':
    smaller_dataloader = load_mnist_with_indexed_dataloader(subset_size=500, batch_size=100)
    for images, labels in smaller_dataloader:
        print(f"Shape of a batch of images: {images.shape}")
        print(f"Shape of a batch of labels: {labels.shape}")
        break # lets just look at the first batch

```

Here, we’re still loading the full MNIST dataset. However, instead of creating a separate `Subset` object, we generate indices using the same `numpy` methodology and then feed these indices as the `sampler` in the `DataLoader`. This allows the `DataLoader` to only iterate over these chosen samples. The key is using `SubsetRandomSampler`, a specific sampler designed to sample from a subset based on the passed indices. This approach is effective when you need to control the iteration order and batching behavior, without creating explicit `Subset` objects.

**Method 3:  Manual Slicing with `torch.utils.data.TensorDataset`**

This approach directly manipulates the underlying data tensors to create a subsampled dataset. While less elegant than the previous two methods, it’s useful when you want very fine-grained control or need to interface with data already loaded as tensors.

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import numpy as np


def load_smaller_mnist_tensors(subset_size=1000):
    """Loads MNIST as tensors and returns a smaller subset."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_tensor = torch.stack([sample[0] for sample in full_dataset])
    labels_tensor = torch.tensor([sample[1] for sample in full_dataset])
    indices = np.random.choice(len(data_tensor), subset_size, replace=False)

    subset_data = data_tensor[indices]
    subset_labels = labels_tensor[indices]
    tensor_dataset = TensorDataset(subset_data, subset_labels)
    return tensor_dataset


if __name__ == '__main__':
    smaller_mnist_tensor = load_smaller_mnist_tensors(500)
    print(f"Number of samples in the tensor dataset: {len(smaller_mnist_tensor)}")
    print(f"Example sample shape: {smaller_mnist_tensor[0][0].shape}") # (1,28,28)
    print(f"Example label: {smaller_mnist_tensor[0][1]}")

```

In this method, the full dataset is loaded and the image and label data are converted to tensors by manually iterating through the dataset and using `torch.stack` to combine them. Indices are selected using `numpy` and then the tensors are sliced to obtain our smaller subset of images and labels. We package them into a `TensorDataset` which is a convenient wrapper for paired tensors. Although this method involves a higher initial memory usage by bringing all the data into tensors at first, it demonstrates how you can efficiently sample from loaded tensors when necessary. This is less efficient than method 1 and 2 due to that initial load, especially for very large datasets, but it is extremely flexible.

**Considerations and Resources**

Choosing the best method depends heavily on the specific use case. Method 1 is typically the preferred approach for most scenarios, as it’s clean, efficient, and well-integrated with PyTorch’s dataset handling. Method 2 offers flexibility in batching and selection, while method 3 caters to more specialized use-cases where data is already in tensor format or extremely fine-grained control is required.

For deeper insight into `torch.utils.data` and related modules, I recommend the official PyTorch documentation, of course. But for a more theoretical understanding of data sampling techniques, refer to “Pattern Recognition and Machine Learning” by Christopher Bishop. It is not a PyTorch-specific resource but its chapter on sampling distributions and random sampling methods offers a fundamental understanding of how the random indices, in the examples, are calculated. Furthermore, “Deep Learning” by Ian Goodfellow, Yoshua Bengio, and Aaron Courville includes in-depth discussions on data loading and manipulation strategies within deep learning frameworks. Understanding the concepts from these texts will allow you to better optimize your data loading and preprocessing pipeline in PyTorch and beyond.

In essence, creating smaller MNIST datasets in PyTorch is quite flexible. The techniques illustrated provide practical, real-world strategies that you'll find invaluable in your deep learning journey. The important thing is to know the tools available and understand their strengths and limitations, tailoring the approach to your specific needs.
