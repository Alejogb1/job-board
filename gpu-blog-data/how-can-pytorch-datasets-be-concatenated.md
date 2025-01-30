---
title: "How can PyTorch datasets be concatenated?"
date: "2025-01-30"
id: "how-can-pytorch-datasets-be-concatenated"
---
Concatenating PyTorch datasets efficiently and correctly requires understanding that datasets in PyTorch are not inherently designed for direct merging. They are, primarily, interfaces for accessing data points, each defined by the `__getitem__` and `__len__` methods. Therefore, naive concatenation attempts will often break these fundamental assumptions, particularly when dealing with datasets of differing characteristics. I encountered this directly when building a multi-modal image classification pipeline that combined synthetic data with real-world captures.

The core problem arises from PyTorch’s `Dataset` class expecting an integer index input to `__getitem__` that corresponds to a specific entry. When combining datasets, maintaining this index validity across merged datasets becomes crucial. Simply placing datasets end-to-end without adjusting indices will cause out-of-bounds errors.

A direct solution involves creating a composite dataset class that encapsulates multiple datasets, effectively acting as a wrapper. This wrapper manages indexing by internally delegating requests to the correct dataset instance. Here is how I typically approach this, starting with the basic concept:

**1. The Composite Dataset Class:**

I begin by defining a custom dataset class which maintains a list of dataset objects. The `__init__` method accepts the individual datasets as arguments, storing them, and calculating prefix lengths for efficient indexing. The crucial component here is the `__getitem__` method. It iterates through the stored dataset lengths, determining which dataset contains the requested index, then fetches the data from that dataset. The `__len__` method simply accumulates the lengths of the contained datasets.

```python
import torch
from torch.utils.data import Dataset

class ConcatenatedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = [0]
        for dataset in self.datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        for i, length in enumerate(self.cumulative_lengths[1:]):
            if index < length:
                return self.datasets[i][index - self.cumulative_lengths[i]]
        raise IndexError("Index out of bounds")

# Example Usage
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset1 = SimpleDataset([1, 2, 3])
dataset2 = SimpleDataset([4, 5, 6, 7])

concat_dataset = ConcatenatedDataset([dataset1, dataset2])

print(f"Length of concatenated dataset: {len(concat_dataset)}") # Output: 7
print(f"Element at index 3: {concat_dataset[3]}") # Output: 4
print(f"Element at index 6: {concat_dataset[6]}") # Output: 7
```

In the provided code, `ConcatenatedDataset` is instantiated with two `SimpleDataset` objects. The lengths and indices are calculated to map correctly into each constituent dataset. Observe that accessing index 3 retrieves the first element from `dataset2`, since the first dataset's length was 3. This example showcases the fundamental logic of index mapping when datasets are combined in this way.

**2. Handling Differing Data Transformations:**

Real-world datasets often require different preprocessing steps. For instance, one dataset might necessitate normalization, while another may require specific augmentation. To manage this, I extend the `ConcatenatedDataset` to incorporate per-dataset transforms using a dictionary. This refinement allows for customized preprocessing, making the concatenated dataset compatible for training when different data formats are in use.

```python
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TransformedConcatenatedDataset(Dataset):
    def __init__(self, datasets, transforms_dict):
        self.datasets = datasets
        self.transforms_dict = transforms_dict
        self.cumulative_lengths = [0]
        for dataset in self.datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

        if len(datasets) != len(transforms_dict):
            raise ValueError("Number of transforms must match number of datasets")


    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        for i, length in enumerate(self.cumulative_lengths[1:]):
            if index < length:
                item = self.datasets[i][index - self.cumulative_lengths[i]]
                if i in self.transforms_dict:
                   item = self.transforms_dict[i](item)
                return item
        raise IndexError("Index out of bounds")

# Example Usage with transformations

class SimpleDatasetTransformed(Dataset):
    def __init__(self, data):
       self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset3 = SimpleDatasetTransformed([1,2,3])
dataset4 = SimpleDatasetTransformed([4,5,6])

transform1 = transforms.Lambda(lambda x: x * 2)
transform2 = transforms.Lambda(lambda x: x + 10)

transforms_dict = {0:transform1, 1:transform2}

transformed_concat = TransformedConcatenatedDataset([dataset3, dataset4], transforms_dict)


print(f"Transformed Element at index 0: {transformed_concat[0]}") # Output: 2
print(f"Transformed Element at index 4: {transformed_concat[4]}") # Output: 15
```

Here, I demonstrate the addition of a `transforms_dict` within the constructor. This dictionary associates transforms with individual datasets based on their index in the datasets list. When an element is requested, the transformation specific to its origin dataset is applied before being returned. This makes it possible to combine datasets with dissimilar inputs within a single training loop.

**3. Handling Custom Samplers for Imbalanced Datasets:**

When dealing with imbalanced datasets, standard sequential sampling is often suboptimal. We may want to oversample specific datasets to address the class imbalances. To accomplish this, custom samplers can be applied on a per-dataset level before concatenation. I've used weighted random sampling extensively, which I'll briefly illustrate here.

```python
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data import DataLoader
from collections import Counter

class SimpleImbalancedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
data2 = [11, 12, 13]
labels2 = [0, 1, 2]


imbalanced_dataset1 = SimpleImbalancedDataset(data1, labels1)
imbalanced_dataset2 = SimpleImbalancedDataset(data2, labels2)


def create_weighted_sampler(dataset):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for _, label in dataset]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


sampler1 = create_weighted_sampler(imbalanced_dataset1)
sampler2 = create_weighted_sampler(imbalanced_dataset2)

concat_dataset_2 = ConcatenatedDataset([imbalanced_dataset1, imbalanced_dataset2])

loader1 = DataLoader(imbalanced_dataset1, batch_size=2, sampler=sampler1)
loader2 = DataLoader(imbalanced_dataset2, batch_size=1, sampler=sampler2)


print("Sample from dataset 1 using weighted sampler")
for inputs, labels in loader1:
    print(f"Input:{inputs}, label:{labels}")
    break

print("Sample from dataset 2 using weighted sampler")
for inputs, labels in loader2:
    print(f"Input:{inputs}, label:{labels}")
    break


#Note: Cannot directly apply samplers to the combined dataset using dataloaders in this context, since each dataloader expects unique samplers for each base dataset
```

Here, I present a basic example illustrating the application of a weighted random sampler. The `create_weighted_sampler` method calculates weights based on class frequencies within a dataset. I apply these samplers when creating dataloaders for the two datasets. Note that custom samplers are configured at the dataloader level, rather than applying it to the combined `ConcatenatedDataset` object directly in this example. A more advanced combined sampler would be required to properly sample from the concatenated data using dataset-specific sampling methodologies.

**Resource Recommendations:**

For a deeper understanding of PyTorch datasets and data loading, consult the official PyTorch documentation; this provides comprehensive information about the `Dataset` and `DataLoader` classes. Study research papers discussing techniques for handling class imbalance in machine learning, for instance publications focusing on cost-sensitive learning or resampling methods for details on creating effective samplers. Finally, examine community-driven tutorial platforms which offer practical examples of building and training with datasets, which often demonstrate the construction of more advanced concatenation strategies for complex modeling tasks. Careful study of these resources will greatly improve your capacity to manage and combine datasets within a training pipeline.
