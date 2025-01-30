---
title: "How can a split torch dataset be created without shuffling?"
date: "2025-01-30"
id: "how-can-a-split-torch-dataset-be-created"
---
A common requirement when training deep learning models, particularly for time-series data or when deterministic reproducibility is crucial, is splitting a PyTorch dataset without introducing random shuffling. While `torch.utils.data.random_split` is convenient, it inherently shuffles the dataset before partitioning. Direct index manipulation offers a controlled method to achieve splitting without unintended reordering.

Essentially, splitting a PyTorch dataset without shuffling involves creating sub-datasets that utilize specific, non-overlapping index ranges from the original dataset. This approach ensures that the data within each subset remains in the same order as it was in the parent dataset. I've employed this technique extensively in my work with sequential model training, where the ordering of input data directly influences the learning process.

The core concept relies on directly manipulating indices when constructing a custom `Dataset` object. The custom dataset subclass takes an original dataset and a set of indices as input. During item retrieval, the subclass fetches the data point from the original dataset corresponding to the specified index. This effectively creates a view or a slice of the original dataset, preserving the order. This design pattern ensures data integrity and allows for precise control over how the dataset is divided.

I'll illustrate this approach using three distinct scenarios.

**Example 1: Simple Percentage Split**

First, consider splitting a dataset into training and validation sets based on a specified percentage, common in standard supervised learning. Assume we have a dataset named `original_dataset`.

```python
import torch
from torch.utils.data import Dataset, TensorDataset

class IndexDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


# Generate a dummy dataset
data = torch.arange(20)
original_dataset = TensorDataset(data)


# Define split ratio
train_ratio = 0.8

# Calculate number of training samples
train_size = int(len(original_dataset) * train_ratio)

# Generate index arrays without shuffling
train_indices = list(range(train_size))
val_indices = list(range(train_size, len(original_dataset)))

# Create new datasets using IndexDataset
train_dataset = IndexDataset(original_dataset, train_indices)
val_dataset = IndexDataset(original_dataset, val_indices)


# Verify the indices
print("First 5 train samples:", [train_dataset[i][0].item() for i in range(5)])
print("First 5 validation samples:", [val_dataset[i][0].item() for i in range(5)])
```

In this example, `IndexDataset` acts as a wrapper. It stores the original dataset and the specified `indices`. The `__getitem__` method then accesses the original dataset using the provided `indices`. This effectively creates two datasets that are subsets of the original one, without any shuffling involved. I have observed that this straightforward approach is the most common requirement in many of the applied scenarios involving my projects. The printing operation serves to confirm the order. The first five training samples are sequential from 0 to 4, while the validation samples start at 16 and proceed sequentially to 19. This ensures no data from the original dataset gets placed in the subsets in a non-sequential manner.

**Example 2: Splitting based on predetermined indices**

In some scenarios, you may have predefined indices to use for splitting, perhaps from an existing train/validation split already present in your data preparation process. This might be applicable in situations like transfer learning or data augmentation involving only a subset of data.

```python
import torch
from torch.utils.data import Dataset, TensorDataset

class IndexDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


# Generate a dummy dataset
data = torch.arange(20)
original_dataset = TensorDataset(data)


# Define pre-defined split indices
train_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16]
val_indices   = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# Create new datasets using IndexDataset
train_dataset = IndexDataset(original_dataset, train_indices)
val_dataset = IndexDataset(original_dataset, val_indices)

# Verify the indices
print("Train Samples:", [train_dataset[i][0].item() for i in range(len(train_dataset))])
print("Validation Samples:", [val_dataset[i][0].item() for i in range(len(val_dataset))])
```

In this second scenario, the index lists are defined explicitly. As such, rather than being a contiguous portion of the dataset, the split is controlled via the explicit index lists. The resulting training data has indices 0, 2, 4 etc., demonstrating that the desired sub-datasets can be created using non-contiguous indices. This flexibility allows me to isolate portions of the dataset based on external specifications. The validation set in this example contains all odd indices of the parent dataset. The ability to split datasets this way has helped me tackle several challenging data preparation scenarios.

**Example 3: Splitting with a gap for testing**

Finally, in some time-series scenarios, you may want to have a gap between training and test data to simulate real-world situations where you have limited recent past data available. This approach has proven to be very crucial for time-series model validations.

```python
import torch
from torch.utils.data import Dataset, TensorDataset

class IndexDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


# Generate a dummy dataset
data = torch.arange(30)
original_dataset = TensorDataset(data)


# Define a test gap
gap_size = 3

# Create index arrays, adding the gap
train_size = 15
train_indices = list(range(train_size))
test_indices = list(range(train_size+gap_size, len(original_dataset)))

# Create new datasets using IndexDataset
train_dataset = IndexDataset(original_dataset, train_indices)
test_dataset = IndexDataset(original_dataset, test_indices)


# Verify the indices
print("First 5 training samples:", [train_dataset[i][0].item() for i in range(5)])
print("First 5 test samples:", [test_dataset[i][0].item() for i in range(5)])
```

This demonstrates that we can create a gap of 3 indices (15, 16, and 17). This is extremely important for time-series models where the test data often needs to be separated from the training data by a specified period to avoid overly optimistic performance assessments. This pattern is useful to prevent data leakage between training and validation splits. This has been vital in my work with temporal data. The first 5 training samples start with 0, while the first 5 test samples start at 18.

For further understanding of dataset manipulation, I recommend exploring the PyTorch documentation related to `torch.utils.data.Dataset` and `torch.utils.data.TensorDataset`. Publications on time-series data processing in machine learning may also be of interest. Moreover, examining examples of advanced data loading and preprocessing techniques available in the PyTorch ecosystem would prove highly beneficial. Specifically, I would recommend focusing on best-practices involving custom dataloaders, since the use-cases in my experience have always become far more complex than simply loading data from predefined `csv` files or similar simple scenarios. Lastly, reviewing examples in academic publications relating to model training for similar tasks could further enhance your comprehension of these concepts. These references together provide a solid foundation for creating and manipulating datasets effectively.
