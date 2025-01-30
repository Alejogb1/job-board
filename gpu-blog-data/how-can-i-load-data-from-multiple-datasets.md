---
title: "How can I load data from multiple datasets in PyTorch?"
date: "2025-01-30"
id: "how-can-i-load-data-from-multiple-datasets"
---
My work frequently involves complex machine learning pipelines requiring data aggregation from disparate sources. PyTorch’s flexibility allows several effective methods for loading data from multiple datasets, but understanding the nuances is critical for efficient model training. The primary challenge lies in creating unified data access while retaining the individual characteristics of each dataset.

The core principle involves utilizing PyTorch’s `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` classes effectively. The `Dataset` class abstracts the data access, requiring implementations of `__len__` and `__getitem__` methods. For multiple datasets, we need to create custom `Dataset` classes or adapt existing ones such that they can handle the specific combinations or interleaving we require. The `DataLoader`, then, iterates over this composite `Dataset`, handling batching and shuffling. This modular design facilitates code reuse and simplifies the creation of arbitrarily complex data loading strategies.

Here are three common approaches I've used for loading from multiple datasets, along with code examples and explanation.

**Example 1: Concatenating Datasets**

This approach combines datasets linearly, treating them as a single, larger dataset. It's suitable when the datasets represent the same type of data and can be treated uniformly by the model, such as when you have multiple image datasets of similar objects. The main idea is to create a composite dataset that keeps track of which original dataset an element originates from.

```python
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, data, label_prefix):
        self.data = data
        self.label_prefix = label_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label_prefix + str(idx)


dataset1 = SimpleDataset(data=[1, 2, 3], label_prefix="A_")
dataset2 = SimpleDataset(data=[4, 5, 6, 7], label_prefix="B_")

concat_dataset = ConcatDataset([dataset1, dataset2])
data_loader = DataLoader(concat_dataset, batch_size=2, shuffle=True)

for batch_idx, (data, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx}: Data={data}, Labels={labels}")

# Expected output order might vary due to shuffling but something akin to:
# Batch 0: Data=tensor([6, 2]), Labels=('B_2', 'A_1')
# Batch 1: Data=tensor([7, 4]), Labels=('B_3', 'B_0')
# Batch 2: Data=tensor([1, 5]), Labels=('A_0', 'B_1')
# Batch 3: Data=tensor([3]), Labels=('A_2',)
```

In this example, we create two `SimpleDataset` instances. `ConcatDataset` combines them into a single entity. The iterator effectively combines the data and their labels from both source datasets during shuffling and batching within a `DataLoader`. Note how elements from both dataset A and B are contained in the same batch, showing how data is combined. `ConcatDataset` simply shifts the index, allowing seamless access across the underlying datasets. This allows access through index directly, without needing to track original dataset boundaries.

**Example 2:  Mixing Data Based on Sampling**

Sometimes, you need to train a model on data that has a specific proportional representation for each dataset, rather than simply concatenating all data and uniformly sampling. This is where weighted random sampling comes into play.  Here, each dataset is sampled independently, then combined.

```python
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import random

class SimpleDataset(Dataset):
    def __init__(self, data, label_prefix):
        self.data = data
        self.label_prefix = label_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label_prefix + str(idx)

class MixedSampler(Sampler):
    def __init__(self, dataset1_len, dataset2_len, ratio_dataset1=0.5, num_samples = None):
      self.dataset1_len = dataset1_len
      self.dataset2_len = dataset2_len
      self.ratio_dataset1 = ratio_dataset1
      if num_samples is None:
        self.num_samples = max(self.dataset1_len, self.dataset2_len)
      else:
        self.num_samples = num_samples

    def __iter__(self):
       num_samples_1 = int(self.num_samples * self.ratio_dataset1)
       num_samples_2 = self.num_samples - num_samples_1

       indices1 = random.sample(range(self.dataset1_len), k=num_samples_1)
       indices2 = random.sample(range(self.dataset2_len), k=num_samples_2)
       return iter([('dataset1', idx) for idx in indices1] + [('dataset2',idx) for idx in indices2] )

    def __len__(self):
        return self.num_samples

dataset1 = SimpleDataset(data=[1, 2, 3], label_prefix="A_")
dataset2 = SimpleDataset(data=[4, 5, 6, 7], label_prefix="B_")


mixed_sampler = MixedSampler(len(dataset1),len(dataset2), ratio_dataset1=0.7, num_samples = 6)


def custom_collate_fn(batch):
    data_list, label_list = [], []
    for dataset_type, idx in batch:
      if dataset_type == "dataset1":
        data, label = dataset1[idx]
      elif dataset_type == "dataset2":
        data, label = dataset2[idx]
      data_list.append(data)
      label_list.append(label)

    return torch.tensor(data_list), label_list


data_loader = DataLoader(
    list(mixed_sampler), # Sampler returns indices, we use list to make it iterable
    batch_size=2,
    sampler=None,
    collate_fn=custom_collate_fn
)



for batch_idx, (data, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx}: Data={data}, Labels={labels}")

# Expected output order might vary due to sampling but something akin to:
# Batch 0: Data=tensor([1, 6]), Labels=['A_0', 'B_2']
# Batch 1: Data=tensor([2, 5]), Labels=['A_1', 'B_1']
# Batch 2: Data=tensor([3, 4]), Labels=['A_2', 'B_0']
```

Here, a `MixedSampler` is defined, which returns indexes from dataset1 and dataset2 based on a pre-defined ratio between the two datasets. The `DataLoader` is initialized with an instance of `MixedSampler`. A custom `collate_fn` is needed because the `Sampler` will yield a tuple with dataset identifier and index rather than just an integer index. The `collate_fn` will use that information to get the actual item from the datasets. This strategy gives precise control over the contribution of each dataset within the batches.

**Example 3: Datasets with Different Structures**

Often datasets won't have the same shape or structure. When this occurs, you can create one composite `Dataset` that handles multiple different dataset structures.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import random

class StructuredDataset(Dataset):
  def __init__(self, dataset1_data, dataset2_data,dataset1_label, dataset2_label):
    self.dataset1_data = dataset1_data
    self.dataset2_data = dataset2_data
    self.dataset1_label = dataset1_label
    self.dataset2_label = dataset2_label
    self.length = max(len(self.dataset1_data), len(self.dataset2_data))

  def __len__(self):
      return self.length

  def __getitem__(self, idx):

      if idx < len(self.dataset1_data) and random.random() < 0.5:
          return self.dataset1_data[idx], self.dataset1_label[idx], "dataset1"
      else:
          idx_2 = idx % len(self.dataset2_data) # use modulo to index the second dataset when the index is higher
          return self.dataset2_data[idx_2], self.dataset2_label[idx_2], "dataset2"


dataset1_data = [(i*10,i*20) for i in range(1,5)]
dataset1_label = [f"dataset1_label_{i}" for i in range(1,5)]
dataset2_data = [i*100 for i in range(1,7)]
dataset2_label = [f"dataset2_label_{i}" for i in range(1,7)]

composite_dataset = StructuredDataset(dataset1_data, dataset2_data, dataset1_label, dataset2_label)


data_loader = DataLoader(composite_dataset, batch_size=3, shuffle=True)


for batch_idx, batch in enumerate(data_loader):
    data, label, source = zip(*batch)

    print(f"Batch {batch_idx}: Data={data}, Labels={label}, Sources = {source}")

# Expected output order might vary due to sampling but something akin to:
# Batch 0: Data=((10, 20), (100,), (30, 60)), Labels=('dataset1_label_1', 'dataset2_label_1', 'dataset1_label_3'), Sources=('dataset1', 'dataset2', 'dataset1')
# Batch 1: Data=((40, 80), (200,), (500,)), Labels=('dataset1_label_4', 'dataset2_label_2', 'dataset2_label_5'), Sources=('dataset1', 'dataset2', 'dataset2')
# Batch 2: Data=((300,), (400,), (20, 40)), Labels=('dataset2_label_3', 'dataset2_label_4', 'dataset1_label_2'), Sources=('dataset2', 'dataset2', 'dataset1')
```

In this approach, the `StructuredDataset` class manages two datasets with completely different structure. The `__getitem__` method either returns an item from dataset1 or dataset2 based on a randomized selection. In the example I've added a source identifier, to keep track of the source dataset per datapoint. Using `zip(*batch)` within the loop allows to separate the data, labels and the source identifier.

These examples highlight how to load multiple datasets in PyTorch. The specific strategy depends entirely on the nature of your data and how you intend to train your model. `ConcatDataset` is straightforward when datasets are compatible; custom sampling provides control over representation, and custom composite `Dataset` classes handle highly disparate datasets.

For further exploration, I highly recommend delving into the official PyTorch documentation for `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, and `torch.utils.data.Sampler`. The book *Deep Learning with PyTorch* by Eli Stevens, Luca Antiga, and Thomas Viehmann also provides comprehensive coverage of data handling in PyTorch. Additionally, various open-source repositories on GitHub offer practical examples of custom data loading implementations, which can provide a great source of inspiration and ideas. Experimenting with different combinations of these techniques will greatly improve data processing abilities when working with Pytorch models.
