---
title: "How to do a Random split a PyTorch dataset of type TensorDataset?"
date: "2024-12-14"
id: "how-to-do-a-random-split-a-pytorch-dataset-of-type-tensordataset"
---

alright, so you're looking to split a pytorch `tensordataset` into random subsets, i get it. been there, done that a bunch of times. it's a pretty common task when you're training machine learning models and need to create your training, validation, and testing splits. it's a pretty crucial step in the whole machine learning process.

i remember the first time i had to do this. i was working on this image recognition project. i had this huge dataset loaded up in memory as a `tensordataset`, it was a real pain. initially i just did a naive indexing approach, and everything worked fine on my small toy dataset, but then, with the real data it started to blow up in my face. i just thought that if i just picked the first 80% of the data for training and the next 10% for validation and the rest for testing, it would be enough. oh man, i was naive back then. i didn’t even shuffle the data, i was using a dataset with an implicit sort order (a-z by subject) so i ended up with all the pictures of the letter 'a' in my training set and all the 'z's in my test, it was a complete joke. i quickly realized that i had a big bias and no generalization. it was clear i needed proper random splitting. lets just say i spent a long night fixing that mess.

so, let's talk about some of the ways you can handle this. the core problem is how to get those random index splits, and then use them effectively with your `tensordataset`.

first, and the most straightforward method, is using `torch.randperm` to generate random indices. this is usually what i go for when i am not handling large amounts of data. it works like this:

```python
import torch
from torch.utils.data import TensorDataset, Subset

def random_split_tensordataset(dataset, train_ratio=0.8, val_ratio=0.1):
    dataset_len = len(dataset)
    indices = torch.randperm(dataset_len)

    train_len = int(dataset_len * train_ratio)
    val_len = int(dataset_len * val_ratio)
    test_len = dataset_len - train_len - val_len

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len+val_len]
    test_indices = indices[train_len+val_len:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

# example
data_tensor = torch.randn(100, 10)
label_tensor = torch.randint(0, 2, (100,))
my_dataset = TensorDataset(data_tensor, label_tensor)

train_dataset, val_dataset, test_dataset = random_split_tensordataset(my_dataset)

print(f"train dataset size: {len(train_dataset)}")
print(f"validation dataset size: {len(val_dataset)}")
print(f"test dataset size: {len(test_dataset)}")
```
here’s the breakdown of what’s going on:

1.  `torch.randperm(dataset_len)`: generates a tensor of random permutations of integers from 0 up to the length of your dataset. that's the random order you need.
2.  we calculate the size of each split (`train_len`, `val_len`, `test_len`) based on the given ratios, remember to account for integers after the operation using int casting.
3.  we slice the index tensor to get the indices for each subset.
4.  we create subsets using `torch.utils.data.subset`, this creates a view of the original dataset using just those indices, which avoids copying the whole dataset and saves memory and time.

this approach is fine for most cases, especially if your dataset fits comfortably in memory, and it is pretty fast. if you have a really large dataset you might have to use a different strategy, but more on that later.

now, let's say you want to enforce reproducibility. well, you can seed `torch`'s random number generator, like this:

```python
import torch
from torch.utils.data import TensorDataset, Subset

def random_split_tensordataset_seed(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    torch.manual_seed(seed)
    dataset_len = len(dataset)
    indices = torch.randperm(dataset_len)

    train_len = int(dataset_len * train_ratio)
    val_len = int(dataset_len * val_ratio)
    test_len = dataset_len - train_len - val_len

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len+val_len]
    test_indices = indices[train_len+val_len:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

# example
data_tensor = torch.randn(100, 10)
label_tensor = torch.randint(0, 2, (100,))
my_dataset = TensorDataset(data_tensor, label_tensor)

train_dataset, val_dataset, test_dataset = random_split_tensordataset_seed(my_dataset)

print(f"train dataset size: {len(train_dataset)}")
print(f"validation dataset size: {len(val_dataset)}")
print(f"test dataset size: {len(test_dataset)}")

train_dataset2, val_dataset2, test_dataset2 = random_split_tensordataset_seed(my_dataset)

print(f"train dataset2 size: {len(train_dataset2)}")
print(f"validation dataset2 size: {len(val_dataset2)}")
print(f"test dataset2 size: {len(test_dataset2)}")
```

the key change here is that i added `torch.manual_seed(seed)` at the beginning of the function, and then if you call the function twice with the same dataset and same seed, you get the exact same split. this is essential if you want to track the progress on a specific split, or to debug something you have previously trained. if you don't set this, your splits will be different every time which can sometimes mess you up. it's kind of a no-brainer, but some people forget to do it.

now lets say your dataset is too big to fit into memory, meaning that you cannot load it all at once or to build the `tensordataset` directly. then you probably already have a dataset object, or some data stream that allows you to read individual samples or batches of data, then you can't use `torch.utils.data.subset` and `torch.randperm` directly to split the `tensordataset` as you do not have it fully loaded at any single point in time.

you have a few strategies here, one that i often use, is to store indices of the split in files, and then use a custom pytorch dataset, you have to implement `__getitem__` which does the loading of the individual samples from the external stream, or directly from disk. here’s an example on how you would handle that:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, indices, transform=None):
        self.data_dir = data_dir
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        data = np.load(os.path.join(self.data_dir, f'data_{file_idx}.npy'))
        label = np.load(os.path.join(self.data_dir, f'label_{file_idx}.npy'))
        
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).long()
        
        if self.transform:
            data = self.transform(data)
            
        return data, label

def create_dummy_files(data_dir, num_files=100):
  os.makedirs(data_dir, exist_ok=True)
  for i in range(num_files):
    np.save(os.path.join(data_dir, f'data_{i}.npy'), np.random.rand(10, 10))
    np.save(os.path.join(data_dir, f'label_{i}.npy'), np.random.randint(0, 2))

def create_splits(data_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    num_files = len(os.listdir(data_dir)) // 2
    indices = np.random.permutation(num_files)

    train_len = int(num_files * train_ratio)
    val_len = int(num_files * val_ratio)

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len+val_len]
    test_indices = indices[train_len+val_len:]

    return train_indices, val_indices, test_indices

# --- Example ---
data_dir = "dummy_data"
create_dummy_files(data_dir)

train_indices, val_indices, test_indices = create_splits(data_dir)

train_dataset = CustomDataset(data_dir, train_indices)
val_dataset = CustomDataset(data_dir, val_indices)
test_dataset = CustomDataset(data_dir, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"train dataset size: {len(train_dataset)}")
print(f"validation dataset size: {len(val_dataset)}")
print(f"test dataset size: {len(test_dataset)}")

```
here is what is going on:

1.  `customdataset`: this is our dataset class. it has a `__getitem__` method which loads data from the file based on the provided index, from an external source (like files on disk). the example here uses `numpy` files, but you could easily use hdf5 or anything else.
2.  `create_dummy_files`: this creates some dummy data in files to simulate a dataset too big to load in memory.
3.  `create_splits`: this creates indices and stores them using `numpy` files, but you could easily change it to use python's `pickle` or any other format, this is usually what i do if i am handling large amounts of data that do not fit into memory. it has the added benefit that you can track the splits that you have used between training runs.
4. we create the three datasets, and then we use `dataloader` to load the data in batches to do the actual training.

when dealing with large datasets this strategy is what you need. the key takeaway is that you don't need to load the whole dataset to do the split, you just need to track the indices.

in terms of resources, the pytorch documentation is always the place to start, it's pretty good, but it may not have all the practical details you need. for a deeper understanding of data loading and data pipelines, i would recommend looking into 'python data science handbook' by jake vanderplas, which covers numpy and how to handle large datasets. also the deep learning book by goodfellow, bengio and courville goes into the details of data handling in machine learning.

so, there you have it. a few ways to do random splits with `tensordataset`. each has its place, depending on your project's needs. choose the method that fits your situation and happy coding. let me know if you have other questions.
