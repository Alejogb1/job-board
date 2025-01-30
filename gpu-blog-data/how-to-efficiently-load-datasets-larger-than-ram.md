---
title: "How to efficiently load datasets larger than RAM using PyTorch DataLoader?"
date: "2025-01-30"
id: "how-to-efficiently-load-datasets-larger-than-ram"
---
Handling datasets exceeding available RAM during PyTorch model training presents a significant challenge. My experience working on large-scale genomic data analysis, specifically whole-genome sequencing projects involving terabyte-sized datasets, has underscored the critical need for efficient data loading strategies.  The core issue stems from the inability to load the entire dataset into memory, necessitating techniques that process data in smaller, manageable chunks.  This necessitates a deep understanding of PyTorch's `DataLoader` and its capabilities beyond simple batching.

The key to efficiently loading datasets larger than RAM lies in leveraging the `DataLoader`'s functionalities in conjunction with custom data loading mechanisms and appropriate memory management practices.  Directly loading the entire dataset into memory is simply infeasible; instead, we need to implement a strategy that streams data from disk to memory as needed, using appropriately sized batches to optimize processing efficiency and minimize memory footprint.

**1. Clear Explanation:**

The standard `DataLoader` readily handles smaller datasets. However, for large datasets, we need to customize its `dataset` argument.  Instead of providing a dataset object holding the entire dataset in memory, we provide a custom dataset class that implements the `__getitem__` and `__len__` methods.  These methods become responsible for reading and returning data chunks from disk (or other persistent storage) upon request.  Critically, this process needs to be optimized to minimize disk I/O, leveraging techniques like memory mapping and efficient data serialization formats.  Further optimization can be achieved by using multiprocessing to parallelize the data loading process.

The choice of data format is crucial. Formats like HDF5 or Parquet, designed for efficient storage and retrieval of large datasets, are highly advantageous. These formats support direct access to individual data entries without needing to read the entire file, significantly reducing the I/O overhead.  Using standard CSV or text files is strongly discouraged for datasets of this magnitude due to their inherent inefficiency.

Furthermore, pin-memory usage (`pin_memory=True`) within the `DataLoader` should always be considered. This feature transfers data directly to the GPU memory, bypassing the CPU, significantly speeding up training when using a GPU.  However, it is crucial to balance the memory usage of pin-memory with the actual amount of RAM available.

**2. Code Examples with Commentary:**

**Example 1: Using HDF5 with a Custom Dataset:**

```python
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, data_key, target_key):
        self.file = h5py.File(hdf5_path, 'r')
        self.data = self.file[data_key]
        self.targets = self.file[target_key]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        target = torch.tensor(self.targets[idx])
        return data, target

    def __del__(self):
        self.file.close()

hdf5_path = 'my_large_dataset.hdf5'  # Path to your HDF5 file
dataset = HDF5Dataset(hdf5_path, 'data', 'targets')
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=4)

# Training loop using the dataloader
for epoch in range(num_epochs):
    for data, target in dataloader:
        # Your training logic here
        ...
```

This example demonstrates using HDF5. The `HDF5Dataset` class handles loading data from the HDF5 file in chunks, ensuring that only the necessary data is loaded into memory at any given time. The `__del__` method explicitly closes the HDF5 file.  `num_workers` utilizes multiprocessing to accelerate data loading.


**Example 2:  Memory Mapping with NumPy:**

```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MemoryMappedDataset(Dataset):
    def __init__(self, data_path, target_path, batch_size):
        self.data = np.memmap(data_path, dtype='float32', mode='r')
        self.targets = np.memmap(target_path, dtype='int64', mode='r')
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        data = torch.from_numpy(self.data[start:end])
        target = torch.from_numpy(self.targets[start:end])
        return data, target

data_path = 'my_data.dat' # Pre-processed data in binary format
target_path = 'my_targets.dat' # Pre-processed targets in binary format
dataset = MemoryMappedDataset(data_path, target_path, 32)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4) #batch_size=1 as dataset already handles batching

#Training loop
for epoch in range(num_epochs):
    for data, target in dataloader:
        #Your training logic
        ...
```

This utilizes NumPy's `memmap` for memory-mapped file access. The data is not loaded entirely into memory; instead, it's accessed directly from the disk, significantly reducing memory consumption.  Note that batching is handled within `__getitem__` here.


**Example 3:  Parquet with Dask:**

```python
import dask.dataframe as dd
import torch
from torch.utils.data import Dataset, DataLoader

# Assuming parquet files are already created. This requires preprocessing beforehand
ddf = dd.read_parquet('my_large_dataset.parquet')

class ParquetDataset(Dataset):
    def __init__(self, ddf, batch_size):
        self.ddf = ddf
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ddf) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch = self.ddf.loc[start:end].compute() #compute brings data into memory in chunks
        data = torch.tensor(batch['data'].values)
        target = torch.tensor(batch['targets'].values)
        return data, target


dataset = ParquetDataset(ddf, 32)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4) #batch_size=1 because of chunking inside ParquetDataset

#Training Loop
for epoch in range(num_epochs):
    for data, target in dataloader:
        #Your training logic
        ...
```

This illustrates using the Dask library with Parquet. Dask enables parallel and out-of-core computation on large datasets. It efficiently handles the data loading and processing, distributing the workload across multiple cores and minimizing memory usage.


**3. Resource Recommendations:**

For further in-depth understanding, consult the official PyTorch documentation, focusing on the `DataLoader` class and its parameters.  Explore documentation related to HDF5, Parquet, and NumPy's `memmap` functionality for detailed explanations of their respective capabilities.  Familiarity with parallel programming concepts and libraries like multiprocessing or Dask will significantly aid in optimizing your data loading pipelines.  Consider reviewing publications and tutorials focusing on large-scale machine learning, as they offer insights into practical approaches for handling large datasets within limited memory resources.  Exploring advanced techniques like data sharding and distributed training might prove valuable for extremely large datasets exceeding the capacity of even distributed memory systems.
