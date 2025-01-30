---
title: "Why is PyTorch DataLoader with np.load slow on an SSD?"
date: "2025-01-30"
id: "why-is-pytorch-dataloader-with-npload-slow-on"
---
The observed performance bottleneck when using `torch.utils.data.DataLoader` with `np.load` on an SSD stems from a combination of factors, primarily related to I/O latency and process management, rather than the SSD's inherent throughput capability. Specifically, the default behavior of `np.load`, coupled with PyTorch's multiprocessing approach, often creates contention and inefficient resource utilization.

My experience working on a large-scale medical imaging project highlighted this issue. Initially, we relied on a straightforward setup: pre-saved NumPy arrays on an SSD and a custom dataset class that utilized `np.load` within the `__getitem__` method, coupled with a PyTorch DataLoader. We noticed surprisingly sluggish loading times, despite the SSD’s advertised speed. This prompted an investigation that revealed several interconnected factors at play.

The core problem lies in the way `np.load` interacts with Python's Global Interpreter Lock (GIL). `np.load` is largely a CPU-bound operation, involving reading binary data and reconstructing NumPy arrays. When multiple worker processes, as enabled by `DataLoader`’s `num_workers` parameter, call `np.load` concurrently, they frequently compete for the GIL. Even though data is being read from a fast SSD, only one process can truly execute NumPy operations at a time within the same Python interpreter. This effectively serializes the loading process, despite utilizing multiple workers, and negates the benefits of multi-processing. The SSD, while capable of delivering high read speeds, spends most of the time waiting for the GIL to release. The situation is further exacerbated when dataset files are large, leading to longer loading times for each worker.

Additionally, the default implementation of `DataLoader` involves spawning new Python processes. Each new process incurs an overhead of copying the entire dataset object and the required libraries in memory. If your dataset is particularly complex, this process creation overhead, which involves a `fork` operation in Unix-like systems, can be non-trivial and contribute significantly to the overall loading time, especially when the data loading time itself is already suboptimal due to the GIL. The overhead also includes initializing `torch` for each worker process.

Finally, disk access patterns play a role. While SSDs are very fast for sequential reads, they are less optimal for a large number of small, random reads. If the file layout on the disk is not contiguous, or if there are other processes simultaneously accessing the SSD, this further exacerbates the loading bottleneck. `np.load` directly reads from the disk, and if the files are fragmented, reading each small file in parallel from multiple workers results in numerous scattered requests which can slow down reads.

Let’s illustrate this with a few examples.

**Example 1: Naive Implementation**

This is a typical but inefficient initial approach. Assume the existence of a folder named `data_folder` with a set of NumPy files (`data_0.npy`, `data_1.npy`, etc.).

```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.filenames[idx])
        data = np.load(filepath)
        return torch.from_numpy(data) # Convert to Torch Tensor


if __name__ == '__main__':
    # Assumed folder structure of 'data_folder' is already present.
    dataset = MyDataset("data_folder")
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

    for batch in dataloader:
       # Processing of batch data here
       pass
```

In this example, every time a data element is requested through the `__getitem__` function, `np.load` is called within a worker process. This leads to GIL contention as previously described, resulting in suboptimal performance. Multiple worker processes simultaneously try to load data, contending for access to `np.load` which is CPU-bound and GIL-limited.

**Example 2: Using Shared Memory**

One way to alleviate the GIL bottleneck is to load the data once in the main process and transfer it to worker processes through shared memory. This requires an additional layer of management of the shared data array.

```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing
import shared_memory_manager as shm_manager  # Assume this is available


class MyDatasetSharedMemory(Dataset):
    def __init__(self, data_dir, shared_memory_name):
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.shared_memory_name = shared_memory_name
        self.shm = shm_manager.attach_shm(shared_memory_name)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return torch.from_numpy(self.shm[idx]) # Access data through shared memory


def load_data_to_shm(data_dir):
        filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        # Get the shape of first loaded array as data is assumed to be same shape
        dummy = np.load(os.path.join(data_dir,filenames[0]))
        shm_name, shm = shm_manager.create_shm(len(filenames), dummy.shape, dtype = dummy.dtype)
        for index, filename in enumerate(filenames):
                shm[index] = np.load(os.path.join(data_dir, filename))
        return shm_name


if __name__ == '__main__':
    # Loading all data into a shared memory segment in the main process
    shared_memory_name = load_data_to_shm("data_folder")
    dataset = MyDatasetSharedMemory("data_folder", shared_memory_name)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

    for batch in dataloader:
        # Process the batch of data
        pass

    # Clean up allocated shared memory
    shm_manager.destroy_shm(shared_memory_name)
```

Here the data is loaded into shared memory and thus the `__getitem__` method directly returns the data from shared memory without calling `np.load` again in worker processes which addresses the GIL issue. The `shared_memory_manager` class will manage creation, access, and cleanup of the shared memory regions. This example avoids the GIL-related bottleneck, but introduces a layer of complexity by having to manage memory sharing.

**Example 3: Pre-loading and Indexing**

Another approach is to pre-load all the data into a structure in the main process, typically a list or a dictionary, and have worker processes access this structure using indices or keys, but without requiring shared memory management.

```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class MyDatasetPreloaded(Dataset):
    def __init__(self, preloaded_data):
        self.data = preloaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

def preload_data(data_dir):
    data = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    for filename in filenames:
         data.append(np.load(os.path.join(data_dir, filename)))
    return data

if __name__ == '__main__':
    preloaded_data = preload_data("data_folder")
    dataset = MyDatasetPreloaded(preloaded_data)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

    for batch in dataloader:
        # Process batch data
        pass
```

In this case, all the data is preloaded into a list in the main process’s memory. The worker processes subsequently access this list using their respective indices without needing to call `np.load`. This reduces the processing time, particularly when data is not very large and fits comfortably within system RAM. If the data is extremely large, this approach could run into RAM issues and is not recommended.

For optimal performance, when dealing with data that is larger than available RAM, employing a combination of techniques is often useful. Pre-processing data into smaller, more manageable chunks, potentially in a format that enables efficient sequential reads (e.g., custom binary format, HDF5) would significantly reduce I/O latencies. Further optimization can be achieved by using memory mapping (`numpy.memmap`) to access the files on disk without loading them into RAM completely.

Based on my experiences and the specific challenges with `np.load` within PyTorch’s DataLoader when using SSDs, I recommend exploring the following resources to delve deeper into these concepts.
*  Discussions on Python's GIL and its impact on multi-processing performance are essential to understand.
* Detailed studies on shared memory implementations in Python and their associated performance characteristics.
*  Reviews of best practices in I/O management for high-performance data loading scenarios.
*  Comparative studies of different file formats (e.g., HDF5, custom binary formats) in the context of deep learning data loading.
These resources will provide a more comprehensive understanding of the nuances involved and help in implementing robust and efficient data loading pipelines.
