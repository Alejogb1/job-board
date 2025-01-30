---
title: "How can I load many small files into PyTorch without encountering I/O issues?"
date: "2025-01-30"
id: "how-can-i-load-many-small-files-into"
---
Direct disk access can quickly become a bottleneck when training deep learning models on datasets composed of numerous small files. The inherent latency of individual file reads, especially from mechanical drives, can significantly impede training speed. This is a common issue I encountered while working on a project involving satellite imagery where individual image patches were stored as separate PNG files. My initial approach, directly loading each file during training, caused the GPU to wait excessively for data, resulting in severe underutilization. Efficiently addressing this requires minimizing direct file interaction within the primary training loop.

The core problem lies in the sequential nature of traditional file I/O. When using `torch.utils.data.Dataset` to fetch individual files via its `__getitem__` method, each call results in a separate disk access. The operating system's file system must locate and read each file, a process that adds overhead for every image. In cases where training batches are small (e.g., due to memory constraints) or files are small, this overhead becomes disproportionately large. The solution therefore revolves around preprocessing and optimization to reduce individual file reads.

A better approach involves pre-loading the data or creating an efficient data access mechanism. Three key strategies accomplish this: 1) combining data into fewer, larger files, 2) using memory-mapped files, and 3) employing a dataset class with optimized data loading. I have successfully utilized variations of all of these.

**1. Combining Data into Fewer, Larger Files:**

Instead of loading individual images, I first combined them into larger archive files. Formats like HDF5 and TFRecord are ideal for this purpose. These file types allow for the storage of multidimensional data (like images) along with metadata in a structured, efficient manner. By grouping multiple small images into one larger file, the number of disk read operations is significantly reduced.

Consider this example of using HDF5:

```python
import h5py
import os
import numpy as np
from PIL import Image

def create_hdf5_dataset(image_dir, output_file):
    images = []
    filenames = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_dir, filename)
            img = Image.open(file_path)
            img_array = np.array(img)
            images.append(img_array)
            filenames.append(filename)

    images_array = np.stack(images) # Stacks a list of arrays into one array with an additional dimension
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('images', data=images_array)
        hf.create_dataset('filenames', data=np.array(filenames, dtype=np.string_))


image_dir = 'path/to/your/small/images' # Replace with your actual path
output_file = 'combined_dataset.hdf5'
create_hdf5_dataset(image_dir, output_file)

```

**Commentary:** The `create_hdf5_dataset` function iterates through images in the specified directory and stores them within an HDF5 file. The `np.stack` function ensures all images are combined along an additional dimension (typically the number of images dimension), allowing for batch processing in subsequent stages. This pre-processing step combines multiple small reads into one larger write operation. The `filenames` dataset is useful for tracking images contained in the archive if needed during debugging.

The training loop can then load batches directly from the HDF5 file:

```python
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.images = self.hdf5_file['images']
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image) # Apply transformations if needed
        return torch.from_numpy(image).float() # Ensure data is a torch.Tensor

hdf5_file = 'combined_dataset.hdf5'
dataset = HDF5Dataset(hdf5_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example training loop
for batch in dataloader:
    # Process your batch
    pass
```

**Commentary:** `HDF5Dataset` reads data directly from the HDF5 file. Importantly, it loads only the current batch data, avoiding unnecessary loading. Furthermore, the HDF5 file keeps the data in a structure optimized for this kind of access pattern. Note that, I omitted complex transforms for brevity but they are easily added into the `__getitem__`. Using this approach, the time spent reading files is significantly less and the data loading is more efficient.

**2. Using Memory-Mapped Files:**

Another strategy, particularly useful when dealing with very large datasets that don’t fit entirely into memory, is to leverage memory-mapped files. Memory mapping allows treating a file as a large array in memory. The OS manages loading parts of the file on demand, effectively avoiding loading the entire dataset into memory. While not as efficient as keeping the data in RAM entirely, this works especially well when processing parts of the dataset that fit within physical memory.

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def create_memmap_dataset(image_dir, output_file, data_type):
    images = []
    filenames = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_dir, filename)
            img = Image.open(file_path)
            img_array = np.array(img,dtype=data_type)
            images.append(img_array)
            filenames.append(filename)

    all_images = np.stack(images)
    shape = all_images.shape
    memmap = np.memmap(output_file, dtype=data_type, mode='w+', shape=shape)
    memmap[:] = all_images[:]
    del memmap

image_dir = 'path/to/your/small/images'
output_file = 'memmap_dataset.dat'
data_type = 'uint8'  # Ensure this is the correct type
create_memmap_dataset(image_dir, output_file,data_type)

class MemmapDataset(Dataset):
    def __init__(self, memmap_file, shape, data_type, transform=None):
        self.memmap = np.memmap(memmap_file, dtype=data_type, mode='r', shape=shape)
        self.transform = transform
    def __len__(self):
        return self.memmap.shape[0]
    def __getitem__(self, idx):
        image = self.memmap[idx]
        if self.transform:
            image = self.transform(image)
        return torch.from_numpy(image).float()

memmap_file = 'memmap_dataset.dat'
shape = all_images.shape
dataset = MemmapDataset(memmap_file,shape, data_type)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Process your batch
    pass
```

**Commentary:** The `create_memmap_dataset` functions are similar to HDF5 setup, but generates a binary file mapped to memory. The `MemmapDataset` class accesses parts of that file using slicing and avoids loading the entire file into the main memory at once. The underlying OS memory management handles the paging of data from disk as needed. The file is memory mapped in the `__init__` function using `np.memmap()`. This significantly speeds up data access, particularly in cases where the dataset is larger than available RAM. However, it requires more initial setup time.

**3. Optimized Dataset Class and Pre-Loading**

A final approach involves creating a custom dataset class that performs pre-loading or buffering. This approach might not reduce disk reads as much as the previous two, but allows for more control over the data loading pipeline. If the dataset size permits, one can load everything in memory initially and avoid disk reads entirely.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

class PreloadedImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.images = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(image_dir, filename)
                self.image_paths.append(file_path)
        self.load_all_images()
        self.transform = transform

    def load_all_images(self):
        for file_path in self.image_paths:
            img = Image.open(file_path)
            img_array = np.array(img)
            self.images.append(img_array)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return torch.from_numpy(image).float()

image_dir = 'path/to/your/small/images'
dataset = PreloadedImageDataset(image_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Process your batch
    pass

```

**Commentary:** The `PreloadedImageDataset` loads all the data during initialization. Once all images are loaded, it only reads from memory. This approach is most effective when the dataset is manageable in RAM and does not require complex data manipulation. This approach might not be ideal for very large datasets due to the memory implications, but works well for datasets of a moderate size.

**Resource Recommendations:**

For further study, I recommend exploring the official documentation and examples provided by the HDF5 and TensorFlow projects. These resources provide comprehensive details on file formats and their performance characteristics. Additionally, research into Python's `multiprocessing` module and how it interacts with PyTorch's `DataLoader` can improve the performance further by parallelizing image loading. The PyTorch documentation itself contains extensive information on data handling best practices. Consider exploring research papers on methods for optimizing file I/O specifically for deep learning applications. Understanding the trade-offs between these methods can lead to optimal data loading pipelines.
