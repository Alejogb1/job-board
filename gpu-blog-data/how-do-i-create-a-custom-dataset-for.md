---
title: "How do I create a custom dataset for PyTorch?"
date: "2025-01-30"
id: "how-do-i-create-a-custom-dataset-for"
---
Creating custom datasets is a cornerstone of effective deep learning with PyTorch. I’ve built numerous models ranging from image classification to time series forecasting, and I've consistently found that understanding the underlying mechanisms of dataset creation is crucial for flexibility and efficiency. PyTorch provides an abstract `Dataset` class, and inheriting from it allows you to tailor data loading to the specifics of your project. This involves defining how your data is stored and accessed. Fundamentally, you're responsible for two key aspects: `__len__`, which provides the size of your dataset, and `__getitem__`, which fetches a single data point based on its index.

A basic dataset implementation must override these methods within a class inheriting from `torch.utils.data.Dataset`. This class will become the interface between your raw data and the PyTorch data loading pipeline, enabling features like batching, shuffling, and multi-processing. Let's explore an example to clarify.

**Code Example 1: A Simple In-Memory Dataset**

This example demonstrates a dataset where all data is held directly in memory, suitable for small datasets or quick prototyping.

```python
import torch
from torch.utils.data import Dataset

class InMemoryDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example usage
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
labels = [0, 1, 0]

dataset = InMemoryDataset(data, labels)
print(f"Dataset size: {len(dataset)}") # Output: Dataset size: 3
sample, label = dataset[1]
print(f"Sample at index 1: {sample}, Label: {label}") # Output: Sample at index 1: tensor([4., 5., 6.]), Label: 1
```

In this example, `InMemoryDataset` initializes itself with lists that are converted to PyTorch tensors during construction. The `__len__` method simply returns the length of the data tensor. The `__getitem__` method returns the data and label at the requested index as a tuple. While convenient, holding an entire dataset in memory can be prohibitive for large-scale applications. Therefore, often you would rather stream data from disk.

**Code Example 2: A File-Based Dataset**

This example shows how to read data from files on disk, which scales much better for large datasets. Assume you have data files `data_0.txt`, `data_1.txt` etc., each containing comma-separated numbers. Similarly, labels are stored in a file with one label per line.

```python
import torch
from torch.utils.data import Dataset
import os

class FileBasedDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('data_')])
        self.labels = self._load_labels(label_file)

    def _load_labels(self, label_file):
        with open(label_file, 'r') as f:
           return [int(line.strip()) for line in f]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        with open(file_path, 'r') as f:
            data = torch.tensor([float(x) for x in f.read().strip().split(',')], dtype=torch.float32)
        label = self.labels[idx]
        return data, label

# Example usage (assuming data files are created elsewhere)
data_dir = 'data_files'  # Assume a 'data_files' directory with data_0.txt, data_1.txt etc.
label_file = 'labels.txt'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

with open(os.path.join(data_dir, 'data_0.txt'), 'w') as f:
    f.write("1,2,3")
with open(os.path.join(data_dir, 'data_1.txt'), 'w') as f:
    f.write("4,5,6")
with open(os.path.join(data_dir, 'data_2.txt'), 'w') as f:
    f.write("7,8,9")
with open(label_file, 'w') as f:
    f.write("0\n1\n0")

dataset = FileBasedDataset(data_dir, label_file)
print(f"Dataset size: {len(dataset)}") # Output: Dataset size: 3
sample, label = dataset[1]
print(f"Sample at index 1: {sample}, Label: {label}") #Output: Sample at index 1: tensor([4., 5., 6.]), Label: 1
```

Here, the `FileBasedDataset` uses the directory and label file locations. The `_load_labels` method reads each line from the label file and converts it to an integer, while in `__getitem__`, the data is loaded only when requested, and converted from the string in the file to a PyTorch tensor. The data files are assumed to contain comma separated numerical strings, which are loaded and cast to float. This pattern is a good approach for working with large data collections that won’t fit into RAM.

**Code Example 3: Applying Transformations**

A vital part of data handling is often applying transforms. `torchvision.transforms` provides many useful transformations, and we can apply these on-the-fly within `__getitem__`. This example incorporates a transform.

```python
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.labels = self._load_labels(label_file)
        self.transform = transform

    def _load_labels(self, label_file):
         with open(label_file, 'r') as f:
             return [int(line.strip()) for line in f]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB') # Load image as PIL image
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

# Example usage (assuming images are created elsewhere)
image_dir = 'image_files' # Assume 'image_files' directory with images.
label_file = 'image_labels.txt'

if not os.path.exists(image_dir):
    os.makedirs(image_dir)
# Create dummy images for demonstration. Note, in a real scenario these would be actual image files
dummy_image_data = np.zeros((10, 10, 3), dtype=np.uint8)
img = Image.fromarray(dummy_image_data)
img.save(os.path.join(image_dir, 'image_0.png'))

dummy_image_data = np.ones((10, 10, 3), dtype=np.uint8) * 255
img = Image.fromarray(dummy_image_data)
img.save(os.path.join(image_dir, 'image_1.png'))

with open(label_file, 'w') as f:
    f.write("0\n1")


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageDataset(image_dir, label_file, transform=transform)
print(f"Dataset size: {len(dataset)}") # Output: Dataset size: 2
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}, Label: {label}") # Output: Sample shape: torch.Size([3, 32, 32]), Label: 0
```

The `ImageDataset` loads image files using PIL, and the provided transform is applied before returning the image tensor and the associated label. Transformations include resizing, converting the PIL image to a PyTorch tensor, and normalizing the tensor’s values. This flexibility allows for preprocessing that's integral to the training process.

These examples demonstrate the core concepts required when implementing a custom PyTorch dataset. I have found that while this provides a lot of flexibility, it is essential to carefully consider the trade-offs of holding data in memory versus reading it from a file. Data transformations and augmentation should also be integrated into the dataset, and proper error handling to prevent issues.

For further exploration, I recommend reviewing the official PyTorch documentation on `torch.utils.data.Dataset`, as well as exploring tutorials focusing on image data processing with `torchvision.datasets`. Texts that provide a detailed introduction to deep learning, especially those covering PyTorch, are also very helpful. Lastly, examining open-source repositories where sophisticated custom datasets are employed will offer further insights into more advanced use cases.
