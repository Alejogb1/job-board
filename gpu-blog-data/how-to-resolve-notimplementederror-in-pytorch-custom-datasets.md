---
title: "How to resolve 'NotImplementedError' in PyTorch custom datasets?"
date: "2025-01-30"
id: "how-to-resolve-notimplementederror-in-pytorch-custom-datasets"
---
A `NotImplementedError` within a PyTorch custom dataset typically arises when essential abstract methods inherited from the `torch.utils.data.Dataset` class are not defined within the custom dataset implementation. Specifically, these methods are `__len__` which dictates the size of the dataset, and `__getitem__` which retrieves a specific data point and its corresponding label. Failure to provide concrete implementations for these results in the base class throwing this exception when PyTorch attempts to utilize your custom dataset.

My experience has consistently shown that developers often encounter this issue when initially creating custom datasets, particularly when porting code from other frameworks or using boilerplate templates without fully understanding the underlying mechanisms. The `Dataset` class is an abstract base class, defining the interface but requiring concrete implementation for practical use.

**Explanation**

PyTorch's `Dataset` class offers a structured method for handling data in machine learning workflows. It decouples data storage and access from the training loop, making the code more modular and maintainable. When you create a custom dataset by inheriting from `torch.utils.data.Dataset`, you inherit the structure but must implement the core functionality. The two pivotal methods that must be implemented are:

*   `__len__(self)`: This method should return an integer representing the total number of samples (data points) within the dataset. This value is crucial for the PyTorch DataLoader which relies on this to iterate correctly through the entire dataset, for tasks such as shuffling, batching, and setting epoch lengths.

*   `__getitem__(self, idx)`: This method must implement data retrieval logic based on a given integer index (`idx`). It must return a data point, potentially in conjunction with a target or label. The return type usually comprises a PyTorch tensor or a tuple of tensors and/or auxiliary information, but is not strictly limited to it. The index passed to this method is always an integer and falls within the range defined by `__len__`.

The `NotImplementedError` emerges when either of these methods is missing or if it contains only a pass statement, as it indicates to PyTorch that the necessary logic is missing from the user-defined dataset. Correctly implementing these methods resolves the error and enables PyTorch to use the dataset correctly during training, validation, and inference.

**Code Examples**

Here, I will demonstrate three distinct dataset implementations that rectify the error, showcasing how these fundamental abstract methods are brought to life in different scenarios:

**Example 1: Simple Numerical Dataset**

This dataset example involves generating a set of numerical data and corresponding labels, a simple yet fundamental case:

```python
import torch
from torch.utils.data import Dataset

class SimpleNumberDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 5) # 5 features for each sample
        self.labels = torch.randint(0, 2, (num_samples,)) # Binary labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example usage
dataset = SimpleNumberDataset(num_samples=200)
first_item_data, first_item_label = dataset[0]
print("Data:", first_item_data.shape, "Label:", first_item_label)
print(f"Dataset size: {len(dataset)}")

```

*   **Commentary:** This example demonstrates a straightforward dataset where the data and labels are created and stored within the dataset class. The `__len__` method returns the predefined number of samples, and `__getitem__` retrieves data and its label given the sample index, as a PyTorch tensor pair. Using it with a DataLoader will enable batching of these pairs.

**Example 2: Loading Data from Disk**

This showcases a dataset where data is loaded from file paths stored in a list. This is closer to real-world scenarios:

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class DiskBasedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')] # List .npy files
        self.labels = {f: int(f.split('_')[0]) for f in self.file_list} # Labels are part of filename

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path)  # Load .npy data
        label = self.labels[file_name] # Get Label from filename
        return torch.from_numpy(data).float(), torch.tensor(label).long()

# Example Usage (Dummy data)
if not os.path.exists("dummy_data"):
    os.makedirs("dummy_data")
    for i in range(5):
        data = np.random.rand(10, 10)
        np.save(f"dummy_data/{i}_sample.npy", data)


dataset = DiskBasedDataset(data_dir='dummy_data')
sample_data, sample_label = dataset[2]
print(f"Loaded data shape: {sample_data.shape}, label: {sample_label}")
print(f"Total files: {len(dataset)}")
```

*   **Commentary:** In this case, the dataset loads data stored in `.npy` format files from a directory. `__len__` returns the number of files in the directory, and `__getitem__` reads the data from the file, extracts labels from filenames, then converts the data and label into PyTorch tensors. Note the use of torch.from_numpy and .float() to ensure compatible data types in PyTorch.

**Example 3: A Complex Dataset with Image Transformation**

This dataset shows how transformation can be incorporated before returning data, which is a common requirement for image data:

```python
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))] # Images
        self.transform = transform if transform else transforms.ToTensor()
        self.labels = {f: int(f.split('_')[0]) for f in self.image_list}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB') # Load Image and ensure RGB
        if self.transform:
            image = self.transform(image)  # Apply Transformation
        label = self.labels[image_name]
        return image, torch.tensor(label).long()

# Example Usage (Dummy Data)
if not os.path.exists("dummy_images"):
    os.makedirs("dummy_images")
    for i in range(3):
        img = Image.new('RGB', (64, 64), color = (i*50, 100, i*20))
        img.save(f"dummy_images/{i}_sample.png")

transform_composition = transforms.Compose([
     transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize Images
])

dataset = ImageDataset(image_dir='dummy_images', transform = transform_composition)
sample_image, sample_label = dataset[1]
print(f"Transformed Image Shape: {sample_image.shape}, Label: {sample_label}")
print(f"Total Images: {len(dataset)}")

```

*   **Commentary:** This dataset is tailored for image data.  It uses `PIL` to load images and employs the `torchvision.transforms` library to apply transformations, including resizing, converting to tensor, and normalization. This demonstrates how custom preprocessing steps can be seamlessly integrated into a custom dataset. Both `__len__` and `__getitem__` are still essential in this complex scenario.

**Resource Recommendations**

For developers seeking to deepen their understanding of PyTorch datasets and related concepts, I recommend consulting:

*   The official PyTorch documentation, which is comprehensive and provides clear explanations of all framework components. The tutorials related to data loading and working with custom datasets are particularly helpful.
*   Books focused on deep learning with PyTorch. These publications frequently offer in-depth explanations and varied examples, which build intuition and practical skills.
*   Online courses about deep learning, typically accompanied by practical coding assignments, which provide hands-on experience.

By carefully implementing the `__len__` and `__getitem__` methods, developers can effectively create custom datasets tailored to specific needs, while avoiding the `NotImplementedError` when dealing with PyTorch. Focusing on these two methods is key when using custom datasets in conjunction with the PyTorch DataLoader and training loop.
