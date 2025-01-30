---
title: "How can I use `random_split()` in PyTorch on a dataset located across multiple subfolders?"
date: "2025-01-30"
id: "how-can-i-use-randomsplit-in-pytorch-on"
---
PyTorch's `random_split()` function, while straightforward for in-memory datasets, presents a challenge when dealing with image or data collections scattered across subfolders. The function operates on a PyTorch Dataset object, assuming a linear indexable structure. Direct application to a folder-based dataset without a defined dataset class leads to errors or ineffective splitting. I've encountered this numerous times while working on image classification tasks involving complex data structures and learned the following approach is effective.

The core issue lies in the fact that `random_split()` takes a dataset object, not file paths. Therefore, to utilize it effectively, I must first create a custom PyTorch Dataset class that understands how to retrieve data from those subfolders, converting the file system structure into an indexable form that `random_split()` can operate on. This custom Dataset class will hold the paths to all data and implements `__len__` to specify dataset size and `__getitem__` to load data for any given index.

The primary goal is to build the index of all data and provide functionality to map an integer index to actual data loading operations. This approach decouples dataset management from the splitting logic, aligning well with PyTorch's modular design. By implementing a custom dataset, the random splitting becomes a two-step process: first, creating the dataset object and second, using `random_split()` on the created dataset.

Below are three specific examples of how to address the task, along with code and commentary:

**Example 1: Basic Image Dataset with Subfolders**

This example demonstrates the most common case: image data grouped into subfolders, with each subfolder representing a different class.

```python
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import os

class SubfolderImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        class_names = os.listdir(root_dir)

        for class_index, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                  filepath = os.path.join(class_dir, filename)
                  self.image_paths.append(filepath)
                  self.labels.append(class_index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    # Assuming a folder structure like 'data/class_a/image1.jpg', 'data/class_b/image2.png', etc.
    data_root = 'data'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SubfolderImageDataset(data_root, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
```

This code defines a `SubfolderImageDataset` class that iterates through subfolders in a given `root_dir`, loads images, and stores them with corresponding labels. The key part is storing `image_paths` which then allows for lazy loading of images in `__getitem__()`.  The example then creates an instance of the dataset, defines standard transforms, splits the dataset, and prints the resulting set sizes. The core benefit of this implementation is its generality. It handles a varying number of classes and dynamically adjusts to the total number of images present.

**Example 2: Dataset with CSV Index File**

This expands upon the previous example to include the use of a CSV index file when the dataset structure is more complex, such as having additional metadata. This is useful when dealing with complex labelling or experimental conditions.

```python
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class CSVIndexedImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data_df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        image_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(image_path).convert('RGB')
        label = row['label'] # Assuming label column exists

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    # Assume the 'data_index.csv' contains 'filename' and 'label' columns.
    data_root = 'data'
    csv_index_path = 'data_index.csv'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CSVIndexedImageDataset(csv_index_path, data_root, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
```

This dataset implementation reads metadata from `data_index.csv` and uses it to locate images and labels. By relying on an index file, this allows much greater flexibility with the dataset, as arbitrary mappings can be stored in the csv. The key change here is the replacement of direct directory listing with a DataFrame.

**Example 3: Advanced Dataset with Data Augmentation**

This builds on Example 1 but extends the transform pipeline with different transforms for training and validation datasets. In a real world scenario this would be essential for effective training.

```python
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import os

class SubfolderAugmentedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        class_names = os.listdir(root_dir)

        for class_index, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                  filepath = os.path.join(class_dir, filename)
                  self.image_paths.append(filepath)
                  self.labels.append(class_index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    # Assuming the same folder structure as in example 1.
    data_root = 'data'

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = SubfolderAugmentedDataset(data_root, transform=None)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform


    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

```

The key modification here is that after the random split the transform is updated. This allows for different transform pipelines for training and validation data to occur, enhancing the robustness of training. This shows how transforms can be managed differently after the split. By not performing the transforms in the dataset initialization, we can modify them for train and validation splits.

In summary, handling datasets organized into subfolders effectively requires encapsulating data access logic within a custom `Dataset` class. This separation of concerns allows the flexibility of PyTorch's `random_split()` function to be utilized effectively, regardless of data organization. Data can be loaded directly from file paths, through index files, or by any logic appropriate for the data structure. These examples should give a solid foundation for using `random_split` with arbitrarily complex folder structures.

For further investigation, consider researching topics such as custom Dataset classes in PyTorch, transforms from the `torchvision` library, best practices for handling large image datasets, and considerations for multi-GPU processing. Look into official PyTorch documentation, blog posts and articles on reproducible research and data management, and textbooks specializing in deep learning with PyTorch.
