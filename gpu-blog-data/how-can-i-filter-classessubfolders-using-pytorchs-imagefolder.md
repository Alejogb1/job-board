---
title: "How can I filter classes/subfolders using PyTorch's ImageFolder?"
date: "2025-01-30"
id: "how-can-i-filter-classessubfolders-using-pytorchs-imagefolder"
---
PyTorch's `torchvision.datasets.ImageFolder` facilitates loading datasets structured into class-specific subfolders, yet it lacks direct built-in parameters for filtering these subfolders beyond simple whitelisting. The behavior I’ve consistently observed is that `ImageFolder` automatically infers class labels based on the directory names within a root directory. When a dataset contains unwanted class directories, custom logic is required during the loading phase to handle this scenario effectively.

**Understanding `ImageFolder`'s Structure and Behavior**

`ImageFolder` expects a specific directory layout: a root directory containing subdirectories, where each subdirectory represents a distinct class. Each subdirectory, in turn, holds image files belonging to that class. Upon instantiation, `ImageFolder` scans this root directory, identifies the subdirectories, and assigns a numerical class index to each. The dataset yields a tuple: (image tensor, class index). This automated class-mapping makes it convenient for standard image classification tasks, but poses a challenge when there is a need to selectively load certain classes or subfolders. Without further modification, it loads *all* subfolders it finds within the root.

**Implementing Custom Filtering Logic**

The core concept I’ve used in several of my projects involves creating a modified dataset class that inherits from `torch.utils.data.Dataset` instead of directly relying on `ImageFolder`. This approach provides granular control over the folder discovery and image loading process, allowing us to implement the necessary filtering criteria. By manually processing the directory structure and then loading the images based on specified filters, we can achieve the desired behavior. The custom class will take a root directory, as well as a list of acceptable folder names. The implementation I have frequently opted for uses `os.scandir` for efficient file system traversal, and image loading via `PIL` to match `ImageFolder`’s approach.

**Code Example 1: Basic Filtering by Subfolder Names**

This example demonstrates a custom dataset class, `FilteredImageFolder`, which filters subfolders based on a list of allowed folder names.

```python
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FilteredImageFolder(Dataset):
    def __init__(self, root_dir, allowed_folders, transform=None):
        self.root_dir = root_dir
        self.allowed_folders = allowed_folders
        self.image_paths = []
        self.class_to_idx = {}
        self.transform = transform
        self._load_images()

    def _load_images(self):
        class_idx = 0
        for entry in os.scandir(self.root_dir):
            if entry.is_dir() and entry.name in self.allowed_folders:
                self.class_to_idx[entry.name] = class_idx
                for image_entry in os.scandir(entry.path):
                    if image_entry.is_file() and image_entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append((image_entry.path, class_idx))
                class_idx += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, class_idx = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx

if __name__ == '__main__':
    # Example Usage
    root_directory = "my_image_data" # Assumes a folder with subdirectories
    allowed_classes = ["cats", "dogs"]
    transform =  torchvision.transforms.ToTensor()

    dataset = FilteredImageFolder(root_directory, allowed_classes, transform=transform)
    print(f"Number of images loaded: {len(dataset)}")
    print(f"Class indices: {dataset.class_to_idx}")
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"Shape of first image: {image.shape}")
        print(f"Label of first image: {label}")
```

**Commentary:** This initial implementation demonstrates the basic mechanics. The constructor takes the root directory and a list of allowed subfolder names. The `_load_images` method iterates over the subdirectories, filters them based on the allowed list, and then gathers the image file paths along with their corresponding class indices. The `__getitem__` method then opens the image at the determined file path and applies the appropriate transformations. The `if __name__ == '__main__'` block shows how it would be invoked and prints basic statistics for verification, along with the first image and label.

**Code Example 2: Filtering by Complex Criteria Using a Function**

In more complex scenarios, you may need a more sophisticated filtering criteria beyond simple whitelisting, like using prefixes, specific naming conventions, or combinations. This example expands on the previous one to allow a filtering function for more dynamic class selection.

```python
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision

class FilteredImageFolderFn(Dataset):
    def __init__(self, root_dir, filter_fn, transform=None):
        self.root_dir = root_dir
        self.filter_fn = filter_fn # Function accepts subdir name, returns bool
        self.image_paths = []
        self.class_to_idx = {}
        self.transform = transform
        self._load_images()


    def _load_images(self):
        class_idx = 0
        for entry in os.scandir(self.root_dir):
            if entry.is_dir() and self.filter_fn(entry.name):
                self.class_to_idx[entry.name] = class_idx
                for image_entry in os.scandir(entry.path):
                    if image_entry.is_file() and image_entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append((image_entry.path, class_idx))
                class_idx += 1


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image_path, class_idx = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx


if __name__ == '__main__':
    # Example Usage
    root_directory = "my_image_data"
    # Function to only include folders that start with 'a'
    def folder_filter(folder_name):
       return folder_name.startswith('a')

    transform = torchvision.transforms.ToTensor()
    dataset = FilteredImageFolderFn(root_directory, folder_filter, transform=transform)
    print(f"Number of images loaded: {len(dataset)}")
    print(f"Class indices: {dataset.class_to_idx}")
    if len(dataset) > 0:
       image, label = dataset[0]
       print(f"Shape of first image: {image.shape}")
       print(f"Label of first image: {label}")
```

**Commentary:** Here, a `filter_fn` is introduced in the constructor. This function takes the subfolder name as input and returns a boolean value indicating whether the folder should be included. This allows for highly flexible filtering, enabling us to implement almost any logical combination or pattern for inclusion based on folder naming. The function ‘folder_filter’ is designed for example only and only uses the prefix 'a', but it can be altered to any filter desired. The rest of the implementation follows the same structure.

**Code Example 3: Handling Empty Subdirectories**

It is often the case that certain subdirectories might be empty. If not handled, they will cause exceptions, or worse, create an unintended zero-sized dataset. This example shows how we can gracefully ignore them.

```python
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision


class FilteredImageFolderEmpty(Dataset):
    def __init__(self, root_dir, allowed_folders, transform=None):
        self.root_dir = root_dir
        self.allowed_folders = allowed_folders
        self.image_paths = []
        self.class_to_idx = {}
        self.transform = transform
        self._load_images()


    def _load_images(self):
        class_idx = 0
        for entry in os.scandir(self.root_dir):
            if entry.is_dir() and entry.name in self.allowed_folders:
                image_count = 0
                self.class_to_idx[entry.name] = class_idx
                for image_entry in os.scandir(entry.path):
                    if image_entry.is_file() and image_entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append((image_entry.path, class_idx))
                        image_count+=1
                if image_count > 0:
                    class_idx += 1
                else:
                    del self.class_to_idx[entry.name]


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image_path, class_idx = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx



if __name__ == '__main__':
    # Example Usage
    root_directory = "my_image_data"
    allowed_classes = ["cats", "dogs", "empty_folder"] #Assume empty_folder is an empty directory

    transform = torchvision.transforms.ToTensor()
    dataset = FilteredImageFolderEmpty(root_directory, allowed_classes, transform=transform)
    print(f"Number of images loaded: {len(dataset)}")
    print(f"Class indices: {dataset.class_to_idx}")
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"Shape of first image: {image.shape}")
        print(f"Label of first image: {label}")
```

**Commentary:** This example builds on the first one, but incorporates a counter for the images in each folder. If after scanning all entries in the folder, the image count is zero, the dictionary entry for the class and the class index is removed. This prevents the creation of a class with no associated data. As long as a class does have data, the usual process of generating the image paths continues.

**Resource Recommendations:**

For more detailed information on file system traversal, explore documentation for the Python `os` and `os.path` modules, particularly `os.scandir` and `os.path.join`. Further detail on the `torch.utils.data.Dataset` class and subclassing requirements is available within the PyTorch documentation. The Python Imaging Library (PIL), specifically its `Image` module, is essential for loading image files, and its documentation provides usage information for the `Image.open` function and related functionalities.
