---
title: "How to batch load all images from a folder using PyTorch?"
date: "2025-01-30"
id: "how-to-batch-load-all-images-from-a"
---
Efficiently loading images from a directory for PyTorch training requires careful consideration of data access, pre-processing, and memory management. I've encountered this scenario frequently during my work developing custom image classification models for satellite imagery, where datasets often involve thousands of high-resolution files. The naive approach of loading all images into memory at once is rarely feasible; instead, a strategy combining a custom dataset class with PyTorch's `DataLoader` offers significant performance advantages.

The core mechanism involves inheriting from `torch.utils.data.Dataset` and overriding its essential methods: `__len__` and `__getitem__`. The `__len__` method defines the total number of data samples (images in our case), while `__getitem__` is responsible for fetching a single sample at a specified index. Inside `__getitem__`, we perform necessary image loading and transformations before returning the processed image as a PyTorch tensor. This design allows `DataLoader` to efficiently sample and batch the data without keeping all the images in memory simultaneously.

Here's a breakdown of the process followed by specific code examples demonstrating the approach:

**1. Data Organization:**
Begin by ensuring your image data resides in a designated folder with a consistent naming scheme. For example, all images could be directly located within the folder or organized into subfolders by category. This choice will influence how the image paths are collected and accessed.

**2. Custom Dataset Class Implementation:**
A custom class inheriting from `torch.utils.data.Dataset` forms the backbone of this process. It stores the list of image file paths during initialization and uses `__getitem__` to load an image, apply transformations, and return the tensor.

**3. Image Loading and Preprocessing:**
Within `__getitem__`, libraries like `PIL` (Pillow) handle image loading and PyTorch's `torchvision.transforms` module provides a flexible way to apply standard transformations (e.g., resizing, normalization, augmentation).

**4. DataLoader Configuration:**
Finally, the custom dataset class is passed to the `torch.utils.data.DataLoader`. We specify parameters like batch size, whether to shuffle data, and the number of worker processes for parallel data loading. This enables efficient batched access of the image data during training.

Now, let’s delve into code examples:

**Example 1: Basic Implementation (No Transformations)**
This example loads images and converts them to tensors without applying any specific transformations. It provides a foundational understanding of the process.

```python
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageFolderDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.image_paths = sorted(self.image_paths)  # Sort for deterministic behavior

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure consistent 3 channel images
        image_tensor = torch.tensor(image.getdata(), dtype=torch.float32).reshape(image.size[1], image.size[0], 3).permute(2, 0, 1)
        return image_tensor

if __name__ == '__main__':
    # Create a directory and some dummy image files for this example
    image_dir = "dummy_images"
    os.makedirs(image_dir, exist_ok=True)
    for i in range(5):
        img = Image.new('RGB', (64, 64), color = 'red')
        img.save(os.path.join(image_dir, f"image_{i}.jpg"))

    image_dataset = ImageFolderDataset(image_dir)
    data_loader = DataLoader(image_dataset, batch_size=2, shuffle=True)

    for batch_idx, batch in enumerate(data_loader):
        print(f"Batch {batch_idx}, Shape: {batch.shape}")
    
    # Clean up dummy images and directory
    for i in range(5):
        os.remove(os.path.join(image_dir, f"image_{i}.jpg"))
    os.rmdir(image_dir)
```

*   **Commentary:** The `ImageFolderDataset` class stores the list of image paths, loads them using PIL, converts the image to a tensor and returns it. Notice that the tensor is reshaped to a height x width x 3 tensor using `reshape()` and then permuted to the channel x height x width representation with `permute()` which is expected by PyTorch.  The dummy image generation part was added to make this example executable without requiring an actual image dataset.

**Example 2: Including Image Transformations**
This example demonstrates the application of transformations, including resizing and normalization. These transformations are essential for training deep learning models.

```python
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageFolderDatasetWithTransforms(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.image_paths = sorted(self.image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

if __name__ == '__main__':
     # Create a directory and some dummy image files for this example
    image_dir = "dummy_images"
    os.makedirs(image_dir, exist_ok=True)
    for i in range(5):
        img = Image.new('RGB', (64, 64), color = 'blue')
        img.save(os.path.join(image_dir, f"image_{i}.jpg"))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_dataset = ImageFolderDatasetWithTransforms(image_dir, transform=transform)
    data_loader = DataLoader(image_dataset, batch_size=2, shuffle=True)

    for batch_idx, batch in enumerate(data_loader):
        print(f"Batch {batch_idx}, Shape: {batch.shape}")

    # Clean up dummy images and directory
    for i in range(5):
         os.remove(os.path.join(image_dir, f"image_{i}.jpg"))
    os.rmdir(image_dir)
```

*   **Commentary:** This revised dataset class incorporates transformations via `torchvision.transforms`. This allows for applying multiple transformations such as resizing to the same size for consistent training and normalizing the images using a transformation before returning the tensors. Common normalization parameters for ImageNet are used here.

**Example 3: Utilizing Subfolders for Category Classification**
This example demonstrates loading images from a directory that is structured with subfolders representing categories. It is particularly useful when training a multi-class image classifier.

```python
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class CategoricalImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if os.path.isfile(os.path.join(class_dir,filename)):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    # Create a directory and some dummy image files for this example
    root_dir = "dummy_images"
    os.makedirs(root_dir, exist_ok=True)
    categories = ['cats', 'dogs']
    for cat in categories:
        os.makedirs(os.path.join(root_dir, cat), exist_ok=True)
    for i in range(2):
        img = Image.new('RGB', (64, 64), color = 'green')
        img.save(os.path.join(root_dir, categories[0], f"image_{i}.jpg"))
        img = Image.new('RGB', (64, 64), color = 'yellow')
        img.save(os.path.join(root_dir, categories[1], f"image_{i}.jpg"))
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_dataset = CategoricalImageFolderDataset(root_dir, transform=transform)
    data_loader = DataLoader(image_dataset, batch_size=2, shuffle=True)

    for batch_idx, (batch_images, batch_labels) in enumerate(data_loader):
        print(f"Batch {batch_idx}, Image Shape: {batch_images.shape}, Labels: {batch_labels}")
    
    # Clean up dummy images and directory
    for cat in categories:
         for i in range(2):
             os.remove(os.path.join(root_dir, cat, f"image_{i}.jpg"))
    os.rmdir(os.path.join(root_dir,categories[0]))
    os.rmdir(os.path.join(root_dir,categories[1]))
    os.rmdir(root_dir)
```

*   **Commentary:**  This version scans subfolders to determine classes. It generates labels that are integer representation of the category that the image belongs to. The `__getitem__` method now returns a tuple of image tensor and category label.

**Resource Recommendations:**
To deepen understanding of this topic, I recommend the following:

1.  **PyTorch Documentation:** Comprehensive documentation on PyTorch datasets, dataloaders, and image transformations. This should be your primary resource.
2.  **Image Processing Libraries Documentation:**  Consult the documentation for libraries such as Pillow (PIL) for image manipulation and loading.
3.  **Online Machine Learning Courses:** Many high-quality online courses provide lectures and examples that cover dataset creation and data loading.

By implementing a custom dataset class as shown, combined with the power of PyTorch's DataLoader, we can efficiently handle large datasets of images for deep learning training, regardless of whether the images are within a single directory or organized into subfolders by category. This provides control over image loading, preprocessing, and batching, and enables scalable model development.
