---
title: "How can I split a dataset folder into train and test sets using PyTorch?"
date: "2025-01-26"
id: "how-can-i-split-a-dataset-folder-into-train-and-test-sets-using-pytorch"
---

In my experience managing deep learning projects, correctly splitting datasets into training and testing subsets is crucial for model evaluation and generalization. An improper split can lead to misleading performance metrics and a model that overfits the training data, performing poorly on unseen examples. PyTorch, while primarily a deep learning framework, does not provide a direct utility for folder-based dataset splitting. Instead, we typically leverage Python's standard library and libraries like `torchvision` to accomplish this.

The fundamental challenge lies in mapping file system organization into a structure that PyTorch's `DataLoader` can understand and efficiently process. The typical approach involves identifying the files within the original folder, shuffling them, and then copying them into dedicated training and testing folders. This allows us to use `torchvision.datasets.ImageFolder` or a custom dataset class with `DataLoader` to create data iterators for each subset. This process, when implemented carefully, ensures a consistent and reproducible split, vital for model development.

Let me illustrate with some practical code. The most basic method involves manually copying files using Python’s `shutil` and `os` modules. Assume you have a directory `data` containing image files in various subfolders, representing different classes:

```python
import os
import shutil
import random

def split_dataset(data_dir, train_dir, test_dir, split_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir,f))]
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        train_images, test_images = images[:split_index], images[split_index:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy2(img, os.path.join(train_dir, class_name, os.path.basename(img)))
        for img in test_images:
            shutil.copy2(img, os.path.join(test_dir, class_name, os.path.basename(img)))


if __name__ == '__main__':
  data_dir = 'data'  # Path to your main dataset directory
  train_dir = 'train' # Path to train set output directory
  test_dir = 'test' # Path to test set output directory
  split_dataset(data_dir, train_dir, test_dir)
  print('Splitting completed')
```

This Python code snippet performs several operations. The function `split_dataset` iterates over the subdirectories within the `data_dir`, which are assumed to represent classes. Inside, it gathers all image paths, shuffles them randomly using `random.shuffle`, and calculates the split point based on `split_ratio` (defaulting to 80% for training). The file paths are copied to respective `train_dir` and `test_dir` subdirectories. The `shutil.copy2` function is used to preserve metadata such as timestamps.

This approach is straightforward, but its downside is that it physically duplicates the dataset, taking up additional disk space. A more efficient approach is creating a metadata file that records the file paths and their corresponding labels. This avoids making copies of the data and can be used to construct a custom dataset. Here’s how we might do that:

```python
import os
import random
import json

def create_metadata(data_dir, output_file, split_ratio=0.8):
    metadata = {'train': [], 'test': []}

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir,f))]
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        train_images, test_images = images[:split_index], images[split_index:]

        for img in train_images:
            metadata['train'].append({'image': img, 'label': class_name})
        for img in test_images:
             metadata['test'].append({'image': img, 'label': class_name})

    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    data_dir = 'data'  # Path to your main dataset directory
    output_file = 'dataset_metadata.json' # Path to save the json metadata file
    create_metadata(data_dir, output_file)
    print('Metadata file saved')
```
This code generates a JSON metadata file (`dataset_metadata.json`). Each entry in the 'train' and 'test' keys of this JSON object consists of a dictionary with 'image' (full file path) and 'label' (the sub-directory name) keys. The `create_metadata` function shuffles image file paths per class before populating the metadata, and the data is not physically copied, addressing the disk space concern of the first method. To load this metadata, you would define a custom `Dataset` class that reads the JSON file, reads the image using a library like Pillow or OpenCV, performs any required transformations, and return data as a torch tensor. This approach promotes modularity and avoids disk duplication.

Finally, to utilize this metadata with PyTorch’s dataloaders, consider this sketch of a custom dataset loader:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, metadata_file, transform=None, subset='train'):
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.image_data = self.metadata[subset]
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
      image_path = self.image_data[idx]['image']
      label = self.image_data[idx]['label']
      image = Image.open(image_path).convert('RGB')
      if self.transform:
        image = self.transform(image)
      label_to_idx = {'class_a':0,'class_b':1, 'class_c':2}  #Example label mapping for dataset
      label_idx = label_to_idx[label]
      return image, label_idx


if __name__ == '__main__':
    metadata_file = 'dataset_metadata.json'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CustomImageDataset(metadata_file, transform, subset='train')
    test_dataset = CustomImageDataset(metadata_file, transform, subset='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for images, labels in train_loader:
        print(f"Train batch: Images shape {images.shape}, Labels: {labels}")
        break # Process one batch
    for images, labels in test_loader:
        print(f"Test batch: Images shape {images.shape}, Labels: {labels}")
        break # Process one batch
```

The `CustomImageDataset` class reads the metadata file, takes the subset (either 'train' or 'test') as a parameter, and loads the images during the `__getitem__` method. The `transforms` variable applies a series of transformations including resizing, cropping, and normalization which are necessary for training typical convolutional neural networks. The data loaders then enable training or testing by pulling batches of transformed image data and their respective labels. Note that you can customize this class further by using a library such as scikit-learn to achieve stratified splitting, if needed.

For individuals looking to expand their understanding beyond these code examples, I recommend consulting several key resources. For a broader understanding of file manipulation, reviewing documentation for Python’s `os`, `shutil`, and `json` modules is valuable. The PyTorch documentation's section on `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` will solidify how custom datasets integrate into a typical training pipeline. For image loading and manipulation, reviewing the Pillow library documentation will be beneficial. Lastly, the torchvision library documentation will provide comprehensive details on prebuilt transforms and dataset classes. Together these resources offer a complete view for implementing efficient and reliable dataset splitting for PyTorch projects.
