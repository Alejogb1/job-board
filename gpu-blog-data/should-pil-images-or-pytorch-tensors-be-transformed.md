---
title: "Should PIL Images or PyTorch tensors be transformed?"
date: "2025-01-30"
id: "should-pil-images-or-pytorch-tensors-be-transformed"
---
Directly transforming PIL (Pillow) Images or PyTorch tensors involves a trade-off between convenience, performance, and the specific needs of a computer vision pipeline. Having built several object detection systems, I've navigated this decision often. The ideal approach isn't universally defined, depending heavily on the subsequent processing steps and the data loading mechanisms employed.

**Explanation of the Core Differences**

PIL Images, at their core, are Python objects that represent images in a human-readable format. They support a wide range of image manipulation operations – resizing, color space conversions, applying filters, and more. They store pixel data as a NumPy array internally, offering a convenient and intuitive interface for common image processing tasks. However, these operations often rely on interpreted Python code, which can introduce performance bottlenecks when dealing with large datasets or high-resolution images. Additionally, PIL Images aren't readily compatible with the computational backbone of neural networks, which predominantly rely on numerical tensor operations.

PyTorch tensors, on the other hand, are the fundamental data structures in the PyTorch framework. They're optimized for numerical computations on both CPU and GPU. Tensors are typically multi-dimensional arrays, capable of representing everything from image pixel data to the weights and gradients of a neural network. Transformations applied directly to tensors benefit from PyTorch's optimized kernels, allowing for significant speedups when training deep learning models. However, while tensors can directly store image data, they lack the image-specific functionalities provided by PIL.

The key difference lies in the target operation. If you intend to perform general image manipulation outside of the context of model training, such as saving the image in a specific format, applying custom filters or augmentations not directly available in PyTorch, PIL provides the necessary tools. If the goal is to pass the image data to a neural network for training or inference, tensors are the appropriate data representation, and efficient transformations are paramount.

**Practical Implications and Transformation Strategies**

The decision hinges on the workflow. A common scenario in training deep learning models involves the following steps: loading an image file, applying pre-processing steps like resizing and normalization, and finally passing the data to the model. The location of the PIL to Tensor conversion within this workflow significantly impacts overall performance.

If data loading is a significant bottleneck, performing initial transformations in PIL and then converting to a tensor can introduce a substantial overhead. Furthermore, augmentations (like rotations, flips, and random crops) often need to be applied during training, meaning applying augmentation once in PIL and converting before training would not suffice.

Therefore, my experience indicates that delaying the conversion to tensors and applying as many transforms as possible in tensor space is most efficient. This ensures that all computational steps relevant to model training are executed within the optimized PyTorch framework, maximizing GPU utilization and minimizing CPU bottlenecks.

**Code Examples and Commentary**

The following code examples demonstrate the different approaches, and I'll comment on each to clarify the reasoning behind them.

**Example 1: Transforming in PIL (Less Efficient for Training)**

```python
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Load image using PIL
image_pil = Image.open("example.jpg")

# Define PIL transformations
pil_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations in PIL and convert to tensor
transformed_image_tensor = pil_transforms(image_pil)

print(transformed_image_tensor.shape) # Should be torch.Size([3, 256, 256])
```

In this example, transformations such as resizing and rotation are applied directly on the PIL image before converting it to a tensor. Although readable, this approach incurs an overhead as PIL operations are not optimized for GPU usage and are executed prior to conversion. The `transforms.ToTensor()` step is used to convert PIL to a tensor, and normalization is performed on tensor as well.

**Example 2: Transforming Tensors Directly (More Efficient for Training)**

```python
from PIL import Image
import torch
from torchvision import transforms

# Load image using PIL
image_pil = Image.open("example.jpg")

# Convert to tensor first
image_tensor = transforms.ToTensor()(image_pil)

# Define tensor transformations
tensor_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations on tensors
transformed_image_tensor = tensor_transforms(image_tensor)

print(transformed_image_tensor.shape) # Should be torch.Size([3, 256, 256])

```
Here, the PIL image is immediately converted to a tensor and all further transformations, including resizing, rotation, and normalization are performed directly on the tensor object. This leverages the optimized routines in PyTorch.  Note that `transforms.Resize` etc. can operate on both tensors and PIL images. The major difference lies where the data is, either in a Python PIL object or a PyTorch Tensor object.

**Example 3: Custom Tensor Transformations within a Dataset Class**

```python
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)  # Convert to tensor immediately

        if self.transform:
            image = self.transform(image)
        
        return image #Return tensor.

# Define transformations (applied within the dataset during training)
tensor_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dummy dataset location
dummy_dir = "./dummy_images/"
os.makedirs(dummy_dir, exist_ok=True)
dummy_image = Image.new('RGB', (64,64), color="red")
dummy_image.save(os.path.join(dummy_dir, "test.png"))


# Instantiate dataset
dataset = CustomImageDataset(dummy_dir, transform=tensor_transform)

# Load a single image from the dataset
image_from_dataset = dataset[0]
print(image_from_dataset.shape) # Should be torch.Size([3, 256, 256])

```

This example shows a common use case. The dataset loader applies transformations at the time of data access. The PIL image is converted immediately to a tensor before other transforms are applied.  This allows for on-the-fly data augmentation during training.

**Resource Recommendations**

For further understanding, I recommend exploring the official documentation for the Pillow library. Specifically, focus on the core image object and the various image manipulation methods it provides. In addition, the torchvision library’s documentation outlines the different transforms available. Pay close attention to the data types that each transform method accepts, whether PIL images, NumPy arrays, or tensors. Understanding the limitations and benefits of each will provide the necessary foundation for writing high-performance computer vision code.
Finally, the PyTorch tutorials regarding custom datasets are extremely valuable. Pay specific attention to how the transforms are applied within the `__getitem__` method and how this ties into the overall training loop. These resources should help in building an effective workflow for any computer vision project.
