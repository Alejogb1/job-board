---
title: "How to one-hot encode a grayscale image for semantic segmentation in PyTorch?"
date: "2025-01-30"
id: "how-to-one-hot-encode-a-grayscale-image-for"
---
The immediate challenge in preparing a grayscale image for semantic segmentation using PyTorch lies in its inherent single-channel nature versus the multi-channel output needed for pixel-wise classification. Semantic segmentation, at its core, predicts a class label for every pixel, requiring an input format that can represent the various classes. Since grayscale images provide only a single intensity value per pixel, directly feeding them into a standard semantic segmentation model trained on multi-channel (e.g., RGB) data won't work. We must transform our single-channel input into a multi-channel, one-hot encoded format.

My experience building a system to automatically identify cancerous tissue from medical scans revealed that even seemingly straightforward preprocessing steps are critical for robust model performance. One-hot encoding, in this context, is the process of converting a scalar value representing a class into a vector where each position indicates the presence or absence of a specific class. In our case, the grayscale intensity values represent distinct classes based on an established mapping, or, if continuous, are binned into classes. The goal is to generate a separate channel for each identified class.

Here’s how I approached this challenge in my work, using PyTorch and its associated ecosystem.

**Explanation of the One-Hot Encoding Process**

1.  **Class Determination and Mapping:** Before performing one-hot encoding, a crucial step is identifying the number of unique classes present in the grayscale image (or if continuous, defining the class bins). Assuming that the pixel intensity levels are associated with semantic classes, we need to establish a mapping between those intensity values and class labels. For example, in a histological image, pixel values might correspond to different tissue types like epithelium, stroma, and immune cells, each corresponding to a discrete intensity level, or range. If there is no direct mapping, the grayscale intensities must be thresholded or clustered into the needed number of classes.

2.  **Creating the One-Hot Encoding Tensor:** After determining the number of classes, we create a target tensor to hold the one-hot encoding. This tensor will have dimensions of `(num_classes, height, width)`, where height and width are the spatial dimensions of the image.

3.  **Populating the Tensor:** For each pixel in the original image, we identify its corresponding class using the predefined mapping or binning. The correct channel index in the one-hot encoded tensor representing that class is then set to 1 at the pixel's spatial location, and all other channels remain at 0. Conceptually, for every pixel in the original image, the corresponding pixel location in the one-hot tensor will be all zeros except at the position that corresponds to the ground-truth class assignment of that original pixel. This can be performed efficiently using indexing in PyTorch tensors.

4.  **Data Type Considerations:** It's essential to ensure the output tensor has the correct data type for model processing. Typically, a floating-point type (`torch.float32`) is required for convolutional layers.

**Code Examples**

**Example 1: One-Hot Encoding with a Discrete Class Mapping**

This example assumes the pixel intensities correspond to pre-defined classes, where we use direct indexing.

```python
import torch

def one_hot_encode_discrete(image, num_classes, class_mapping):
  """
  Performs one-hot encoding based on discrete pixel values and a class mapping.

  Args:
    image (torch.Tensor): Grayscale image tensor with shape (height, width).
    num_classes (int): Number of classes.
    class_mapping (dict): Dictionary mapping pixel values to class indices.

  Returns:
    torch.Tensor: One-hot encoded tensor of shape (num_classes, height, width).
  """
  height, width = image.shape
  encoded_image = torch.zeros((num_classes, height, width), dtype=torch.float32)
  for pixel_value, class_idx in class_mapping.items():
      encoded_image[class_idx, image == pixel_value] = 1.0

  return encoded_image


# Example Usage
image = torch.tensor([[1, 2, 0],
                      [2, 0, 1],
                      [0, 1, 2]], dtype=torch.int64)

num_classes = 3
class_mapping = {0: 0, 1: 1, 2: 2} # Map pixel values to class indices
encoded_image = one_hot_encode_discrete(image, num_classes, class_mapping)
print(encoded_image)
```

In this function, the `class_mapping` is used to determine the class index associated with each unique pixel intensity level, and then sets the target channel to 1 based on the pixels that match that particular intensity level.

**Example 2: One-Hot Encoding with Bins**

This function performs the one-hot encoding using binning instead of discrete mapping.

```python
import torch

def one_hot_encode_bins(image, num_classes, bins):
  """
  Performs one-hot encoding using bins for class assignment.

  Args:
    image (torch.Tensor): Grayscale image tensor with shape (height, width).
    num_classes (int): Number of classes (equal to number of bins).
    bins (list): List of bin edges (e.g., [0, 0.25, 0.5, 0.75, 1.0]).

  Returns:
    torch.Tensor: One-hot encoded tensor of shape (num_classes, height, width).
  """
  height, width = image.shape
  encoded_image = torch.zeros((num_classes, height, width), dtype=torch.float32)
  for i in range(num_classes):
        if i == 0:
            mask = (image >= bins[i]) & (image < bins[i+1])
        elif i == num_classes -1:
             mask = (image >= bins[i]) & (image <= bins[i+1])
        else:
            mask = (image >= bins[i]) & (image < bins[i+1])
        encoded_image[i, mask] = 1.0
  return encoded_image

#Example usage
image = torch.tensor([[0.1, 0.6, 0.2],
                      [0.8, 0.3, 0.1],
                      [0.4, 0.2, 0.9]], dtype=torch.float32)
num_classes = 4
bins = [0.0, 0.25, 0.5, 0.75, 1.0]

encoded_image = one_hot_encode_bins(image, num_classes, bins)
print(encoded_image)
```

This example uses a series of bins to classify the input pixel intensities. A pixel's intensity will determine which of the n bins it falls into, and then the channel associated with that bin will have the pixel set to 1. Note that this function supports n bins, creating a one-hot tensor with n channels, even if not all of the bins are populated in the source image.

**Example 3: Integration in a Dataset**

This illustrates how to incorporate one-hot encoding during data loading using PyTorch’s `Dataset` class.

```python
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, image_dir, num_classes, class_mapping, transform=None):
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.class_mapping = class_mapping
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Load as grayscale
        image = np.array(image)
        image = torch.from_numpy(image).long()
        if self.transform:
             image = self.transform(image)

        encoded_image = one_hot_encode_discrete(image, self.num_classes, self.class_mapping)

        return encoded_image, encoded_image.sum(axis=0)


# Example Dataset Setup
image_dir = 'example_images' #Directory of .png grayscale images
if not os.path.exists(image_dir):
     os.makedirs(image_dir)
#Create example image file
example_image = np.array([[1, 2, 0],
                    [2, 0, 1],
                    [0, 1, 2]], dtype=np.uint8)
Image.fromarray(example_image).save(os.path.join(image_dir, 'example_image.png'))

num_classes = 3
class_mapping = {0: 0, 1: 1, 2: 2}
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()])
dataset = GrayscaleSegmentationDataset(image_dir, num_classes, class_mapping, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for encoded_image, label in dataloader:
    print("One-Hot Encoded Image Shape:", encoded_image.shape)
    print("Example label", label)
    break
```

In this example, the `GrayscaleSegmentationDataset` loads the grayscale images and applies the transforms. Within the `__getitem__` method the loaded image is one-hot encoded and then returned along with the original image. The `transforms.ToTensor()` transforms the numpy array to a torch tensor and performs the important operation of normalizing the pixel values between 0 and 1. Note that I included resizing operations here to match how images will be processed in the model, and how data scientists will approach the problem in practice.

**Resource Recommendations**

For a deeper understanding of PyTorch tensor operations, consult the official PyTorch documentation. Furthermore, examine examples of custom datasets within the PyTorch ecosystem. Study the typical transforms applied to data for training neural networks. The fundamental principles of one-hot encoding are also covered in numerous online resources and machine learning textbooks, which will provide a more in depth understanding of the concepts at play here.
