---
title: "How can dataset objects be used for training with multiple image inputs?"
date: "2025-01-30"
id: "how-can-dataset-objects-be-used-for-training"
---
Dealing with multiple image inputs for deep learning models, a common scenario in tasks like image fusion or stereo vision, requires careful construction of dataset objects that go beyond simple image loading. The standard image dataset abstractions often assume a one-to-one relationship between input and label.  When training on multiple images, we need to either manipulate existing abstractions or build custom ones to ensure proper data loading, pairing, and pipelining. My experience with building a multi-modal medical imaging system for tumor classification highlighted this challenge, where combining MRI, CT scans, and pathology images was essential for robust predictions.

The fundamental issue lies in the structure expected by model training routines. Most deep learning libraries (TensorFlow, PyTorch, etc.) iterate over a dataset object, where each item returned typically comprises either a single input-output pair or, in the case of multiple inputs, a list of inputs paired with a single output. Therefore, we must reshape our data in accordance to this expectation. This can be achieved by overloading the dataset objects in such a way that each 'item' returned contains multiple images. There are two main approaches: modification of existing dataset classes provided by libraries, or implementation of custom dataset classes.

Let us first consider extending the functionality of an existing dataset class, specifically addressing a common scenario in TensorFlow, using `tf.data.Dataset`. In the case where all input images share the same naming convention with a simple numerical extension, we can utilize the `tf.data.Dataset.from_tensor_slices()` functionality. Let’s suppose our data directory has a structure where a single training instance is represented by "image_01.png", "image_02.png", and "image_03.png", for instance for three modalities, with corresponding labels stored in a separate vector.

Here's a snippet of how that might look, along with the essential explanation:

```python
import tensorflow as tf
import os
import numpy as np

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3) # adjust to image type
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256,256])
    return image

def create_dataset_with_multiple_inputs(image_paths_list, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_list, labels))

    def process_entry(image_paths, label):
        images = [load_image(path) for path in image_paths]
        return images, label

    dataset = dataset.map(process_entry, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Simulated File Structure Example:
image_paths = [
    ["image_01_1.png","image_01_2.png","image_01_3.png"],
    ["image_02_1.png","image_02_2.png","image_02_3.png"],
    ["image_03_1.png","image_03_2.png","image_03_3.png"]
    ]

labels = [0,1,0]
for index, path_list in enumerate(image_paths):
    for file in path_list:
        if not os.path.exists(file):
            np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8).tofile(file)

dataset = create_dataset_with_multiple_inputs(image_paths, labels)

for images, label in dataset.take(1):
   print(f"Number of input images:{len(images)}")
   print(f"shape of first image:{images[0].shape}")
   print(f"label:{label}")
```

In this code, `create_dataset_with_multiple_inputs` takes a list of lists of image paths (where each inner list contains paths for one instance's multiple inputs) alongside labels. The core of the process lies in the `process_entry` function which loads, decodes, and resizes the individual images into a list of tensors and then returns the list with corresponding label. `map` then applies this function to each element in the tensor slices, creating an efficiently operating pipeline for data loading. This allows the model to receive the inputs as a list of tensors, each of which can be fed into corresponding input layers.

Alternatively, for more complex scenarios, or if you prefer direct control, implementing a custom dataset class is beneficial.  This is often the approach when data sources are not disk-based, or if more complex preprocessing is required than a simple path-based read. Let's illustrate this using the PyTorch framework, which employs the `torch.utils.data.Dataset` abstract class:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image

class MultiImageDataset(Dataset):
    def __init__(self, image_paths_list, labels, transform=None):
        self.image_paths_list = image_paths_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_paths = self.image_paths_list[idx]
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB') # Adjust to Image type
            if self.transform:
              image = self.transform(image)
            images.append(image)

        label = self.labels[idx]
        return images, label

# Example Usage:
# Simulated File Structure Example:
image_paths = [
    ["image_01_1.png","image_01_2.png","image_01_3.png"],
    ["image_02_1.png","image_02_2.png","image_02_3.png"],
    ["image_03_1.png","image_03_2.png","image_03_3.png"]
    ]

labels = [0,1,0]
for index, path_list in enumerate(image_paths):
    for file in path_list:
        if not os.path.exists(file):
          Image.fromarray(np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)).save(file)
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = MultiImageDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


for images, labels in dataloader:
    print(f"Number of input images:{len(images[0])}")
    print(f"Shape of images: {[image.shape for image in images]}")
    print(f"Labels: {labels}")
    break

```

Here, `MultiImageDataset` inherits from `torch.utils.data.Dataset`. The `__init__` method initializes the paths and labels. The `__len__` method returns the total number of samples. Crucially, `__getitem__` retrieves multiple images from provided paths, applies the transformation if provided, and packages these images along with corresponding label. This implementation gives fine-grained control over data loading and transformations. Note the utilization of the `torchvision.transforms` module for performing the transformation on images, this is a common practice. The output is a batch of lists of images with corresponding labels ready to be consumed by a neural network.

Lastly, let's consider a modification of the previous approach, but instead of loading images individually, imagine we are dealing with pre-processed tensors and wish to package them into a Dataset. For this, we use PyTorch’s `TensorDataset` and modify it for our use case. This method is useful when images are preprocessed and loaded into memory already.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class MultiTensorDataset(TensorDataset):
    def __init__(self, image_tensors_list, labels):
        super(MultiTensorDataset,self).__init__(*[torch.tensor(image_tensors) for image_tensors in image_tensors_list ],torch.tensor(labels))
        self.num_tensors = len(image_tensors_list)

    def __getitem__(self,idx):
      tensors_tuple = super().__getitem__(idx)
      tensors_list = [tensors_tuple[i] for i in range(self.num_tensors)]
      return tensors_list, tensors_tuple[-1]


# Example Usage:
# Generate some dummy tensors.
image_tensors_1 = [np.random.rand(3, 256, 256).astype(np.float32) for _ in range(3)]
image_tensors_2 = [np.random.rand(3, 256, 256).astype(np.float32) for _ in range(3)]
image_tensors_3 = [np.random.rand(3, 256, 256).astype(np.float32) for _ in range(3)]

image_tensors_list = [image_tensors_1,image_tensors_2,image_tensors_3]
labels = [0, 1, 0]


dataset = MultiTensorDataset(image_tensors_list, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for images, labels in dataloader:
  print(f"Number of input images:{len(images[0])}")
  print(f"Shape of images:{[image.shape for image in images]}")
  print(f"Labels: {labels}")
  break
```

In this example, `MultiTensorDataset` inherits from `TensorDataset`. The `__init__` method transforms the list of numpy tensors into torch tensors, and then constructs a `TensorDataset` via the super function call. The `__getitem__` method overloads the standard `__getitem__` method to return a list of tensors instead of a tuple. The tensors are already in memory and do not need transformations, this method is beneficial when the data is preprocessed and stored in memory beforehand, as tensors. This approach avoids redundant loading and preprocessing if those operations are external to data loading pipeline.

When embarking on projects involving multi-input image training, I would recommend investing time in mastering the data loading capabilities of the chosen deep learning framework. Reading the official documentation and exploring resources such as the TensorFlow guide on `tf.data` or the PyTorch documentation on custom datasets will be beneficial. Also, practical coding examples from GitHub repositories that deal with similar problems are valuable in understanding implementations. Online tutorials by research groups who specialize in specific application areas can also be highly informative. It's crucial to select and modify a method that is best suited to the specific data characteristics and processing requirements in hand. Efficient data loading and processing is the key for efficient use of computational resources during training.
