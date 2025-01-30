---
title: "How can I create a dataset similar to CIFAR-10 with multiple images?"
date: "2025-01-30"
id: "how-can-i-create-a-dataset-similar-to"
---
The core challenge in creating a multi-image dataset analogous to CIFAR-10 lies in structuring data for efficient processing by machine learning models. CIFAR-10's strength derives from its well-defined, labeled format: 32x32 color images grouped into ten distinct classes. Replicating this structure for multiple images per instance necessitates careful attention to data representation and loading techniques.

My experience building computer vision models has revealed several key aspects to address. First, unlike CIFAR-10, which stores each image as a discrete data point, multiple image datasets require a method to aggregate related images. Second, we need to manage label association, ensuring the correct mapping of images to their corresponding class. Finally, input pipelines must be optimized to load this grouped data efficiently during training. The following discussion elaborates on these points and offers practical approaches.

A fundamental aspect is how to represent groups of images. A common method utilizes a directory-based approach. Imagine each sample representing a scene, such as a collection of birds in different poses. We can create a directory structure where each class (e.g., “birds”) has its subdirectory, and within each subdirectory, each sample has its own subdirectory holding all relevant images. For instance, we may have a structure like “dataset/birds/sample_001/image_01.png,” “dataset/birds/sample_001/image_02.png,” etc. This organization reflects the inherent grouping, facilitating intuitive dataset management.

Now, consider the label associations. In CIFAR-10, each image has a single label. With multiple images per sample, we must consider whether a single class label applies to the entire collection (e.g., all images in "sample_001" belong to the "birds" class), or if each image within a sample might require its own label (e.g., a sample with a bird and a nest; one image showing the bird and the other showing the nest, each with a different label). For the sake of this discussion, let's assume that the collection of images shares a single common class label.

During the data loading process, we leverage the hierarchical file structure. Input pipelines are crucial for both efficiency and adaptability. Let's examine some code using Python and common deep learning libraries, specifically TensorFlow and PyTorch, illustrating how to load this structure and adapt to model expectations.

**Code Example 1: TensorFlow Data Loading with `tf.data.Dataset`**

```python
import tensorflow as tf
import os
import numpy as np

def load_multiple_image_dataset_tf(root_dir, image_size, batch_size):
  """Loads a multi-image dataset using TensorFlow's tf.data.Dataset.

  Args:
      root_dir: Path to the root directory of the dataset.
      image_size: Tuple (height, width) for resizing images.
      batch_size: Desired batch size.

  Returns:
    A tf.data.Dataset object.
  """

  def load_sample(sample_dir):
    images = []
    for filename in os.listdir(sample_dir):
      if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(sample_dir, filename)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, image_size)
        images.append(image)
    return tf.stack(images)  # stack images in one sample to [num_images, H, W, 3]

  def create_label_pairs(root_dir):
    sample_dirs = []
    labels = []
    class_dirs = sorted(os.listdir(root_dir))
    for label, class_dir in enumerate(class_dirs):
      class_path = os.path.join(root_dir, class_dir)
      if os.path.isdir(class_path):
          for sample_dir in os.listdir(class_path):
            sample_path = os.path.join(class_path, sample_dir)
            if os.path.isdir(sample_path):
              sample_dirs.append(sample_path)
              labels.append(label)
    return sample_dirs, labels

  sample_dirs, labels = create_label_pairs(root_dir)
  dataset = tf.data.Dataset.from_tensor_slices((sample_dirs, labels))

  dataset = dataset.map(lambda sample_dir, label: (load_sample(sample_dir), label),
                          num_parallel_calls=tf.data.AUTOTUNE)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset
```

This code defines `load_multiple_image_dataset_tf`, which takes the root directory, desired image size, and batch size. The `create_label_pairs` function constructs a list of sample directories along with their corresponding class labels. A separate function `load_sample` takes a sample directory and loads all images within it as a tensor. It decodes and resizes each image, and returns the stack of images. `tf.data.Dataset` is then used to efficiently load and prefetch the data. Note the crucial usage of `tf.stack` to create a tensor structure where the sample dimension represents all the images from the sample. We can now efficiently iterate through the dataset to retrieve batches that contain image stacks and the appropriate class label.

**Code Example 2: PyTorch Data Loading with `torch.utils.data.Dataset`**

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class MultipleImageDataset(Dataset):
  """PyTorch Dataset for multi-image samples."""

  def __init__(self, root_dir, image_size, transform=None):
    self.root_dir = root_dir
    self.image_size = image_size
    self.transform = transform
    self.sample_dirs = []
    self.labels = []
    class_dirs = sorted(os.listdir(root_dir))
    for label, class_dir in enumerate(class_dirs):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            for sample_dir in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_dir)
                if os.path.isdir(sample_path):
                    self.sample_dirs.append(sample_path)
                    self.labels.append(label)

  def __len__(self):
    return len(self.sample_dirs)

  def __getitem__(self, idx):
    sample_dir = self.sample_dirs[idx]
    images = []
    for filename in os.listdir(sample_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(sample_dir, filename)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

    images = torch.stack(images)
    label = self.labels[idx]
    return images, label

def load_multiple_image_dataset_pt(root_dir, image_size, batch_size):
    transform = transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
    ])

    dataset = MultipleImageDataset(root_dir, image_size, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
```

In this PyTorch version, we define a custom `MultipleImageDataset` class that inherits from `torch.utils.data.Dataset`. The dataset constructor populates lists of sample directories and their labels similar to the TensorFlow example. The `__getitem__` method, when given an index, loads all images from the corresponding directory, converts each to a tensor, stacks them, and also provides its label. The `load_multiple_image_dataset_pt` function uses `DataLoader` to create a generator for iterating through batches. It also incorporates `torchvision.transforms` for preprocessing. The returned dataloader yields batches of images and their labels.

**Code Example 3: NumPy based Data Generation and Handling**

```python
import os
import numpy as np
from PIL import Image

def load_multiple_image_dataset_np(root_dir, image_size):
  """Loads multiple images into numpy arrays with associated labels."""
  images = []
  labels = []
  class_dirs = sorted(os.listdir(root_dir))
  for label, class_dir in enumerate(class_dirs):
      class_path = os.path.join(root_dir, class_dir)
      if os.path.isdir(class_path):
          for sample_dir in os.listdir(class_path):
              sample_path = os.path.join(class_path, sample_dir)
              if os.path.isdir(sample_path):
                sample_images = []
                for filename in os.listdir(sample_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(sample_path, filename)
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize(image_size)
                        image = np.array(image)
                        sample_images.append(image)
                images.append(np.array(sample_images))
                labels.append(label)
  images = np.array(images)
  labels = np.array(labels)
  return images, labels

# Example Usage:
root_dir = "dataset"
image_size = (32,32)
images, labels = load_multiple_image_dataset_np(root_dir,image_size)

print(f"Loaded {len(images)} samples, shape of a single sample of image stack: {images[0].shape}")
print(f"Loaded {len(labels)} corresponding labels")
```

This example using NumPy takes a more direct approach to data handling. It generates the data array and labels using pure Python and NumPy. The `load_multiple_image_dataset_np` function recursively scans through the directory structure, opens, resizes and converts the images into NumPy arrays. It stacks the array representing images of one sample together and store it. Finally, it returns the complete set of image stacks as well as the associated labels. The usage is quite simple, requiring only the dataset's root directory and the desired image size. This approach would not be suitable for very large datasets given they will all be loaded into the main memory at once. However, it demonstrates the fundamental structure and can serve as a stepping stone to constructing customized data loading logic.

When working with similar multi-image datasets, several best practices apply. Experiment with different batch sizes, depending on available resources and the model's complexity. The use of preprocessing such as image augmentation, is beneficial for improving the generalization of the model. For large-scale datasets, using a more sophisticated method that uses memory mapping will be necessary. For this task, research on using `tf.data.TFRecordDataset` and `h5py` in conjunction with the above methods.

In summary, creating a multi-image dataset requires thoughtful structuring of the file system, careful management of label associations, and an understanding of data loading approaches. By applying the techniques above, along with careful management and optimization, you can generate and use complex data efficiently.
