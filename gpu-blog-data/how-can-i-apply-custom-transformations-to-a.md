---
title: "How can I apply custom transformations to a custom PyTorch dataset?"
date: "2025-01-30"
id: "how-can-i-apply-custom-transformations-to-a"
---
Custom transformations on PyTorch datasets are frequently required to tailor input data to specific model architectures or training regimes. A core understanding lies in PyTorch's `torch.utils.data.Dataset` and `torchvision.transforms` modules, which, when combined correctly, facilitate flexible and efficient data preprocessing pipelines. In my experience, a common pitfall is implementing transformations within the `Dataset`'s `__getitem__` method directly, which can lead to inefficient on-the-fly processing; it's almost always preferable to leverage `torchvision.transforms` or a custom transformation class instead.

The `torch.utils.data.Dataset` class serves as an abstract base class for creating custom datasets. When you define a custom dataset, you generally override two key methods: `__len__` which returns the size of the dataset, and `__getitem__` which retrieves a sample given an index. The goal is to return the raw data along with the corresponding label in `__getitem__`. Transformations, which should be kept separate from data retrieval, are applied *after* the raw data has been fetched but *before* it's used for training. This separation of concerns is crucial for code organization and efficient data loading, especially during multi-threaded training when you're using a `torch.utils.data.DataLoader`.

The `torchvision.transforms` module provides a range of pre-built transformation functions commonly used in image processing (e.g., resizing, cropping, normalization). These transformations can be composed using `transforms.Compose`, which allows you to apply multiple operations sequentially. When custom transformations are needed, they should be implemented as a class, inheriting from `torchvision.transforms.transforms.Transform`.

The custom transformation class must, at a minimum, implement the `__call__` method. This method takes the raw data as input and returns the transformed data. Encapsulating transformations in this way allows for modular design and easy reuse across different datasets. The `__call__` method represents the heart of the transformation, defining the specific processing to be done on the input data before it gets to the model for training.

Let’s illustrate with a few examples. Suppose I have a custom dataset of image-text pairs. First, a scenario using pre-built transformations:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_file, transform=None):
        self.image_dir = image_dir
        with open(text_file, 'r') as f:
            self.text_labels = [line.strip() for line in f]
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
      image_file = self.image_files[idx]
      image_path = os.path.join(self.image_dir, image_file)
      image = Image.open(image_path).convert('RGB')
      label = self.text_labels[idx % len(self.text_labels)] # cycle labels to ensure there are as many samples
      if self.transform:
          image = self.transform(image)
      return image, label


if __name__ == '__main__':
    # Assume we have a 'images' dir and a 'labels.txt' file for this example
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dummy_image_dir = "dummy_images"
    if not os.path.exists(dummy_image_dir):
        os.makedirs(dummy_image_dir)

    # Create dummy images
    for i in range(10):
        dummy_image = Image.new("RGB", (512,512), color = (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        dummy_image.save(os.path.join(dummy_image_dir, f"image_{i}.jpg"))
    
    with open("dummy_labels.txt", "w") as f:
        for i in range(5):
            f.write(f"Label_{i}\n")


    dataset = ImageTextDataset(image_dir=dummy_image_dir, text_file="dummy_labels.txt", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
      print(images.shape)
      print(labels)
      break # we just want to ensure it runs for a simple demo
```

In this example, I'm utilizing `transforms.Compose` to resize, crop, convert to a tensor, and normalize the images. The `ImageTextDataset` fetches the image from disk and then applies the composed transformations. This is a very common pattern for many image-based tasks.

Next, consider a scenario where you need a custom transformation:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import random

class RandomRectangle(transforms.transforms.Transform):
    def __init__(self, rectangle_size):
        self.rectangle_size = rectangle_size

    def __call__(self, img):
        draw = ImageDraw.Draw(img)
        width, height = img.size
        x1 = random.randint(0, width - self.rectangle_size)
        y1 = random.randint(0, height - self.rectangle_size)
        x2 = x1 + self.rectangle_size
        y2 = y1 + self.rectangle_size
        draw.rectangle((x1, y1, x2, y2), fill='black')
        return img


class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_file, transform=None):
        self.image_dir = image_dir
        with open(text_file, 'r') as f:
            self.text_labels = [line.strip() for line in f]
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        label = self.text_labels[idx % len(self.text_labels)]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
     # Assume we have a 'images' dir and a 'labels.txt' file for this example
    transform = transforms.Compose([
        RandomRectangle(rectangle_size=50),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dummy_image_dir = "dummy_images"
    if not os.path.exists(dummy_image_dir):
        os.makedirs(dummy_image_dir)

    # Create dummy images
    for i in range(10):
        dummy_image = Image.new("RGB", (512,512), color = (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        dummy_image.save(os.path.join(dummy_image_dir, f"image_{i}.jpg"))
    
    with open("dummy_labels.txt", "w") as f:
        for i in range(5):
            f.write(f"Label_{i}\n")

    dataset = ImageTextDataset(image_dir=dummy_image_dir, text_file="dummy_labels.txt", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
      print(images.shape)
      print(labels)
      break
```

Here, I've created a custom transformation called `RandomRectangle`, which draws a random black rectangle on the image. This transformation, encapsulated in a class and inheriting from `transforms.transforms.Transform`, works in combination with the pre-built transforms within the `transforms.Compose` object. This modular approach keeps the code clean and allows for complex transformations that might not exist natively in `torchvision`.

Finally, let’s add another transformation, this time on the textual data:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import random


class RandomRectangle(transforms.transforms.Transform):
    def __init__(self, rectangle_size):
        self.rectangle_size = rectangle_size

    def __call__(self, img):
        draw = ImageDraw.Draw(img)
        width, height = img.size
        x1 = random.randint(0, width - self.rectangle_size)
        y1 = random.randint(0, height - self.rectangle_size)
        x2 = x1 + self.rectangle_size
        y2 = y1 + self.rectangle_size
        draw.rectangle((x1, y1, x2, y2), fill='black')
        return img

class TextToTensor(transforms.transforms.Transform):
   def __init__(self, vocab):
       self.vocab = vocab

   def __call__(self, text):
       tokens = text.split()
       tensor = torch.tensor([self.vocab.get(token, 0) for token in tokens], dtype=torch.long)
       return tensor


class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_file, transform=None, text_transform=None):
        self.image_dir = image_dir
        with open(text_file, 'r') as f:
            self.text_labels = [line.strip() for line in f]
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
      image_file = self.image_files[idx]
      image_path = os.path.join(self.image_dir, image_file)
      image = Image.open(image_path).convert('RGB')
      label = self.text_labels[idx % len(self.text_labels)]

      if self.transform:
          image = self.transform(image)
      if self.text_transform:
          label = self.text_transform(label)
      return image, label



if __name__ == '__main__':
     # Assume we have a 'images' dir and a 'labels.txt' file for this example
    image_transform = transforms.Compose([
        RandomRectangle(rectangle_size=50),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    vocab = {"Label_0": 1, "Label_1": 2, "Label_2": 3, "Label_3": 4, "Label_4":5}
    text_transform = TextToTensor(vocab=vocab)

    dummy_image_dir = "dummy_images"
    if not os.path.exists(dummy_image_dir):
        os.makedirs(dummy_image_dir)

    # Create dummy images
    for i in range(10):
        dummy_image = Image.new("RGB", (512,512), color = (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        dummy_image.save(os.path.join(dummy_image_dir, f"image_{i}.jpg"))
    
    with open("dummy_labels.txt", "w") as f:
        for i in range(5):
            f.write(f"Label_{i}\n")

    dataset = ImageTextDataset(image_dir=dummy_image_dir, text_file="dummy_labels.txt", transform=image_transform, text_transform=text_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
      print(images.shape)
      print(labels)
      break
```

In this instance, I have introduced a `TextToTensor` transformation for the text data. This demonstrates that transformations are not restricted to image data and can be applied to other input modalities. It also shows how you can handle potentially disparate transform operations separately from the core dataset logic.

When implementing custom transformations, it is crucial to consider several factors. Firstly, carefully consider the order of transformations, as some transformations rely on the outputs of other transformations. Secondly, pay attention to the data types and formats being passed between transformations. For instance, most image-based transforms in `torchvision` are designed to operate on `PIL.Image` objects, or PyTorch tensors. Thirdly, make sure you keep all transformations and dataset logic in the CPU as much as possible, then transfer it all to the GPU within the model training loop to maximize data loading and GPU training performance, leveraging `pin_memory` on the `DataLoader` where possible.

For further learning, I would recommend consulting resources that discuss advanced techniques for data loading, specifically focusing on topics like data augmentation and normalization best practices. Articles that cover different types of dataset classes can also offer useful insight, such as those found in PyTorch’s documentation. Studying the implementations of popular transformation libraries can also teach you the proper ways to create and compose custom transformations for your data-centric tasks.
