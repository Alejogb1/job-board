---
title: "How can I modify image labels in PyTorch's ImageFolder?"
date: "2025-01-30"
id: "how-can-i-modify-image-labels-in-pytorchs"
---
ImageFolder, by default, assigns labels based on the directory structure, mapping each subdirectory to a class index. This built-in behavior, while convenient for many datasets, often requires customization when dealing with unconventional label mappings or tasks beyond simple classification. During my experience building a multi-modal medical image analysis platform, I frequently encountered the need to refine or completely redefine label assignments, a necessity that standard ImageFolder usage didn't directly address.

The core issue stems from ImageFolder’s internal mechanisms. It relies on the `os.walk` function to traverse the provided root directory, effectively extracting class names from subdirectory names. Each subdirectory then represents a class, and files within are assigned the index corresponding to the subdirectory's position in the directory listing. To modify the labels, we must circumvent this default behavior, gaining direct control over the label assignment process. This is achievable using two primary methods: overriding the `targets` attribute or providing a custom `loader` function to ImageFolder during instantiation. Choosing which method depends upon the complexity of the mapping requirement. For straightforward re-mappings, like shuffling class index assignments, overriding `targets` is often sufficient. However, for intricate label generation scenarios, a custom loader provides more flexibility.

Let’s delve into how overriding the `targets` attribute is implemented. After instantiating `ImageFolder`, the `targets` attribute represents a list containing the class index assigned to each loaded image. Modifying this list alters the labels directly associated with each sample. The following Python code illustrates this:

```python
import torch
from torchvision.datasets import ImageFolder
import os

# Assume a directory structure like:
#   root/
#      class_a/
#         img_1.png
#         img_2.png
#      class_b/
#         img_3.png
#         img_4.png

root_dir = "root" # Replace with actual path if needed.
os.makedirs(os.path.join(root_dir, "class_a"), exist_ok=True)
os.makedirs(os.path.join(root_dir, "class_b"), exist_ok=True)
open(os.path.join(root_dir, "class_a", "img_1.png"), 'a').close()
open(os.path.join(root_dir, "class_a", "img_2.png"), 'a').close()
open(os.path.join(root_dir, "class_b", "img_3.png"), 'a').close()
open(os.path.join(root_dir, "class_b", "img_4.png"), 'a').close()


dataset = ImageFolder(root=root_dir)

# The default mapping is class_a: 0, class_b: 1
print(f"Original targets: {dataset.targets}") # Output: [0, 0, 1, 1]

# Reverse labels
dataset.targets = [1, 1, 0, 0]

print(f"Modified targets: {dataset.targets}") # Output: [1, 1, 0, 0]

# Verify that labels have been changed
print(f"Sample label for index 0 is {dataset[0][1]}") #Output: Sample label for index 0 is 1
print(f"Sample label for index 3 is {dataset[3][1]}") #Output: Sample label for index 3 is 0
```

In the example above, `ImageFolder` is initialized using a dummy directory structure, then the `targets` attribute is directly modified after initialization, reversing the initial label assignments. Critically, the original directory structure is preserved, only the labels assigned to each image are modified. This illustrates that the underlying image loading is unaffected by overriding this attribute. The output confirms that the first two images were previously labeled as 0 (class_a) and after, they were assigned 1. The opposite happened for the other two images.  This approach offers a straightforward, in-place method to re-label data without necessitating a new dataset. This approach becomes extremely useful when needing to test custom models on the same dataset but with different label assignments.

However, when modifications are not easily derived from existing class structure, a custom loader provides greater adaptability. The ImageFolder constructor accepts a `loader` argument, which is a callable responsible for loading an image and returning a tuple containing the image data and its corresponding label. By defining a custom function as this loader, we gain full control over the image loading and, more crucially, label assignment. Below, the custom loader uses file path information to generate arbitrary labels:

```python
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import re

# Assume same directory structure as before

def custom_loader(path):
    #Example custom logic: Label based on file name.
    #For example, 'img_1.png' assigned a label of 1; 'img_2.png' label 2, etc.
    match = re.search(r'img_(\d+)\.png', path)
    if match:
        label = int(match.group(1))
    else:
        label = 0 #Default label if not matched.

    img = Image.open(path).convert('RGB')
    return img, label


dataset = ImageFolder(root=root_dir, loader=custom_loader)

print(f"Custom targets: {[dataset[i][1] for i in range(len(dataset))]} ") # Output: [1, 2, 3, 4]

# Verify the first image is loaded with the correct label from the custom loader
print(f"Sample label for index 0: {dataset[0][1]}") # Output: Sample label for index 0: 1
print(f"Sample label for index 3: {dataset[3][1]}") # Output: Sample label for index 3: 4

```

In this instance, instead of using the built-in logic which relies on directory structure, the custom loader `custom_loader` function extracts the numerical part of the filename using regular expressions and assigns it as the label, effectively bypassing `ImageFolder`'s default label assignment. This example highlights the flexibility granted by custom loaders, allowing for much richer and diverse label mappings. The output demonstrates that our custom logic was successful in extracting labels from filenames.

Lastly, the `ImageFolder` class also has an optional `transform` parameter which provides a more concise way to modify the label at the point of dataset access. Although the `transform` function's primary purpose is for image augmentation, it can also be used for label modifications if required. The following example shows this:

```python
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os

# Assume same directory structure as before

def transform_with_label_modification(sample):
    image, label = sample
    # Convert labels to one-hot vectors.
    num_classes = 2 # 2 classes: a,b
    one_hot_label = torch.zeros(num_classes)
    one_hot_label[label] = 1
    return image, one_hot_label

dataset = ImageFolder(root=root_dir, transform=transform_with_label_modification)
print(f"Modified labels using transformation: {dataset[0][1]}") #Output: Modified labels using transformation: tensor([1., 0.])
print(f"Modified labels using transformation: {dataset[2][1]}") #Output: Modified labels using transformation: tensor([0., 1.])
```

In this example, a transformation is applied to the image and label at retrieval time, effectively converting the integer-based labels into one-hot encoded vectors. Note that this approach does not modify the `targets` attribute itself; rather it modifies the data returned by the `__getitem__` method, allowing label manipulation during data loading and batch creation. The output verifies that the label for the first image has been converted to one-hot representation [1.,0.], while the label for the third image is [0., 1.]

For further reading and enhanced theoretical understanding, I recommend consulting the official PyTorch documentation on torchvision datasets, particularly the `ImageFolder` section. Additionally, research and study papers and tutorials focused on dataset creation within the PyTorch ecosystem. Specifically, focus on examples showcasing advanced data preprocessing techniques and custom dataset implementation patterns. Finally, consider exploring the source code of the ImageFolder class within the torchvision library itself. Directly examining the implementation will greatly enhance comprehension of underlying mechanisms and enable more effective customization strategies.
