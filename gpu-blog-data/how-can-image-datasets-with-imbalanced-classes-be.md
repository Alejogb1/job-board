---
title: "How can image datasets with imbalanced classes be balanced?"
date: "2025-01-30"
id: "how-can-image-datasets-with-imbalanced-classes-be"
---
Dealing with imbalanced image datasets is a recurring challenge in computer vision projects. Specifically, when one or more classes significantly outnumber others, standard machine learning models tend to exhibit a bias towards the majority classes, resulting in poor classification performance on the minority classes. My experience across several projects – from defect detection in manufacturing to rare disease identification in medical imaging – has repeatedly demonstrated the critical need for effective balancing techniques to mitigate this issue.

The core problem stems from the loss function used during model training. When presented with a disproportionately large number of examples from a majority class, the gradient updates will primarily optimize for that class, leading the model to become overly sensitive to its features and under-sensitive to the minority classes. Consequently, the model’s decision boundary becomes skewed, producing inaccurate predictions on instances from the underrepresented categories. Effective balancing strategies aim to counteract this tendency.

I’ve found that balancing can be broadly approached in two primary ways: data-level techniques and algorithm-level techniques. Data-level techniques modify the training dataset to create a more balanced representation, whereas algorithm-level techniques modify the learning algorithm itself to accommodate the imbalance. These strategies are not mutually exclusive and are often employed in combination to achieve optimal results.

Data-level approaches usually involve either oversampling minority classes or undersampling majority classes, or a combination of both. Oversampling generates new synthetic examples of minority classes, while undersampling involves discarding examples from the majority classes. Which method, or combination thereof, is most effective is often problem-specific. I tend to start with oversampling as it retains all the original data.

Here is a code example utilizing simple random oversampling using Python with NumPy and Pillow for handling image loading:

```python
import os
import numpy as np
from PIL import Image
import random

def oversample_images(image_dir, output_dir, class_to_oversample, oversample_factor):
    """Oversamples a specified class in an image dataset.

    Args:
        image_dir (str): The root directory containing image subfolders by class.
        output_dir (str): The output directory to store oversampled images.
        class_to_oversample (str): The subdirectory name of the class to oversample.
        oversample_factor (int): The number of times to replicate images (integer, minimum 1).
    """
    class_dir = os.path.join(image_dir, class_to_oversample)
    if not os.path.exists(class_dir):
        raise ValueError(f"Class directory not found: {class_dir}")

    output_class_dir = os.path.join(output_dir, class_to_oversample)
    os.makedirs(output_class_dir, exist_ok=True)


    image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    if not image_paths:
        raise ValueError(f"No images found in the directory: {class_dir}")


    for i, path in enumerate(image_paths):
      image = Image.open(path)
      for j in range(oversample_factor):
        new_filename = f"{os.path.splitext(os.path.basename(path))[0]}_{j}.{os.path.splitext(os.path.basename(path))[1]}"
        new_path = os.path.join(output_class_dir, new_filename)
        image.save(new_path)


# Example Usage:
# oversample_images("images/", "oversampled_images", "class_a", 3)
```
This function loads images from the specified class within `image_dir`, replicates them according to the `oversample_factor`, and saves the augmented versions to `output_dir`. The new filenames are appended with a number to differentiate the copies. This simple oversampling method can increase the overall size of your dataset considerably if `oversample_factor` is large and should be used with careful consideration.  Note that simply replicating the original images may not always be ideal; it can cause the model to overfit to the few unique examples.

A more sophisticated oversampling technique is to utilize image augmentation during the replication step, effectively increasing the diversity of the generated images. This involves applying transformations like rotations, flips, crops, and color adjustments. Here is a modification to the above function which incorporates random image rotations:

```python
import os
import numpy as np
from PIL import Image
import random

def oversample_images_augmented(image_dir, output_dir, class_to_oversample, oversample_factor):
    """Oversamples a specified class in an image dataset, using rotations as augmentation.

    Args:
        image_dir (str): The root directory containing image subfolders by class.
        output_dir (str): The output directory to store oversampled images.
        class_to_oversample (str): The subdirectory name of the class to oversample.
        oversample_factor (int): The number of times to replicate images (integer, minimum 1).
    """
    class_dir = os.path.join(image_dir, class_to_oversample)
    if not os.path.exists(class_dir):
        raise ValueError(f"Class directory not found: {class_dir}")

    output_class_dir = os.path.join(output_dir, class_to_oversample)
    os.makedirs(output_class_dir, exist_ok=True)

    image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    if not image_paths:
        raise ValueError(f"No images found in the directory: {class_dir}")


    for i, path in enumerate(image_paths):
        image = Image.open(path)
        for j in range(oversample_factor):
           new_filename = f"{os.path.splitext(os.path.basename(path))[0]}_{j}.{os.path.splitext(os.path.basename(path))[1]}"
           new_path = os.path.join(output_class_dir, new_filename)
           angle = random.uniform(-25,25)
           rotated_image = image.rotate(angle)
           rotated_image.save(new_path)

# Example Usage:
# oversample_images_augmented("images/", "augmented_images", "class_a", 3)
```
This modified function now applies random rotations (between -25 to 25 degrees) to each image during the oversampling process. This leads to a more diverse training set and helps to reduce overfitting. Additional transformations can be easily added into the augmentation loop as needed.

At the algorithm-level, techniques such as class weighting are effective. Class weighting adjusts the loss function such that misclassifications of the minority classes are penalized more heavily than misclassifications of the majority classes. This is done by assigning a weight to each class based on its inverse frequency in the dataset.

Here's how one might implement class weights using PyTorch:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

# Assume we have a custom Dataset subclass called MyImageDataset which yields (image, label) tuples
# and a pre-trained model called MyModel which uses cross entropy loss

class MyImageDataset(Dataset): # This example only covers labels, image loading left to the user
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
         # Load images here given the index in self.labels
         return  torch.rand(3, 128, 128) , torch.tensor(self.labels[idx])

def get_class_weights(labels):
   class_counts = np.bincount(labels)
   total_samples = len(labels)
   weights = total_samples / (len(class_counts) * class_counts)
   return torch.tensor(weights, dtype=torch.float)


# Example usage:
labels = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2]
dataset = MyImageDataset(labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Assume an existing MyModel class and model object
model = nn.Linear(10,3) # Replace with your model class

# Calculate class weights using the dataset
class_weights = get_class_weights(labels)

# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters())

# Training loop (simplified)
for epoch in range(5):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images.view(images.size(0), -1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} completed")
```

This code snippet demonstrates how class weights can be calculated based on the inverse class frequencies. In this example, the `get_class_weights` calculates those weights based on a label array.  These weights are then used to parameterize the `nn.CrossEntropyLoss` function, and thus influence the impact of each class on gradient updates during training. The model is a simple linear example to clarify how the weights and loss function are implemented.

Effective balancing is not a one-size-fits-all problem, and experimentation with various techniques is crucial. I often start by calculating the class distributions and then apply combinations of simple oversampling with class weights, followed by more complex oversampling if required. Undersampling can also be valuable if computational resources are limited.

For further reading on balancing techniques I would suggest exploring research literature on the topic of imbalance learning. Publications often delve deeper into more sophisticated oversampling techniques such as SMOTE (Synthetic Minority Oversampling Technique) and its variants, as well as advanced undersampling techniques. Additionally, resources related to cost-sensitive learning delve into more complex ways to modify the loss function for imbalanced datasets. Finally, there are several good introductory textbooks on machine learning that offer clear explanations of common imbalanced dataset problems, along with basic strategies to resolve them.
