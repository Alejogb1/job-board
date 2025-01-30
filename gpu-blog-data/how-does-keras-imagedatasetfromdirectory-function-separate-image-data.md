---
title: "How does Keras' image_dataset_from_directory() function separate image data from labels?"
date: "2025-01-30"
id: "how-does-keras-imagedatasetfromdirectory-function-separate-image-data"
---
Keras’ `image_dataset_from_directory()` function elegantly handles the common task of loading image datasets organized in a directory structure, implicitly separating image data from their corresponding labels based primarily on the directory organization. I've extensively used this function in various machine learning projects involving image classification, and have come to appreciate its streamlined design. The function leverages a convention where the subdirectory names directly correspond to the class labels.

Fundamentally, the separation mechanism relies on the hierarchical structure of the file system, not any metadata embedded within the image files themselves. When you call `image_dataset_from_directory()`, it scans the specified directory. Inside that directory, it expects to find subdirectories. Each of these subdirectories is interpreted as representing a distinct class. The images residing within each subdirectory are considered samples belonging to that particular class. The labels are therefore derived from the names of these subdirectories. For example, a directory structure such as:

```
dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.png
│   └── cat3.jpeg
└── dogs/
    ├── dog1.jpg
    ├── dog2.png
    └── dog3.jpeg
```

will be interpreted as having two classes: `cats` and `dogs`. Images like `cat1.jpg`, `cat2.png`, and `cat3.jpeg` would be grouped under the label `cats`, while `dog1.jpg`, `dog2.png`, and `dog3.jpeg` would belong to the label `dogs`. Internally, the function maps each of these subdirectory names to an integer, creating a one-hot encoded label matrix when labels are required. This encoding is determined by the order in which the subdirectory names are encountered during the directory traversal. This is important when later interpreting model predictions, as those predictions will output probabilities based on the integer labels assigned. While it is often sufficient in many situations, it is always beneficial to keep a copy of your labels in their string format.

The `labels` argument can modify this behavior. When set to "inferred," which is the default, the directory structure defines the labels. Alternatively, you could specify `labels="int"`, which still uses the subdirectory structure but assigns integer labels to the classes numerically based on alphabetical ordering of the subdirectory names. The `label_mode` argument plays a crucial role here. Setting it to `"int"` results in integer labels for each class instance. Setting it to `"categorical"` encodes the labels into one-hot format. Setting it to `"binary"` assumes exactly two classes, which is very helpful to ensure the label format is as required for two-class classifiers. Setting it to `None` generates only the image batches without associated labels. Lastly, the `class_names` argument, lets you use customized ordering of labels beyond the lexicographical ordering of subdirectory names. This allows you to map subdirectory names to numerical labels in any desired order which is useful if your directories aren't in the desired alphabetical format.

The actual processing involves the following steps: Initially, `image_dataset_from_directory()` validates the existence of the provided directory. Then, it iterates over the subdirectories, ensuring only subdirectories are considered class labels, not files. For each image within a class folder, it reads and decodes it using the provided image size and color channel parameters, applying any transformations as specified. Finally, it creates a `tf.data.Dataset` object, which is a standard tensor format that Keras uses for model training and evaluation. Each element in this dataset is a batch containing image tensors as input and their corresponding encoded labels. This avoids needing to load the entire dataset into memory at once, which is a significant advantage when working with large datasets.

Here are three code examples illustrating these concepts:

**Example 1: Basic Usage with Inferred Labels (Default)**

```python
import tensorflow as tf
from tensorflow import keras
import os

# Assume a directory structure like 'dataset/cats/cat1.jpg' and 'dataset/dogs/dog1.jpg'
dataset_path = 'dataset'

# Create dummy directories and images
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    os.makedirs(os.path.join(dataset_path, 'cats'))
    os.makedirs(os.path.join(dataset_path, 'dogs'))
    open(os.path.join(dataset_path, 'cats', 'cat1.jpg'), 'a').close()
    open(os.path.join(dataset_path, 'dogs', 'dog1.jpg'), 'a').close()

dataset = keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(180, 180),
    batch_size=32,
    label_mode="int"
)

# Verify label shape and type in the dataset
for images, labels in dataset.take(1):
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Label batch data type: {labels.dtype}")

# Clean up dummy directories and images
import shutil
shutil.rmtree(dataset_path)
```

This first example demonstrates the basic usage. It automatically infers the class labels `cats` and `dogs` from the directory names and outputs integer class labels which are useful when building classifiers.

**Example 2: Using Specified `class_names`**

```python
import tensorflow as tf
from tensorflow import keras
import os

# Assume a directory structure like 'dataset/class_a/image1.jpg' and 'dataset/class_b/image2.jpg'
dataset_path = 'dataset'

# Create dummy directories and images
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    os.makedirs(os.path.join(dataset_path, 'class_a'))
    os.makedirs(os.path.join(dataset_path, 'class_b'))
    open(os.path.join(dataset_path, 'class_a', 'image1.jpg'), 'a').close()
    open(os.path.join(dataset_path, 'class_b', 'image2.jpg'), 'a').close()

dataset = keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(180, 180),
    batch_size=32,
    label_mode="int",
    class_names=['class_b', 'class_a']
)

# Verify the assigned numerical labels
for images, labels in dataset.take(1):
    print(f"First label in batch: {labels[0]}")

# Clean up dummy directories and images
import shutil
shutil.rmtree(dataset_path)
```

Here, the subdirectory names, `class_a` and `class_b`, are mapped to integer labels, but their numerical order (which is usually alphabetic), is swapped by the explicit `class_names` argument to `image_dataset_from_directory()`. This is important if you need to associate classes to specific numerical encodings for downstream model training or interpretation purposes. If class_a comes before class_b, without the specification of class\_names, class\_a would be mapped to 0, and class\_b would be mapped to 1. In this case, we swap them via `class_names`, therefore class\_b is 0 and class\_a is 1.

**Example 3: `label_mode` set to "categorical"**

```python
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np


# Assume a directory structure like 'dataset/class_1/image_1.jpg', 'dataset/class_2/image_2.jpg', 'dataset/class_3/image_3.jpg'
dataset_path = 'dataset'

# Create dummy directories and images
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    os.makedirs(os.path.join(dataset_path, 'class_1'))
    os.makedirs(os.path.join(dataset_path, 'class_2'))
    os.makedirs(os.path.join(dataset_path, 'class_3'))
    open(os.path.join(dataset_path, 'class_1', 'image_1.jpg'), 'a').close()
    open(os.path.join(dataset_path, 'class_2', 'image_2.jpg'), 'a').close()
    open(os.path.join(dataset_path, 'class_3', 'image_3.jpg'), 'a').close()

dataset = keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(180, 180),
    batch_size=32,
    label_mode="categorical"
)

# Verify the labels are one-hot encoded
for images, labels in dataset.take(1):
    print(f"Label batch shape: {labels.shape}")
    print(f"First label in batch: {labels[0]}")


# Clean up dummy directories and images
import shutil
shutil.rmtree(dataset_path)
```

This third example showcases the usage of the `label_mode` argument set to `"categorical"`. This causes the labels to be transformed into a one-hot encoded format. For three classes, each label will be a vector of length 3. The advantage of this format is that it is well suited for training multi-class classifiers.

For further exploration and a deep dive into the usage of `image_dataset_from_directory`, consider reviewing the official Keras documentation, TensorFlow’s data API documentation, which explains `tf.data.Dataset` objects, and consult any good text on machine learning image processing with Keras, which often have well-detailed examples. These resources are extremely valuable for understanding the underlying mechanisms. Moreover, exploring tutorials focusing on data loading in Keras can provide valuable context. This is because the function has a lot of functionality and you may require more advanced use cases such as loading from a database, pre-processing of images, or more sophisticated training regimes.
