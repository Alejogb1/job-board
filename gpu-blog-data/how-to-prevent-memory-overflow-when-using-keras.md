---
title: "How to prevent memory overflow when using Keras' Sequence class?"
date: "2025-01-30"
id: "how-to-prevent-memory-overflow-when-using-keras"
---
Keras' `Sequence` class provides a memory-efficient way to handle large datasets during deep learning training, addressing a common challenge when data exceeds available RAM. The core issue arises from how the default data loading mechanism of many Keras model fitting operations functions; it attempts to load the entire dataset into memory at once. `Sequence` circumvents this by enabling data loading in batches, on-demand, during training. My experience working on a large-scale image classification project revealed just how essential this pattern becomes when dealing with hundreds of thousands of high-resolution images. I’ll describe this further, along with practical coding examples.

The `Sequence` class, itself, is an abstract base class provided by Keras; to use it effectively, one must inherit from it and implement two key methods: `__len__()` and `__getitem__()`. The `__len__()` method must return the number of batches, not the number of samples, within the sequence. The `__getitem__(idx)` method is responsible for generating and returning a single batch of data associated with the provided batch index `idx`. Keras calls these methods during training to obtain the data batches, preventing the system from attempting to load the entire dataset into memory. This lazy-loading mechanism is critical for avoiding memory overflows. The key is to implement these methods so they efficiently load and process only the required data batch.

The primary cause of memory overflow issues with data loading stems from attempting to perform operations on the entire dataset simultaneously, often while using `numpy` arrays. When this happens, memory demands increase exponentially alongside the dataset size. The `Sequence` class avoids this by treating the dataset as an iterable of batches, allowing for more granular control over the loading process, particularly useful when data must be loaded from files, databases, or a remote location.

Here are some specific code examples, built around my work with medical image datasets. Let's assume we have a collection of medical image files along with their associated ground truth labels. Each image resides as a single `.png` file, and the file names are stored in a list. I'll provide three different implementation approaches to exemplify several techniques that address potential issues and improve efficiency.

**Example 1: Basic Sequence implementation**

This illustrates the core functionality of a `Sequence` subclass. It reads image file paths and labels, then constructs a batch of images, represented as a `numpy` array.

```python
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

class ImageSequence(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, target_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
      return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]

        images = []
        for image_path in batch_paths:
            img = tf.keras.utils.load_img(image_path, target_size = self.target_size, color_mode = "grayscale")
            img_array = tf.keras.utils.img_to_array(img)
            images.append(img_array)

        return np.array(images), np.array(batch_labels)

# Example Usage:
image_files = [f for f in os.listdir("images") if f.endswith(".png")]
image_paths = ["images/" + i for i in image_files]
labels = np.random.randint(0, 2, len(image_files))
BATCH_SIZE = 32
TARGET_SIZE = (256,256)
sequence = ImageSequence(image_paths, labels, BATCH_SIZE, TARGET_SIZE)

# Model Fitting:
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(sequence, steps_per_epoch=len(sequence), epochs=3)
```
This basic example defines an image loader that iterates through a list of image paths. The `__getitem__` method loads an image, converts it to an array, and constructs batches. `steps_per_epoch` has been explicitly provided for demonstration. In practice, during regular fitting via the `fit()` method on `keras` models, providing `steps_per_epoch` is optional. When not provided, its value is derived from the `__len__()` method.

**Example 2: Adding Data Augmentation**

This adds image augmentation to demonstrate how transformations can be integrated while maintaining memory efficiency, performed only as data is loaded.

```python
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers

class AugmentingImageSequence(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, target_size, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment

        self.augmentation_layers = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))
        ])

    def __len__(self):
      return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]

        images = []
        for image_path in batch_paths:
            img = tf.keras.utils.load_img(image_path, target_size=self.target_size, color_mode="grayscale")
            img_array = tf.keras.utils.img_to_array(img)

            if self.augment:
                 img_array = self.augmentation_layers(img_array[tf.newaxis, ...]) # Expand dims for layer
                 img_array = tf.squeeze(img_array, axis=0) # Remove added dims

            images.append(img_array)

        return np.array(images), np.array(batch_labels)

# Example Usage:
image_files = [f for f in os.listdir("images") if f.endswith(".png")]
image_paths = ["images/" + i for i in image_files]
labels = np.random.randint(0, 2, len(image_files))
BATCH_SIZE = 32
TARGET_SIZE = (256,256)
sequence = AugmentingImageSequence(image_paths, labels, BATCH_SIZE, TARGET_SIZE, augment=True)

# Model Fitting:
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(sequence, steps_per_epoch=len(sequence), epochs=3)
```
This extended version of the sequence class demonstrates the application of on-the-fly data augmentation.  The core logic remains the same, but we create a set of augmentation layers, using the `tf.keras.layers` API, that apply rotation, zoom, and flipping to the images before being used for training. This ensures that no augmented images are stored persistently; instead, augmentations are applied in memory as needed, eliminating the memory footprint of pre-computed augmentations. The layers are provided with `tf.newaxis` to expand their dimensionality to fit the `tf.keras.Sequential` API requirements before it is squashed back to its original shape for returning to the `images` list.

**Example 3: Preprocessing within Sequence**

This example further integrates more preprocessing steps, for instance, normalizing pixel values, illustrating a method to integrate dataset-specific preparation steps.

```python
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers

class PreprocessingImageSequence(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, target_size, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment

        self.augmentation_layers = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))
        ])

    def __len__(self):
      return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]

        images = []
        for image_path in batch_paths:
            img = tf.keras.utils.load_img(image_path, target_size=self.target_size, color_mode = "grayscale")
            img_array = tf.keras.utils.img_to_array(img)
            img_array = img_array / 255.0  # Normalize the pixels

            if self.augment:
                img_array = self.augmentation_layers(img_array[tf.newaxis, ...]) # Expand dims for layer
                img_array = tf.squeeze(img_array, axis=0) # Remove added dims

            images.append(img_array)

        return np.array(images), np.array(batch_labels)

# Example Usage:
image_files = [f for f in os.listdir("images") if f.endswith(".png")]
image_paths = ["images/" + i for i in image_files]
labels = np.random.randint(0, 2, len(image_files))
BATCH_SIZE = 32
TARGET_SIZE = (256,256)
sequence = PreprocessingImageSequence(image_paths, labels, BATCH_SIZE, TARGET_SIZE, augment = True)

# Model Fitting:
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(sequence, steps_per_epoch=len(sequence), epochs=3)
```
The major change here is the line `img_array = img_array / 255.0`, where images are normalized to the range [0,1]. This is a common operation in image preprocessing, and by integrating it within the `Sequence` class, it is performed on batches of data as they are generated during training. This approach avoids processing the entire dataset in advance, thereby reducing initial memory consumption.

For further exploration, I suggest consulting the Keras API documentation, particularly the information on `keras.utils.Sequence` and `tf.data.Dataset`. Additionally, research into techniques like custom data loading and preprocessing, optimized loading strategies for different file types, and how data augmentation can be applied effectively. Understanding these concepts is essential for managing memory effectively when training large models with sizable datasets.
