---
title: "How to use tf.keras.utils.Sequence with `model.fit()` in TensorFlow 2?"
date: "2025-01-30"
id: "how-to-use-tfkerasutilssequence-with-modelfit-in-tensorflow"
---
Implementing `tf.keras.utils.Sequence` for feeding data to `model.fit()` in TensorFlow 2 offers a memory-efficient alternative to loading entire datasets into memory, particularly when dealing with large-scale or dynamically generated data. The core benefit derives from its ability to generate batches of data on-demand, thus circumventing memory limitations prevalent when working with datasets that exceed available RAM. This approach is critical for large image datasets, text corpora, and other scenarios where preloading is impractical.

The `tf.keras.utils.Sequence` class provides a structure for implementing a data generator compatible with Keras model training. It demands the implementation of two core methods: `__len__`, which dictates the number of batches per epoch, and `__getitem__`, which retrieves a specific batch of data. I've relied heavily on this approach in projects involving real-time data augmentation and complex synthetic data generation, which are too large to hold in memory simultaneously.

To illustrate, I'll dissect the implementation using three distinct code examples. These examples progressively introduce complexity and showcase different aspects of how `Sequence` can be leveraged effectively.

**Example 1: Simple Numerical Data Generation**

This initial example demonstrates the core mechanics of `Sequence` with a straightforward numerical dataset. Imagine a scenario where each data point is a pair of random numbers, and the goal is a trivial regression. Although a simplified use case, it clarifies the foundational concepts.

```python
import tensorflow as tf
import numpy as np

class SimpleSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, num_samples):
        self.batch_size = batch_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = (idx + 1) * self.batch_size
        inputs = np.random.rand(self.batch_size, 2)
        targets = np.sum(inputs, axis=1, keepdims=True)
        return inputs, targets


# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,))
])
model.compile(optimizer='adam', loss='mse')

# Sequence instantiation
batch_size = 32
num_samples = 1000
data_sequence = SimpleSequence(batch_size, num_samples)

# Model training
model.fit(data_sequence, epochs=5)
```

In this example, `SimpleSequence` generates random input pairs and their sums as targets. The `__len__` method defines the number of batches, calculated by integer division, meaning any remaining samples are effectively ignored. The `__getitem__` method uses NumPy for the generation of a batch based on an index. Importantly, the model can consume this `Sequence` directly via `model.fit()`. There is no need to provide data in a single tensor.

**Example 2: Image Data with Labels**

This example expands on the prior implementation by showing how to work with image data and labels, which is a far more typical use case. Here, the data will not be randomly generated, but assumed to be a dataset where each image has a file path. I’ll avoid using actual filepaths in this example to focus on the core mechanics of the class.

```python
import tensorflow as tf
import numpy as np
import os  # Added to simulate a file list

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, image_size=(64, 64)):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = min((idx + 1) * self.batch_size, len(self.file_paths))
        batch_paths = self.file_paths[start_index:end_index]
        batch_labels = self.labels[start_index:end_index]

        batch_images = []
        for _ in batch_paths:  # Replace path with actual image loading here
            image = np.random.rand(self.image_size[0], self.image_size[1], 3)  # Simulated image
            batch_images.append(image)

        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        return batch_images, batch_labels

# Model definition (simplified for example)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generate sample data (simulated)
num_images = 200
file_paths = [f"image_{i}.png" for i in range(num_images)]  # Filepaths are only symbolic here
labels = np.random.randint(0, 10, num_images)
batch_size = 32
image_sequence = ImageSequence(file_paths, labels, batch_size)

# Model training
model.fit(image_sequence, epochs=5)

```

Here, `ImageSequence` takes file paths and labels as input. Critically, the loop in `__getitem__` where images would be loaded is replaced by a simulated data generation using `np.random.rand`. This is for demonstration only; In a real application, the correct image loading mechanism would need to be implemented within the loop, typically using libraries such as `PIL` or `cv2`. I have also switched to the ceil function in `__len__` to ensure no remaining files are left over after the divisions.

**Example 3: Sequence with Data Augmentation**

This final example integrates data augmentation within the `Sequence` class. Augmentation often involves applying random transformations to each batch, such as rotations, flips, and color adjustments. Generating augmented images on the fly prevents storing multiple versions of the same image.

```python
import tensorflow as tf
import numpy as np
import random # For random augmentation choices

class AugmentedImageSequence(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, image_size=(64, 64)):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = min((idx + 1) * self.batch_size, len(self.file_paths))
        batch_paths = self.file_paths[start_index:end_index]
        batch_labels = self.labels[start_index:end_index]

        batch_images = []
        for _ in batch_paths: # Load actual images here
            image = np.random.rand(self.image_size[0], self.image_size[1], 3) # Simulated loading
            # Begin augmentation block
            if random.random() > 0.5:
                image = np.fliplr(image)
            if random.random() > 0.5:
                angle = random.randint(-15,15)
                rows, cols, _ = image.shape
                rotation_matrix = np.float32([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                                          [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0]])
                image = tf.keras.preprocessing.image.apply_affine_transform(image, rotation_matrix[:2, :2],
                                                           row_axis=0, col_axis=1, channel_axis=2,
                                                          fill_mode='nearest', cval=0.0)
            batch_images.append(image)

        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        return batch_images, batch_labels


# Model definition remains the same as example 2
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data generation and instantiation as example 2
num_images = 200
file_paths = [f"image_{i}.png" for i in range(num_images)]
labels = np.random.randint(0, 10, num_images)
batch_size = 32
image_sequence = AugmentedImageSequence(file_paths, labels, batch_size)

# Model training
model.fit(image_sequence, epochs=5)
```

In this version, random horizontal flips and rotations are applied to each image using `random.random()`. The key here is that the augmentation is happening within `__getitem__`, and thus is performed every time a batch is requested during training. For simplicity and readability, the example includes relatively crude transformations, and more sophisticated augmentation methods could be applied. The actual `tf.keras.preprocessing.image` tools are used here, although they could be replaced with the user's preference of augmentation tools.

Through this sequence of examples, the `tf.keras.utils.Sequence` class's utility for handling dynamically generated or large datasets becomes evident. This structured approach to batching is crucial for efficient training, especially in scenarios where dataset size would otherwise cause memory overload.

For further exploration, I would recommend studying TensorFlow’s official documentation on custom data loading, specifically around `tf.data.Dataset` as it offers another powerful tool, and researching image loading using `PIL` or `cv2` for a robust data pipeline when working with images. Investigating the implementation of common augmentation strategies would also benefit any project utilizing large or custom datasets. Finally, carefully consider resource management within your sequence methods, particularly in multi-threaded environments.
