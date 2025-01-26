---
title: "How can Python generators be used to feed data to a neural network?"
date: "2025-01-26"
id: "how-can-python-generators-be-used-to-feed-data-to-a-neural-network"
---

Python generators offer a memory-efficient strategy for providing data to neural networks, particularly when dealing with datasets that exceed available RAM. Instead of loading an entire dataset into memory, a generator yields data batches on demand, enabling the training of deep learning models with large-scale inputs. This technique, known as data streaming, is crucial for optimizing training performance and preventing resource exhaustion.

The core advantage of a generator lies in its "lazy evaluation." Rather than computing and storing a complete sequence, a generator produces each element only when requested, using the `yield` keyword. This behavior contrasts sharply with lists or other iterable containers that require pre-computation and memory allocation for all their elements at once. For neural network training, where mini-batches of data are processed sequentially, this "on-demand" data delivery proves immensely beneficial.

Typically, in a neural network training loop, we iterate over a dataset multiple times (epochs). Each epoch involves processing the data in batches. A standard approach involves splitting the dataset into batches and loading each batch sequentially. If the dataset is vast, this can lead to significant memory constraints. Instead, a generator can produce each batch on the fly, mitigating the need to hold the entire dataset in memory at any single time. Furthermore, generators can incorporate data augmentation steps, preprocessing steps and even complex data loading logic directly into the data stream, promoting code modularity and maintainability.

Here's how I've practically applied generators within my neural network training workflows.

**Code Example 1: Basic Image Batching with a Generator**

This example demonstrates the core concept of a generator yielding mini-batches of image data from a hypothetical directory structure.

```python
import os
import numpy as np
from PIL import Image

def image_batch_generator(image_dir, batch_size, image_size):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    while True:
        np.random.shuffle(image_files) # Shuffle images each epoch
        for i in range(0, num_images, batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            for file in batch_files:
                image_path = os.path.join(image_dir, file)
                try:
                     image = Image.open(image_path).resize(image_size)
                     image = np.array(image, dtype=np.float32)/255.0
                     batch_images.append(image)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
            if batch_images:
                yield np.array(batch_images)
```

*Commentary:* This function `image_batch_generator` takes a directory path, a batch size, and the desired image dimensions as input. It begins by listing image files within the specified directory. The infinite `while True` loop ensures the generator does not exhaust and instead shuffles and iterates through files again. For each iteration, it takes a slice from the shuffled list of files, forming the mini-batch. It then opens each file as a PIL Image, resizes it, converts it to a numpy array normalized to [0,1] range, and appends it to the batch. Crucially, the batch of images is yielded using `yield`, which allows the calling function to iterate over the data without having to load all images in memory upfront. Error handling is included to account for corrupted files, which is often encountered in real-world image datasets.

**Code Example 2: Generators with Labels and Data Augmentation**

This example extends the previous one by incorporating label loading and basic data augmentation, using simple horizontal flips.

```python
import os
import numpy as np
from PIL import Image
import random

def image_batch_generator_augmented(image_dir, label_mapping, batch_size, image_size):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    while True:
        np.random.shuffle(image_files)
        for i in range(0, num_images, batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            batch_labels = []
            for file in batch_files:
                image_path = os.path.join(image_dir, file)
                label = label_mapping.get(file, None)
                if label is None:
                   print(f"Warning: No label found for image {file}")
                   continue

                try:
                    image = Image.open(image_path).resize(image_size)
                    image = np.array(image, dtype=np.float32)/255.0

                    # Simple augmentation: horizontal flip
                    if random.random() > 0.5:
                        image = np.fliplr(image)

                    batch_images.append(image)
                    batch_labels.append(label)
                except Exception as e:
                    print(f"Error loading or processing image {image_path}: {e}")

            if batch_images:
                yield np.array(batch_images), np.array(batch_labels)
```

*Commentary:*  The `image_batch_generator_augmented` function takes an additional parameter, `label_mapping`, assumed to be a dictionary mapping image filenames to corresponding class labels. For each image, this function retrieves the label using the filename as key. A simple augmentation step involving a horizontal flip with 50% probability is added. The function now yields a tuple containing the batch of images and their corresponding labels, demonstrating how to incorporate both data inputs and targets for supervised learning. The function still includes error handling to gracefully manage files which might be missing labels.

**Code Example 3: Using Generators with Keras/TensorFlow**

This example highlights a very common use case, integrating a generator with a Keras model, making use of TensorFlow's `tf.data.Dataset.from_generator` for efficient data pipelining.

```python
import tensorflow as tf
import numpy as np


def generate_sample_data(num_samples, image_size=(64, 64), num_classes=10):
    images = np.random.rand(num_samples, image_size[0], image_size[1], 3).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    return images, labels


def batch_generator(batch_size, images, labels):
    num_samples = images.shape[0]
    while True:
      for i in range(0,num_samples, batch_size):
           batch_images = images[i:i+batch_size]
           batch_labels = labels[i:i+batch_size]
           yield batch_images, batch_labels


def train_model_with_generator(num_samples, batch_size, epochs, image_size = (64, 64), num_classes = 10):
   images, labels = generate_sample_data(num_samples, image_size, num_classes)
   gen = batch_generator(batch_size, images, labels)

   input_shape = (image_size[0], image_size[1],3)
   model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ])
   model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

   ds = tf.data.Dataset.from_generator(lambda: gen, output_signature=(
    tf.TensorSpec(shape=(None, image_size[0], image_size[1],3), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)))
   model.fit(ds, epochs = epochs, steps_per_epoch = num_samples//batch_size)
```

*Commentary:* This example integrates the batch generator (simplified for clarity, since in a real world scenario this would be reading from files, using labels etc.) with a simple keras model. First, sample data is generated along with labels, to simulate realistic image and labels. The `tf.data.Dataset.from_generator` function turns our Python generator `batch_generator` into a TensorFlow Dataset. This is essential for efficient integration with Keras models as it allows the use of TensorFlow's highly optimized data pipelines, which handles data prefetching and caching transparently. The generator's output signature, specifying the expected shape and data types, is passed to `from_generator`. The Keras model's training loop now uses the `tf.data.Dataset` object and processes batches using steps_per_epoch specified by the dataset length and batch size. This implementation shows how a generator can be effectively incorporated into a modern deep learning framework.

**Resource Recommendations:**

For more detailed knowledge, I recommend exploring the documentation for Python’s `yield` keyword and generator expressions. Additionally, resources dedicated to TensorFlow's `tf.data` module, especially the `tf.data.Dataset.from_generator` method, are highly useful. It is also beneficial to review tutorials on data loading and preprocessing techniques for machine learning tasks as they often encompass generator-based solutions. Deep learning resources frequently cover this topic as well since it is fundamental for training with large datasets.
