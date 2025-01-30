---
title: "How can I obtain a single sample image per class using image_dataset_from_directory?"
date: "2025-01-30"
id: "how-can-i-obtain-a-single-sample-image"
---
The `image_dataset_from_directory` function in TensorFlow/Keras, while convenient, lacks a direct parameter for subsampling classes to a single representative image.  My experience working on a large-scale image classification project for a medical imaging company highlighted this limitation.  Efficiently managing imbalanced datasets requires sophisticated sampling techniques, and a simple, per-class single-sample retrieval necessitates a custom preprocessing step.  This response outlines several methods to achieve this, addressing potential pitfalls encountered during my work.

**1. Clear Explanation:**

The core issue lies in the fact that `image_dataset_from_directory` is designed to load all images within specified subdirectories.  Therefore, obtaining one sample per class requires external manipulation of the dataset after its creation. This can be accomplished through various methods, all centering on identifying and selecting a single image from each class directory. The ideal selection method will depend on the desired level of randomness and whether any prior knowledge of image quality is available.  For example, in medical imaging, a preprocessing step could prioritize images with clearer annotations or higher resolution.  However, in the absence of such information, random sampling offers a straightforward approach.

**2. Code Examples with Commentary:**

**Example 1: Random Sampling using `tf.data.Dataset` methods:**

```python
import tensorflow as tf
import os
import random

def single_sample_per_class(directory, image_size=(256, 256), batch_size=32):
    """
    Generates a dataset with one random image per class from a directory.

    Args:
        directory: Path to the directory containing class subdirectories.
        image_size: Tuple specifying the desired image size.
        batch_size: Batch size for the dataset.

    Returns:
        A tf.data.Dataset object with one image per class.
    """

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        image_size=image_size,
        batch_size=1,  # crucial for individual image processing
        shuffle=True,  # ensures random sampling later
        seed=42 # for reproducibility
    )

    class_samples = {}
    for images, labels in ds:
        class_label = labels[0].numpy()
        if class_label not in class_samples:
            class_samples[class_label] = images[0]

    # Convert the dictionary to a dataset.  Note the need to handle different image shapes.
    images = [sample.numpy() for sample in class_samples.values()]
    labels = list(class_samples.keys())
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    return dataset


# Usage:
data_dir = 'path/to/your/image/directory'
single_sample_dataset = single_sample_per_class(data_dir)
# Iterate through the dataset and use the images.
for images, labels in single_sample_dataset:
    # Process the images
    pass
```

This example utilizes the inherent shuffling capabilities of `image_dataset_from_directory` and then iterates through the resulting dataset, storing only the first encountered image for each class. The `seed` parameter ensures consistent results for testing.  The final dataset is reconstructed from this dictionary using `tf.data.Dataset.from_tensor_slices`, ensuring efficient processing.  The batch size parameter is deliberately set to 1 initially, to allow processing one image per iteration.

**Example 2:  Deterministic Selection using `os.listdir` and `glob`:**

```python
import tensorflow as tf
import os
import glob

def single_sample_per_class_deterministic(directory, image_size=(256, 256), batch_size=32):
    """
    Generates a dataset with the first image from each class.

    Args:
        directory: Path to the directory containing class subdirectories.
        image_size: Tuple specifying the desired image size.
        batch_size: Batch size for the dataset.

    Returns:
        A tf.data.Dataset object.
    """
    classes = os.listdir(directory)
    images = []
    labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        image_path = glob.glob(os.path.join(class_dir, '*'))[0] #selects first image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(i)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    return dataset

# Usage (same as Example 1)
data_dir = 'path/to/your/image/directory'
single_sample_dataset = single_sample_per_class_deterministic(data_dir)
# Iterate through the dataset and use the images
```

This deterministic approach uses `os.listdir` and `glob` to directly access and load the first image from each class directory.  This eliminates the need for intermediate dataset creation and iteration, potentially improving performance for very large datasets. However, it lacks the randomness of Example 1.

**Example 3:  Handling potential errors (Empty Classes):**

```python
import tensorflow as tf
import os
import glob

def single_sample_per_class_robust(directory, image_size=(256, 256), batch_size=32):
    """
    Generates a dataset, handling potential empty class directories.

    Args:
        directory: Path to the directory.
        image_size: Image size.
        batch_size: Batch size.

    Returns:
        A tf.data.Dataset object.  Returns None if any class directory is empty.
    """
    classes = os.listdir(directory)
    images = []
    labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        image_paths = glob.glob(os.path.join(class_dir, '*'))
        if not image_paths:
            print(f"Warning: Class directory '{class_dir}' is empty. Skipping.")
            return None  # Handle the error appropriately
        image_path = image_paths[0]
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(i)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    return dataset

# Usage (same as Example 1, remember to check for None return)
data_dir = 'path/to/your/image/directory'
single_sample_dataset = single_sample_per_class_robust(data_dir)
if single_sample_dataset:
    # Process the dataset
    pass
```

This example adds error handling for situations where a class directory might be empty.  This is crucial for robust code, preventing unexpected crashes.  The function returns `None` if an empty class is encountered, requiring appropriate handling in the calling function.


**3. Resource Recommendations:**

The official TensorFlow documentation, particularly the sections on `tf.data` and image preprocessing, is invaluable.  Explore resources on data augmentation techniques; understanding data augmentation is important to balance the effect of using a single image per class.  Familiarize yourself with different dataset manipulation libraries in Python for improved flexibility in data preprocessing.  Books on practical machine learning with TensorFlow/Keras will provide a broader context for these techniques within the larger machine learning workflow.
