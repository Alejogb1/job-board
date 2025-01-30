---
title: "How can I use Keras' image_dataset_from_directory with non-standard directory structures?"
date: "2025-01-30"
id: "how-can-i-use-keras-imagedatasetfromdirectory-with-non-standard"
---
Handling non-standard directory structures with Keras' `image_dataset_from_directory` often requires a nuanced understanding of its underlying data loading mechanism.  My experience working on a large-scale medical image classification project highlighted the limitations of the default structure and necessitated the development of custom solutions. The key insight is that the function relies on a predictable pattern: subdirectories within the main directory represent classes, and images within those subdirectories are assumed to belong to the corresponding class.  Deviation from this necessitates explicit control over the data loading process.

The primary challenge stems from `image_dataset_from_directory`'s inherent assumption of a class-per-subdirectory structure.  When dealing with datasets organized differently – perhaps with multiple levels of subdirectories, or with class labels encoded in filenames rather than directory names – the default function fails to interpret the data correctly.  To overcome this, we must leverage the flexibility offered by TensorFlow and Keras by directly utilizing the `tf.data.Dataset` API for building custom data pipelines.

This involves manually constructing a `tf.data.Dataset` object, iterating through the directory structure, and creating tuples of image paths and corresponding labels.  We then utilize the `tf.data.Dataset.map` function to load and preprocess the images, mirroring the functionality of `image_dataset_from_directory`, but with the necessary adaptations for non-standard organization.

**Explanation:**

The solution involves three core steps: directory traversal, label assignment, and data pipeline construction.  Firstly, we need a function to recursively traverse our directory, identifying images and associating them with their respective classes. This involves defining a function which receives the base directory and a dictionary mapping directory paths or filenames to class labels. This function should then generate a list of (image_path, label) tuples.  Secondly, a label assignment strategy must be carefully designed based on the specific organizational scheme.  If labels are encoded in filenames, regular expressions may be necessary.  If labels are scattered across multiple levels of directories, a path parsing mechanism is required to extract the relevant class information.  Finally, using these tuples, a `tf.data.Dataset` is constructed using `tf.data.Dataset.from_tensor_slices`. This dataset is then mapped using a function that loads and preprocesses images. This allows for a degree of control over image augmentation or other preprocessing steps which may be different from what the `image_dataset_from_directory` offers by default.  Batching and prefetching are then applied using `batch` and `prefetch` methods, optimizing the data loading process.

**Code Examples:**


**Example 1: Multi-level Directory Structure**

```python
import tensorflow as tf
import os
import pathlib

def load_images_multilevel(data_dir, label_map):
    dataset = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                relative_path = os.path.relpath(root, data_dir)
                label = label_map.get(relative_path, None) # Handle missing labels
                if label is not None:
                  dataset.append((path, label))
    return dataset

data_dir = 'path/to/images'
label_map = {
    'class_a/subclass_1': 0,
    'class_a/subclass_2': 1,
    'class_b': 2
}

dataset = load_images_multilevel(data_dir, label_map)

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3) #Adjust if needed
    image = tf.image.resize(image, [224, 224]) #Resize images
    return image, label

dataset = tf.data.Dataset.from_tensor_slices(dataset)
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

This example demonstrates handling a dataset with multiple subdirectory levels. The `label_map` explicitly defines the correspondence between directory paths and class labels.


**Example 2: Filename-based Labeling**

```python
import re
import tensorflow as tf
import os

def load_images_filename(data_dir, label_regex):
  dataset = []
  for root, _, files in os.walk(data_dir):
      for file in files:
          if file.endswith(('.jpg', '.jpeg', '.png')):
              path = os.path.join(root, file)
              match = re.search(label_regex, file)
              if match:
                  label = int(match.group(1)) # Assuming label is a number in the filename
                  dataset.append((path, label))
  return dataset

data_dir = 'path/to/images'
label_regex = r'class(\d+)_'

dataset = load_images_filename(data_dir, label_regex)

#Preprocessing function remains the same as Example 1
dataset = tf.data.Dataset.from_tensor_slices(dataset)
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

```
This example shows how to extract labels from filenames using regular expressions.  The `label_regex` captures the class label embedded within the filename.


**Example 3:  Custom Preprocessing with Augmented Data**

```python
import tensorflow as tf
import os

#... (load_images_multilevel or load_images_filename function from previous examples) ...

def augmented_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_flip_left_right(image) #augmentation
    image = tf.image.random_brightness(image, max_delta=0.2) #augmentation
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2) #augmentation
    image = tf.image.central_crop(image, central_fraction=0.8)
    image = tf.image.resize(image, [224, 224])
    return image, label

#... (rest of dataset construction, similar to previous examples) ...
```
This example extends the preprocessing step to include data augmentation techniques, demonstrating the flexibility offered by the custom approach.


**Resource Recommendations:**

The official TensorFlow documentation, particularly the sections on `tf.data.Dataset` and image loading, are invaluable.  Thorough study of the TensorFlow tutorials on data input pipelines will provide the necessary foundational knowledge.  Understanding the concepts of functional programming and lazy evaluation in Python is also highly beneficial.  Finally, exploring resources on regular expressions will assist in handling complex filename patterns.
