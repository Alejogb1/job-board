---
title: "How can I load color images using TensorFlow based on file names instead of directory structures?"
date: "2025-01-30"
id: "how-can-i-load-color-images-using-tensorflow"
---
TensorFlow’s primary image loading utilities, especially `tf.keras.utils.image_dataset_from_directory`, inherently rely on a directory structure where subfolders represent class labels. This structure isn't always ideal; many datasets are provided with a manifest file containing image filenames and their corresponding labels. To load color images based on filenames instead of relying on a directory structure, one must construct a custom data loading pipeline using `tf.data.Dataset`. This allows for greater flexibility and control over the image loading process.

My experience developing computer vision applications has frequently involved dealing with datasets that don't conform to the strict directory hierarchies expected by standard TensorFlow functions. I’ve had to load images from datasets where annotations were stored separately from the images themselves, in a file listing filenames and corresponding labels, or when the dataset lacked a natural class structure. The key is to leverage TensorFlow’s `tf.data` API to build a custom pipeline that reads the image files directly, independent of directory structures. This process essentially involves creating a `tf.data.Dataset` from a list of image file paths and then processing each image individually.

The fundamental process involves these key steps: first, one would acquire a list of image file paths and their corresponding labels. This list may be generated manually, read from a file like a CSV, or obtained from a database. Second, the lists of filenames and labels are used to create a `tf.data.Dataset` using `tf.data.Dataset.from_tensor_slices`. This function transforms the python lists into tensors. Third, a custom function is mapped to the dataset to read and decode the images. Finally, any additional transformations or data augmentation can be added to the pipeline. This approach provides complete control over how the data is processed and formatted for consumption by a TensorFlow model.

Let’s demonstrate this with three examples.

**Example 1: Simple Image Loading**

This example illustrates loading a list of images with associated labels from a list of filenames. Here, we assume we already have lists of `image_paths` and `labels`.

```python
import tensorflow as tf

# Assume these lists are pre-populated
image_paths = ['image1.jpg', 'image2.png', 'image3.jpeg']
labels = [0, 1, 0]

def load_and_preprocess_image(image_path, label):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_string, channels=3) # Force 3 channels for color images
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224]) # Standardize size
    return image, label


dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) #Load in parallel
dataset = dataset.batch(32) # Batch size
dataset = dataset.prefetch(tf.data.AUTOTUNE) #Prefetch next batch
```

This code first initializes example lists for image file paths and their labels. The `load_and_preprocess_image` function reads a single image file, decodes it into a color image (3 channels), resizes it to 224x224 pixels, and converts its datatype to `float32`, a common requirement for model training. The function returns the preprocessed image and its corresponding label as a tuple. The `tf.data.Dataset.from_tensor_slices` function creates a dataset that holds pairs of filenames and their labels. We map our custom function to apply the transformation to each element, batch it for efficient training, and use prefetching to optimize pipeline performance. The `num_parallel_calls` parameter lets TensorFlow load multiple images in parallel. This example demonstrates a fundamental loading pipeline.

**Example 2: Loading from a CSV File**

Often, image paths and labels are stored in a CSV file. This example demonstrates how to load this data. We’ll assume a simple CSV file format with “filename” and “label” columns, and we'll use the python `csv` library for parsing.

```python
import tensorflow as tf
import csv

csv_path = 'image_manifest.csv' # Example CSV file

def load_csv_manifest(csv_path):
    image_paths = []
    labels = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
           image_paths.append(row['filename'])
           labels.append(int(row['label'])) # Convert label string to integer
    return image_paths, labels

image_paths, labels = load_csv_manifest(csv_path)

def load_and_preprocess_image(image_path, label):
  image_string = tf.io.read_file(image_path)
  image = tf.image.decode_image(image_string, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [224, 224])
  return image, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

In this example, the `load_csv_manifest` function handles reading the CSV file and populating lists with image filenames and their corresponding labels. The remainder of the code mirrors the previous example, using these lists to create a dataset and apply the same image preprocessing. This shows how to adapt the loading approach to handle structured data sources. Note that error handling such as file existence checks are omitted for brevity.

**Example 3: Adding Data Augmentation**

This example extends the previous scenario by incorporating basic data augmentation to increase the robustness of a model and mitigate the risk of overfitting. Augmentation typically includes manipulations such as random flips, rotations, or color adjustments.

```python
import tensorflow as tf
import csv

csv_path = 'image_manifest.csv'

def load_csv_manifest(csv_path):
    image_paths = []
    labels = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
           image_paths.append(row['filename'])
           labels.append(int(row['label']))
    return image_paths, labels


image_paths, labels = load_csv_manifest(csv_path)


def load_and_preprocess_image(image_path, label):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    # Data Augmentation (randomly flip left/right)
    image = tf.image.random_flip_left_right(image)
    return image, label


dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

Here, the `load_and_preprocess_image` function is augmented with `tf.image.random_flip_left_right`, which randomly flips the image horizontally. Similar augmentation techniques can be added by leveraging the image manipulation operations available in `tf.image`. This demonstrates how the loading pipeline can be extended to incorporate further data processing and transformations, tailoring the training input for best results.

In terms of resources, it's recommended to review the official TensorFlow documentation on `tf.data.Dataset`, specifically focusing on `tf.data.Dataset.from_tensor_slices`, `tf.io.read_file`, `tf.image.decode_image`, and other functions within `tf.image`. Exploring relevant TensorFlow tutorials on custom data loading pipelines can also be useful. Additionally, familiarity with Python's file handling and `csv` library is beneficial for loading data from external sources. For advanced data augmentation techniques, investigating the broader scope of `tf.image` is recommended. Understanding the implications of data augmentation on model generalization, especially for the specific use case, is crucial. Finally, optimizing I/O operations for faster data loading, often involving the use of `tf.data.AUTOTUNE` or `tf.data.cache`, is very effective for large datasets. This method allows one to load color images effectively, providing the required flexibility when relying on filenames rather than direct folder structures.
