---
title: "How to use image_dataset_from_directory and get labels from CSV?"
date: "2024-12-16"
id: "how-to-use-imagedatasetfromdirectory-and-get-labels-from-csv"
---

Alright, let’s tackle this. I remember back in 2018, working on a large-scale image classification project for a medical diagnosis tool, we ran into a very similar challenge. We had thousands of medical images spread across directories, but the corresponding labels were meticulously recorded in a separate CSV file. Feeding those disparate datasets into our TensorFlow models directly wasn't an option; we needed to bridge that gap efficiently. `tf.keras.utils.image_dataset_from_directory` is fantastic for structured image folders, but its built-in labeling mechanism is filename based. When you've got an external label source like a CSV, you need a little extra finesse.

The crux of the issue revolves around the need to couple directory-based images with CSV-based labels. The standard workflow for `image_dataset_from_directory` assumes labels are derived from the subdirectory names. When your ground truth labels reside elsewhere, you have to craft a mechanism that allows TensorFlow to access and use those CSV entries correctly. My approach, and the one I've refined over the years, involves two primary steps: preprocessing the CSV and creating a custom data loader that ties the preprocessed labels to the images loaded through `image_dataset_from_directory`.

First, preprocessing the CSV file is essential. Think of this as preparing your data for consumption. We can’t just point a data loader at a raw CSV; we need to extract the relevant information and organize it into a format that's easily accessible. Typically, I would use pandas to read the csv, map image filenames to labels, and organize it into either a dictionary or, more efficiently, two numpy arrays – one for filenames, and the other for corresponding labels.

```python
import pandas as pd
import numpy as np
import os

def preprocess_labels_csv(csv_path, image_directory):
    """
    Reads a csv file and creates numpy arrays of filenames and their corresponding labels.

    Args:
        csv_path (str): The path to the csv file.
        image_directory (str): The base path to the images.

    Returns:
        tuple: Filenames and labels as numpy arrays.
    """
    df = pd.read_csv(csv_path)
    # Assume the CSV has a column named 'image_id' and 'label'
    filenames = df['image_id'].values
    labels = df['label'].values
    
    # Ensure image paths are correct by prepending the directory path.
    # Assuming that image_id values are image names only without a parent folder
    filenames = np.array([os.path.join(image_directory, filename) for filename in filenames])

    return filenames, labels

# Example usage (replace with your paths)
csv_file = 'path/to/labels.csv'
image_dir = 'path/to/images'

filenames, labels = preprocess_labels_csv(csv_file, image_dir)

#Optional but helpful - check if the length of filenames is equal to labels
if len(filenames) != len(labels):
    raise ValueError("Number of filenames does not match the number of labels.")

print(f"Number of image filenames: {len(filenames)}")
print(f"Number of labels: {len(labels)}")


```

This function, `preprocess_labels_csv`, reads the CSV, extracts filenames and labels, and importantly, prepends the image directory to ensure proper file paths. The numpy arrays, `filenames` and `labels`, provide efficient lookups for the subsequent step. The explicit check for equal lengths is critical to prevent errors down the line. I've debugged so many model pipelines where this initial length mismatch caused headaches.

Now, here's where the custom data loader comes into play. The challenge is to avoid relying on `image_dataset_from_directory` for labels. Rather, we use it just to get the image data, and then manually marry that with our label array from step 1. I'll be using `tf.data.Dataset` API to craft a custom dataset that achieves this. The core of this method is creating a mapping function that loads the images with `tf.io.read_file`, decodes them using `tf.image.decode_jpeg`, and then, at the same time, fetches the labels using the filename in our `filenames` array. A critical implementation detail is using a lookup table (created from the preprocessed numpy arrays) to correlate filepaths to labels.

```python
import tensorflow as tf


def create_custom_dataset(filenames, labels, image_size=(256, 256), batch_size=32):
    """
    Creates a tf.data.Dataset that loads images and associated labels from preprocessed lists.

    Args:
        filenames (np.array): Array of image file paths.
        labels (np.array): Array of labels.
        image_size (tuple, optional): Desired image size. Defaults to (256, 256).
        batch_size (int, optional): Batch size for the dataset. Defaults to 32.

    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """

    # Create a lookup table. This ensures mapping using file path instead of directory.
    file_label_dict = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(filenames, labels), -1)

    def load_image_and_label(filepath):
      file_contents = tf.io.read_file(filepath)
      image = tf.image.decode_jpeg(file_contents, channels=3)
      image = tf.image.resize(image, image_size)
      image = tf.cast(image, tf.float32) / 255.0
      label = file_label_dict.lookup(filepath) #Fetch label by looking up the filename.
      return image, label

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Example usage:
custom_dataset = create_custom_dataset(filenames, labels)

# Verify shape of the loaded data
for images, labels in custom_dataset.take(1):
  print("Shape of Images batch:", images.shape)
  print("Shape of Labels batch:", labels.shape)


```

In `create_custom_dataset`, `tf.lookup.StaticHashTable` forms a rapid look up mechanism connecting filepaths to labels.  The `load_image_and_label` function then uses this table to access the label corresponding to the image filepath. This method avoids unnecessary re-processing of the label data each time an image is fetched, ensuring efficiency.  The `tf.data.Dataset` API is employed here to make sure you can efficiently prefetch batches, so data loading becomes asynchronous.  Prefetching is non-negotiable in any serious model training setup.

Lastly, for verification, I would always recommend inspecting a sample batch to ensure data shapes are correct. This helps prevent surprises down the line when your model training begins.

```python

import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Mock Data Generation (For illustrative purposes)
def generate_mock_data(num_images=200, base_dir='mock_images'):
    os.makedirs(base_dir, exist_ok=True)
    image_data = {}

    for i in range(num_images):
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        img_filename = f'img_{i:03d}.jpg'
        image_path = os.path.join(base_dir, img_filename)
        tf.keras.utils.save_img(image_path, img)
        image_data[img_filename] = np.random.randint(0, 4)  # Mock labels

    df = pd.DataFrame(list(image_data.items()), columns=['image_id', 'label'])
    return df, base_dir

# Use the mock data generator
mock_df, mock_img_dir = generate_mock_data()
mock_csv_path = 'mock_labels.csv'
mock_df.to_csv(mock_csv_path, index=False)

mock_filenames, mock_labels = preprocess_labels_csv(mock_csv_path, mock_img_dir)
mock_dataset = create_custom_dataset(mock_filenames, mock_labels)


for images, labels in mock_dataset.take(1):
  print("Shape of Mock Images batch:", images.shape)
  print("Shape of Mock Labels batch:", labels.shape)
  print("Sample Labels", labels)
```

This additional snippet uses a mock data generator to simulate the real-world scenario. This is incredibly helpful for testing and validation purposes. I always construct minimal versions of my pipelines to isolate the source of any issues. The mock data generation, the CSV creation, and the testing loop are essential parts of my workflow and I encourage you to do the same.

For deepening your understanding, I highly recommend exploring the *TensorFlow Data API guide* on the TensorFlow website and also diving into chapter 3 of the *Deep Learning with Python* (2nd ed.) by François Chollet for excellent insights into data loading and preprocessing techniques. Both of these provide an invaluable conceptual understanding and practical guidance for effectively managing data. *Programming TensorFlow* by Ian Goodfellow is another great resource if you prefer a more hands-on approach.
