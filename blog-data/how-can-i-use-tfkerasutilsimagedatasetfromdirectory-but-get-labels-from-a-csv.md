---
title: "How can I use tf.keras.utils.image_dataset_from_directory, but get labels from a CSV?"
date: "2024-12-23"
id: "how-can-i-use-tfkerasutilsimagedatasetfromdirectory-but-get-labels-from-a-csv"
---

Okay, let's tackle this. I’ve seen this scenario pop up quite a bit over the years, especially when dealing with datasets that aren't neatly organized into subdirectories reflecting their classes. You're wanting to leverage the convenience of `tf.keras.utils.image_dataset_from_directory` for image loading but need to pull your labels from a separate CSV file – perfectly reasonable. The direct usage of `image_dataset_from_directory` assumes labels are implicit in directory structure, which is limiting when dealing with datasets having complex annotations. Here’s how we can accomplish this by sidestepping that constraint and creating a custom data pipeline.

Instead of trying to force `image_dataset_from_directory` to do something it wasn’t designed for, we’ll use it to load the file paths and then augment that with label information from your CSV file using `tf.data`. Think of `image_dataset_from_directory` here primarily as a method for finding images, and we will handle labels separately.

The core challenge lies in ensuring each image path loaded by `image_dataset_from_directory` is correctly associated with its corresponding label from your CSV. We’ll use `tf.data.Dataset.from_tensor_slices` to create a dataset from file paths and labels read from the CSV, and then construct a new dataset capable of reading image data and matching labels.

First, I’ll need to detail the workflow. In my experience, such hybrid approaches have proven resilient across varied projects, ranging from simple image classification tasks to more sophisticated ones. The strategy is usually this: parse the csv into file paths and labels, then use those to assemble a `tf.data.Dataset`. Let's break it down step-by-step, including how to handle different scenarios. We will explore three solutions which use varying assumptions about the structure of your dataset.

**Solution 1: Simple matching by file name (no path in csv)**

This scenario assumes that your CSV contains filenames that directly match the filenames of the images in your directory. The CSV will essentially be two columns, the first containing the filename of the image and the second column will contain the label for that image. For example, your csv might have 'image1.jpg,cat' as a row. The image 'image1.jpg' should be in the directory read by `image_dataset_from_directory`. This is the simplest case.

Here's a practical example using tensorflow:

```python
import tensorflow as tf
import pandas as pd
import os

def create_dataset_from_csv_simple(image_dir, csv_path, image_size=(256, 256), batch_size=32):
    # 1. Load file paths using image_dataset_from_directory for file discovery
    initial_dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels=None, # We don't use directory based labels
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False # we will shuffle later so we don't load in the wrong order
    )

    # 2. Read CSV, using Pandas for ease of use
    df = pd.read_csv(csv_path, header=None) # Assumes no header
    filenames = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()

    # Ensure only file names are used from csv, and they should match the directory
    # This step is crucial to match labels to filenames from tf.data
    image_paths = []
    for batch in initial_dataset:
        for path in batch.numpy():
            image_paths.append(path.decode('utf-8').split('/')[-1])

    # Correcting for common issue where relative path is used for matching csv entries to the initial_dataset
    # Assuming the last directory used in initial_dataset is the target for filename matching
    # This can also be improved using more complex filename parsing
    
    corrected_labels = []
    for path in image_paths:
        try:
            idx = filenames.index(path)
            corrected_labels.append(labels[idx])
        except ValueError:
            corrected_labels.append(None) # Handle case where image doesn't exist in CSV
            print(f'Warning: Image "{path}" has no label in CSV.')

    # remove images without labels
    valid_indices = [i for i, label in enumerate(corrected_labels) if label is not None]
    corrected_labels = [corrected_labels[i] for i in valid_indices]
    image_paths = [image_paths[i] for i in valid_indices]

    # create new tf.data.Dataset
    new_dataset = tf.data.Dataset.from_tensor_slices((image_paths, corrected_labels))

    def load_and_preprocess_image(filename, label):
        file_path = os.path.join(image_dir, filename)
        image = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(image, channels=3) # or decode_png if necessary
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0 # Scale pixel values
        return image, label

    new_dataset = new_dataset.map(load_and_preprocess_image)
    new_dataset = new_dataset.shuffle(buffer_size=len(corrected_labels)).batch(batch_size)

    return new_dataset

# Example usage
image_directory = 'path/to/your/images' # Replace with your image directory
csv_file = 'path/to/your/labels.csv'  # Replace with your CSV
dataset = create_dataset_from_csv_simple(image_directory, csv_file)

# Iterate through it
for images, labels in dataset.take(1):
    print(images.shape, labels)
```

In this snippet, I first load all image paths using the default directory approach, but without labels. Then, I parse the CSV, ensuring both paths and labels are collected. Crucially, the path is extracted to file name using a `.split('/')[-1]`, since `image_dataset_from_directory` can create full paths. Then a new `tf.data.Dataset` is generated to handle image loading and resizing. The `load_and_preprocess_image` function handles loading and preprocessing. We ensure we don't process files without labels by dropping them using valid_indices. This code is relatively simple, but assumes file names line up between the csv and image directory.

**Solution 2: Matching by file path (full path in csv)**

The previous approach assumes that file names line up exactly, which isn't always the case. Often you may have csvs that contain a complete path to the image which should match the output from `image_dataset_from_directory`. This scenario assumes your csv includes the complete path to each image, for example, `/path/to/your/images/image1.jpg,cat` as a row.

```python
import tensorflow as tf
import pandas as pd
import os

def create_dataset_from_csv_full_path(image_dir, csv_path, image_size=(256, 256), batch_size=32):

    initial_dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels=None, # We don't use directory based labels
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False # we will shuffle later so we don't load in the wrong order
    )

    # 2. Read CSV, using Pandas for ease of use
    df = pd.read_csv(csv_path, header=None)
    paths_from_csv = df.iloc[:, 0].tolist()
    labels_from_csv = df.iloc[:, 1].tolist()

    paths_from_directory = []
    for batch in initial_dataset:
        for path in batch.numpy():
            paths_from_directory.append(path.decode('utf-8'))

    corrected_labels = []
    for path in paths_from_directory:
        try:
            idx = paths_from_csv.index(path)
            corrected_labels.append(labels_from_csv[idx])
        except ValueError:
            corrected_labels.append(None) # Handle case where image doesn't exist in CSV
            print(f'Warning: Image "{path}" has no label in CSV.')

    # remove images without labels
    valid_indices = [i for i, label in enumerate(corrected_labels) if label is not None]
    corrected_labels = [corrected_labels[i] for i in valid_indices]
    image_paths = [paths_from_directory[i] for i in valid_indices]

    new_dataset = tf.data.Dataset.from_tensor_slices((image_paths, corrected_labels))

    def load_and_preprocess_image(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(image, channels=3) # or decode_png if necessary
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    new_dataset = new_dataset.map(load_and_preprocess_image)
    new_dataset = new_dataset.shuffle(buffer_size=len(corrected_labels)).batch(batch_size)


    return new_dataset

# Example usage
image_directory = 'path/to/your/images' # Replace with your image directory
csv_file = 'path/to/your/labels.csv'  # Replace with your CSV
dataset = create_dataset_from_csv_full_path(image_directory, csv_file)

# Iterate through it
for images, labels in dataset.take(1):
    print(images.shape, labels)

```
This version is very similar to the first, except we do not split the path before matching them to labels read from the csv.

**Solution 3: Using a dictionary for fast lookups (full path or file name)**

When you have datasets containing many files the linear search in the previous two examples can be slow. Using a dictionary lookup provides fast label mapping and performs very well for large datasets.

```python
import tensorflow as tf
import pandas as pd
import os

def create_dataset_from_csv_dict(image_dir, csv_path, image_size=(256, 256), batch_size=32):
    initial_dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    df = pd.read_csv(csv_path, header=None)
    label_mapping = dict(zip(df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()))

    image_paths = []
    for batch in initial_dataset:
        for path in batch.numpy():
            image_paths.append(path.decode('utf-8'))
    
    corrected_labels = []
    for path in image_paths:
       try:
           if path in label_mapping:
              corrected_labels.append(label_mapping[path])
           else:
              corrected_labels.append(label_mapping[path.split('/')[-1]]) # Attempt filename match
       except (KeyError, AttributeError) as e:
           corrected_labels.append(None)
           print(f"Warning: could not find label for {path}")

    valid_indices = [i for i, label in enumerate(corrected_labels) if label is not None]
    corrected_labels = [corrected_labels[i] for i in valid_indices]
    image_paths = [image_paths[i] for i in valid_indices]

    new_dataset = tf.data.Dataset.from_tensor_slices((image_paths, corrected_labels))

    def load_and_preprocess_image(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(image, channels=3) # or decode_png if necessary
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    new_dataset = new_dataset.map(load_and_preprocess_image)
    new_dataset = new_dataset.shuffle(buffer_size=len(corrected_labels)).batch(batch_size)

    return new_dataset

# Example usage
image_directory = 'path/to/your/images'
csv_file = 'path/to/your/labels.csv'
dataset = create_dataset_from_csv_dict(image_directory, csv_file)

for images, labels in dataset.take(1):
    print(images.shape, labels)
```

This code is similar to the previous two approaches. It first loads the paths using `image_dataset_from_directory`, then generates a dictionary from the csv. The dictionary lookup avoids the linear search and is faster, especially for larger datasets. We also perform error handling for missing or misaligned labels between the image paths read and the csv file. This example covers the two most common cases for reading image data where csv is used to store labels.

**Key Considerations**

*   **Performance:** For extremely large datasets, consider using `tf.data.AUTOTUNE` for performance optimization when mapping or shuffling.
*   **Data Augmentation:** After building your dataset, integrate `tf.keras.layers` for data augmentation within the pipeline if needed.
*   **Error Handling:** Include checks for missing files or labels to improve the resilience of your system.
*   **File Types:** Adapt the image decoding step (e.g., `tf.io.decode_jpeg`, `tf.io.decode_png`) based on the image formats you are using.

**Recommended Resources**

To dive deeper into data pipelines with TensorFlow, I recommend:

*   **"TensorFlow 2.0 Quick Start Guide" by Mark Hodson:** This book provides a solid foundation in TensorFlow 2.0 concepts, including effective data handling.
*   **The official TensorFlow documentation:** The 'tf.data' documentation provides a wealth of information and best practices. Pay special attention to the API details for `tf.data.Dataset`, `tf.data.Dataset.from_tensor_slices` and `tf.data.Dataset.map`.
*   **The Keras API documentation:** For understanding `tf.keras.utils.image_dataset_from_directory` itself. Understanding the intention behind its design is important in avoiding misuse.
*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** This book is an excellent resource for understanding the complete process of machine learning, including data preprocessing.

I hope this helps. Let me know if you have further questions or more specific requirements. This approach should be a good starting point for most common scenarios.
