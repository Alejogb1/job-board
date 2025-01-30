---
title: "How can external datasets be fed into TensorFlow?"
date: "2025-01-30"
id: "how-can-external-datasets-be-fed-into-tensorflow"
---
TensorFlow, at its core, operates on a computational graph, demanding data to be structured and formatted consistently for effective processing. Loading external datasets, therefore, becomes a crucial first step in any practical machine learning project. I've personally spent countless hours wrangling diverse data sources for my work in predictive modeling, and the techniques I've honed revolve around TensorFlow's robust data pipeline capabilities.

The primary means of injecting external data into TensorFlow is through the `tf.data` API. This API constructs efficient data pipelines that can handle large datasets, out-of-memory data, and complex preprocessing steps. Rather than loading an entire dataset into memory at once, the `tf.data` API treats data as a stream, processing it in batches, which is particularly vital when dealing with large datasets that exceed available RAM.

The fundamental object within `tf.data` is the `tf.data.Dataset`. We create these objects from existing data sources, including NumPy arrays, lists, or even file paths. Subsequently, a variety of operations can be chained onto the `Dataset` object, allowing for data transformation, batching, shuffling, and prefetching.

Let's consider a few illustrative scenarios.

**Scenario 1: Loading Data from NumPy Arrays**

Imagine we have a small dataset stored in NumPy arrays. Suppose these arrays represent features (input data) and corresponding labels (target values). For demonstration, consider a simplified problem of predicting house prices with two features.

```python
import tensorflow as tf
import numpy as np

# Sample data
features = np.array([[1500, 2], [1800, 3], [2200, 4], [1600, 2], [2000, 3]], dtype=np.float32) # sqft, bedrooms
labels = np.array([250000, 320000, 400000, 280000, 350000], dtype=np.float32)

# Create a tf.data.Dataset from the NumPy arrays
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Batch the data for training.
batch_size = 2
batched_dataset = dataset.batch(batch_size)


# Iterate through the batched data.
for feature_batch, label_batch in batched_dataset:
    print("Feature batch:", feature_batch.numpy())
    print("Label batch:", label_batch.numpy())
```

In this instance, `tf.data.Dataset.from_tensor_slices` creates a dataset where each element is a tuple of a feature vector and its corresponding label. The `batch` operation groups the elements into batches of size two. Note the `dtype=np.float32` declaration to ensure data types are compatible with TensorFlow. Iterating through the dataset yields feature and label batches ready for model consumption. This approach is suitable for small- to medium-sized datasets that can be loaded entirely into memory.

**Scenario 2: Loading Data from Files (CSV)**

The most common scenario involves handling datasets stored in files, often CSV files. We can use `tf.data.experimental.make_csv_dataset` to efficiently load and parse CSV data.

```python
import tensorflow as tf
import numpy as np
import os
import tempfile


# Create a sample CSV file in temp directory
with tempfile.TemporaryDirectory() as tmpdir:
    csv_file = os.path.join(tmpdir, 'housing.csv')
    with open(csv_file, 'w') as f:
      f.write("square_feet,bedrooms,price\n")
      f.write("1500,2,250000\n")
      f.write("1800,3,320000\n")
      f.write("2200,4,400000\n")
      f.write("1600,2,280000\n")
      f.write("2000,3,350000\n")

    # Define column names and data types
    column_names = ['square_feet', 'bedrooms', 'price']
    feature_columns = column_names[:-1]
    label_column = column_names[-1]
    column_types = [tf.float32, tf.int32, tf.float32]

    # Create a CSV dataset
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_file,
        batch_size=2,
        column_names=column_names,
        column_defaults=[tf.constant([], dtype=t) for t in column_types], # Provide defaults
        label_name=label_column,
        num_epochs=1,
        shuffle=False
    )


    # Iterate through the dataset
    for features, labels in dataset:
        print("Features:", features)
        print("Labels:", labels)
```

Here, we utilize `tf.data.experimental.make_csv_dataset` to directly ingest the CSV data. The `column_names` parameter specifies the CSV column headers, and `column_defaults` provides the default values to be used when parsing the data. The `label_name` parameter identifies the target variable, separating it from the feature columns.  The `batch_size` parameter dictates the number of records per batch.  It is important to note that the function automatically parses string data to the designated data types. In the event of missing values in the CSV, column defaults are applied. If a type can not be inferred, the `column_types` argument should also be provided.

**Scenario 3: Handling Image Data**

For image data, a slightly more involved process is needed involving file path manipulation, decoding image data, and optionally applying preprocessing transformations.

```python
import tensorflow as tf
import numpy as np
import os
import tempfile
from PIL import Image

#Create directory and sample images

with tempfile.TemporaryDirectory() as tmpdir:
    images_dir = os.path.join(tmpdir, 'images')
    os.makedirs(images_dir)

    # Sample 2 images
    img_data_1 = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    img_data_2 = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    image_1_path = os.path.join(images_dir, 'image_1.png')
    image_2_path = os.path.join(images_dir, 'image_2.png')

    Image.fromarray(img_data_1).save(image_1_path)
    Image.fromarray(img_data_2).save(image_2_path)

    all_image_paths = [image_1_path, image_2_path]
    labels = [0,1] # Create corresponding labels

    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image, channels=3) # Decode image file to tensor
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Convert to float32 for model input
        return image, label

    # Dataset from file paths
    path_dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, labels))
    # Apply the preprocessing function to all image/label pairs
    image_dataset = path_dataset.map(load_and_preprocess_image)

    batch_size = 2

    # Batch the data
    batched_dataset = image_dataset.batch(batch_size)
    # Iterate through the dataset.
    for image_batch, label_batch in batched_dataset:
        print("Image batch shape:", image_batch.shape)
        print("Label batch:", label_batch.numpy())
```

This example constructs a dataset using paths to image files, applies a custom preprocessing function (`load_and_preprocess_image`) to decode and format the image data, and then batches the processed images for training.  Specifically, `tf.io.read_file` reads the contents of the file, `tf.io.decode_png` decodes the PNG image and produces a tensor, and `tf.image.convert_image_dtype` converts the image to `tf.float32`. Notice the use of `dataset.map` to apply a function to every element of the dataset, which enables flexibility in how the data is processed.

These examples highlight the fundamental techniques for feeding external datasets into TensorFlow using the `tf.data` API.  The process involves creating `tf.data.Dataset` objects, transforming those objects, and then iterating over them during model training. While these examples showcase basic functionalities, the `tf.data` API can handle more complex data formats and incorporate sophisticated data augmentation and preprocessing techniques.

To enhance understanding further, I recommend exploring the official TensorFlow documentation for the `tf.data` module. The "TensorFlow Data API guide" will be useful as a primary resource. The TensorFlow "Performance Guide" can offer insights into building performant data pipelines that optimize GPU utilization. Finally, I would recommend reading the "Effective Data Pipelines in TensorFlow" white paper by Google for deeper knowledge on data handling and pipeline best practices. These resources provide the comprehensive foundation necessary to handle various data formats and build robust training pipelines.
