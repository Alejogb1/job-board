---
title: "How can I convert a large .tif image dataset with labels in a separate CSV file to a usable format for a TensorFlow CNN in Python 3?"
date: "2025-01-30"
id: "how-can-i-convert-a-large-tif-image"
---
The challenge in processing large .tif image datasets paired with external CSV labels for TensorFlow CNNs lies primarily in efficient data loading and preprocessing. Standard in-memory approaches become impractical at scale, necessitating generator-based workflows that perform these operations lazily. I’ve encountered this exact problem while developing a remote sensing model for urban land cover classification, where we were dealing with multi-band .tif imagery and corresponding geographic labels stored in a CSV.

The core concept involves creating a custom Python generator that reads image paths and corresponding labels from the CSV, loads image data using a library like `rasterio`, and yields processed image-label pairs. This approach avoids loading the entire dataset into memory at once, allowing us to train on datasets exceeding available RAM. TensorFlow's `tf.data.Dataset` API then becomes the mechanism to construct an efficient pipeline from this generator, enabling batching, shuffling, and prefetching.

Here’s how to break down the implementation. First, the image filenames and labels from the CSV must be extracted and paired. This CSV, assuming it contains at least two columns, one for the image filename (or relative path), and another for the label (either an integer class or a categorical label encoded as a string), can be loaded with `pandas`. The generator function itself requires the image paths and their corresponding labels as inputs. It will iterate through each pair, utilizing `rasterio` to open and read the image data, followed by preprocessing steps like resizing and standardization, and returns both the processed image and the label for use in training.

Let's look at some code examples:

**Example 1: Data Loading Generator**

```python
import rasterio
import pandas as pd
import numpy as np
import tensorflow as tf

def data_generator(image_paths, labels, target_size=(256, 256)):
    """Yields preprocessed images and labels."""
    for image_path, label in zip(image_paths, labels):
        try:
            with rasterio.open(image_path) as src:
                image = src.read()
                # Transpose the image to channels-last format for Tensorflow
                image = np.transpose(image, (1, 2, 0))
                # Assuming images have 3 or more channels.
                if image.shape[2] < 3:
                    # Handle monochrome images or less than 3 channels with a gray scale copy.
                    image = np.stack((image[:,:,0],image[:,:,0],image[:,:,0]), axis=2)
                image = tf.image.resize(image, target_size).numpy()
                image = image / 255.0  # Normalize pixel values
                yield image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

# Example usage:
# Assuming your CSV has columns 'image_path' and 'label'
csv_path = "path/to/your/labels.csv"
df = pd.read_csv(csv_path)
image_paths = df['image_path'].tolist()
labels = df['label'].tolist()


```

This function `data_generator` takes lists of `image_paths` and `labels` as input, and for each image path, attempts to open the `.tif` using `rasterio`. It reads the raster, transposes it to channel-last order, and handles cases where it has less than 3 channels. Then resizes the image to the `target_size` and normalizes it. Finally it yields the preprocessed image and its corresponding label. Error handling ensures that processing doesn't crash on a bad file. This generator is efficient because the image is loaded, processed, and yielded one at a time.

**Example 2: Constructing the `tf.data.Dataset`**

```python
def create_tf_dataset(image_paths, labels, batch_size=32, target_size=(256, 256)):
    """Creates a TensorFlow dataset from the generator."""
    generator = lambda: data_generator(image_paths, labels, target_size=target_size)
    output_signature = (
       tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32),
       tf.TensorSpec(shape=(), dtype=tf.int32) # Or tf.string for categorical labels
    )
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

# Example Usage:
dataset = create_tf_dataset(image_paths, labels)

# Test iterating over the dataset
for images, labels in dataset.take(1):
  print(f"Image batch shape: {images.shape}, Label batch shape: {labels.shape}")
```

Here, `create_tf_dataset` encapsulates the entire dataset creation process. It first defines a lambda to create a generator function. Subsequently, the `tf.data.Dataset.from_generator` method constructs a dataset from the generator with the `output_signature` parameter specifies the shape and data type of the generator's output. Critically, `.batch()` sets the batch size and `.prefetch(tf.data.AUTOTUNE)` preloads the next batch to improve pipeline throughput. The example usage shows how to retrieve the dataset for use. I specified an `int32` label type, but this would need to be a `tf.string` in a case of categorical data.

**Example 3: Handling Categorical Labels**

```python
def create_tf_dataset_categorical(image_paths, labels, batch_size=32, target_size=(256, 256)):
    """Creates a TensorFlow dataset with string labels and one-hot encoding."""
    generator = lambda: data_generator(image_paths, labels, target_size=target_size)
    output_signature = (
       tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32),
       tf.TensorSpec(shape=(), dtype=tf.string)
    )
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Unique labels to determine encoding size
    unique_labels = sorted(list(set(labels)))
    num_classes = len(unique_labels)

    # One hot encode labels
    def one_hot_encode(image, label):
        label_index = tf.where(tf.equal(unique_labels, label))[0][0]
        label_encoded = tf.one_hot(label_index, depth = num_classes)
        return image, label_encoded

    dataset_encoded = dataset.map(one_hot_encode)
    return dataset_encoded

# Example Usage with categorical labels:
# Assuming you have string based labels, like 'forest', 'urban', 'water'.
dataset_categorical = create_tf_dataset_categorical(image_paths, labels)

# Test iterating over the dataset
for images, labels in dataset_categorical.take(1):
    print(f"Image batch shape: {images.shape}, Label batch shape: {labels.shape}")
    print(f"One hot encoded label sample: {labels[0]}")
```
This extension `create_tf_dataset_categorical` allows for string based, categorical label processing.  It is similar to the previous function, however it initializes the output signature with the `tf.string` data type. After the initial dataset is constructed, it gets passed to a `map` operation that will transform the generator's string label output. Before this can occur, however, the unique labels need to be determined and encoded in one-hot fashion so that it may be used in training.  The `one_hot_encode` function creates one-hot encoded labels using  `tf.one_hot`, which is suitable for multi-class classification tasks. This final encoded dataset is then returned for use.

In conclusion, this method relies on combining a data generator to process `.tif` images with the powerful `tf.data.Dataset` API for efficient and scalable training with TensorFlow CNNs. It avoids loading the entire dataset into memory, enables batching and prefetching, and handles both numerical and categorical labels, demonstrating the flexibility required for real-world image processing tasks. Further development may include data augmentations incorporated into the pipeline.

For more in-depth learning on these techniques, consult the TensorFlow documentation focusing on `tf.data`, especially the sections about `tf.data.Dataset.from_generator`, and working with custom data loading pipelines. Additionally, explore the `rasterio` library's documentation thoroughly for best practices in handling geospatial raster data. Finally, documentation on `pandas` provides more options for flexible loading and management of label data.
