---
title: "How to extract batches from a list of images in TensorFlow?"
date: "2025-01-30"
id: "how-to-extract-batches-from-a-list-of"
---
Handling image datasets in deep learning often requires the efficient creation of batches for training, validation, and testing. Specifically with TensorFlow, the process involves structuring your data as a `tf.data.Dataset`, enabling parallel processing and streamlined input pipelines. Without proper batching, memory limitations become a substantial bottleneck, particularly with large image datasets.

When dealing with a list of filepaths to images, the most common and performant approach in TensorFlow centers around the `tf.data` API. I’ve frequently encountered performance issues with naive looping, and leveraging the `tf.data.Dataset` pipeline consistently yields significant speedups. The key idea is to move the data processing, including file reading, decoding, and augmentation, into the TensorFlow graph, minimizing the back and forth between Python and the native C++ backend. This also allows TensorFlow to perform these operations in parallel across multiple CPU cores or even onto a GPU.

The core concept is to first create a `tf.data.Dataset` from your list of image file paths. Then, a series of transformations are applied to this dataset, beginning with the file reading and decoding of the images, and concluding with batching. This method offers far greater control and scalability than traditional approaches. The initial dataset created is an *input dataset*; each element corresponds to a file path to an image. We map the reading and decoding operations over this dataset which transforms it to a dataset of images.

Below, I will demonstrate three examples showcasing common use cases, starting from a basic example with no additional processing, then moving onto one involving image resizing, and finally one that includes label loading alongside the images.

**Example 1: Basic Image Batching**

The first example showcases the fundamental steps needed to convert a list of file paths into batches of decoded images. In this case, I’m not incorporating any image processing like resizing or augmentation.

```python
import tensorflow as tf
import os

# Simulate a list of image filepaths (replace with your actual data)
image_paths = [f"image_{i}.jpg" for i in range(100)]
for path in image_paths:
    open(path, 'a').close() # Create dummy files

batch_size = 32
num_epochs = 1

def load_and_decode_image(image_path):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    return image

# Create a tf.data.Dataset from the list of paths
dataset = tf.data.Dataset.from_tensor_slices(image_paths)

# Map the loading and decoding function
dataset = dataset.map(load_and_decode_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch the dataset
batched_dataset = dataset.batch(batch_size)

# Prefetch for improved performance
batched_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)

# Iterate over the dataset (simulating model training)
for epoch in range(num_epochs):
    for batch in batched_dataset:
        print(f"Batch shape: {batch.shape}") # Example operation

# Clean dummy files
for path in image_paths:
    os.remove(path)
```

In this code, `tf.data.Dataset.from_tensor_slices()` transforms the list of file paths into a `tf.data.Dataset`. `map()` then applies the `load_and_decode_image` function to each path, reading and decoding the image.  `num_parallel_calls=tf.data.AUTOTUNE` instructs TensorFlow to determine the optimal number of parallel processes for the mapping.  `batch(batch_size)` bundles multiple images into a batch. The `prefetch(tf.data.AUTOTUNE)` operation allows TensorFlow to prepare the next batch in parallel as the current batch is processed. Finally, we iterate over the batched dataset, printing the shape of each batch. The dummy files are cleaned at the end.

**Example 2: Image Batching with Resizing**

In this example, the decoding function is modified to include resizing. This is crucial when input images are of varying dimensions.

```python
import tensorflow as tf
import os

image_paths = [f"image_{i}.jpg" for i in range(100)]
for path in image_paths:
    open(path, 'a').close() # Create dummy files

batch_size = 32
image_size = (224, 224)
num_epochs = 1

def load_and_preprocess_image(image_path):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, image_size)
    return image

dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
batched_dataset = dataset.batch(batch_size)
batched_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)

for epoch in range(num_epochs):
    for batch in batched_dataset:
        print(f"Batch shape: {batch.shape}") # Example operation

for path in image_paths:
    os.remove(path)

```

The `load_and_preprocess_image` function now incorporates `tf.image.resize()` to scale each image to the specified `image_size`. The rest of the code remains similar, demonstrating the flexibility of `tf.data.Dataset`.  This ensures all images within the batch have consistent dimensions suitable for neural network input.

**Example 3: Image Batching with Corresponding Labels**

Often images are paired with labels. This example demonstrates how to incorporate this within a dataset pipeline. In a real use case, these labels would come from a metadata file like a `.csv`.

```python
import tensorflow as tf
import os
import numpy as np

image_paths = [f"image_{i}.jpg" for i in range(100)]
for path in image_paths:
    open(path, 'a').close() # Create dummy files

# Generate dummy labels
labels = np.random.randint(0, 10, size=(100,))

batch_size = 32
image_size = (224, 224)
num_epochs = 1

def load_and_preprocess_image_with_label(image_path, label):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, image_size)
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image_with_label, num_parallel_calls=tf.data.AUTOTUNE)
batched_dataset = dataset.batch(batch_size)
batched_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)


for epoch in range(num_epochs):
    for images, labels_batch in batched_dataset:
        print(f"Image batch shape: {images.shape}, Label batch shape: {labels_batch.shape}")

for path in image_paths:
    os.remove(path)
```

Here, I've used `tf.data.Dataset.from_tensor_slices` with both image paths and labels. `load_and_preprocess_image_with_label` now returns a tuple of the processed image and corresponding label. The iteration over the batch now unpacks a tuple of images and labels. This demonstrates how easily paired inputs can be handled in a `tf.data.Dataset`.

**Resource Recommendations**

When expanding on these concepts, I'd suggest focusing on the official TensorFlow documentation for `tf.data`, particularly the guides on "Better performance with the tf.data API." There are comprehensive explanations for more advanced features, such as caching and data augmentation integration within the dataset pipeline. Look into methods for handling more complex use cases like asynchronous processing and memory management when dealing with extremely large datasets. The tutorials provided on the TensorFlow website for image classification provide excellent practical applications of this framework as well. Additionally, exploring specific examples involving efficient image augmentation and pre-processing within the `tf.data` pipeline will provide further insight into optimizing performance and resource utilization. The core principle is to leverage the TensorFlow graph to parallelize as much of the image loading, decoding, and processing as possible.
