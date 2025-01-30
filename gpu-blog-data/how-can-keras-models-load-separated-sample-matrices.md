---
title: "How can Keras models load separated sample matrices efficiently?"
date: "2025-01-30"
id: "how-can-keras-models-load-separated-sample-matrices"
---
Efficiently loading separated sample matrices into Keras models, particularly when dealing with large datasets, necessitates careful consideration of data loading mechanisms. My experience implementing deep learning pipelines for image segmentation, where I often faced multi-terabyte datasets, highlighted that naively loading all data into memory before training is impractical, if not impossible. Instead, utilizing Keras' `tf.data` API or custom generators offers substantial performance improvements by allowing for batched loading and on-the-fly preprocessing. This approach sidesteps the memory limitations encountered when trying to load entire datasets.

The problem specifically arises when sample matrices, such as images or sequences, are not stored contiguously but are instead dispersed across disk or multiple files. This is quite common when working with raw datasets or preprocessed data where individual samples are stored separately. For example, in medical imaging, each patient’s scan might reside in a dedicated file. Directly feeding this data to Keras requires a strategy to efficiently fetch, preprocess, and batch these disparate samples.

Keras models inherently function on batches of tensors; therefore, the loading process must be designed to consistently provide these batches. We can abstract this functionality into a data pipeline. The `tf.data` API in TensorFlow offers a performant way to achieve this. Using `tf.data.Dataset` objects, we can create data pipelines that handle the loading, shuffling, and preprocessing of samples efficiently. Alternatively, custom data generator functions can achieve a similar result but require more manual control over data loading. Both methods avoid loading the entire dataset into memory simultaneously, a critical optimization when dealing with large data.

The core process consists of several steps: first, identify the paths to all sample data. Second, define a loading function that accepts a file path and returns the sample data as a tensor, which might require decoding, resizing, or type conversion. Third, create a `tf.data.Dataset` or construct a generator that applies the loading function to a batch of paths. Finally, configure shuffling, prefetching, and batching parameters.

Here's a demonstration using `tf.data` with a hypothetical scenario where each image is stored as a separate `.png` file alongside its corresponding label in a separate `.npy` file. We begin with image and label paths:

```python
import tensorflow as tf
import numpy as np
import os

# Assume file structure:
# data/
#   images/
#     image_001.png
#     image_002.png
#     ...
#   labels/
#     label_001.npy
#     label_002.npy
#     ...

# Simulate a file structure for testing
if not os.path.exists("data/images"):
    os.makedirs("data/images")
    os.makedirs("data/labels")
    for i in range(10):
      dummy_image = np.random.randint(0,255,(64,64,3), dtype=np.uint8)
      dummy_label = np.random.randint(0, 2, (10,), dtype=np.int32)
      tf.keras.utils.save_img(f"data/images/image_{i:03}.png", dummy_image)
      np.save(f"data/labels/label_{i:03}.npy", dummy_label)

image_paths = [f"data/images/image_{i:03}.png" for i in range(10)]
label_paths = [f"data/labels/label_{i:03}.npy" for i in range(10)]

def load_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.io.decode_png(image, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)  
  image = tf.image.resize(image, [128, 128])
  return image

def load_label(label_path):
  label = tf.io.read_file(label_path)
  label = tf.io.decode_raw(label, tf.int32)
  return label

def load_pair(image_path, label_path):
  return load_image(image_path), load_label(label_path)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=5)
dataset = dataset.batch(batch_size=2)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# Iterate through the batched dataset (for demonstration)
for images, labels in dataset.take(2):
  print("Batch of images shape:", images.shape)
  print("Batch of labels shape:", labels.shape)
```

This example illustrates the use of `tf.data.Dataset.from_tensor_slices` to convert the lists of file paths into a dataset. The `map` operation then applies the `load_pair` function to read and preprocess images and labels, using `num_parallel_calls` for parallel processing to speed things up. `shuffle`, `batch`, and `prefetch` finalize the data pipeline. I’ve personally noticed that `num_parallel_calls=tf.data.AUTOTUNE` provides optimal performance, as TensorFlow intelligently determines the number of parallel threads.

Alternative to `tf.data`, Python generators provide a more hands-on approach. This method requires crafting a function that yields batches of data. This approach is typically employed when `tf.data` encounters complexities, such as during real-time processing.

```python
import numpy as np
import os
import tensorflow as tf

# Assume file structure from previous example is present.

image_paths = [f"data/images/image_{i:03}.png" for i in range(10)]
label_paths = [f"data/labels/label_{i:03}.npy" for i in range(10)]

def load_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.io.decode_png(image, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)  
  image = tf.image.resize(image, [128, 128])
  return image

def load_label(label_path):
  label = tf.io.read_file(label_path)
  label = tf.io.decode_raw(label, tf.int32)
  return label


def data_generator(image_paths, label_paths, batch_size):
  num_samples = len(image_paths)
  while True:
    indices = np.random.permutation(num_samples)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_image_paths = [image_paths[idx] for idx in batch_indices]
        batch_label_paths = [label_paths[idx] for idx in batch_indices]

        batch_images = [load_image(img_path) for img_path in batch_image_paths]
        batch_labels = [load_label(label_path) for label_path in batch_label_paths]

        yield np.stack(batch_images), np.stack(batch_labels)

batch_size = 2
generator = data_generator(image_paths, label_paths, batch_size)
# Example of using the generator with a model:
# model.fit(generator, steps_per_epoch = len(image_paths)//batch_size, epochs=1)

# Print two batches for example purposes.
for i, (images, labels) in enumerate(generator):
  print("Batch of images shape:", images.shape)
  print("Batch of labels shape:", labels.shape)
  if i ==1:
    break
```

Here the generator function yields batches of processed image-label pairs. This method provides greater control as it uses normal python iteration and does not have the limitation of `tf.data` datasets, which require tensor-based operations.

Finally, a more advanced case involves variable-length sequences. When each sample has a different length, padding is required to achieve a tensor-based batch, or using Ragged Tensors, which Keras supports. Here we’ll demonstrate how to do padding, with data consisting of NumPy arrays.

```python
import tensorflow as tf
import numpy as np
# Simulate data with varying sequence lengths
sequences = [np.random.rand(np.random.randint(5, 20), 10) for _ in range(10)]
labels = np.random.randint(0, 2, 10)


def pad_sequence(seq, label):
    return tf.convert_to_tensor(seq, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.int32)

def pad_and_batch(sequence_list, label_list, batch_size):
  padded_sequence = [pad_sequence(seq,label) for seq, label in zip(sequence_list,label_list)]
  dataset = tf.data.Dataset.from_tensor_slices(padded_sequence)
  padded_batch = dataset.padded_batch(batch_size,
                                   padding_values = (tf.constant(0.0, dtype = tf.float32),tf.constant(0, dtype = tf.int32) ),
                                    padded_shapes = ((None, 10),()))
  return padded_batch
batch_size = 2

dataset = pad_and_batch(sequences, labels, batch_size)
for batch_seq, batch_label in dataset.take(2):
  print(batch_seq.shape, batch_label.shape)

```
In this example, `tf.data.Dataset.from_tensor_slices` is used to create a dataset from a list of tensors, and then `padded_batch` is used to batch data together. This is a more complex use case and was often used by me for my research in time series data and sequence modeling, where every sequence may have a different length. The use of `padded_shapes = ((None, 10),())` allows for the padding to be a maximum sequence length of each batch.

For further reading, I would recommend focusing on the official TensorFlow documentation regarding the `tf.data` API, which provides a detailed overview of its capabilities. The “Effective TensorFlow 2” book (available in a PDF format online) also offers a broad overview of how to create high-performance pipelines. Articles detailing performance tuning for `tf.data` pipelines can be found on numerous AI blogs and online educational platforms. Exploring resources concerning data loading for specific data types (like images or time-series data) is beneficial too.
