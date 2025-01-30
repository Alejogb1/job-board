---
title: "Why is TensorFlow Keras training slower on GPU than CPU?"
date: "2025-01-30"
id: "why-is-tensorflow-keras-training-slower-on-gpu"
---
TensorFlow Keras training performance discrepancies between GPU and CPU, specifically where a GPU underperforms, frequently stem from data loading and processing bottlenecks. A common, and often overlooked, scenario involves a CPU-bound pipeline feeding the GPU, preventing the GPU from achieving its full computational potential.

The core issue arises from how TensorFlow utilizes computational resources. CPUs are optimized for general-purpose operations, including handling intricate data preprocessing tasks like image decoding, augmentation, and batch shuffling. GPUs, conversely, excel at massively parallel floating-point computations, perfectly suited for the matrix multiplications and convolutions inherent in deep learning models. The training loop typically involves a cycle of fetching data, preprocessing it, feeding it into the model for forward and backward passes, and updating the model's weights. If the CPU cannot prepare the data quickly enough to keep the GPU busy, the GPU spends considerable time idle, waiting for inputs. The result is underutilization of GPU resources and slower training times than expected, potentially even slower than a CPU-based training. This scenario often presents itself where data preprocessing becomes a serial bottleneck, negating the benefits of parallel GPU processing.

Let's consider a hypothetical example. I've spent considerable time optimizing neural networks for image classification. One instance saw me working with a large image dataset of 200,000 256x256 color images. Initially, I experienced unexpectedly sluggish performance when training a convolutional neural network with a GeForce RTX 3080 Ti GPU. Profiling revealed the data loading pipeline was the bottleneck. The images were being read from disk, resized, and augmented on the CPU within a Python loop before being converted to tensors and passed to the GPU. While the model's backpropagation and parameter updates happened very quickly on the GPU, the model was often sitting idle awaiting the next batch of processed images.

Here's a first simplified example in TensorFlow, demonstrating the basic data loading with an image augmentation function applied on the CPU:

```python
import tensorflow as tf
import numpy as np
import time

def augment_image(image):
  image = tf.image.random_brightness(image, max_delta=0.2)
  image = tf.image.random_flip_left_right(image)
  return image

def load_data(batch_size, num_images=1000):
    images = np.random.randint(0, 256, size=(num_images, 256, 256, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=(num_images,))

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    def process_example(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = augment_image(image)
        return image, label

    dataset = dataset.map(process_example).batch(batch_size)
    return dataset

batch_size = 32
dataset = load_data(batch_size)

start_time = time.time()
for i, (images, labels) in enumerate(dataset):
  if i > 10:
    break;
  pass # Placeholder for model training
end_time = time.time()
print(f"Data loading and augmentation on CPU time: {end_time - start_time:.4f} seconds")

```
In this example, the `load_data` function simulates loading images and applying augmentation. The `augment_image` function, executed during dataset mapping using `.map()`, runs on the CPU for each batch, before the batch of data is passed to the GPU, if training was taking place. It's important to observe the timing here, because if data loading and preprocessing are taking a significant portion of the time, this indicates a CPU bottleneck.

To improve this, we need to transfer some of the preprocessing burden to the GPU. TensorFlow provides mechanisms to perform preprocessing using TensorFlow operations directly, which can leverage the GPU’s processing power. We also need to configure the dataset for asynchronous data loading. By using `tf.data.AUTOTUNE`, TensorFlow automatically determines the number of prefetch buffers, parallel calls, and other parameters to optimally manage data loading.

Below is an improved code example that moves the `augment_image` logic to GPU using TensorFlow ops, and utilizes `AUTOTUNE`:

```python
import tensorflow as tf
import numpy as np
import time

def augment_image(image):
  image = tf.image.random_brightness(image, max_delta=0.2)
  image = tf.image.random_flip_left_right(image)
  return image

def load_data(batch_size, num_images=1000):
    images = np.random.randint(0, 256, size=(num_images, 256, 256, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=(num_images,))

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    def process_example(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = augment_image(image) # Augmentation happens on GPU
        return image, label

    dataset = dataset.map(process_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

batch_size = 32
dataset = load_data(batch_size)

start_time = time.time()
for i, (images, labels) in enumerate(dataset):
    if i > 10:
      break;
    pass # Placeholder for model training
end_time = time.time()
print(f"Data loading and augmentation on GPU time: {end_time - start_time:.4f} seconds")

```
By moving the augment function to the `map` operation and setting `num_parallel_calls` to `tf.data.AUTOTUNE`, we tell Tensorflow to make use of multiple threads for CPU-bound tasks, and to execute the processing tasks on the device where it’s most efficient. Further, using `prefetch(tf.data.AUTOTUNE)` ensures that new data is loaded while the current batch is being processed, reducing idle time. The time to load and process a batch of data should decrease significantly compared to the first example.

A critical aspect often overlooked is the format of the data itself. Storing data in optimized formats that can be loaded quickly (like TFRecords) will help improve data loading speed. While the previous examples use numpy arrays for simplicity, using TFRecords is a standard practice for large-scale machine learning projects.

Here's a third example, which illustrates how to use a basic TFRecord dataset, and shows how the mapping operation could apply augmentations (for example, using `tf.image.random_flip_left_right` for demonstration). TFRecord support is more intricate, and typically would have associated helper methods that perform reading and writing, but the below outlines the dataset creation. This example still leverages the concept of performing augmentation through the `map()` operation in the `tf.data.Dataset` pipeline.

```python
import tensorflow as tf
import numpy as np
import time

def augment_image(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_flip_left_right(image)
    return image

def create_tfrecord_dataset(num_images=1000, output_file="sample.tfrecord"):
    with tf.io.TFRecordWriter(output_file) as writer:
        for _ in range(num_images):
            image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
            label = np.random.randint(0, 10)
            example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                  }
              ))
            writer.write(example.SerializeToString())
    return output_file

def load_tfrecord_dataset(batch_size, tfrecord_file):
    def _parse_function(serialized_example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        image = tf.io.decode_raw(example['image'], tf.uint8)
        image = tf.reshape(image, (256, 256, 3))
        image = tf.cast(image, tf.float32) / 255.0
        image = augment_image(image)
        label = tf.cast(example['label'], tf.int32)
        return image, label

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

tfrecord_file = create_tfrecord_dataset()
batch_size = 32
dataset = load_tfrecord_dataset(batch_size, tfrecord_file)

start_time = time.time()
for i, (images, labels) in enumerate(dataset):
    if i > 10:
      break;
    pass # Placeholder for model training
end_time = time.time()

print(f"Data loading and augmentation from TFRecord time: {end_time - start_time:.4f} seconds")
```

In this third example, the `create_tfrecord_dataset` creates a sample TFRecord file using synthetic data. The `load_tfrecord_dataset` function demonstrates how to use the `TFRecordDataset` and apply data parsing and augmentation within the mapping operation, which then allows TensorFlow to delegate these tasks to the GPU. Asynchronous prefetching is still enabled with `prefetch`. This example demonstrates a more efficient way to handle data loading, reducing the likelihood of a CPU bottleneck, and is closer to the typical setup in real-world deep learning projects.

For further exploration, I recommend looking into TensorFlow’s official documentation on the `tf.data` API. Specifically, investigate features such as `tf.data.Dataset.map`, `tf.data.Dataset.batch`, `tf.data.Dataset.prefetch`, and `tf.data.AUTOTUNE` for performance optimization. Additionally, examine guides on using TFRecords for efficiently storing and loading data, along with TensorFlow profiling tools that allow you to diagnose bottlenecks within the training loop by visualizing operations. The use of a proper profiling tool is critical in diagnosing where the bottleneck may be occurring. Consider also looking into mixed-precision training, as this can also speed up the actual training time.
