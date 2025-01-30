---
title: "How can I batch a list of images for TensorFlow classification?"
date: "2025-01-30"
id: "how-can-i-batch-a-list-of-images"
---
Image classification, especially with large datasets, necessitates efficient handling of input data to avoid memory exhaustion and to maximize GPU utilization. I've often encountered this during my work on real-time image analysis systems for robotics applications. Simply loading all images into memory at once is impractical. Therefore, we use a technique called batching which involves dividing the dataset into smaller groups called batches, which are then processed sequentially. This is crucial for optimizing TensorFlow model training.

The primary mechanism for implementing batching in TensorFlow is through the `tf.data` API. This API provides a robust and flexible system for building complex data input pipelines. Before diving into the code, understand that each image needs to be loaded, possibly resized, normalized, and converted into a tensor, the fundamental data structure in TensorFlow. We don't perform these operations on the entire dataset at once. Instead, the `tf.data.Dataset` object handles this lazily, only performing the transformations when a batch is requested during training.

Here’s how the basic pipeline works: first we create a `tf.data.Dataset` from a list of file paths. Then we use methods like `.map()` to apply operations to each file path, resulting in image tensors.  Finally, the `.batch()` method groups these tensors into batches. I’ve found that this significantly improves training speed and resource utilization compared to manual batching approaches.

Now, let’s explore the code examples.

**Example 1: Basic Image Loading and Batching**

This demonstrates the fundamental principles using a minimal example.

```python
import tensorflow as tf
import os

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224]) # Resizing to a common size
    return image

def create_dataset(image_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Fictional image paths (replace with your actual image paths)
image_paths = [f"image_{i}.jpg" for i in range(100)]
# Create dummy images to run the code
for path in image_paths:
    with open(path, 'w') as f:
      f.write("dummy image content")


batch_size = 32
batched_dataset = create_dataset(image_paths, batch_size)

for batch in batched_dataset:
    print("Batch shape:", batch.shape)
    # This is where you would pass the batch to your model.

#Clean dummy images
for path in image_paths:
    os.remove(path)
```

Here, `load_and_preprocess_image` loads the image from the given path, decodes it using `tf.image.decode_jpeg`, converts it to a floating point tensor, and resizes it to a 224x224 size—a common input shape for many pre-trained image classification models. The `create_dataset` function takes a list of image paths and a `batch_size` as inputs. It first converts the list into a `tf.data.Dataset`. Then `dataset.map` is applied, which applies `load_and_preprocess_image` to each path. Importantly, `num_parallel_calls=tf.data.AUTOTUNE` instructs TensorFlow to use an optimized number of threads for parallel processing of image loading, which can significantly speed up the process, especially when dealing with numerous images.  Next, the dataset is batched using `.batch(batch_size)`. Finally, `prefetch(tf.data.AUTOTUNE)` instructs TensorFlow to prepare the next batch while the current batch is being processed by the model, further enhancing performance.

The for loop iterates over the `batched_dataset` and prints each batch’s shape, which should be `(batch_size, 224, 224, 3)`. If fewer than batch\_size images remain, the last batch will have fewer elements. This is standard behavior.

**Example 2: Using Labels**

Often, your dataset includes labels or classifications for each image. This example demonstrates how to incorporate labels into your dataset.

```python
import tensorflow as tf
import os

def load_and_preprocess_image_with_label(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image, label

def create_labeled_dataset(image_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image_with_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Fictional image paths and labels
image_paths = [f"image_{i}.jpg" for i in range(100)]
labels = [i % 5 for i in range(100)] # Sample labels (e.g., 0, 1, 2, 3, 4)

# Create dummy images to run the code
for path in image_paths:
    with open(path, 'w') as f:
      f.write("dummy image content")

batch_size = 32
labeled_dataset = create_labeled_dataset(image_paths, labels, batch_size)

for image_batch, label_batch in labeled_dataset:
    print("Image batch shape:", image_batch.shape)
    print("Label batch shape:", label_batch.shape)
    # Here you would pass the image and label batches to your model.

#Clean dummy images
for path in image_paths:
    os.remove(path)
```

In this example, the function `create_labeled_dataset` receives the `image_paths` and `labels` as separate lists. `tf.data.Dataset.from_tensor_slices` now receives a tuple of these two lists, and therefore creates a dataset where each element consists of an image path and the corresponding label. The `load_and_preprocess_image_with_label` function now takes both the path and the label as arguments, and returns both the image tensor and the label. During iteration over the `labeled_dataset`, each batch consists of two items: a batch of images and a batch of labels, enabling us to pass the inputs and targets to the model simultaneously.  The batch shapes outputted are `(batch_size, 224, 224, 3)` and `(batch_size,)` respectively.

**Example 3: Handling Different Image Formats and Data Augmentation**

This final example introduces techniques for handling different image formats and applying data augmentation.

```python
import tensorflow as tf
import os

def load_and_preprocess_image_advanced(image_path, label):
    image = tf.io.read_file(image_path)
    try:
        image = tf.image.decode_jpeg(image, channels=3) # Attempt to decode as JPEG
    except tf.errors.InvalidArgumentError:
        try:
            image = tf.image.decode_png(image, channels=3)  # Attempt to decode as PNG
        except tf.errors.InvalidArgumentError:
            raise ValueError(f"Unsupported image format for: {image_path}")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])

    # Data augmentation (random flip and rotation)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, label

def create_augmented_dataset(image_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image_advanced, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Fictional image paths and labels
image_paths = [f"image_{i}.jpg" if i%2==0 else f"image_{i}.png" for i in range(100)] # mixture of jpg and png
labels = [i % 5 for i in range(100)]

# Create dummy images to run the code
for path in image_paths:
    with open(path, 'w') as f:
      f.write("dummy image content")


batch_size = 32
augmented_dataset = create_augmented_dataset(image_paths, labels, batch_size)

for image_batch, label_batch in augmented_dataset:
    print("Image batch shape:", image_batch.shape)
    print("Label batch shape:", label_batch.shape)
    # Here you would pass the image and label batches to your model.
#Clean dummy images
for path in image_paths:
    os.remove(path)
```

In `load_and_preprocess_image_advanced`, we first attempt to decode an image as a JPEG. If the operation fails with an `InvalidArgumentError` (meaning it is probably not a JPEG), we attempt to decode it as a PNG. We raise an exception if neither succeeds. Additionally, we've implemented basic data augmentation with random left-right flips and random rotations to expose our model to varied inputs. This augmentation occurs every time the data is fetched and helps generalize the model.

In all three examples, using `tf.data.AUTOTUNE` with both `num_parallel_calls` and `prefetch` allows TensorFlow to dynamically select the best values for the number of parallel calls and the number of prefetched batches for the hardware configuration, greatly simplifying code and boosting overall training performance.

For further reading, I recommend exploring the official TensorFlow documentation on the `tf.data` API. The TensorFlow Data API guide provides extensive information on building sophisticated pipelines for handling different data formats. The guide on preprocessing images will also be beneficial. Also, review the documentation for specific image decoding functions like `tf.io.decode_jpeg` and `tf.io.decode_png`, and for augmentation operations within the `tf.image` module. I have found studying these resources essential for building efficient image processing systems.
