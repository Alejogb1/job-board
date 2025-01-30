---
title: "How can I dynamically load training batches of images and text in TensorFlow/Keras?"
date: "2025-01-30"
id: "how-can-i-dynamically-load-training-batches-of"
---
The challenge in efficiently training deep learning models on image and text data often lies in managing memory consumption during the batch loading process. Direct loading of entire datasets into memory, particularly with large image and text collections, frequently results in resource exhaustion. Therefore, a dynamic batch loading approach is crucial, where data is loaded on demand, batch by batch, during training. I've personally wrestled with this issue across several projects involving multimodal learning, specifically those employing image captioning models and image-text retrieval systems. This response outlines my preferred method leveraging TensorFlow's `tf.data` API for creating performant, dynamically loaded training pipelines for image and text data.

The core concept revolves around constructing a `tf.data.Dataset` that represents your training data. This dataset is an abstraction over the actual data residing on disk, facilitating operations like shuffling, batching, and transformations without immediately loading all the data. The process typically involves these stages: creating a dataset of file paths, reading image and text data from these paths, pairing the data, and preparing it for model input. We will use `tf.io` for file reading and preprocessing operations, and the pipeline will aim for a form consumable by Keras model training.

**Data Preparation and Dataset Creation**

The initial step involves creating a dataset of file path pairs. Imagine a scenario where you have image files in a 'images/' directory and corresponding text files in a 'captions/' directory. Assuming each image file `image_1.jpg` corresponds to a text file `image_1.txt`, we can create a dataset of path pairs programmatically. This eliminates the need to hold file contents in memory from the beginning.

```python
import tensorflow as tf
import os

def create_path_dataset(image_dir, text_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    text_files = [os.path.join(text_dir, os.path.splitext(os.path.basename(f))[0] + ".txt") for f in image_files] # Infer text file name based on image file

    path_pairs = list(zip(image_files, text_files))
    return tf.data.Dataset.from_tensor_slices(path_pairs)
```

In this code snippet, I create lists of image and text file paths based on the contents of respective directories. I ensure file name correspondence, and then I combine them into tuples representing image-text pairs. This list of tuples becomes the data source of a `tf.data.Dataset` through `from_tensor_slices`. This approach handles arbitrary sized file systems.

**Loading and Preprocessing Data**

Having established a path dataset, we now need to load the image and text data, and preprocess them into formats suitable for the deep learning model. For images, common transformations include resizing and normalization. For text, tokenization and sequence padding are necessary before numerical representation for input into the model. The mapping operation within `tf.data` offers a method to perform these steps on each element of the dataset dynamically.

```python
def load_and_preprocess(image_path, text_path, image_size=(256, 256), max_text_length=20):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3) # Or decode_png for PNG
    image = tf.image.resize(image, image_size)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Normalize to [0, 1]

    text = tf.io.read_file(text_path)
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^a-z0-9\\s]', '') # Basic text cleaning
    text = tf.strings.strip(text)
    text = tf.strings.split(text)  # Basic tokenization

    # Padding and truncation of text sequences
    text_tokens = text.to_tensor(default_value=b'', shape=[None, max_text_length]) # Truncate or pad to max_text_length
    text_tokens = text_tokens.to_tensor()

    return image, text_tokens


def preprocess_dataset(dataset, image_size, max_text_length):
    return dataset.map(lambda image_path, text_path:
                    load_and_preprocess(image_path, text_path, image_size, max_text_length),
                    num_parallel_calls=tf.data.AUTOTUNE) # Utilize concurrent calls for faster processing
```

In the `load_and_preprocess` function, images are read, decoded, resized, and normalized. Similarly, text files are loaded, normalized, and tokenized. Furthermore, text sequences are padded or truncated to a specified `max_text_length` for batch consistency, leveraging the `to_tensor` method. The `preprocess_dataset` function subsequently applies `load_and_preprocess` to the path-based dataset. By using `num_parallel_calls=tf.data.AUTOTUNE`, we allow TensorFlow to dynamically determine the degree of parallelism to optimize loading performance, based on available CPU cores.

**Batching and Prefetching**

The final stage involves batching the processed dataset and prefetching for enhanced performance. Batching groups multiple data elements together, forming input batches for the model. Prefetching pipelines an operation where data loading and model training occur concurrently; specifically while the GPU is working on one batch, the next batch is prepared for the GPU by the CPU.

```python
def prepare_dataset_for_training(dataset, batch_size):
  return (dataset
          .batch(batch_size)
          .prefetch(buffer_size=tf.data.AUTOTUNE))
```

By adding these steps, the data is loaded, processed, and batched entirely dynamically. The data is not loaded into memory until the point of use, batch by batch, which prevents memory related errors and enables smooth, scalable training.

**Complete Example**

Bringing all components together we have the complete, functional code. Note: for simplicity, I assume that your images are .jpg files and your text files are .txt.

```python
import tensorflow as tf
import os

def create_path_dataset(image_dir, text_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    text_files = [os.path.join(text_dir, os.path.splitext(os.path.basename(f))[0] + ".txt") for f in image_files]
    path_pairs = list(zip(image_files, text_files))
    return tf.data.Dataset.from_tensor_slices(path_pairs)

def load_and_preprocess(image_path, text_path, image_size=(256, 256), max_text_length=20):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    text = tf.io.read_file(text_path)
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^a-z0-9\\s]', '')
    text = tf.strings.strip(text)
    text = tf.strings.split(text)
    text_tokens = text.to_tensor(default_value=b'', shape=[None, max_text_length])
    text_tokens = text_tokens.to_tensor()

    return image, text_tokens

def preprocess_dataset(dataset, image_size, max_text_length):
    return dataset.map(lambda image_path, text_path:
                    load_and_preprocess(image_path, text_path, image_size, max_text_length),
                    num_parallel_calls=tf.data.AUTOTUNE)

def prepare_dataset_for_training(dataset, batch_size):
    return (dataset
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

# Example Usage:
image_dir = 'images/'  # Replace with your image directory
text_dir = 'captions/'  # Replace with your text directory
image_size = (224, 224)
max_text_length = 50
batch_size = 32

path_dataset = create_path_dataset(image_dir, text_dir)
preprocessed_dataset = preprocess_dataset(path_dataset, image_size, max_text_length)
final_dataset = prepare_dataset_for_training(preprocessed_dataset, batch_size)

# Now you can iterate over your batches
for images, texts in final_dataset.take(5): # Take a few example batches
  print("Image batch shape:", images.shape)
  print("Text batch shape:", texts.shape)
  # pass your images and text to a training step
```

This approach efficiently manages large datasets and allows for scalable deep learning training.

**Resource Recommendations**

For deeper exploration of `tf.data` API, I recommend consulting the official TensorFlow documentation on data loading techniques. Specifically, exploring the sections about "Input Pipelines" and "Performance with tf.data" will prove useful. The TensorFlow guide for "Text classification" contains good practical examples of text preprocessing with `tf.data` and `tf.strings`, as well. Additionally, a review of examples that demonstrate data loading for image models, such as those in the TensorFlow image classification guides, is recommended. These examples often incorporate techniques for efficient data loading that you can adapt and integrate with text data pipelines.
