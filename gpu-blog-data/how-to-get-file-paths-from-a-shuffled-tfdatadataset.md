---
title: "How to get file paths from a shuffled tf.data.Dataset?"
date: "2025-01-26"
id: "how-to-get-file-paths-from-a-shuffled-tfdatadataset"
---

Retrieving file paths from a shuffled `tf.data.Dataset` presents a challenge because the shuffle operation typically obscures the original ordering and access to those paths. Specifically, the `tf.data.Dataset` API, by design, abstracts away the underlying data source, enhancing performance through pipelining and parallel processing, at the cost of direct access to filenames after dataset creation and shuffling. I've encountered this particular issue numerous times when debugging and building custom data pipelines for image processing and natural language tasks. I remember one project where the image augmentation parameters had to be dynamically adjusted based on the specific training split, requiring me to peek at the original file name.

The primary reason for this difficulty stems from `tf.data.Dataset`'s focus on efficient, distributed data loading. When you use methods like `tf.data.Dataset.from_tensor_slices`, `tf.data.Dataset.list_files`, or even custom data loading functions, the underlying file paths are often used to *create* the dataset. Once the dataset is created and subsequent operations, such as `shuffle`, `batch`, or `map`, are applied, those original paths are essentially transformed into dataset elements; they are no longer directly addressable as an ordered sequence of filenames associated with elements of the processed dataset. The operations within a data pipeline are intended to manipulate tensor data within the dataset, obscuring knowledge of how individual data instances relate back to the filesystem. Thus, direct access to the filepaths after these processing steps generally requires implementing specific mechanisms to retain that link.

The most straightforward approach to maintain file path information is to include the file path itself as part of the dataset. When constructing the initial dataset, you must ensure that file paths are explicitly loaded or stored alongside the actual data. For example, instead of just loading the image data, you might load the image and also store the filename in a tensor, effectively creating a tuple or dictionary-like structure within the dataset. During your training pipeline, the filename remains available, even after the dataset is shuffled and processed with functions like `map` and `batch`. This approach avoids the need to trace back through the pipeline or access file information outside of the `tf.data.Dataset` object itself, preserving the integrity of the dataset abstraction.

The following code samples illustrate how to accomplish this with slightly different strategies. I've structured them to demonstrate data types commonly used in such situations.

**Example 1: Including filenames with image data**

This example demonstrates how to explicitly incorporate file paths while reading image data from disk using a `tf.data.Dataset`.

```python
import tensorflow as tf
import os

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label, image_path

def create_image_dataset_with_filepaths(image_dir, label_map):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    labels = [label_map[f.split('.')[0]] for f in os.listdir(image_dir) if f.endswith('.jpg')]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    return dataset

# Sample image directory and label mapping for demonstration
image_dir = "sample_images" # Assuming a directory of jpgs exists
label_map = {"image1": 0, "image2": 1, "image3": 0}
os.makedirs(image_dir, exist_ok=True)
for i in range(1,4):
    with open(os.path.join(image_dir, f'image{i}.jpg'), 'wb') as f:
        f.write(b'dummy image content')


dataset = create_image_dataset_with_filepaths(image_dir, label_map)
for image, label, path in dataset.take(3):
    print(f"Image path: {path.numpy().decode()}, Label: {label.numpy()}")

```

In this example, the `create_image_dataset_with_filepaths` function constructs a dataset from a list of file paths and their corresponding labels. Importantly, the `load_and_preprocess_image` function not only reads and processes the image data but also passes the file path along. This ensures that the file path is maintained as part of the dataset element. The `take(3)` method is used to show the first three elements along with their associated image path and label to verify the process. Note the filepath is included as a `tf.string` tensor and must be decoded to a standard string for printing using `path.numpy().decode()`.

**Example 2: Preserving paths within a text dataset**

Here, we deal with a text dataset, where file paths might be necessary for tracking associated meta-data.

```python
import tensorflow as tf
import os

def load_and_preprocess_text(file_path, label):
  text = tf.io.read_file(file_path)
  text = tf.strings.substr(text, 0, 100)  # Truncate text example
  return text, label, file_path

def create_text_dataset_with_filepaths(text_dir, label_map):
    text_paths = [os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith('.txt')]
    labels = [label_map[f.split('.')[0]] for f in os.listdir(text_dir) if f.endswith('.txt')]
    dataset = tf.data.Dataset.from_tensor_slices((text_paths, labels))
    dataset = dataset.map(load_and_preprocess_text)
    dataset = dataset.shuffle(buffer_size=len(text_paths))
    return dataset

#Sample text directory and label mapping
text_dir = "sample_text" # Assuming a directory of txt exists
label_map = {"text1": 0, "text2": 1, "text3": 0}
os.makedirs(text_dir, exist_ok=True)
for i in range(1,4):
    with open(os.path.join(text_dir, f'text{i}.txt'), 'w') as f:
        f.write(f'This is sample text content for text file number {i}.')


dataset = create_text_dataset_with_filepaths(text_dir, label_map)
for text, label, path in dataset.take(3):
    print(f"Text path: {path.numpy().decode()}, Label: {label.numpy()}")
```

The approach mirrors the image example; file paths are explicitly passed through the processing pipeline. This allows users to access the file path even when working with datasets based on text files. The dataset operation and printing process are similar to the image dataset.

**Example 3: Using `tf.data.Dataset.from_generator`**

When more customized data loading is needed, the `tf.data.Dataset.from_generator` method can be used to maintain access to file paths.

```python
import tensorflow as tf
import os
import numpy as np

def generator_with_filepaths(data_dir, label_map):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dat')]
    for path in file_paths:
        label = label_map[os.path.basename(path).split('.')[0]]
        data = np.loadtxt(path)  # Assuming data is numeric in .dat files
        yield data, label, path


def create_generator_dataset_with_filepaths(data_dir, label_map):
  gen = lambda: generator_with_filepaths(data_dir, label_map)
  dataset = tf.data.Dataset.from_generator(gen,
                                            output_signature=(
                                              tf.TensorSpec(shape=(2,), dtype=tf.float64),
                                              tf.TensorSpec(shape=(), dtype=tf.int32),
                                              tf.TensorSpec(shape=(), dtype=tf.string)))

  dataset = dataset.shuffle(buffer_size=len(os.listdir(data_dir)))
  return dataset


# Sample data directory and label mapping
data_dir = "sample_data" # Assuming a directory of dat files
label_map = {"data1": 0, "data2": 1, "data3": 0}
os.makedirs(data_dir, exist_ok=True)
for i in range(1,4):
    np.savetxt(os.path.join(data_dir, f'data{i}.dat'), np.random.rand(2))

dataset = create_generator_dataset_with_filepaths(data_dir, label_map)
for data, label, path in dataset.take(3):
    print(f"Data path: {path.numpy().decode()}, Label: {label.numpy()}")

```

Here, a custom Python generator, `generator_with_filepaths`, yields data, labels, and the file paths. The `tf.data.Dataset.from_generator` method creates a dataset using this generator and the output signature is defined using the `tf.TensorSpec` objects to specify the output data type. The file paths are generated as part of this process and remain accessible in the processed dataset. Again the `decode()` call is needed to print the paths.

In summary, to retrieve file paths from a shuffled `tf.data.Dataset`, you must explicitly retain them within the dataset itself, which usually involves adding an additional tensor to store the path during the initial dataset creation and processing. This design choice offers flexibility and maintains the benefits of `tf.data.Dataset`'s optimized data loading.

For further reading and expanding one's understanding, I would recommend reviewing the official TensorFlow documentation on:  `tf.data.Dataset` creation, including `from_tensor_slices` and `from_generator`; the usage of `tf.data.Dataset.map` for data transformation; and the concept of `tf.TensorSpec`. Specifically, pay attention to how these parts interact when building custom data pipelines.
