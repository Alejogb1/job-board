---
title: "How can I load images and labels using filenames in TensorFlow?"
date: "2025-01-30"
id: "how-can-i-load-images-and-labels-using"
---
Loading images and their corresponding labels based on filenames in TensorFlow hinges on creating a robust data pipeline that efficiently reads, decodes, and preprocesses files into a format suitable for model training. The core of this operation involves using TensorFlow’s `tf.data` API, which provides a structured way to handle large datasets without overwhelming memory, especially when dealing with image data. In my experience managing image recognition projects, I've found that correctly structuring this pipeline is often the key differentiator in training performance.

The process can be broken down into several crucial steps: acquiring a list of image file paths and their associated labels, creating a `tf.data.Dataset` from this list, defining functions for image decoding and preprocessing, and batching the data for efficient model training. The filenames themselves act as the connection between the raw image files on disk and their categorical or numerical representations for supervised learning. The power of this method lies in its flexibility; you can easily adapt this system to various naming conventions and data organizations, provided you implement the proper string parsing logic.

Specifically, the initial stage involves gathering file paths and their corresponding labels. Often, I use a function that iterates through a directory structure, identifying image files and extracting label information from file or directory names. This extracted data is then stored in lists or data structures suitable for use with TensorFlow. The fundamental component here is `tf.data.Dataset.from_tensor_slices`, which converts these initial lists of filenames and labels into a data object manageable within a TensorFlow pipeline.

After creating the dataset, the next critical step involves defining functions that can efficiently read the raw image bytes from a given file path and transform them into tensors. `tf.io.read_file` performs the first part, loading the image as a raw byte string. Subsequently, `tf.io.decode_image` is utilized to decode the image, converting the byte string to a numerical tensor representing the image's pixel data. This tensor might be a 3D tensor for color images (height, width, channels) or a 2D tensor for grayscale images.

Beyond decoding, preprocessing is crucial to standardize image data, which contributes to better model performance and training stability. Operations frequently required include resizing images to a uniform size, casting the pixel data to a specific floating-point data type for calculations, and possibly data augmentation to introduce variety during training. These transformations are encapsulated in a function that can be applied to each image in the dataset. This use of a map function within the `tf.data.Dataset` allows these preprocessing steps to operate efficiently on batches of data.

Finally, data is batched and potentially shuffled to create training mini-batches. Batching ensures that data is processed in reasonable chunks, avoiding overwhelming system memory. Shuffling is performed before each training epoch to introduce variability into training and mitigate overfitting. The final output is an efficient data pipeline capable of delivering image tensors and their corresponding labels to a model for training, evaluation, or inference.

Below are three code examples demonstrating different aspects of loading images and labels based on filenames:

**Example 1: Basic Image Loading and Label Creation**

This example shows how to load images from a single directory and generate labels based on a simple integer mapping:

```python
import tensorflow as tf
import os

def load_images_from_directory(image_dir, label_mapping):
    image_paths = []
    labels = []

    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(image_dir, filename))
            label = label_mapping.get(filename.split('_')[0], 0) # Assume label encoded in the filename like "cat_001.jpg"
            labels.append(label)

    return image_paths, labels

def decode_and_preprocess_image(image_path, label):
    image_string = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_string, channels=3) # Assumes color images
    image = tf.image.resize(image, [64, 64])
    image = tf.cast(image, tf.float32) / 255.0 # Scale values to 0-1 range
    return image, label

image_dir = "images_directory" # Replace with your image directory
label_map = {'cat':0, 'dog':1, 'bird':2} # Example mapping for classes
image_paths, labels = load_images_from_directory(image_dir, label_map)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(decode_and_preprocess_image)
dataset = dataset.batch(32)


for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch:", labels)
```

In this example, `load_images_from_directory` reads the images in the specified directory. The function extracts the label by parsing the filename, using a dictionary called `label_map`. The extracted paths and labels are passed to `tf.data.Dataset.from_tensor_slices`. Each element is processed by `decode_and_preprocess_image` which decodes the bytes and prepares the image before being batch processed. This demonstrates a common structure for simple classification tasks.

**Example 2: Loading Images with Labels from Subdirectories**

This example highlights loading image data from multiple subdirectories, with each subdirectory representing a class:

```python
import tensorflow as tf
import os

def load_images_from_subdirs(base_dir):
    image_paths = []
    labels = []
    class_names = os.listdir(base_dir)
    class_names.sort()

    for i, class_name in enumerate(class_names):
      class_path = os.path.join(base_dir, class_name)
      if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_path, filename))
                labels.append(i) # Use index of the subdirectory name as label

    return image_paths, labels


def decode_and_preprocess_image(image_path, label):
    image_string = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_string, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = tf.image.convert_image_dtype(image, tf.float32) # Alternative scaling
    return image, label


base_dir = "images_subdirs" # Replace with your base directory
image_paths, labels = load_images_from_subdirs(base_dir)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(decode_and_preprocess_image)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)


for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch:", labels)
```

In this second example, the structure reads images from a base directory where each subdirectory is a different class. This method of organizing data is prevalent for image classification projects. Labels are assigned based on the order in which the subdirectories are listed using the enumerate method. This shows how to handle datasets structured with folder-based categorization. The dataset is additionally shuffled using the `shuffle` method, which is necessary for training.

**Example 3: Utilizing a Lambda for Preprocessing**

This example emphasizes more flexibility by using a lambda function for inline image preprocessing with a custom resize configuration :

```python
import tensorflow as tf
import os

def load_image_paths(image_dir):
    image_paths = []
    for filename in os.listdir(image_dir):
      if filename.endswith(('.jpg', '.jpeg', '.png')):
          image_paths.append(os.path.join(image_dir, filename))
    return image_paths

def extract_label_from_filename(filename):
  #Assume a filename format like "image_01_class_label_2.jpg" where class is a word and label is the integer
  parts = filename.split('_')
  return int(parts[-1].split(".")[0])

image_dir = "images_custom_format" # Replace with your image directory
image_paths = load_image_paths(image_dir)

labels = [extract_label_from_filename(os.path.basename(path)) for path in image_paths]

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

resize_dims = (256, 256)
def read_and_preprocess(image_path, label):
    image_string = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_string, channels=3)
    image = tf.image.resize(image, resize_dims)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

dataset = dataset.map(read_and_preprocess)
dataset = dataset.batch(32)

for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch:", labels)
```

This third example showcases an alternative structure.  The label extraction is separated into a separate function and an example filename structure is given. Instead of being encapsulated within the function passed to the map method as in the prior examples, the decoding and preprocessing logic is wrapped in a named function (`read_and_preprocess`) called within `map`. The use of a named function, coupled with a resize dimension variable shows more flexibility for potential use within different project contexts.

For further exploration and understanding of these concepts, I recommend consulting the TensorFlow documentation on the `tf.data` API, especially related to creating and transforming datasets. Refer to official tutorials on image processing with TensorFlow which provide detailed guidance on image manipulation with TensorFlow. Additionally, reading through examples from the TensorFlow example repositories on platforms like GitHub is useful.
