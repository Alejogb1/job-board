---
title: "How to load custom images into TensorFlow?"
date: "2025-01-30"
id: "how-to-load-custom-images-into-tensorflow"
---
TensorFlow's image loading pipeline is often a performance bottleneck when dealing with custom datasets, particularly if those images are stored in diverse formats or require preprocessing. Efficiently loading these images directly impacts model training speed and overall experimentation turnaround time. I've encountered this challenge firsthand while developing a remote sensing application, where satellite imagery often comes in formats TensorFlow doesn't readily ingest. The standard `tf.keras.utils.image_dataset_from_directory` proves inadequate for non-standard scenarios. Therefore, creating a tailored data loading mechanism becomes crucial.

The fundamental problem arises from the fact that TensorFlow expects image data to conform to a specific tensor format: typically a three-dimensional tensor representing height, width, and color channels (e.g., RGB or grayscale). This presupposes that your images are already in a format that TensorFlow’s built-in decoders can handle, such as PNG, JPEG, or BMP. When this isn't the case, you must explicitly decode and potentially preprocess the images before feeding them to your model. This is achieved using a combination of TensorFlow operations and, when necessary, external libraries like Pillow or OpenCV for decoding less common formats.

The solution involves building a custom function to process individual image files and integrate this function into a TensorFlow data pipeline using `tf.data.Dataset` API.  Specifically,  `tf.io.read_file` retrieves the raw bytes of an image file. Following this, depending on the file type, various decoding operations may be needed.  For standard formats like PNG and JPEG, `tf.image.decode_png` and `tf.image.decode_jpeg` functions work directly. In the case of custom formats, or if more advanced decoding is required, you would leverage a library like Pillow to decode the image and convert it into a NumPy array which is then converted to TensorFlow tensor using `tf.convert_to_tensor`. Finally, the resulting tensor will require reshaping and type conversion for further use in the pipeline.

Here’s the core process, illustrated through code examples:

**Example 1: Loading standard format images (PNG)**

```python
import tensorflow as tf
import numpy as np

def load_image_png(file_path):
  """Loads a PNG image from a file path.

  Args:
    file_path: A string tensor representing the path to the image file.

  Returns:
    A float32 tensor representing the decoded image, scaled to [0, 1].
  """
  raw_image = tf.io.read_file(file_path)
  decoded_image = tf.image.decode_png(raw_image, channels=3)
  decoded_image = tf.image.convert_image_dtype(decoded_image, dtype=tf.float32)
  return decoded_image

# Example usage within a dataset pipeline
image_paths = tf.constant(["image1.png", "image2.png", "image3.png"])
image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
loaded_images_dataset = image_dataset.map(load_image_png)

for image_tensor in loaded_images_dataset.take(2):
   print("Shape:", image_tensor.shape, "Data Type:", image_tensor.dtype)

```
In this example, `tf.io.read_file` reads the raw bytes. `tf.image.decode_png` then handles the decoding process.  I specify channels=3 to explicitly obtain a 3-channel image (R, G, B); if working with grayscale images, this would be 1. The `tf.image.convert_image_dtype` operation normalizes the pixel intensities to the range of [0, 1] by dividing by 255 as a standard practice. This ensures all data is scaled to the same numerical range, improving training stability. The subsequent `Dataset` construction demonstrates how the `load_image_png` function is mapped onto a collection of paths for easy loading into training pipeline.

**Example 2: Loading an image from a non-standard format using Pillow.**

```python
import tensorflow as tf
import numpy as np
from PIL import Image
import io

def load_image_nonstandard(file_path):
  """Loads a non-standard image format using Pillow.

  Args:
    file_path: A string tensor representing the path to the image file.

  Returns:
    A float32 tensor representing the decoded image, scaled to [0, 1].
  """
  raw_image = tf.io.read_file(file_path)
  image = Image.open(io.BytesIO(raw_image.numpy()))
  image_array = np.array(image)
  image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
  image_tensor = image_tensor/255.0 # normalizes pixel range
  return image_tensor

# Example Usage in a dataset pipeline
image_paths = tf.constant(["image1.tif", "image2.tif", "image3.tif"])
image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
loaded_images_dataset = image_dataset.map(load_image_nonstandard)

for image_tensor in loaded_images_dataset.take(2):
    print("Shape:", image_tensor.shape, "Data Type:", image_tensor.dtype)


```
This example showcases a common situation where TensorFlow's built-in decoding operations aren’t sufficient.  I use Pillow to open the image (it's critical to use the io.BytesIO wrapper since `tf.io.read_file` reads the raw bytes from the file path). The image is then converted to a NumPy array, which is in turn converted into a TensorFlow tensor using `tf.convert_to_tensor`. If necessary, you can perform additional pre-processing using the Pillow library before converting into a NumPy array. Again, the pixel intensities are normalized by dividing by 255.

**Example 3:  Loading images with labels from a directory structure**

```python
import tensorflow as tf
import os
import numpy as np

def load_image_with_label(file_path, label):
  """Loads an image and its associated label.

  Args:
    file_path: A string tensor representing the path to the image file.
    label: A int tensor representing the label associated with the image.

  Returns:
    A tuple containing:
      - A float32 tensor representing the decoded image, scaled to [0, 1].
      - A int tensor representing the label.
  """

  image = load_image_png(file_path)
  return image, label

# Example Usage with an image path and labels

image_paths = tf.constant([
    "data/class_a/image1.png",
    "data/class_a/image2.png",
    "data/class_b/image1.png",
    "data/class_b/image2.png"
])
labels = tf.constant([0, 0, 1, 1])

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image_with_label)

for image, label in dataset.take(2):
    print("Image Shape:", image.shape, "Label:", label.numpy())


```
In this instance, I demonstrate loading not just images, but also associated labels. This is common during classification and supervised learning. I’ve combined the image loading function (`load_image_png`) with a mapping operation that also includes associated labels. The labels, often encoded as integer values, are provided in this example, but they could also be derived from file names or folder structures as required. The `tf.data.Dataset.from_tensor_slices` is given a tuple representing pairs of filepaths and their corresponding labels. `load_image_with_label` is then called to map this into tuples containing the loaded images and their labels.

The key takeaway is to encapsulate your image loading process within a function, making it suitable for use with `tf.data.Dataset` API.  This enables efficient data handling, including features like caching, batching, and shuffling.

For further exploration into this area, several resources can prove useful. Begin by consulting the official TensorFlow documentation specifically for `tf.io`, `tf.image`, and `tf.data.Dataset`.  This provides foundational knowledge on working with data input pipelines. I’d recommend delving into more specific examples and tutorials on advanced dataset construction that feature custom loading functions.  Furthermore, exploring the capabilities of image manipulation libraries such as OpenCV or Pillow will prove invaluable when working with non-standard or complex image formats. Studying real-world implementations, often found in research papers and open-source projects, can illustrate practical ways to optimize these custom loading pipelines for large-scale training datasets. Finally, familiarizing yourself with efficient data streaming techniques within TensorFlow can significantly boost the performance of your workflows.
