---
title: "How does TensorFlow's `image_dataset_from_directory` differ from PIL's image loading?"
date: "2025-01-30"
id: "how-does-tensorflows-imagedatasetfromdirectory-differ-from-pils-image"
---
TensorFlow’s `tf.keras.utils.image_dataset_from_directory` and Pillow (PIL)'s image loading offer fundamentally different approaches to image data handling, serving distinct purposes within a machine learning workflow. My experience building image classification pipelines has highlighted these contrasts. PIL focuses on individual image manipulation and processing, while `image_dataset_from_directory` is designed for efficient, large-scale data loading and preprocessing for deep learning model training.

PIL's image handling is centered around opening and manipulating single image files. Using `PIL.Image.open(filepath)`, an image is loaded into memory as a PIL Image object. This object offers methods for various operations: resizing, color adjustments, format conversions, and more. It provides fine-grained control at the pixel level, enabling intricate image editing or analysis before it’s potentially needed for model training. Crucially, each image is processed individually, requiring explicit iteration if handling a dataset in a folder structure. The results are typically NumPy arrays representing pixel values that you would then manually incorporate into a larger dataset structure.

`tf.keras.utils.image_dataset_from_directory`, conversely, abstracts away much of this manual processing. It directly consumes a directory structure where subdirectories are interpreted as class labels. Its primary function is to create a `tf.data.Dataset` object, which is optimized for TensorFlow's data pipeline. This method handles batching, shuffling, and parallelized loading from disk in a way that PIL does not.  It efficiently prepares the data to be streamed directly into the model training process. The output isn't a collection of NumPy arrays, but rather a  TensorFlow `Dataset` object, specifically designed for iterative processing through model training loops. Data is typically loaded lazily. The method can also handle image resizing and label assignment automatically using its various parameters.

Let’s examine three code examples to illuminate these practical differences.

**Example 1: PIL-based image loading and pre-processing**

```python
from PIL import Image
import numpy as np
import os

def load_and_preprocess_image_pil(filepath, target_size=(224, 224)):
  try:
    img = Image.open(filepath)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array
  except Exception as e:
    print(f"Error loading image {filepath}: {e}")
    return None


data_dir = "path/to/my/image_data" # Replace with your data directory
images = []
labels = []

for class_name in os.listdir(data_dir):
  class_path = os.path.join(data_dir, class_name)
  if os.path.isdir(class_path):
      for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path,image_name)
        img_array = load_and_preprocess_image_pil(image_path)
        if img_array is not None:
          images.append(img_array)
          labels.append(class_name)


# Manually convert to NumPy arrays after processing.
images_np = np.array(images)
labels_np = np.array(labels)
```
This code block illustrates the manual effort involved with PIL. It requires: 1) explicit looping through the directory structure, 2) error handling for individual image loads, 3) image pre-processing with PIL's resizing and conversion methods, 4) explicit label assignments, and 5) subsequent organization into NumPy arrays for use in training. The entire data set is loaded into memory.

**Example 2: TensorFlow's `image_dataset_from_directory` for data loading and preprocessing**

```python
import tensorflow as tf

data_dir = "path/to/my/image_data" # Replace with your data directory

image_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
)

# Accessing a batch of the data
for images, labels in image_dataset.take(1):
  print("Images batch shape: ", images.shape) # e.g. (32, 224, 224, 3)
  print("Labels batch shape: ", labels.shape) # e.g (32, num_classes)
```
This second example shows the simplicity and efficiency of using TensorFlow's method. The dataset loading is reduced to a single function call, automatically handling label inference and batching. The output, `image_dataset`, is an iterator which allows the user to efficiently stream batches into model training. Preprocessing occurs on-the-fly during data loading and without loading the entire dataset into memory. The images and labels are converted to TensorFlow tensors.

**Example 3: Custom preprocessing with `image_dataset_from_directory` using `map`**

```python
import tensorflow as tf

data_dir = "path/to/my/image_data" # Replace with your data directory


def preprocess(image,label):
  image = tf.image.convert_image_dtype(image, tf.float32) # Convert to float32 and normalize [0,1]
  image = tf.image.random_brightness(image, max_delta=0.2) # Add random brightness
  return image,label

image_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
)

image_dataset_augmented = image_dataset.map(preprocess)

# Accessing a batch of augmented data
for images, labels in image_dataset_augmented.take(1):
    print("Images batch shape:", images.shape) #e.g. (32, 224, 224, 3)
    print("Labels batch shape:", labels.shape) # e.g. (32, num_classes)
```
In this third example, I illustrate how we can utilize the `map` function on the TensorFlow `Dataset` to further customize pre-processing. Here, I show how we can add additional normalization and image augmentation. Critically, the base image loading still happens with `image_dataset_from_directory`, offering the same advantages over PIL. The added transformations occur using TensorFlow operations within a functional programming approach. This is ideal for incorporating custom augmentations as part of the data loading process.

The crucial distinction is that PIL offers detailed, individual image manipulation capabilities while `image_dataset_from_directory` is optimized for preparing batched, labeled, and processed data for machine learning workflows within TensorFlow. While PIL might be used when initially exploring image data or performing specialized image processing outside the training loop, it is not well suited to efficiently load and process large datasets for model training. `image_dataset_from_directory` excels in that domain.

Choosing between these methods depends on the intended use case. If the task involves complex image manipulation and the application is not directly for a TensorFlow model, PIL is the appropriate tool. However, for most image-based machine learning, `image_dataset_from_directory` significantly streamlines data loading and pre-processing, saving considerable effort and often improving training efficiency.

For further exploration, I recommend consulting the official TensorFlow documentation on `tf.data` for detailed insight on the structure and behavior of datasets. Referencing resources on Keras preprocessing layers will also expand your understanding of image preprocessing within TensorFlow. Additionally, examining examples of image classification models in the TensorFlow tutorials will provide context for where and how `image_dataset_from_directory` is applied in a practical model training workflow. Finally, the Pillow documentation provides detailed explanations for each of its image manipulation functions.
