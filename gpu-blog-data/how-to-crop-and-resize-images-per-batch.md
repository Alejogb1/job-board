---
title: "How to crop and resize images per batch in TensorFlow Datasets?"
date: "2025-01-30"
id: "how-to-crop-and-resize-images-per-batch"
---
TensorFlow Datasets (TFDS) provides a powerful mechanism for building efficient data pipelines, and handling image transformations like cropping and resizing directly within the dataset loading process significantly improves performance by parallelizing these operations on the data pipeline level rather than sequentially applying them to entire batches afterward. Based on my experience architecting several large-scale image recognition models, performing such operations efficiently is crucial, especially when dealing with large datasets.

The process fundamentally involves mapping a transformation function to each element of the TFDS dataset. This mapping occurs before batching, ensuring that transformations are integrated into the pipeline. While TFDS offers pre-defined datasets, often requiring customization for research needs like cropping to a specific aspect ratio, resizing to a target input size, or combining both, necessitates creating a map function using TensorFlow operations.

The core of this process relies on the `tf.image` module, which includes functions such as `tf.image.crop_to_bounding_box`, `tf.image.resize`, and `tf.image.random_crop`. The key is to construct a mapping function that reads each element (usually a dictionary containing the image and potentially other metadata), applies the desired transformations, and then returns a modified dictionary. This function is then passed to the `dataset.map()` function.

Let's consider a scenario involving a dataset where the original images are of varying sizes and I need to first crop each image to a square aspect ratio (keeping the center) and then resize them to a consistent 224x224 pixels, commonly used in image classification. Here’s how I would accomplish this.

**Code Example 1: Cropping to a Square Aspect Ratio and Resizing**

```python
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image(element):
    image = element['image']  # Extract the image tensor
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    
    # Calculate the smaller dimension to ensure square cropping
    min_dim = tf.minimum(height, width)
    
    # Calculate offsets for center cropping
    offset_height = (height - min_dim) // 2
    offset_width = (width - min_dim) // 2

    # Apply the crop
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, min_dim, min_dim)

    # Resize the image to the target size
    resized_image = tf.image.resize(cropped_image, [224, 224])

    # Return the transformed image, keeping original structure
    element['image'] = resized_image
    return element

# Load the dataset using TFDS (replace 'your_dataset' with actual name)
dataset = tfds.load('your_dataset', split='train')

# Apply transformation using map
preprocessed_dataset = dataset.map(preprocess_image)

# Batch the transformed dataset
batched_dataset = preprocessed_dataset.batch(32)

# You can then iterate over the batched dataset
for batch in batched_dataset.take(1):
    print(batch['image'].shape) # Output shape should be (32, 224, 224, 3) assuming 3 channel color

```

In this example, the `preprocess_image` function performs the core work. First, it fetches the image from the dictionary element. Then, it dynamically calculates the smaller dimension of the original image using `tf.minimum()`. It uses this value to compute the necessary offset for center cropping with `tf.image.crop_to_bounding_box()`. Subsequently, `tf.image.resize()` transforms the cropped image to 224x224 pixels. Finally, the processed image replaces the original image inside the element dictionary before being returned.

The `dataset.map(preprocess_image)` call applies this function to each element of the dataset, and the batch operation combines multiple elements into batches for efficient processing.

Sometimes you might want to apply a random crop, mimicking a data augmentation. In such instances `tf.image.random_crop` is extremely effective.

**Code Example 2: Random Cropping and Resizing**

```python
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image_random_crop(element):
    image = element['image']
    # Define the desired target size for random crop
    crop_size = [256, 256, tf.shape(image)[-1]] # Channels kept

    # Perform random cropping
    cropped_image = tf.image.random_crop(image, crop_size)

    # Resize the image after the random crop
    resized_image = tf.image.resize(cropped_image, [224, 224])

    element['image'] = resized_image # Replace original image
    return element

# Load the dataset
dataset = tfds.load('your_dataset', split='train')

# Apply the random cropping and resizing transformation
preprocessed_dataset = dataset.map(preprocess_image_random_crop)

# Batch the dataset
batched_dataset = preprocessed_dataset.batch(32)

# Example usage
for batch in batched_dataset.take(1):
    print(batch['image'].shape) # Output shape should be (32, 224, 224, 3)

```

Here, the `preprocess_image_random_crop` function introduces randomness, which is crucial for robust models. I establish the target crop size, which maintains the number of channels, and perform the random crop using `tf.image.random_crop()`. As a design choice, I typically use a crop dimension larger than the resize size to allow for variations within the image. Then I resize as in the previous example. The rest of the process, including loading and batching, remains analogous.

In some cases, you might have images where the aspect ratio is critical. Instead of distorting it through direct resizing, you can pad the image. Let's assume the need for a resize to 224x224 after the square-crop but also the addition of padding to fit a 256x256 window.

**Code Example 3: Aspect Ratio Preserving Resizing with Padding**

```python
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image_pad(element):
     image = element['image']
     height = tf.shape(image)[0]
     width = tf.shape(image)[1]
    
    # Square cropping step same as before
     min_dim = tf.minimum(height, width)
     offset_height = (height - min_dim) // 2
     offset_width = (width - min_dim) // 2
     cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, min_dim, min_dim)
     
    # Perform resize keeping the aspect ratio
     resized_image = tf.image.resize(cropped_image, [224, 224])
     
     # Determine padding size to fit in a 256x256 window
     pad_height = (256 - tf.shape(resized_image)[0]) // 2
     pad_width = (256 - tf.shape(resized_image)[1]) // 2
     
     paddings = [[pad_height, pad_height], [pad_width, pad_width], [0, 0]]
     padded_image = tf.pad(resized_image, paddings, "CONSTANT")
     
     element['image'] = padded_image
     return element
   
# Load the dataset
dataset = tfds.load('your_dataset', split='train')

# Map the padding and resize function
preprocessed_dataset = dataset.map(preprocess_image_pad)

# Batch dataset
batched_dataset = preprocessed_dataset.batch(32)

# Example Usage
for batch in batched_dataset.take(1):
    print(batch['image'].shape) #Output shape is expected to be (32, 256, 256, 3)

```
In this example, I pad the resized image. The `preprocess_image_pad` function first performs the square crop and resize operation. After resizing, I calculate the amount of padding necessary to achieve a target size of 256x256 pixels. `tf.pad` is used to add the padding around the image, preserving aspect ratio during the resize. The padding is `CONSTANT`, usually 0, creating black borders.

These examples highlight the flexibility of `tf.image` and `dataset.map` in TFDS. It's essential to choose the right combination of these operations based on the specific needs of the model.

For further learning, the official TensorFlow documentation for `tf.image` is invaluable and provides details on various image manipulation functions. The TensorFlow Datasets documentation also offers comprehensive guides on creating custom datasets, especially for when modifying an existing one is insufficient, as well as for handling various aspects of large-scale dataset management. Moreover, examining example scripts in official TensorFlow models repositories can provide insightful real-world implementation patterns that leverage the power of the mapping function with TFDS. Finally, exploration of the different parameters available in these functions, such as interpolation methods for resize, helps to fine-tune transformations for specific scenarios.
