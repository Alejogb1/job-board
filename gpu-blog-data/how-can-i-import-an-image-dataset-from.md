---
title: "How can I import an image dataset from subdirectories using TensorFlow?"
date: "2025-01-30"
id: "how-can-i-import-an-image-dataset-from"
---
Image datasets are frequently structured in subdirectories, with each subdirectory representing a unique class. This hierarchical organization facilitates clear dataset management and aligns well with the common practice of storing labeled data. When using TensorFlow for image analysis, efficient and accurate ingestion of such datasets is critical. I've personally spent considerable time refining this process for various projects, and have observed that the `tf.keras.utils.image_dataset_from_directory` function offers a streamlined, robust approach.

My focus here will be on using this specific function, as it abstracts away much of the lower-level data loading and preprocessing logic. This minimizes the potential for error and allows for greater concentration on model design and training, particularly in complex vision tasks. However, it’s also worth noting that there are alternative methods involving manual file listings and custom `tf.data.Dataset` pipelines. These might be preferable when dealing with highly specific requirements, or for very large datasets that can benefit from asynchronous reading and distributed processing. However, I’ve found that for common use cases the efficiency and clarity of `image_dataset_from_directory` generally outweighs the flexibility offered by manual options.

`tf.keras.utils.image_dataset_from_directory` simplifies the creation of a TensorFlow `tf.data.Dataset` from image files arranged in a directory structure where each subdirectory corresponds to a class label. This method automatically infers labels, shuffles the dataset, and can apply image resizing and other transformations during the loading process. The dataset it returns consists of batches of image tensors and corresponding labels, ready for model training or evaluation. The input directory should follow a predictable pattern, typically with each subdirectory containing images belonging to a specific class. Crucially, files within these subdirectories should be images recognized by TensorFlow’s image decoding functions, typically JPG or PNG formats. If other formats are present, TensorFlow will either skip them (potentially leaving less data available than expected) or throw an exception if an attempt to decode them is made.

The key parameters of this function include: the root directory of the dataset, the image size, batch size, and optionally the labels mode (e.g. “int”, "categorical", or "binary"). Setting these correctly is essential for achieving the desired data input pipeline. The `image_size` argument controls image resizing, and can be used to ensure all images have consistent dimensions regardless of their source resolutions, which is crucial for many models. The `batch_size` defines how many images will be grouped together in each batch during training. The labels mode specifies how class labels are represented: integer IDs for single-label classification, categorical one-hot encoded vectors for multi-class classification, and binary for a two-class problem. Additional arguments can control shuffling, validation splits, and color modes. Improper configurations of any of these parameters are a common source of errors, particularly regarding dataset compatibility with downstream modeling tasks.

The function returns a `tf.data.Dataset` object. This object behaves like a Python iterator, but it is optimized for TensorFlow, performing operations like data loading and preprocessing efficiently. When the dataset is iterated, it produces batches of images and their corresponding labels as tensors. This streamlined data pipeline, encapsulated by the `tf.data.Dataset` structure, enables TensorFlow’s high-performance training and evaluation workflows. It also allows for data augmentation techniques to be integrated via the `map` method, although that is not the focus here. Further transformations of the dataset using the `tf.data` API, like caching or prefetching, can also significantly improve the training process, as I have personally seen with large datasets where IO becomes a bottleneck.

Below, I will provide three code examples illustrating the practical use of `tf.keras.utils.image_dataset_from_directory`.

**Example 1: Basic Image Classification Dataset**

This example demonstrates the most common usage scenario: loading a dataset for a multi-class image classification task with explicit integer labels, standard batch size, and standardized image size. I’ve set an explicit seed here, and recommend that whenever random components are involved, that a random seed is specified to ensure reproducible results.

```python
import tensorflow as tf
import os

# Assumes a directory structure:
# dataset/
#   class_a/
#     image1.jpg
#     image2.jpg
#   class_b/
#     image3.jpg
#     image4.png

dataset_dir = 'dataset' # Replace with actual path
image_size = (128, 128)
batch_size = 32
seed = 42

image_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    seed=seed,
    shuffle=True
)

# Print class names for reference
class_names = image_dataset.class_names
print("Class names:", class_names)

# Verify the batch shape and label shape for an example batch
for images, labels in image_dataset.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
```

This code first imports the necessary modules (`tensorflow` and `os`). It then defines the path to the dataset directory, and the desired image size and batch size, along with the random seed for reproducibility.  The key line `tf.keras.utils.image_dataset_from_directory(...)` constructs the dataset object. The `labels='inferred'` parameter means that the function infers the labels from the subdirectory names. `label_mode='int'` indicates that labels should be represented by integer IDs. `shuffle=True` ensures the dataset is shuffled before training, a crucial step to improve training and generalization. The last part of the example extracts the list of class names which corresponds to the subdirectories. By iterating once, the code prints out the shape of the example image batches and label batches, to illustrate the format of the data as it's delivered to TensorFlow.

**Example 2: Dataset with Categorical Labels**

Here, I modify the previous example to demonstrate using categorical (one-hot encoded) labels, often needed for multi-class classification models when the output is via a Softmax layer. This is slightly more memory-intensive, so you should use this setting with care and should be conscious of the dimensionality required.

```python
import tensorflow as tf
import os

# Assumes a directory structure:
# dataset/
#   class_a/
#     image1.jpg
#     image2.jpg
#   class_b/
#     image3.jpg
#     image4.png

dataset_dir = 'dataset' # Replace with actual path
image_size = (128, 128)
batch_size = 32
seed = 42

image_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    seed=seed,
    shuffle=True
)

# Print class names for reference
class_names = image_dataset.class_names
print("Class names:", class_names)

# Verify the batch shape and label shape for an example batch
for images, labels in image_dataset.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Batch label example:", labels[0])
```
The changes here are minimal, only replacing `label_mode='int'` with `label_mode='categorical'`. The key difference is in the label output. Where the previous example returned integers (e.g., `0`, `1`, `2`), this example returns one-hot encoded vectors (e.g., `[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]`). The print statement `print("Batch label example:", labels[0])` shows one such label, allowing you to examine the output explicitly. You can clearly see the one-hot encoding being used.

**Example 3: Dataset with Validation Split**

Often when working with a machine learning problem, you will need to keep data aside for validation to ensure your model generalizes well to new, unseen data. Here is an example which demonstrates how to split the available data. I've made use of the `validation_split` parameter and ensure that the shuffling is consistent across the different sets of data.

```python
import tensorflow as tf
import os

# Assumes a directory structure:
# dataset/
#   class_a/
#     image1.jpg
#     image2.jpg
#   class_b/
#     image3.jpg
#     image4.png

dataset_dir = 'dataset'  # Replace with actual path
image_size = (128, 128)
batch_size = 32
seed = 42
validation_split = 0.2

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    validation_split=validation_split,
    subset="training"
)


val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    validation_split=validation_split,
    subset="validation"
)

# Print class names for reference
class_names = train_dataset.class_names
print("Class names:", class_names)

# Verify the batch shape and label shape for example batch
for images, labels in train_dataset.take(1):
    print("Train batch image shape:", images.shape)
    print("Train batch label shape:", labels.shape)

for images, labels in val_dataset.take(1):
    print("Validation batch image shape:", images.shape)
    print("Validation batch label shape:", labels.shape)
```

This snippet introduces the `validation_split` parameter, setting aside 20% of the dataset for validation. It also adds the `subset` parameter, used to select either training (`subset="training"`) or validation (`subset="validation"`) sets, creating two distinct dataset objects that share the same dataset and shuffling seed. This guarantees that the training and validation data are derived from the same randomly shuffled sequence of images. The `take(1)` operation on both datasets allows you to confirm that the sets have a consistent shape. This separation is necessary for evaluating model performance.

For further learning, I recommend consulting resources that provide detailed explanations of the `tf.data` API, specifically how it handles datasets. There are also excellent books on using TensorFlow for deep learning, and tutorials on the official TensorFlow website, which can be particularly helpful for understanding nuances of data pipelines, image preprocessing, and integration with model training. It is beneficial to review any materials on model training with training and validation sets, as this highlights the importance of keeping training and evaluation separate. This practical knowledge will greatly enhance the effective use of `image_dataset_from_directory` and similar functions.
