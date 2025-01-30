---
title: "How do I create a TensorFlow dataset with images and labels?"
date: "2025-01-30"
id: "how-do-i-create-a-tensorflow-dataset-with"
---
Creating a TensorFlow dataset from images and labels requires careful consideration of data organization, preprocessing steps, and efficient data loading strategies.  My experience building large-scale image classification models has highlighted the importance of leveraging TensorFlow's `tf.data` API for optimal performance.  Directly manipulating NumPy arrays for large datasets is inefficient and can lead to out-of-memory errors.  The `tf.data` API provides tools for building highly optimized input pipelines that seamlessly integrate with TensorFlow's training loop.

**1. Data Organization and Preprocessing:**

Before constructing the dataset, ensure your image data and corresponding labels are organized in a structured manner. I typically employ a directory structure where each subdirectory represents a class label, containing images belonging to that class.  For example, a dataset for classifying cats and dogs might look like this:

```
dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

This structure simplifies the process of associating images with their labels.  The filenames themselves might not be directly useful as labels; instead, the directory structure implicitly provides this information.

Furthermore, preprocessing is crucial. Images may require resizing, normalization, and potentially augmentation to improve model performance and robustness.  Resizing ensures uniform input dimensions for the model.  Normalization, often involving subtracting the mean and dividing by the standard deviation of the pixel values, improves training stability. Augmentation techniques, such as random cropping, flipping, and rotations, can artificially increase dataset size and enhance model generalization.  These steps should be incorporated into the dataset pipeline.

**2. Constructing the TensorFlow Dataset:**

TensorFlow's `tf.data` API offers a flexible and efficient way to create datasets.  The `tf.keras.utils.image_dataset_from_directory` function provides a convenient method for constructing datasets from the directory structure described above.  However, for more fine-grained control and complex preprocessing, a custom pipeline built using `tf.data.Dataset.from_tensor_slices` is often preferred.

**3. Code Examples:**

**Example 1: Using `image_dataset_from_directory`:**

```python
import tensorflow as tf

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    labels='inferred',  # Labels are inferred from subdirectory names
    label_mode='categorical', # One-hot encoding for labels
    image_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    shuffle=True,
    seed=42
)

# Basic data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
])

augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

for images, labels in augmented_dataset.take(1):
    print(images.shape)
    print(labels.shape)
```

This example leverages the convenience of `image_dataset_from_directory`.  It automatically infers labels from the directory structure, applies one-hot encoding, and handles image loading and batching.  I've added basic data augmentation to showcase the flexibility of this approach.

**Example 2:  Custom Dataset with `from_tensor_slices`:**

```python
import tensorflow as tf
import numpy as np

# Assume image_data is a NumPy array of shape (N, H, W, C) and labels is a NumPy array of shape (N,)
image_data = np.random.rand(100, 224, 224, 3)
labels = np.random.randint(0, 2, 100) # Binary classification

dataset = tf.data.Dataset.from_tensor_slices((image_data, labels))

# Preprocessing function
def preprocess(image, label):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize(image, [224, 224])
  return image, label

dataset = dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

for images, labels in dataset.take(1):
    print(images.shape)
    print(labels.shape)
```

This example demonstrates building a dataset from raw NumPy arrays using `from_tensor_slices`.  The `preprocess` function handles image type conversion and resizing.  Crucially, `.batch(32)` creates batches of 32 samples, and `.prefetch(tf.data.AUTOTUNE)` allows for asynchronous prefetching of data, significantly improving training speed.  I've used this approach extensively when dealing with datasets not fitting neatly into a directory structure.


**Example 3:  Dataset with complex preprocessing and augmentation:**

```python
import tensorflow as tf

def complex_preprocess(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.random_crop(image, size=[200, 200, 3]) #random crop
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [224, 224])
    image = tf.image.random_brightness(image, max_delta=0.2) #brightness augmentation
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2) #contrast augmentation
    return image, label

# Assuming 'dataset' is created using either of the previous methods
dataset = dataset.map(complex_preprocess).cache().batch(32).prefetch(tf.data.AUTOTUNE)

#The cache() method ensures that the dataset is cached in memory for faster access during subsequent epochs.
```

This example builds upon previous examples, incorporating more sophisticated preprocessing and augmentation steps. The `complex_preprocess` function now includes random cropping, flipping, brightness and contrast adjustments. The `.cache()` method is added to cache the preprocessed dataset in memory, improving performance for subsequent epochs.

**4. Resource Recommendations:**

For a deeper understanding of TensorFlow's `tf.data` API, I recommend consulting the official TensorFlow documentation.  Exploring resources on image preprocessing techniques and data augmentation strategies will further enhance your abilities.  Furthermore, studying best practices for efficient data loading and handling large datasets will be invaluable.  These resources will provide valuable context and advanced techniques beyond the scope of this response.  I've found that a strong grasp of these principles is essential for building robust and efficient image classification models.
