---
title: "How can TensorFlow Datasets be used to create training and testing sets for Keras models?"
date: "2025-01-30"
id: "how-can-tensorflow-datasets-be-used-to-create"
---
TensorFlow Datasets (TFDS) provide a convenient and efficient way to access and manage large-scale datasets, streamlining the process of creating training and testing sets for Keras models. The core benefit lies in TFDS’s pre-built datasets and standardized API, allowing for direct integration with TensorFlow workflows and optimized data loading. I've personally used TFDS extensively in several projects, ranging from image classification to natural language processing, and its impact on reducing boilerplate code and increasing data handling efficiency has been significant.

The fundamental approach involves loading a dataset from TFDS, potentially performing transformations, and then partitioning it into training, validation, and testing subsets. Once loaded, the dataset is structured as a `tf.data.Dataset`, enabling efficient data pipelining through methods such as `map`, `batch`, `shuffle`, and `prefetch`. This integrated nature streamlines the data processing and feeding mechanism to Keras models.

The typical flow starts by identifying the desired dataset from the TFDS catalog. Once selected, we use `tfds.load` to fetch the dataset. This function returns a `tf.data.Dataset` object that is an iterator yielding batches of samples. These samples are dictionaries where keys are determined by dataset documentation (often containing the image data and label).

**Example 1: Loading and Exploring a Basic Dataset (MNIST)**

Let’s consider the classic MNIST dataset. First, install `tensorflow-datasets` if you haven't already, then load it:

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the MNIST dataset
mnist_data, mnist_info = tfds.load('mnist', with_info=True, as_supervised=True)

# Extract the train and test datasets
train_dataset, test_dataset = mnist_data['train'], mnist_data['test']

# Display dataset information
print(mnist_info)
print(f"Training set size: {len(list(train_dataset.as_numpy_iterator()))}")
print(f"Testing set size: {len(list(test_dataset.as_numpy_iterator()))}")

# Inspect the structure of an example
for image, label in train_dataset.take(1):
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")

```

In this example, `tfds.load` fetches the MNIST dataset, automatically splitting it into training and test sets. The `as_supervised=True` argument returns the dataset as tuples of `(image, label)`. The code extracts these splits from the returned dictionary. The `with_info=True` also returns an `mnist_info` object which provides metadata about the dataset. The output demonstrates the structure of the data and dataset size. This provides the initial building blocks for preparing data for a Keras model.

A common pre-processing step is reshaping images and converting them to float values between 0 and 1 for improved model performance. We'll incorporate that in the next example along with creating batches and shuffling the data.

**Example 2: Preprocessing, Batching, and Shuffling**

Building on the MNIST dataset, let's perform data pre-processing steps:

```python
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Convert to float32, range [0, 1]
    image = tf.reshape(image, [-1])  # Flatten the 28x28 image
    return image, label

# Load the MNIST dataset
mnist_data = tfds.load('mnist', as_supervised=True)
train_dataset, test_dataset = mnist_data['train'], mnist_data['test']

# Apply preprocessing, batching, and shuffling to the training set
BATCH_SIZE = 32
SHUFFLE_BUFFER = 60000 # Number of elements to shuffle. Typically equal to size of data.
processed_train_dataset = train_dataset.map(preprocess_image) \
    .shuffle(buffer_size=SHUFFLE_BUFFER) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# Batch the test data (no shuffling needed for test data)
processed_test_dataset = test_dataset.map(preprocess_image) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# Inspect the batched datasets
for image_batch, label_batch in processed_train_dataset.take(1):
    print(f"Training batch image shape: {image_batch.shape}")
    print(f"Training batch label shape: {label_batch.shape}")

for image_batch, label_batch in processed_test_dataset.take(1):
    print(f"Test batch image shape: {image_batch.shape}")
    print(f"Test batch label shape: {label_batch.shape}")

```

Here, the `preprocess_image` function flattens the input image from 28x28 to a vector of length 784 and scales the pixel values to a float between 0 and 1. The `map` function applies this to every example in the datasets. I introduced the `shuffle` transformation on the training data to randomize example order during training. `batch` groups the data into manageable batches. `prefetch(tf.data.AUTOTUNE)` ensures data is available while the model is training. The shapes show how the dataset has been transformed into batches of preprocessed images ready for training.

When building more sophisticated models, we sometimes need to control dataset split ratios. TFDS allows for this using the `split` argument during the initial dataset loading. This allows more control over how datasets are partitioned, which is particularly useful when additional subsets, such as a validation set, are needed.

**Example 3: Using Split API to Define Train, Validation, Test Sets**

Often, it is preferable to have distinct validation sets. Here’s how that is handled with split:

```python
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Convert to float32, range [0, 1]
    image = tf.reshape(image, [-1])  # Flatten the 28x28 image
    return image, label


# Load the MNIST dataset with train/validation/test splits
mnist_data = tfds.load('mnist', as_supervised=True,
                        split=['train[:80%]', 'train[80%:]', 'test']) # Split train data

train_dataset, validation_dataset, test_dataset = mnist_data
# Alternatively:
# train_dataset = tfds.load('mnist', as_supervised=True, split='train[:80%]')
# validation_dataset = tfds.load('mnist', as_supervised=True, split='train[80%:]')
# test_dataset = tfds.load('mnist', as_supervised=True, split='test')

BATCH_SIZE = 32
SHUFFLE_BUFFER = 60000 * 0.8  # Adjust shuffle buffer for smaller train set

# Apply preprocessing, batching, and shuffling to the training set
processed_train_dataset = train_dataset.map(preprocess_image) \
    .shuffle(buffer_size=SHUFFLE_BUFFER) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

processed_validation_dataset = validation_dataset.map(preprocess_image) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# Batch the test data (no shuffling needed for test data)
processed_test_dataset = test_dataset.map(preprocess_image) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# Inspect the batched datasets
for image_batch, label_batch in processed_train_dataset.take(1):
    print(f"Training batch image shape: {image_batch.shape}")
    print(f"Training batch label shape: {label_batch.shape}")

for image_batch, label_batch in processed_validation_dataset.take(1):
    print(f"Validation batch image shape: {image_batch.shape}")
    print(f"Validation batch label shape: {label_batch.shape}")


for image_batch, label_batch in processed_test_dataset.take(1):
    print(f"Test batch image shape: {image_batch.shape}")
    print(f"Test batch label shape: {label_batch.shape}")
```

In this example, the `split` argument of `tfds.load` specifies the desired data division. `'train[:80%]'` uses the first 80% of the original train data for training, and `'train[80%:]'` is used as the validation set. The test data is unchanged. The code demonstrates how to create training, validation, and test sets using slicing of an existing training set, which provides more refined control over the data split.

**Resource Recommendations**

For further exploration, several excellent resources are available. The official TensorFlow documentation provides comprehensive details about TensorFlow Datasets and the `tf.data` API. The TensorFlow tutorials offer practical examples of using TFDS for various machine learning tasks. Additionally, the TFDS catalogue documents each available dataset providing crucial details for loading and using each. Consulting research papers that have released data via TFDS can further enhance understanding of its use in real-world scenarios.

In conclusion, TFDS is an invaluable tool for efficiently managing and preparing data for Keras models, allowing us to rapidly prototype models without worrying about the details of loading and pre-processing data. Using a standardized approach enables more rapid iteration while also promoting more reproducible research.
