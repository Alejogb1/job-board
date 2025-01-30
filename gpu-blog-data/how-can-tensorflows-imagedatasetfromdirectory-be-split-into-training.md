---
title: "How can TensorFlow's `image_dataset_from_directory` be split into training, validation, and testing datasets?"
date: "2025-01-30"
id: "how-can-tensorflows-imagedatasetfromdirectory-be-split-into-training"
---
TensorFlow's `tf.keras.utils.image_dataset_from_directory` function offers a convenient means to generate a dataset from a directory of images structured into class subfolders, but directly implementing a training, validation, and testing split necessitates additional steps beyond its core functionality. I've encountered this scenario frequently in projects involving image classification, and the common approach involves pre-splitting the data directory or utilizing dataset manipulation techniques offered by TensorFlow.

Initially, `image_dataset_from_directory` produces a single `tf.data.Dataset` object encompassing all image data found within the specified directory. To divide this into distinct training, validation, and testing sets, it's important to understand this object's structure. Each element within the dataset is a tuple consisting of a batch of image tensors and a batch of corresponding labels, where the batch size is determined during the function's call. The images are represented as `tf.float32` tensors with pixel values normalized to the range [0, 1], and labels are represented as integer class indices. Consequently, the splitting must happen after this dataset is generated.

There are two primary methods to achieve this split: utilizing a predetermined split in the directory structure or by leveraging TensorFlow dataset manipulation methods after dataset creation. The former approach entails organizing images into separate subdirectories like `train`, `validation`, and `test` beforehand. This provides a direct way for `image_dataset_from_directory` to create the distinct datasets. The latter approach involves creating a single dataset initially and then employing methods like `take()` and `skip()` to partition it. Let's explore both methods, including code illustrations.

**Method 1: Directory-Based Splitting**

This method relies on a pre-existing directory structure where images are grouped into three subfolders representing training, validation, and testing. If a directory like `data/images` exists, it must contain subfolders such as `train`, `validation`, and `test`. Each of these subfolders, in turn, would contain subfolders corresponding to the various image classes.

```python
import tensorflow as tf

# Define image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Paths to train, validation, and test directories
train_dir = "data/images/train"
validation_dir = "data/images/validation"
test_dir = "data/images/test"

# Create datasets using image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42 # Not shuffling validation data during model training
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42 # Not shuffling test data
)

# Print number of batches in each set
print(f"Train dataset: {len(train_dataset)} batches")
print(f"Validation dataset: {len(validation_dataset)} batches")
print(f"Test dataset: {len(test_dataset)} batches")
```

In this example, the `image_dataset_from_directory` is called three times, once for each subdirectory (`train`, `validation`, `test`). The `labels="inferred"` argument ensures labels are extracted from the class subfolder names, and `label_mode="int"` ensures labels are returned as integers. The image size and batch size are specified, and shuffling is enabled only for the training set. The shuffle parameter in the validation and test sets are set to False, because the order of the validation and test datasets does not matter during model training or evaluation. The seed argument ensures consistent shuffling. This method is conceptually straightforward and avoids post-processing, provided the directory structure aligns with the required format.

**Method 2: Dataset Splitting Using `take` and `skip`**

If modifying the file system is inconvenient, the alternative is to create a single dataset and subsequently use `take()` and `skip()` to partition it. This is particularly useful when only a single directory of images is available and pre-splitting is not viable.

```python
import tensorflow as tf

# Define image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Path to the data directory
data_dir = "data/images"

# Create the complete dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# Determine the dataset size
dataset_size = len(dataset)

# Calculate sizes for training, validation, and test sets
train_size = int(0.7 * dataset_size)  # 70% for training
validation_size = int(0.15 * dataset_size) # 15% for validation
test_size = dataset_size - train_size - validation_size # Remaining for testing

# Split the dataset
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size).take(validation_size)
test_dataset = dataset.skip(train_size + validation_size).take(test_size)

# Print number of batches in each set
print(f"Train dataset: {len(train_dataset)} batches")
print(f"Validation dataset: {len(validation_dataset)} batches")
print(f"Test dataset: {len(test_dataset)} batches")
```

In this second example, a single dataset is created using `image_dataset_from_directory` without assuming prior directory splits.  Then, the dataset’s size is calculated, and this size is then used to allocate the appropriate proportion of data to each of the training, validation, and testing datasets.  The first `take` operation takes the initial 70% of the dataset, assigning it to the training set. Subsequent `skip` and `take` operations are used to extract the validation set from the remaining data and the test set from what is left after the validation data is extracted. This offers flexibility in defining split proportions directly in code.

**Method 3: Splitting via `Dataset.cardinality()` and a Custom Split**

This method provides an alternative to calculating the size of the dataset, leveraging the `cardinality()` function, which can be more robust for large datasets where counting may be time consuming. The core concept is similar to Method 2, but the method used to split the dataset is different.

```python
import tensorflow as tf

# Define image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Path to the data directory
data_dir = "data/images"

# Create the complete dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# Get the dataset's cardinality (number of batches)
dataset_size = tf.data.experimental.cardinality(dataset).numpy()

# Calculate sizes for training, validation, and test sets
train_size = int(0.7 * dataset_size)  # 70% for training
validation_size = int(0.15 * dataset_size) # 15% for validation
test_size = dataset_size - train_size - validation_size # Remaining for testing

# Split the dataset
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size).take(validation_size)
test_dataset = dataset.skip(train_size + validation_size).take(test_size)

# Print number of batches in each set
print(f"Train dataset: {len(train_dataset)} batches")
print(f"Validation dataset: {len(validation_dataset)} batches")
print(f"Test dataset: {len(test_dataset)} batches")

```
This method is similar to Method 2, except in how the dataset size is derived. Instead of calculating via the length function, the cardinality of the dataset is used, providing an alternative that may be beneficial in specific scenarios.

**Resource Recommendations**

To gain a more in-depth understanding of related concepts, I would suggest exploring the official TensorFlow documentation. Specifically, focus on the sections regarding `tf.data.Dataset` manipulation, including `take`, `skip`, `shuffle`, and dataset creation functions like `image_dataset_from_directory`. Study tutorials on preparing image data for machine learning. Finally, reviewing examples within TensorFlow’s official GitHub repository that deal with image classification tasks can be immensely helpful. These resources provide the detailed technical explanations and implementation examples that you will need for advanced work with image datasets and model training.
