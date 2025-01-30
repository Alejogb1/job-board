---
title: "How can I create train, validation, and test datasets from an image directory using TensorFlow's `image_dataset_from_directory`?"
date: "2025-01-30"
id: "how-can-i-create-train-validation-and-test"
---
Creating robust machine learning models, particularly those dealing with image data, necessitates a careful division of your dataset into training, validation, and testing subsets. TensorFlow's `image_dataset_from_directory` function offers a convenient way to load image data directly from a directory structure. This function assumes a specific organization: each subdirectory represents a different class. My previous work developing a plant disease detection system required precisely this type of setup, where each folder corresponded to a different plant ailment. A crucial, often overlooked aspect is managing this split effectively. The function, while powerful, requires a degree of manual configuration to achieve the desired three-way partitioning.

The `image_dataset_from_directory` function inherently generates a single `tf.data.Dataset` object. This object, while iterable, does not automatically subdivide into distinct train, validation, and test sets. To achieve that three-way split, one needs to employ a few steps after loading. Primarily, the technique I have found most reliable is to use `tf.data.Dataset.take()` and `tf.data.Dataset.skip()`. We first load the complete dataset from the directory, then extract the required proportions for each subset.

The key parameters for `image_dataset_from_directory` are the directory containing the images, `labels`, `label_mode`, `image_size`, `batch_size`, and `seed`. The `directory` argument should point to the parent directory containing the class subdirectories. If your images are already labeled within subdirectories, set `labels` to 'inferred' and `label_mode` accordingly; otherwise, use the numerical representation. The `image_size` parameter dictates the size to which images are resized, and `batch_size` governs the number of images per batch during training. Notably, the `seed` parameter is indispensable for reproducible results. This ensures that the randomization applied during shuffling remains consistent across runs.

The division into train, validation, and test sets then involves these key steps. First, load the complete dataset. Then, calculate the sizes of the validation and test sets based on the desired percentage. For example, if you want a training set comprising 70% of the total data, a validation set with 15%, and a test set with 15%, you'd use 0.7, 0.15, and 0.15, respectively. Subsequently, utilize the `take()` method to extract the appropriate number of batches for the training set. The validation set is created by first using the `skip()` method to skip past the training set elements, followed by `take()` to acquire the desired proportion. The test set is derived by similarly skipping the training and validation set elements.

Here’s a practical illustration of this approach, based on a hypothetical image set categorized into 'cat', 'dog', and 'bird' folders. I will first demonstrate a scenario with an explicit validation split, then adjust the method to allow for an independent test dataset.

**Code Example 1: Train and Validation Split**

```python
import tensorflow as tf

data_dir = 'path/to/your/image_directory' # Replace with actual path
image_height = 150
image_width = 150
batch_size = 32
seed = 42

# Load the full dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    seed=seed
)

# Calculate the sizes for training and validation
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Split into train and validation sets
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)

print(f"Number of training batches: {len(train_dataset)}")
print(f"Number of validation batches: {len(val_dataset)}")
```

In this first example, the `image_dataset_from_directory` function loads all the images and their associated labels from the provided `data_dir`. The dataset is shuffled using the specified seed to ensure reproducibility. Afterwards, I calculate the split point. The `take` function extracts the initial 80% for the training set, and the remaining 20% are then used for validation via the skip function and by skipping the train data points. This configuration lacks an independent test set, which can be problematic if you need to measure the final performance of your model on unseen data.

**Code Example 2: Train, Validation, and Test Split**

```python
import tensorflow as tf

data_dir = 'path/to/your/image_directory'
image_height = 150
image_width = 150
batch_size = 32
seed = 42

# Load the full dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    seed=seed
)

# Calculate the sizes for training, validation, and test
dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split into train, validation, and test sets
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size).take(val_size)
test_dataset = full_dataset.skip(train_size + val_size)

print(f"Number of training batches: {len(train_dataset)}")
print(f"Number of validation batches: {len(val_dataset)}")
print(f"Number of test batches: {len(test_dataset)}")
```

The second example expands on the first one to include a dedicated test set. Here, I've opted for a 70/15/15 split for train, validation, and test respectively. The calculation to determine the size of each dataset is similar, but now we derive the test dataset size by subtracting the train and validation sizes. The train set is extracted as before using `take`. The validation set is generated by `skipping` over the training elements and then `taking` the next `val_size` elements. The test set is acquired by skipping both the training and validation sets using `skip` and then taking the remainder. This method ensures a rigorous separation between the different dataset parts.

**Code Example 3: Adjusting for Dataset Size Variance**

```python
import tensorflow as tf
import math

data_dir = 'path/to/your/image_directory'
image_height = 150
image_width = 150
batch_size = 32
seed = 42

# Load the full dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    seed=seed
)

# Calculate sizes, handling edge cases with math.floor for int cast
dataset_size = len(full_dataset)
train_size = math.floor(0.7 * dataset_size)
val_size = math.floor(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size


# Split into train, validation, and test sets
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size).take(val_size)
test_dataset = full_dataset.skip(train_size + val_size)


print(f"Number of training batches: {len(train_dataset)}")
print(f"Number of validation batches: {len(val_dataset)}")
print(f"Number of test batches: {len(test_dataset)}")
```

The third example demonstrates a common scenario when dealing with datasets that might not divide perfectly into batches. I use `math.floor` to ensure the `train_size` and `val_size` are integers, as `tf.data.Dataset.take()` and `tf.data.Dataset.skip()` accept integer inputs. This adjustment prevents potential errors stemming from a size calculation that could result in fractions that would not align with batch numbers. This ensures that all sets have valid number of batches, and no data is skipped or missed.

For further understanding of dataset handling in TensorFlow, I recommend reviewing the official TensorFlow documentation regarding `tf.data.Dataset` and `tf.keras.utils.image_dataset_from_directory`. Reading the tutorials about data loading and preprocessing offered by TensorFlow is also highly beneficial. Lastly, various tutorials and examples surrounding image data management using TensorFlow, can provide insights into alternative practices and further understanding. A systematic approach to dataset preparation is crucial for successful machine learning projects. The method outlined provides a robust, reproducible, and efficient workflow for partitioning image datasets.
