---
title: "How can I uniformly split each category in a TensorFlow dataset into training and validation sets?"
date: "2025-01-30"
id: "how-can-i-uniformly-split-each-category-in"
---
The challenge of creating balanced training and validation splits from a categorized TensorFlow dataset frequently arises, particularly when dealing with imbalanced classes or the desire for consistent representation across splits. A naive split may lead to skewed datasets that do not accurately reflect the underlying data distribution, causing misleading evaluation metrics and unreliable model training. I've encountered this multiple times while working with image classification tasks involving varying object counts and have found that utilizing TensorFlow's `tf.data` API coupled with careful filtering and reshaping offers a robust solution.

To achieve uniform splitting per category, I prefer to avoid shuffling the entire dataset at once since the inherent organization by category is the key. I accomplish this by first grouping the data by category and then perform splitting on *each* category independently. This guarantees that each category contributes proportionally to both the training and validation sets. The process generally involves the following logical steps: 1) identifying unique categories within the dataset, 2) filtering the dataset to isolate examples belonging to a single category, 3) splitting that isolated subset into training and validation components, and 4) combining these per-category splits into cohesive training and validation datasets. Let's examine the specific techniques and code needed for implementation.

First, the core approach leverages the flexibility of the `tf.data` API to filter based on a label value and the `take` and `skip` functions to achieve splitting. Here's the general process I follow: I start with a dataset whose entries are assumed to be tuples of (example, label), representing the data and its associated category label. If the input dataset structure is different, adjustments to the filtering process, particularly concerning indexing, will be needed. The process is most effective when labels are integer-encoded, but string labels can also be managed with some modification.

The first code example is an illustration of how to handle a dataset where labels are integers:

```python
import tensorflow as tf

def split_dataset_by_category(dataset, num_categories, train_ratio=0.8):
    """Splits a tf.data.Dataset into training and validation sets, maintaining
    proportionate representation from each category.

    Args:
        dataset: A tf.data.Dataset whose elements are tuples of
          (example, label), where label is an integer representing category.
        num_categories: The total number of distinct categories.
        train_ratio: Proportion of data to include in the training set.

    Returns:
        A tuple (train_dataset, val_dataset).
    """
    train_datasets = []
    val_datasets = []

    for category_idx in range(num_categories):
        # Filter for the current category
        category_dataset = dataset.filter(
            lambda example, label: label == category_idx)

        # Determine sizes
        category_size = len(list(category_dataset)) # Explicitly materialize the dataset for length determination
        train_size = int(category_size * train_ratio)

        # Create train and validation subsets
        train_category_dataset = category_dataset.take(train_size)
        val_category_dataset = category_dataset.skip(train_size)

        train_datasets.append(train_category_dataset)
        val_datasets.append(val_category_dataset)

    # Concatenate the per-category sets
    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    val_dataset = val_datasets[0]
    for ds in val_datasets[1:]:
        val_dataset = val_dataset.concatenate(ds)

    return train_dataset, val_dataset


# Example Usage:
if __name__ == '__main__':
    # Create a dummy dataset (e.g. with 1000 examples, 5 classes)
    num_examples = 1000
    num_categories = 5
    labels = tf.random.uniform([num_examples], minval=0, maxval=num_categories, dtype=tf.int64)
    data = tf.random.normal(shape=(num_examples, 32, 32, 3))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))


    train_ds, val_ds = split_dataset_by_category(dataset, num_categories)
    print(f"Training set size: {len(list(train_ds))}")
    print(f"Validation set size: {len(list(val_ds))}")


```

In this example, the `split_dataset_by_category` function iterates through each category, filters the dataset accordingly, and splits using `take` and `skip`. The `len(list(...))` materializes the dataset into a list to determine the size, an operation that can consume a significant amount of memory for very large datasets and is therefore used judiciously during development. A production system would need to determine sizes without fully materializing. The train and validation subsets are then concatenated to form the final training and validation datasets. In this usage example, a synthetic dataset of 1000 samples, having 5 categories, is generated. The core splitting process ensures each class is represented proportionally in the training and validation splits, assuming an initial even distribution across categories. The final print statements will indicate the respective lengths of the split datasets.

For the second code example, consider a scenario where labels are represented as one-hot encoded vectors rather than integers. The filtering condition must then be modified:

```python
import tensorflow as tf

def split_dataset_by_category_onehot(dataset, num_categories, train_ratio=0.8):
    """Splits a tf.data.Dataset into training and validation sets, maintaining
    proportionate representation from each category when labels are one-hot encoded.

    Args:
        dataset: A tf.data.Dataset whose elements are tuples of
          (example, label), where label is a one-hot encoded vector.
        num_categories: The total number of distinct categories.
        train_ratio: Proportion of data to include in the training set.

    Returns:
        A tuple (train_dataset, val_dataset).
    """
    train_datasets = []
    val_datasets = []

    for category_idx in range(num_categories):
        # Filter based on the one-hot encoded label
        category_dataset = dataset.filter(
             lambda example, label: tf.argmax(label) == category_idx)

        # Determine sizes
        category_size = len(list(category_dataset))
        train_size = int(category_size * train_ratio)

        # Create train and validation subsets
        train_category_dataset = category_dataset.take(train_size)
        val_category_dataset = category_dataset.skip(train_size)

        train_datasets.append(train_category_dataset)
        val_datasets.append(val_category_dataset)


    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    val_dataset = val_datasets[0]
    for ds in val_datasets[1:]:
        val_dataset = val_dataset.concatenate(ds)

    return train_dataset, val_dataset

# Example Usage:
if __name__ == '__main__':
    # Create dummy data with one-hot encoding
    num_examples = 1000
    num_categories = 5

    labels_int = tf.random.uniform([num_examples], minval=0, maxval=num_categories, dtype=tf.int64)
    labels_one_hot = tf.one_hot(labels_int, depth=num_categories)
    data = tf.random.normal(shape=(num_examples, 32, 32, 3))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels_one_hot))


    train_ds, val_ds = split_dataset_by_category_onehot(dataset, num_categories)
    print(f"Training set size: {len(list(train_ds))}")
    print(f"Validation set size: {len(list(val_ds))}")
```
Here, the only modification lies in how the filtering is performed within the loop. Instead of direct equality to the `category_idx` as an integer, the `tf.argmax(label)` function is used to determine the index of the active class within the one-hot encoded vector. The remainder of the code maintains the same logic as the previous example, including the concatenation.

Finally, a third example deals with datasets where labels are represented as strings. This requires mapping the strings to an integer to facilitate the filtering:

```python
import tensorflow as tf
import numpy as np

def split_dataset_by_category_string(dataset, categories, train_ratio=0.8):
    """Splits a tf.data.Dataset into training and validation sets, maintaining
    proportionate representation from each category when labels are strings.

    Args:
        dataset: A tf.data.Dataset whose elements are tuples of
          (example, label), where label is a string representing category.
        categories: A list or numpy array containing the unique string categories.
        train_ratio: Proportion of data to include in the training set.

    Returns:
        A tuple (train_dataset, val_dataset).
    """
    train_datasets = []
    val_datasets = []

    for idx, category_str in enumerate(categories):
        # Filter for the current category
         category_dataset = dataset.filter(
             lambda example, label: label == category_str
         )

        # Determine sizes
         category_size = len(list(category_dataset))
         train_size = int(category_size * train_ratio)


        # Create train and validation subsets
         train_category_dataset = category_dataset.take(train_size)
         val_category_dataset = category_dataset.skip(train_size)

         train_datasets.append(train_category_dataset)
         val_datasets.append(val_category_dataset)


    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    val_dataset = val_datasets[0]
    for ds in val_datasets[1:]:
        val_dataset = val_dataset.concatenate(ds)

    return train_dataset, val_dataset


# Example Usage:
if __name__ == '__main__':

    num_examples = 1000
    categories = np.array(['cat', 'dog', 'bird', 'fish', 'hamster'])

    labels_int = tf.random.uniform([num_examples], minval=0, maxval=len(categories), dtype=tf.int64)
    labels_str = tf.gather(categories, labels_int)
    data = tf.random.normal(shape=(num_examples, 32, 32, 3))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels_str))


    train_ds, val_ds = split_dataset_by_category_string(dataset, categories)
    print(f"Training set size: {len(list(train_ds))}")
    print(f"Validation set size: {len(list(val_ds))}")
```

Here, instead of filtering by an integer, we filter by string, utilizing the string directly in the lambda function. The rest of the implementation follows the same methodology as previous examples.

In terms of further resources, while I avoid specific links, I recommend consulting the official TensorFlow documentation pertaining to `tf.data.Dataset`, specifically focusing on the `filter`, `take`, `skip`, and `concatenate` methods. Further, studying examples within the TensorFlow Tutorials, particularly ones related to dataset manipulation, can provide valuable insight. Deep understanding of dataset building and processing is essential for producing high-quality models. The presented code provides a modular basis for achieving uniform category splits, adapting readily to varied data and label formats as needed in complex machine learning projects.
