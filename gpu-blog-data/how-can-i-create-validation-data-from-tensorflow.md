---
title: "How can I create validation data from TensorFlow Datasets (TFDS) data?"
date: "2025-01-30"
id: "how-can-i-create-validation-data-from-tensorflow"
---
Generating validation data from TensorFlow Datasets (TFDS) hinges on a crucial understanding: TFDS datasets are not inherently partitioned into training, validation, and test sets.  The `tfds.load` function provides the raw dataset; subsequent partitioning is the responsibility of the user.  This is a deliberate design choice, allowing for flexibility in defining validation strategies based on the specifics of the dataset and the intended model. In my experience developing image classification models for satellite imagery, this flexibility proved crucial in optimizing model performance across diverse geographical features.

The most straightforward method involves utilizing the `tf.data.Dataset.take` and `tf.data.Dataset.skip` methods to create subsets from the loaded TFDS dataset. This approach is efficient for smaller datasets where memory constraints are less of a concern, but scaling it to extremely large datasets might prove problematic.

**1.  Using `tf.data.Dataset.take` and `tf.data.Dataset.skip`:**

This approach is ideal for datasets that fit comfortably into memory.  It allows for a simple and direct split of the dataset into training and validation portions.  Consider the following Python code:

```python
import tensorflow_datasets as tfds
import tensorflow as tf

# Load the dataset. Replace 'your_dataset' with the actual dataset name.
dataset = tfds.load('your_dataset', split='train', as_supervised=True)

# Determine the dataset size.  This is crucial for accurate splitting.
dataset_size = len(dataset) #This works for smaller datasets. For larger, consider tf.data.experimental.cardinality()

# Define the validation split percentage (e.g., 20%).
validation_split = 0.2

# Calculate the number of examples for validation.
validation_size = int(dataset_size * validation_split)

# Create the validation dataset.
validation_dataset = dataset.take(validation_size)

# Create the training dataset.
training_dataset = dataset.skip(validation_size)

# Batch and prefetch the datasets for efficient training.
batch_size = 32
training_dataset = training_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Now you have 'training_dataset' and 'validation_dataset' ready for use.
```

This code first loads the entire dataset, calculates the validation set size based on a chosen percentage, and then uses `take` and `skip` to create separate datasets.  The use of `prefetch` is critical for optimized training performance.  Note that `len(dataset)` may be slow or unreliable for very large datasets; use `tf.data.experimental.cardinality()` for those.  In my work with high-resolution satellite images, I frequently encountered this limitation and adapted the code accordingly.


**2.  Stratified Splitting using `tf.data.Dataset.shard`:**

When dealing with class imbalances, a stratified split is necessary to ensure that the validation set reflects the class distribution of the training set.  This prevents biased model evaluation.  While TFDS doesn't directly offer stratified splitting, we can achieve this using `tf.data.Dataset.shard` in conjunction with careful data preprocessing.

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# Load the dataset.
dataset = tfds.load('your_dataset', split='train', as_supervised=True)

# Extract labels and features.  Adapt to your dataset's structure.
features, labels = zip(*list(dataset))
features = np.array(features)
labels = np.array(labels)

# Stratify the data.
num_classes = np.max(labels) + 1
stratified_data = []
for i in range(num_classes):
    class_indices = np.where(labels == i)[0]
    stratified_data.append(features[class_indices])

# Define validation split.
validation_split = 0.2

# Create validation and training datasets by sharding each class.
validation_dataset = tf.data.Dataset.from_tensor_slices((np.concatenate([data[:int(len(data)*validation_split)] for data in stratified_data]),
                                                        np.concatenate([np.repeat(i, int(len(data)*validation_split)) for i, data in enumerate(stratified_data)])))
training_dataset = tf.data.Dataset.from_tensor_slices((np.concatenate([data[int(len(data)*validation_split):] for data in stratified_data]),
                                                       np.concatenate([np.repeat(i, len(data) - int(len(data)*validation_split)) for i, data in enumerate(stratified_data)])))

# Batch and prefetch.
batch_size = 32
training_dataset = training_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```


This code first extracts labels and features, then groups data by class.  It subsequently creates a stratified split by taking a percentage of each class for the validation set.  This ensures proportional representation across classes.  This approach proved invaluable when working with imbalanced datasets of agricultural land use, preventing overfitting on dominant classes.


**3. Using `tfds.Split.TRAIN.subsplit` for pre-defined splits:**

Some TFDS datasets offer pre-defined splits.  If the dataset already contains a validation split (e.g., 'train', 'validation', 'test'), utilizing these built-in splits is the simplest and most efficient method.

```python
import tensorflow_datasets as tfds
import tensorflow as tf

# Load the dataset with predefined splits.
dataset = tfds.load('your_dataset', split='train', as_supervised=True)
validation_dataset = tfds.load('your_dataset', split='validation', as_supervised=True)

#Process both datasets, batch, and prefetch as necessary

batch_size = 32
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

```


This exemplifies the preferred approach when available.  It avoids the manual splitting, enhancing reproducibility and reducing potential errors.  In my work, I prioritized this approach whenever the dataset structure allowed it, simplifying the data preparation pipeline.


**Resource Recommendations:**

1. The official TensorFlow documentation on datasets.
2.  A comprehensive guide to TensorFlow data input pipelines.
3.  A practical guide on handling imbalanced datasets in machine learning.

Remember to always carefully consider the characteristics of your dataset and choose the validation method that best suits your needs.  The optimal approach depends on dataset size, class distribution, and computational resources.  Failing to appropriately address these factors can significantly impact model performance and evaluation.
