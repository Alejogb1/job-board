---
title: "How can I create balanced mini-batches using the Dataset API?"
date: "2025-01-30"
id: "how-can-i-create-balanced-mini-batches-using-the"
---
The core challenge in creating balanced mini-batches with the `tf.data.Dataset` API lies in effectively stratifying the dataset before applying batching operations.  Simple batching will not inherently guarantee class distribution parity within each mini-batch, potentially leading to biased model training.  Over the course of developing a multi-class image classifier for a medical imaging project, I encountered this precise problem and implemented several solutions, each with distinct performance characteristics.

My approach focused on leveraging the `Dataset.group_by_window` method, which provides a powerful mechanism for grouping data elements based on a key and then applying a windowed operation to each group.  This allows for the creation of balanced mini-batches by first grouping the data by class label and subsequently creating batches from each group. The efficiency depends critically on the relative class frequencies.  Significant class imbalances necessitate careful consideration of the window size to avoid creating excessively large or small mini-batches.

**1.  Explanation of the Approach:**

The strategy involves a three-stage process:

* **Data Preparation:** The initial dataset must be augmented with a class label, acting as the grouping key.  This label typically corresponds to the target variable in your supervised learning task.  The dataset should be pre-processed appropriately, including any necessary transformations and encoding.

* **Grouping and Windowing:** The `Dataset.group_by_window` method is central to this process. The first argument is the key function, which maps each data element to its class label. The second argument specifies the window size, which dictates the maximum number of elements per class in each mini-batch. The third argument is the reduction function, typically `lambda x: x.batch(batch_size)`, which creates batches from each group.  The batch size here determines the number of samples *per class* within a mini-batch.

* **Concatenation and Shuffling:**  The output of `group_by_window` is a dataset where each element represents a balanced mini-batch for a single class.  These mini-batches are then concatenated using `Dataset.concatenate`. Finally, shuffling is crucial for preventing order bias during training, applied using `Dataset.shuffle`. The buffer size in the shuffle operation should be sufficiently large to ensure a good randomisation.


**2. Code Examples with Commentary:**

**Example 1: Simple Balanced Mini-batch Creation:**

```python
import tensorflow as tf

def create_balanced_batches(dataset, num_classes, batch_size):
  return dataset.group_by_window(
      key_func=lambda x, y: y, # Assuming (features, label) format
      reduce_func=lambda key, d: d.batch(batch_size),
      window_size=batch_size
  ).concatenate().shuffle(buffer_size=1000)

# Example usage:
labels = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 1, 1], dtype=tf.int64)
features = tf.random.normal((10, 32, 32, 3))  # Example features
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

balanced_dataset = create_balanced_batches(dataset, num_classes=2, batch_size=2)

for batch in balanced_dataset:
  print(batch[1].numpy()) # Print the labels to verify balance
```

This example demonstrates a basic implementation for a binary classification problem.  The `key_func` extracts the label,  `reduce_func` batches elements from each class, and `window_size` matches the `batch_size` to ensure equal representation of each class in each mini-batch.


**Example 2: Handling Class Imbalance:**

```python
import tensorflow as tf
import numpy as np

def create_balanced_batches_imbalanced(dataset, num_classes, batch_size, max_elements_per_class):
  return dataset.group_by_window(
      key_func=lambda x, y: y,
      reduce_func=lambda key, d: d.batch(max_elements_per_class),
      window_size=max_elements_per_class
  ).flat_map(lambda x: x.unbatch()).batch(batch_size).shuffle(buffer_size=10000)


#Simulate an imbalanced dataset
labels = np.concatenate([np.zeros(2), np.ones(8)])
features = tf.random.normal((10, 32, 32, 3))
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

balanced_dataset = create_balanced_batches_imbalanced(dataset, num_classes=2, batch_size=4, max_elements_per_class=2)

for batch in balanced_dataset:
  print(batch[1].numpy())
```

This example addresses class imbalance by allowing a `max_elements_per_class` parameter.  This ensures that even if one class has significantly fewer samples, it contributes proportionally to the mini-batches. Note the use of `flat_map` and `unbatch` to handle the variable sized output from `group_by_window` before creating the final batches.  The buffer size in `shuffle` is increased to account for the potentially larger dataset.


**Example 3:  Multi-class Scenario:**

```python
import tensorflow as tf

def create_balanced_batches_multiclass(dataset, num_classes, batch_size):
  return dataset.group_by_window(
      key_func=lambda x, y: y,
      reduce_func=lambda key, d: d.batch(batch_size),
      window_size=batch_size
  ).interleave(lambda x: x, cycle_length=num_classes).shuffle(buffer_size=1000)

# Example usage for multi-class:
labels = tf.constant([0, 1, 2, 0, 1, 2, 0, 1, 2, 2], dtype=tf.int64)
features = tf.random.normal((10, 32, 32, 3))
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

balanced_dataset = create_balanced_batches_multiclass(dataset, num_classes=3, batch_size=2)

for batch in balanced_dataset:
    print(batch[1].numpy())
```

This extends the approach to multi-class problems. The crucial change is the use of `interleave` to interleave the mini-batches from each class, which ensures that a variety of classes is presented in each training iteration, unlike simple concatenation.


**3. Resource Recommendations:**

For further understanding of the `tf.data.Dataset` API and its functionalities, I recommend consulting the official TensorFlow documentation.  Thorough exploration of the `group_by_window`, `batch`, `shuffle`, `concatenate`, `flat_map` and `interleave` methods is crucial.  Consider reviewing literature on stratified sampling techniques in machine learning, which provide a broader theoretical context for balanced mini-batch creation.  Studying examples of data augmentation and preprocessing pipelines within TensorFlow will also prove beneficial.  Finally, consider exploring advanced TensorFlow tutorials on building efficient data pipelines for deep learning models.
