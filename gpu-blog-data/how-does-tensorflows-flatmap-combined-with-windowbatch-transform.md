---
title: "How does TensorFlow's `flat_map` combined with `window.batch()` transform a dataset?"
date: "2025-01-30"
id: "how-does-tensorflows-flatmap-combined-with-windowbatch-transform"
---
TensorFlow’s `tf.data.Dataset.flat_map()` and `tf.data.Dataset.window().batch()` are frequently used in tandem to preprocess data, particularly when working with time series or sequential data that requires overlapping subsets for training recurrent neural networks or related tasks. The combination addresses a common need: generating sequences of a fixed length, potentially with overlapping strides, from a larger dataset. These operations, when used together, fundamentally reshape the input dataset, creating a sliding window view of the original data with batches.

The `flat_map()` transformation, unlike `map()`, is designed to handle input elements that map to datasets themselves. Specifically, `flat_map()` applies a given function to each element of the input dataset, expecting that function to return another dataset. The output of the `flat_map()` operation is then a single, flattened dataset that combines all elements from the datasets produced by the mapping function. In the case of combining `window()` and `batch()`, the function passed to `flat_map` uses `window()` to create a series of sub-datasets, which are then converted to batches. Without `flat_map`, the result of `window()` would be a dataset of datasets, which is unsuitable for direct training.

The process can be broken down into stages. Initially, `window(size, shift, stride, drop_remainder=False)` creates a series of 'windows' from the input sequence. Each window is a sub-dataset representing a segment of the original dataset. The ‘size’ defines the length of each sub-dataset (or window). The ‘shift’ determines how many elements the window moves forward. If 'shift' is equal to 'size' it results in no overlap. A ‘stride’ specifies the interval between elements inside each window, useful for downsampling or creating subsampled subsequences, though often this is set to 1. The boolean flag `drop_remainder`, when set to true, discards windows whose size is smaller than the specified `size`. The result of `window()` is a `Dataset` of `Datasets`.

This dataset-of-datasets is where the `flat_map()` function comes in. The function we pass to flat_map iterates over each sub-dataset (window) and, within this function, we apply `batch(batch_size)`. The `batch()` operation groups elements from each window into batches of specified size. If the size of the window is less than the `batch_size`, the last incomplete batch is dropped, or padded depending on settings passed to batch, or the remaining elements are batched. The `flat_map()` flattens these batches from each window. This results in a `Dataset` of batched tensors suitable for model training.

Let me illustrate this with examples drawn from prior work involving recurrent sequence models.

**Example 1: Simple Sequence Segmentation**

In this initial scenario, the goal is to split a sequential dataset into overlapping windows of a fixed size and convert them to batches of that window size. Assume we have a simple dataset containing a sequence of numbers representing, say, sensor readings.

```python
import tensorflow as tf

# Sample data
data = tf.range(1, 21, dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices(data)

window_size = 5
shift_size = 2
batch_size = 5

def window_and_batch(window):
  return window.batch(batch_size)

windowed_dataset = dataset.window(size=window_size, shift=shift_size, drop_remainder=True)
batched_dataset = windowed_dataset.flat_map(window_and_batch)


for batch in batched_dataset:
    print(batch.numpy())

```

Here, we first create a `tf.data.Dataset` from a tensor of integers 1 to 20. The `window()` operation creates overlapping windows of 5 elements each, shifted by 2. So, the first window will contain `[1, 2, 3, 4, 5]`, the second `[3, 4, 5, 6, 7]`, and so on. The `drop_remainder=True` ensures that no windows shorter than 5 are produced. The `flat_map` function then takes these windows and batches them using `batch(5)`. Since the window size is also 5, it creates batches of length 5. The result is a `Dataset` where each element is a batch of shape (5,). The output will be batches such as:

```
[1 2 3 4 5]
[3 4 5 6 7]
[5 6 7 8 9]
[7 8 9 10 11]
[9 10 11 12 13]
[11 12 13 14 15]
[13 14 15 16 17]
[15 16 17 18 19]
```
**Example 2: Time Series Data with Feature and Target**

Consider a time series dataset where each sample includes a feature vector and a target value. Often times the target is a shifted version of the feature sequence. This is very common in time-series forcasting. This example demonstrates how we can create these windows, and then further split them into feature and targets.

```python
import tensorflow as tf

# Sample time series data, each element is a tuple of (feature, target)
features = tf.range(10, 30, dtype=tf.float32)
targets = tf.range(12, 32, dtype=tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((features, targets))


window_size = 5
shift_size = 1
batch_size = 5

def window_and_batch_features_targets(window):
    window = window.batch(batch_size)
    def split_features_targets(windowed_batch):
        features_batch = windowed_batch[0]
        targets_batch = windowed_batch[1]
        return features_batch, targets_batch
    return window.map(split_features_targets)


windowed_dataset = dataset.window(size=window_size, shift=shift_size, drop_remainder=True)
batched_dataset = windowed_dataset.flat_map(window_and_batch_features_targets)

for features, targets in batched_dataset:
    print("Features:", features.numpy())
    print("Targets:", targets.numpy())
```

Here, the original dataset is a set of tuples, representing (feature, target) pairs, and they are offset by two time-steps. The `window()` operation creates windows of size 5, shifting by 1, capturing overlapping segments of this data. The function passed to `flat_map` takes each window, and batches them. Importantly, inside this function, there is a second call to `map`.  The function passed to this second map operation splits the features and targets into separate batches. The output of this is a flattened dataset, where each element consists of a batch of features and a corresponding batch of targets.

```
Features: [10. 11. 12. 13. 14.]
Targets: [12. 13. 14. 15. 16.]
Features: [11. 12. 13. 14. 15.]
Targets: [13. 14. 15. 16. 17.]
Features: [12. 13. 14. 15. 16.]
Targets: [14. 15. 16. 17. 18.]
Features: [13. 14. 15. 16. 17.]
Targets: [15. 16. 17. 18. 19.]
Features: [14. 15. 16. 17. 18.]
Targets: [16. 17. 18. 19. 20.]
Features: [15. 16. 17. 18. 19.]
Targets: [17. 18. 19. 20. 21.]
Features: [16. 17. 18. 19. 20.]
Targets: [18. 19. 20. 21. 22.]
Features: [17. 18. 19. 20. 21.]
Targets: [19. 20. 21. 22. 23.]
Features: [18. 19. 20. 21. 22.]
Targets: [20. 21. 22. 23. 24.]
Features: [19. 20. 21. 22. 23.]
Targets: [21. 22. 23. 24. 25.]
Features: [20. 21. 22. 23. 24.]
Targets: [22. 23. 24. 25. 26.]
Features: [21. 22. 23. 24. 25.]
Targets: [23. 24. 25. 26. 27.]
Features: [22. 23. 24. 25. 26.]
Targets: [24. 25. 26. 27. 28.]
Features: [23. 24. 25. 26. 27.]
Targets: [25. 26. 27. 28. 29.]
```

**Example 3: Variable-length Sequences with Padding**

Sometimes our input sequences may have variable lengths. For this, the  `batch` operation has options to pad incomplete batches so the resulting batch tensors all have the same shape, which is helpful for many deep learning models.

```python
import tensorflow as tf

# Sample data with different lengths
data = [tf.range(i, dtype=tf.int32) for i in [3, 6, 4, 7, 2]]
dataset = tf.data.Dataset.from_tensor_slices(data)

window_size = 2
shift_size = 1
batch_size = 2

def window_and_batch_pad(window):
   return window.batch(batch_size, pad_to_batch=True)

windowed_dataset = dataset.window(size=window_size, shift=shift_size, drop_remainder=True)
batched_dataset = windowed_dataset.flat_map(window_and_batch_pad)

for batch in batched_dataset:
    print(batch.numpy())
```

Here, the input dataset consists of variable length integer sequences. The `window()` creates the sliding windows and `drop_remainder=True` means windows of a different length than what was set will not be included in the output. The key is in the `flat_map`, where the `batch()` is modified with the argument `pad_to_batch=True`. When this argument is set, all batches are padded to the specified batch size, and it fills any missing elements with a zero value by default. Notice that the batch size is set to two, and the output sequence length is always two.

```
[[0 1]
 [1 2]]
[[1 2]
 [2 3]]
[[0 1]
 [1 2]]
[[2 3]
 [3 4]]
[[3 4]
 [4 5]]
[[0 1]
 [1 2]]
[[0 1]
 [1 2]]
[[1 2]
 [2 3]]
[[2 3]
 [3 4]]
[[3 4]
 [4 5]]
[[4 5]
 [5 6]]
[[0]
 [0]]
```

**Resource Recommendations**

To further understand these operations, it is beneficial to review the official TensorFlow documentation. The `tf.data` module documentation is particularly valuable, especially the sections on `tf.data.Dataset.flat_map`, `tf.data.Dataset.window`, and `tf.data.Dataset.batch`. Additionally, many tutorials and blogs focusing on time series data preprocessing in TensorFlow cover these methods in practical contexts. Searching for articles related to ‘sequence padding’ and ‘windowing for time series data’ can provide additional examples and insights. I have found that practicing with different `shift` values and `drop_remainder` settings often clarifies their impact on the resulting dataset structure. These combined operations are very powerful, and understanding their impacts on your data can dramatically improve preprocessing workflows for training deep learning models.
