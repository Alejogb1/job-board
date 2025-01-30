---
title: "How can I efficiently load .npy files larger than 20GB using a Keras/Tensorflow dataloader?"
date: "2025-01-30"
id: "how-can-i-efficiently-load-npy-files-larger"
---
Loading large .npy files, especially those exceeding 20GB, directly into memory using traditional methods will quickly exhaust system resources. This constraint necessitates a memory-efficient approach when using TensorFlow and Keras for model training. I’ve faced this challenge during several image segmentation projects involving large volumetric datasets, and the standard `np.load` followed by feeding into a `tf.data.Dataset` proved utterly inadequate. The key is to leverage memory mapping and custom dataset generators.

A memory map allows NumPy to access a large array stored on disk as if it were entirely in memory, without actually loading the whole thing into RAM at once. This works because NumPy accesses only the relevant portions as needed. Combined with a custom `tf.data.Dataset` generator, this approach allows us to feed the data to our Keras model in batches, which circumvents the memory limitations.

First, we use `np.memmap` to create a view of the .npy file. This doesn't load the entire file into memory but establishes a mapping to the file on disk. After this, a custom generator function will use this mapping to yield batches of data. Finally, we create a `tf.data.Dataset` from that generator, which TensorFlow will use to efficiently feed data into our model.

Here’s a conceptual breakdown:

1. **Create Memory Map:** Use `np.memmap` to map the .npy file.
2. **Custom Generator:** Define a Python generator function that accesses specific slices (batches) from the memory map.
3. **TensorFlow Dataset:** Use `tf.data.Dataset.from_generator` to create a TensorFlow dataset from your custom generator.
4. **Data Preprocessing:** Perform any necessary preprocessing operations within the generator to avoid storing processed data in RAM.

Let’s examine this process through a few concrete examples, assuming the .npy files store image data.

**Example 1: Basic Sequential Data Loading**

This example covers a scenario where the data within the .npy file is a simple sequence (e.g., a list of images or vectors), and the batches are extracted sequentially.

```python
import numpy as np
import tensorflow as tf

def create_dataset(filepath, batch_size, data_shape, dtype=np.float32):
  """Creates a tf.data.Dataset from a memmapped .npy file.

  Args:
    filepath: Path to the .npy file.
    batch_size: Desired batch size.
    data_shape: Shape of the individual data items in the .npy file.
    dtype: Data type of the array.

  Returns:
    A tf.data.Dataset object.
  """

  mmap = np.memmap(filepath, dtype=dtype, mode='r', shape=data_shape)
  data_length = mmap.shape[0]

  def generator():
      for i in range(0, data_length, batch_size):
        yield mmap[i:min(i + batch_size, data_length)]

  dataset = tf.data.Dataset.from_generator(
      generator,
      output_signature=tf.TensorSpec(shape=(None,) + data_shape[1:], dtype=tf.float32)
    )
  return dataset.prefetch(tf.data.AUTOTUNE)

# Example Usage:
filepath = 'large_data.npy' # Assumes large_data.npy is a large file
data_shape = (100000, 256, 256, 3) # Example of a dataset of 100,000 images, 256x256x3
batch_size = 32
dataset = create_dataset(filepath, batch_size, data_shape)

for batch in dataset.take(5):  #Example of taking 5 batches
    print(batch.shape)

```

In this code, `create_dataset` is the core function. It first creates a memory map using `np.memmap`, reading the file in read-only mode (`'r'`).  The generator function iterates through the mmap and yields slices corresponding to batches of the specified `batch_size`. The `tf.data.Dataset.from_generator` function converts this generator into a TensorFlow dataset. The use of `prefetch(tf.data.AUTOTUNE)` optimizes data loading. We then demonstrate the use of this dataset to fetch five batches for training.  Crucially, this function assumes data that can be batched sequentially.

**Example 2: Handling Shuffled Data**

If data needs to be shuffled during training, we modify the generator to access random indices within the memory map.

```python
import numpy as np
import tensorflow as tf
import random

def create_shuffled_dataset(filepath, batch_size, data_shape, dtype=np.float32, seed=42):
    """Creates a shuffled tf.data.Dataset from a memmapped .npy file.

    Args:
        filepath: Path to the .npy file.
        batch_size: Desired batch size.
        data_shape: Shape of the individual data items in the .npy file.
        dtype: Data type of the array.
        seed: Random seed for shuffling.

    Returns:
        A tf.data.Dataset object.
    """
    mmap = np.memmap(filepath, dtype=dtype, mode='r', shape=data_shape)
    data_length = mmap.shape[0]
    random.seed(seed)
    indices = list(range(data_length))

    def generator():
        while True:
            random.shuffle(indices) # Reshuffle indices each epoch
            for i in range(0, data_length, batch_size):
                batch_indices = indices[i:min(i + batch_size, data_length)]
                yield mmap[batch_indices]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(None,) + data_shape[1:], dtype=tf.float32)
        ).prefetch(tf.data.AUTOTUNE)
    return dataset


# Example Usage
filepath = 'large_data.npy'
data_shape = (100000, 256, 256, 3)
batch_size = 32
dataset_shuffled = create_shuffled_dataset(filepath, batch_size, data_shape)


for batch in dataset_shuffled.take(5): #Example of taking 5 batches
    print(batch.shape)

```
Here, we introduce the `create_shuffled_dataset` function. In each call to the generator, the `indices` are randomly shuffled before each epoch is processed. Each time the generator is accessed, it uses a new permutation of these indices to create the batches. We continue to utilize `tf.data.AUTOTUNE` to optimize prefetching. The use of `while True` ensures that the data is read infinitely without interruption during training, since we shuffle the indices on each outer loop. This prevents the dataset generator from ever completing, as needed during model training.

**Example 3: Incorporating Preprocessing**

The most crucial aspect is that any preprocessing operations should ideally occur inside the generator. This avoids creating preprocessed arrays in memory. This is demonstrated in the following code, which performs simple rescaling on the data.

```python
import numpy as np
import tensorflow as tf
import random

def create_preprocess_dataset(filepath, batch_size, data_shape, dtype=np.float32, seed=42):
    """Creates a tf.data.Dataset with preprocessing from a memmapped .npy file.

    Args:
      filepath: Path to the .npy file.
      batch_size: Desired batch size.
      data_shape: Shape of the individual data items in the .npy file.
      dtype: Data type of the array.
      seed: Random seed for shuffling.
    Returns:
      A tf.data.Dataset object.
    """
    mmap = np.memmap(filepath, dtype=dtype, mode='r', shape=data_shape)
    data_length = mmap.shape[0]
    random.seed(seed)
    indices = list(range(data_length))


    def generator():
        while True:
            random.shuffle(indices)
            for i in range(0, data_length, batch_size):
                batch_indices = indices[i:min(i + batch_size, data_length)]
                batch = mmap[batch_indices]
                #Preprocess Here
                batch = batch / 255.0 # Example Rescaling
                yield batch

    dataset = tf.data.Dataset.from_generator(
        generator,
         output_signature=tf.TensorSpec(shape=(None,) + data_shape[1:], dtype=tf.float32)
    ).prefetch(tf.data.AUTOTUNE)
    return dataset

# Example usage
filepath = 'large_data.npy'
data_shape = (100000, 256, 256, 3)
batch_size = 32
dataset_preprocessed = create_preprocess_dataset(filepath, batch_size, data_shape)

for batch in dataset_preprocessed.take(5): # Example of taking 5 batches
    print(batch.shape)
    print(batch.max(), batch.min()) # Check that data is between 0 and 1

```

In this example, the core structure from the previous code is preserved. However, we now divide the batch by 255 within the generator function which does the preprocessing operation of rescaling the values from the range of [0, 255] to [0,1]. The rest of the implementation remains consistent.

Several resources provided further details. The NumPy documentation on memory mapping is essential for understanding the core `np.memmap` functionality. The TensorFlow documentation on `tf.data.Dataset` API provides crucial insight into using and optimizing data pipelines. Additionally, guides on creating custom data loaders using generators help to structure your own specific data loading needs.  These, in concert, have provided a framework that allows me to handle even very large datasets in memory-constrained environments efficiently. It is crucial to test the loading speed and throughput of the loader and adjust the level of prefetching accordingly.
