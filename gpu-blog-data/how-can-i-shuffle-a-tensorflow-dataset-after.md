---
title: "How can I shuffle a TensorFlow dataset after each epoch?"
date: "2025-01-30"
id: "how-can-i-shuffle-a-tensorflow-dataset-after"
---
TensorFlow datasets, when used for training deep learning models, often benefit from random shuffling of samples after each complete pass, or epoch, over the data. Failing to shuffle can lead to biases during training, especially if the underlying dataset has an inherent order. I've encountered this issue frequently during my work on time-series forecasting models, where sequential data needs randomization to avoid learning patterns specific to that order. Achieving effective, epoch-based shuffling requires careful manipulation of the TensorFlow dataset pipeline using techniques specifically designed for this purpose.

The fundamental challenge lies in the fact that TensorFlow datasets are designed for efficient, streamed processing, rather than loading the entire dataset into memory. This is particularly true for large-scale datasets, making conventional list-based shuffling operations impractical. Thus, we must leverage the tools within the `tf.data` API that allow for controlled randomization without sacrificing performance.

The core operation for shuffling is achieved using the `.shuffle()` method available on `tf.data.Dataset` objects. However, a simple `.shuffle()` call won't automatically reshuffle after each epoch. Instead, the shuffle buffer would maintain the same order across epochs because the dataset is evaluated once.  To achieve the desired per-epoch shuffling, we must create a new shuffled dataset each epoch. This typically involves creating a new dataset from the original source each time an epoch starts. This can be handled inside the training loop or by using functions designed to return the dataset when needed.

Let's examine how this is implemented in practice. The following examples will illustrate different approaches for handling epoch-based shuffling in `tf.data`. Each example will be self-contained and provide a commented explanation of each step involved.

**Example 1: Shuffling Within a Loop Using a Generator**

This approach involves defining a generator function that produces a new shuffled dataset upon request. This is particularly useful in custom training loops, where we have fine-grained control over each epoch.

```python
import tensorflow as tf
import numpy as np

def create_shuffled_dataset(data, buffer_size, batch_size):
    """Creates a shuffled dataset.

      Args:
          data: The data source, e.g., numpy array.
          buffer_size: The shuffle buffer size.
          batch_size: The batch size.

      Returns:
         A tf.data.Dataset instance.
      """
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    return ds

# Example Usage:
data = np.random.rand(100, 10) # 100 samples with 10 features.
BUFFER_SIZE = 100
BATCH_SIZE = 16
EPOCHS = 3

for epoch in range(EPOCHS):
  print(f"Epoch: {epoch + 1}")
  shuffled_ds = create_shuffled_dataset(data, BUFFER_SIZE, BATCH_SIZE)
  for batch in shuffled_ds:
    # Simulate training step by print the first element
    print(batch[0, :])
```

*Commentary:*
1.  `create_shuffled_dataset(data, buffer_size, batch_size)`: This function takes the data, a buffer size for shuffling, and a batch size as arguments. It creates a `tf.data.Dataset` from the provided data.
2.  `ds.shuffle(buffer_size)`: This line is the critical component. It shuffles the dataset using a buffer. `buffer_size` dictates how many elements are pre-loaded into the buffer before any shuffling occurs; a larger buffer generally results in more randomness.
3.  `ds.batch(batch_size)`:  The shuffled dataset is then batched into the specified size, which are the batches processed during training.
4.  The main training loop iterates through the number of epochs, calls `create_shuffled_dataset` to get the shuffled data each epoch. Then iterates through the batches in the dataset simulating a training step by printing the first element of each batch.

This demonstrates how you effectively get a new shuffling of data for each training epoch.

**Example 2: Shuffling Using a Function for Dataset Re-initialization**

This approach involves using a function designed to return a shuffled dataset, often embedded within a custom training model class. This creates a more object-oriented structure and can be more readable in larger projects.

```python
import tensorflow as tf
import numpy as np

class DatasetShuffler:
    def __init__(self, data, buffer_size, batch_size):
        self.data = data
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def get_shuffled_dataset(self):
        """Creates and returns a shuffled tf.data.Dataset."""
        ds = tf.data.Dataset.from_tensor_slices(self.data)
        ds = ds.shuffle(self.buffer_size)
        ds = ds.batch(self.batch_size)
        return ds

# Example Usage:
data = np.random.rand(100, 10) # 100 samples with 10 features.
BUFFER_SIZE = 100
BATCH_SIZE = 16
EPOCHS = 3

dataset_handler = DatasetShuffler(data, BUFFER_SIZE, BATCH_SIZE)

for epoch in range(EPOCHS):
  print(f"Epoch: {epoch + 1}")
  shuffled_ds = dataset_handler.get_shuffled_dataset()
  for batch in shuffled_ds:
    # Simulate training step by print the first element
    print(batch[0, :])
```
*Commentary:*
1. `DatasetShuffler` class: This encapsulates dataset management, taking data, buffer size, and batch size as initialization parameters.
2.  `get_shuffled_dataset`: This method returns a new shuffled dataset. Its core functionality is identical to `create_shuffled_dataset` in the prior example, but it is contained in the class.
3.  An instance of `DatasetShuffler` is created. The main training loop iterates and uses the instance to retrieve the new shuffled dataset each epoch, showing that the same effect is achieved via a more object oriented approach.

This approach improves organization when the dataset manipulation is part of a larger training framework.

**Example 3: Shuffling Within Keras Model Training**

When using the Keras API, you don't necessarily control the training loop, but you can achieve epoch based shuffling by returning a new dataset each time.

```python
import tensorflow as tf
import numpy as np

def get_shuffled_dataset(data, buffer_size, batch_size):
        """Creates and returns a shuffled tf.data.Dataset."""
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size)
        return ds

# Example Usage:
data = np.random.rand(100, 10) # 100 samples with 10 features.
labels = np.random.randint(0, 2, 100) # 100 binary labels.
BUFFER_SIZE = 100
BATCH_SIZE = 16
EPOCHS = 3

# A simple model.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

for epoch in range(EPOCHS):
  print(f"Epoch: {epoch + 1}")
  shuffled_ds = get_shuffled_dataset(data, BUFFER_SIZE, BATCH_SIZE)
  # Keras doesn't accept the tensorflow dataset directly; must convert to input and target tensors.
  dataset_inputs = []
  dataset_targets = []
  for batch in shuffled_ds:
      dataset_inputs.append(batch)
      dataset_targets.append(labels[0:len(batch)])
  
  dataset_inputs = np.concatenate(dataset_inputs)
  dataset_targets = np.concatenate(dataset_targets)

  model.fit(dataset_inputs, dataset_targets, epochs=1, verbose=0)
  
  # Simulate training step by print a message
  print("Finished Training")
```

*Commentary:*
1.  The core shuffle implementation in `get_shuffled_dataset` remains the same, returning a shuffled dataset.
2.  A Keras sequential model is defined, and compiled.
3.  The training loop iterates through epochs. A new dataset is created each time and then converted back into numpy array data so that the keras model.fit() can use it for training.
4.  `model.fit(dataset_inputs, dataset_targets, epochs=1, verbose=0)` will train on the shuffled data for one epoch.

This demonstrates that with some adjustments, one can shuffle the data for Keras based training, although the more modern way would be to use a custom fit loop as implemented above.

When working with datasets that require more complex preprocessing, it is essential to ensure that your preprocessing steps occur *after* the shuffle operation. Failure to do so could compromise the randomness that you are aiming to achieve.

To deepen understanding of TensorFlow datasets and shuffling, I recommend exploring the official TensorFlow documentation under the `tf.data` API. Also, research papers and tutorials focusing on efficient training methods with TensorFlow are helpful resources. The TensorFlow tutorials on data loading and training loops are especially relevant.  Furthermore, understanding buffer sizes in more detail can fine-tune shuffling operations for optimal performance. Experimenting with various buffer sizes allows for a more thorough comprehension of the concept and its impact.
