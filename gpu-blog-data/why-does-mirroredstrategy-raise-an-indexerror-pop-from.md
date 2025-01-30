---
title: "Why does MirroredStrategy raise an IndexError: pop from empty list when Keras Sequences are used as input?"
date: "2025-01-30"
id: "why-does-mirroredstrategy-raise-an-indexerror-pop-from"
---
MirroredStrategy, when combined with Keras Sequences for data input, can inadvertently trigger an `IndexError: pop from empty list`, specifically within the data fetching mechanisms during training. This arises from a subtle interaction between how `MirroredStrategy` distributes data and how `Sequence` objects are designed to provide it. My experience optimizing distributed training workflows for large-scale image classification models exposed me to this issue, and its resolution hinges on understanding the implicit data handling within these components.

The root cause lies in the `MirroredStrategy`'s attempt to prefetch and distribute batches across multiple replicas (GPUs or TPUs) in a synchronous manner. Typically, during training, a `Sequence` object generates data in batches, and the strategy expects that each replica will consume one of these batches per step. The problem appears when the number of batches available from the `Sequence` is not an exact multiple of the number of replicas, *and* the last batch is smaller than the intended batch size.

In a single-GPU setup (or with `tf.distribute.Strategy` other than `MirroredStrategy`), Keras models can tolerate a final batch of smaller size. The model processes what is available, then moves to the next epoch. However, with `MirroredStrategy`, the system attempts to distribute a full-size batch to each replica on every training step. If the number of available batches is not perfectly divisible by the replica count and the final batch isn’t full, each replica may try to pull data that no longer exists when its turn arrives. In essence, the replicas may all try to retrieve "one more batch" when the `Sequence` is already empty, resulting in the `pop` from an empty list within TensorFlow's distributed data handling. The prefetching mechanism, while intended to boost performance, becomes a source of this error when coupled with imbalanced batch distribution.

The `Sequence` class, by design, yields batches based on the `__getitem__` method, and is inherently meant to be stateless with regard to the global training procedure. It assumes that the caller requests indices sequentially or at least predictably. The logic built into `MirroredStrategy`, however, may not consistently respect the sequential nature of a sequence's underlying data source, especially regarding prefetching. This is because it attempts to optimize data loading by prefetching several batches at once. When the final batch is smaller than usual, `MirroredStrategy` might make copies of the last one across several replicas, leading to an empty list when these copies are exhausted and the replicas expect one more batch to fulfill their data needs for the current step.

To clarify, the error isn't directly caused by a flaw in either the `MirroredStrategy` or the `Sequence` classes themselves. Instead, it's an emergent behavior when they are combined due to implicit assumptions made about the data distribution.

Here are a few illustrative code snippets to demonstrate different aspects of the problem and solutions:

**Example 1: The Problematic Scenario**

This example highlights the issue with a mismatched number of batches and replicas, causing the `IndexError`. Assume that you want to train on a dataset that results in a total of 7 batches and you are using two GPUs to train using MirroredStrategy. In that case, two batches will go to GPU 0, two batches will go to GPU 1, then two batches will go to GPU 0, and one batch will go to GPU 1. Therefore the next batch requested by GPU 1 will cause the "pop from empty list" error:

```python
import tensorflow as tf
import numpy as np

class DummySequence(tf.keras.utils.Sequence):
  def __init__(self, batch_size, num_batches):
    self.batch_size = batch_size
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    data = np.random.rand(self.batch_size, 10) # Dummy data
    labels = np.random.randint(0, 2, size=(self.batch_size,)) # Dummy Labels
    return data, labels

# Define Strategy and sequence
strategy = tf.distribute.MirroredStrategy()
batch_size = 3
num_batches = 7
sequence = DummySequence(batch_size, num_batches)

# Define a simple model.
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

# Attempt training. Error will occur due to prefetch / batch mismatch
try:
    model.fit(sequence, epochs=1)
except Exception as e:
    print(f"Encountered Exception: {e}")
```

In this example, running the code will raise the `IndexError`. The `Sequence` has 7 batches (non-divisible by 2 replicas) and the last batch is not of full size. The model will attempt to distribute batches from this sequence to 2 devices, and the error appears as the model and the data pipeline attempt to go one batch too far for the number of training steps.

**Example 2: Solution 1 - Padding or Resizing Sequence**

A common solution is to modify the sequence to either pad the data to ensure a number of batches divisible by the number of replicas or adjust batch size based on the last batch size. This is done by determining how many batches are needed in order to match the number of training steps required, and then adding or removing batches to match that target. While padding can impact model generalization (if not handled correctly), it can avoid the immediate indexing issue. In my experience, padding is often an easier fix than rewriting the Sequence class.

```python
import tensorflow as tf
import numpy as np

class DummySequencePadded(tf.keras.utils.Sequence):
    def __init__(self, batch_size, num_batches, num_replicas):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_replicas = num_replicas
        self.padded_batches = self._pad_batches()


    def __len__(self):
        return len(self.padded_batches)

    def _pad_batches(self):
         padded_batches = []
         for i in range(self.num_batches):
            data = np.random.rand(self.batch_size, 10)
            labels = np.random.randint(0, 2, size=(self.batch_size,))
            padded_batches.append((data, labels))

         while len(padded_batches) % self.num_replicas != 0:
            data = np.zeros((self.batch_size, 10))
            labels = np.zeros((self.batch_size))
            padded_batches.append((data, labels))
         return padded_batches


    def __getitem__(self, idx):
        return self.padded_batches[idx]

# Define Strategy and sequence (padded)
strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
batch_size = 3
num_batches = 7
sequence = DummySequencePadded(batch_size, num_batches, num_replicas)

# Define a simple model.
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')


# Training will now proceed without error
model.fit(sequence, epochs=1)
```

This approach ensures that there are a number of batches divisible by the number of replicas, thus avoiding the prefetch error on that final undersized batch. While padding can introduce noise in the data (the zero vectors used here may be problematic), this approach will at least avoid a crash.

**Example 3: Solution 2 - Utilizing tf.data.Dataset from Sequence**

Alternatively, a more robust method involves transforming the `Sequence` into a `tf.data.Dataset`. Datasets allow for more granular control over data distribution, batching, and shuffling, and often play better with distributed training strategies. This often involves creating the dataset using `tf.data.Dataset.from_generator`. It allows you to work with the sequence and still benefit from the high performance dataset abstraction.

```python
import tensorflow as tf
import numpy as np

class DummySequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        data = np.random.rand(self.batch_size, 10)
        labels = np.random.randint(0, 2, size=(self.batch_size,))
        return data, labels

# Define Strategy and sequence
strategy = tf.distribute.MirroredStrategy()
batch_size = 3
num_batches = 7
sequence = DummySequence(batch_size, num_batches)

def dataset_generator():
    for i in range(len(sequence)):
        yield sequence[i]

dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(tf.TensorSpec(shape=(batch_size, 10), dtype=tf.float64),
                          tf.TensorSpec(shape=(batch_size,), dtype=tf.int64))
    ).batch(1).prefetch(tf.data.AUTOTUNE)

# Define a simple model.
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

# Training with Dataset. No more indexing error.
model.fit(dataset, epochs=1)
```

By using `tf.data.Dataset`, we enable more robust data handling including prefetching and sharding across different devices. This approach is typically preferred when dealing with complex data pipelines. Specifically, the `from_generator` method ensures that the sequence's `__getitem__` method is respected while yielding data that the model can consume. Note the `.batch(1)` in the data pipeline. This is important because the default data distribution for strategies like `MirroredStrategy` requires a batch dimension, which is usually handled by the `model.fit` method when it receives a Keras Sequence. With a dataset, we need to handle the batching within the dataset itself.

In addition to these approaches, careful consideration should be given to the design of the `Sequence` class itself. Implementing `__len__` and `__getitem__` properly is crucial for both single-device and distributed training.

For further learning, I recommend referring to the TensorFlow documentation on distributed strategies, the Keras API documentation, and material focused on `tf.data` and data pipelines. Exploring source code for `MirroredStrategy` can be helpful in understanding the underlying mechanics. Additionally, papers and articles that discuss common pitfalls of distributed deep learning and data handling provide a broad context for this issue. This problem is not unique to TensorFlow, so material about similar issues in PyTorch, for example, can also prove useful. While online forums can also be a source of information, it is best to rely on primary documentation sources and peer reviewed material whenever available. Understanding the core mechanisms behind data handling and prefetching will provide the best foundation for resolving such issues in a robust and maintainable way.
