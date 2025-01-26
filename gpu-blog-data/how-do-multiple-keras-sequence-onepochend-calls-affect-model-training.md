---
title: "How do multiple Keras Sequence on_epoch_end() calls affect model training?"
date: "2025-01-26"
id: "how-do-multiple-keras-sequence-onepochend-calls-affect-model-training"
---

I've encountered issues with unexpected behavior when using `on_epoch_end()` within multiple Keras `Sequence` instances during a single training process. The critical point is that `on_epoch_end()` callbacks, when implemented as instance methods within `keras.utils.Sequence` subclasses, are invoked *per instance* rather than at the end of a global epoch across all data sources. This distinction is crucial for understanding why seemingly innocuous multi-sequence setups can lead to training artifacts and incorrect evaluation metrics.

The standard Keras `fit()` method progresses through training epochs, where each epoch involves iterating over batches of data provided by the training data generator. When using a single `Sequence` as the training data, the `on_epoch_end()` method is called once after all batches from that sequence have been processed. This aligns with the intuitive notion of an epoch. However, if training is performed using multiple `Sequence` instances, these `on_epoch_end()` calls are decoupled and executed sequentially, in the order the sequences were provided within the fit method. The global understanding of a 'single training epoch' is broken down into local 'sequence epochs.'

The consequence is that actions performed within `on_epoch_end()`, such as shuffling data indices or logging metrics, will be executed independently for each sequence instance. For instance, if shuffling data is intended to introduce variability across the *entire* training data, shuffling will occur only within the limited scope of a single sequence. Similarly, if validation is performed during `on_epoch_end()`, that validation step will be carried out against subsets of the overall training data, leading to inaccurate representations of model performance at the epoch level. This becomes particularly troublesome when the number of sequences is large, because validation is now occurring very frequently and might not provide relevant information.

Here are some concrete examples to illustrate this issue:

**Example 1: Data shuffling within `on_epoch_end()`**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

class ShufflingSequence(keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_indices = self.indices[start:end]
        return self.data[batch_indices], np.ones_like(batch_indices)

    def on_epoch_end(self):
        print("Shuffling data in sequence")
        np.random.shuffle(self.indices)

data1 = np.arange(10)
data2 = np.arange(10, 20)

seq1 = ShufflingSequence(data1, batch_size=2)
seq2 = ShufflingSequence(data2, batch_size=2)


model = keras.models.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit([seq1, seq2], epochs=2, verbose = 0)

#Output with the print statements will show each sequence has a shuffle
# before it is used
```

In this example, we have two `ShufflingSequence` instances. If we observed the output, we would see the “Shuffling data in sequence” print statement called twice within each epoch; once for sequence 1 and again for sequence 2. The shuffling action applies only to its sequence data, not to the combined dataset. If the intention was to shuffle all data jointly, this approach fails.

**Example 2: Validation within `on_epoch_end()` with multiple sequences**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

class ValidationSequence(keras.utils.Sequence):
    def __init__(self, data, batch_size, validation_data):
        self.data = data
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data))
        self.validation_data = validation_data

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_indices = self.indices[start:end]
        return self.data[batch_indices], np.ones_like(batch_indices)


    def on_epoch_end(self):
      print("Performing validation on sequence")
      val_loss = model.evaluate(self.validation_data, verbose = 0)
      print(f"Validation loss on subset: {val_loss}")

data1 = np.arange(10)
data2 = np.arange(10, 20)
val_data = (np.arange(0, 20), np.ones(20))

seq1 = ValidationSequence(data1, batch_size=2, validation_data = (val_data[0][:10], val_data[1][:10]))
seq2 = ValidationSequence(data2, batch_size=2, validation_data = (val_data[0][10:], val_data[1][10:]))


model = keras.models.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit([seq1, seq2], epochs=2, verbose = 0)

#Output of print statements will show validation occurring on partial sets
#of the validation set each time
```

Here, we have two `ValidationSequence` instances, each assigned a portion of the overall validation data. Because the `on_epoch_end()` is called twice, the model is evaluated on different validation *subsets* each time, again giving a misleading picture of model performance for the end of epoch result. It is only at the final epoch and final sequence where the *full* evaluation can be thought of.

**Example 3: Modified global state in `on_epoch_end()`**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

global_counter = 0

class CounterSequence(keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_indices = self.indices[start:end]
        return self.data[batch_indices], np.ones_like(batch_indices)

    def on_epoch_end(self):
        global global_counter
        global_counter +=1
        print(f"Global counter value:{global_counter}")

data1 = np.arange(10)
data2 = np.arange(10, 20)

seq1 = CounterSequence(data1, batch_size=2)
seq2 = CounterSequence(data2, batch_size=2)

model = keras.models.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit([seq1, seq2], epochs=2, verbose = 0)

# The global counter gets updated twice per epoch.
```

This example demonstrates the issues that arise with modifying global variables. Each call to `on_epoch_end()` increments the `global_counter` variable independently, which means the counter reflects not the "global" state for a true epoch, but instead for a sequence-level epoch. This will be confusing as it doesn't necessarily accurately reflect the number of global epochs.

To mitigate this issue, avoid relying on the `on_epoch_end()` of individual `Sequence` instances to perform global tasks. Instead, use callbacks provided by Keras, such as `keras.callbacks.Callback`, which has the `on_epoch_end` method which is called once for all sequences per epoch. These allow you to track metrics or perform actions after each *global* epoch rather than each sequence epoch. When using multiple sequences, it can be easier to define a custom callback. If, however, there is an absolute requirement to use `on_epoch_end()` in a `Sequence`, be mindful that these are local calls per sequence and not to mix local concerns with global concerns, such as validation or metric tracking.

For further exploration, I suggest reviewing the Keras documentation on callbacks and custom data generators. Specifically, exploring the `keras.callbacks.Callback` API provides a better understanding of the global callbacks available during the training process. Additionally, studying the implementation of the `Sequence` class and its integration within the `Model.fit()` method, can be valuable. Understanding the separation of concerns is essential for using multiple `Sequence` instances effectively. Focus on the design of the individual `Sequence` to manage the data and avoid making assumptions about global state. Remember that `on_epoch_end()` will be called per sequence and to design your code accordingly.
