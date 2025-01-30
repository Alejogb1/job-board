---
title: "How does `steps_per_epoch` affect multi-worker distributed training on Google AI Platform?"
date: "2025-01-30"
id: "how-does-stepsperepoch-affect-multi-worker-distributed-training-on"
---
In distributed training using TensorFlow with Google AI Platform, the `steps_per_epoch` parameter, despite its seemingly straightforward definition, directly impacts training efficiency and can lead to subtle but significant performance variations, especially when multiple workers are involved. The core issue stems from how TensorFlow distributes data and calculates metrics across different workers. Specifically, `steps_per_epoch` determines the number of training steps the model undergoes before the framework calculates validation loss, and a mismatch between its value and the actual data distribution can lead to under- or over-utilization of resources.

Fundamentally, a single "step" in TensorFlow training corresponds to one iteration of the optimization algorithm, typically involving a single batch of data. When training on a single device, `steps_per_epoch` is usually chosen such that one epoch involves iterating through the entire dataset exactly once, or nearly once. However, in a multi-worker distributed setting, each worker receives only a fraction of the total dataset. This is accomplished using distributed data strategies like `tf.distribute.MultiWorkerMirroredStrategy`. The division method, whether based on sharding the original dataset or other techniques, introduces complexities when calculating `steps_per_epoch`. If the `steps_per_epoch` is set expecting all data to be processed by one worker, the effective number of training iterations in the distributed setting becomes smaller. Conversely, if `steps_per_epoch` is set according to per-worker data volume but not scaled, total data processed per epoch becomes far larger than anticipated.

The implications are multifaceted. If the value provided to `steps_per_epoch` is inappropriately low compared to per-worker data availability, workers will finish their training before the epoch is technically considered complete. The model will then perform validation on a partial epoch, causing inaccurate loss calculations and reduced utilization of resources, leading to decreased performance. If the `steps_per_epoch` is too high, each worker might loop through their local data multiple times before the epoch concludes, thus repeating data and impacting generalization capabilities. Effective distributed training requires that `steps_per_epoch` be set such that the total number of training steps across all workers per epoch equals the amount that would correspond to a single pass through the full dataset. This often requires a calculation that considers the number of workers.

Now, let us examine some code examples. The following snippet illustrates an incorrect implementation of `steps_per_epoch` and its consequences. Assume we have a dataset of 1000 samples and 2 workers, each receiving 500 samples. We set `steps_per_epoch=1000` assuming all data is processed by a single worker.

```python
import tensorflow as tf

# Assume data preparation is done elsewhere
num_samples = 1000
batch_size = 32
num_workers = 2

# Create a sample dataset for example purposes only
dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal((num_samples, 10)))
dataset = dataset.batch(batch_size)
global_batch_size = batch_size * num_workers

# Assume a distributed strategy is already configured:
strategy = tf.distribute.MultiWorkerMirroredStrategy()


with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    optimizer = tf.keras.optimizers.Adam()

epochs = 10
steps_per_epoch = num_samples // batch_size # Incorrect

model.compile(optimizer=optimizer, loss='mse')


history = model.fit(dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch)

print(history.history) # will have inconsistent number of steps trained per epoch
```

In the above code, setting `steps_per_epoch` to `num_samples // batch_size` calculates the number of batches when all the data is passed through the model, irrespective of the number of workers. This is incorrect; with two workers, each worker receives only half the data, so, each worker would finish iterating through its dataset far faster, thus resulting in each epoch being executed on a fraction of the data. This means that the loss calculations will be incomplete, and the training process isn't utilizing available resources effectively.

The next example provides a correction by adjusting `steps_per_epoch` based on the number of workers involved. We calculate the number of batches each worker needs to process such that the sum across all workers equals one pass over the entire dataset.

```python
import tensorflow as tf

# Assume data preparation is done elsewhere
num_samples = 1000
batch_size = 32
num_workers = 2

# Create a sample dataset for example purposes only
dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal((num_samples, 10)))
dataset = dataset.batch(batch_size)
global_batch_size = batch_size * num_workers

# Assume a distributed strategy is already configured:
strategy = tf.distribute.MultiWorkerMirroredStrategy()


with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    optimizer = tf.keras.optimizers.Adam()

epochs = 10
steps_per_epoch = num_samples // global_batch_size # Correct

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch)

print(history.history) # steps trained per epoch are now consistent
```

In this revised code, we modify the calculation for `steps_per_epoch` to `num_samples // global_batch_size`. Here, the `global_batch_size` is calculated as the `batch_size` multiplied by `num_workers`. This ensures that during one epoch, the model processes the total number of samples in the dataset, regardless of how many workers are involved. As each worker processes data in `batch_size`, multiplying that by the number of workers gives us the effective data process per batch across the distributed system. This provides accurate training and validation cycles.

Lastly, some caution needs to be exercised when the dataset cannot be divided evenly among the workers. The following shows a more robust implementation of calculating `steps_per_epoch`. Here, we calculate and use the total batches available, ensuring a consistent evaluation of the model across all workers.

```python
import tensorflow as tf
import numpy as np

# Assume data preparation is done elsewhere
num_samples = 1001 # Not divisible by batch size for demonstration
batch_size = 32
num_workers = 2

# Create a sample dataset for example purposes only
dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal((num_samples, 10)))
dataset = dataset.batch(batch_size)
global_batch_size = batch_size * num_workers

# Assume a distributed strategy is already configured:
strategy = tf.distribute.MultiWorkerMirroredStrategy()


with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    optimizer = tf.keras.optimizers.Adam()

epochs = 10
total_batches = len(list(dataset.as_numpy_iterator()))
steps_per_epoch = int(np.ceil(total_batches/num_workers)) # Correct for non-even divisions

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch)

print(history.history) # steps trained per epoch are now consistent
```

In this implementation, we iterate through the dataset and calculate the `total_batches`. By dividing this number of batches by the number of workers and rounding up using `np.ceil`, we ensure that every worker participates in training on its portion of the dataset, without skipping batches when data cannot be evenly divided. This is necessary as not every dataset will be cleanly divisible by the global batch size, and this robust approach ensures consistent epoch durations, which is crucial for fair and reliable training results.

In summary, appropriate setting of `steps_per_epoch` in multi-worker distributed training on Google AI Platform requires careful consideration of both the total size of your dataset and the distribution strategy used. Incorrectly setting this value leads to underutilized computational resources, biased loss calculation, and inconsistent training behavior, and will make it harder to obtain performant models.

For further understanding and exploration, I suggest consulting the TensorFlow official documentation for `tf.distribute` strategies and dataset handling. The Keras API documentation offers deep insights into the `model.fit` method and its parameters, which is critical for effectively utilizing `steps_per_epoch`. Google AI Platform’s documentation provides context and guidance for configuring distributed training jobs, including best practices for dataset handling and strategy selection. These resources will significantly help in fine-tuning multi-worker training procedures.
