---
title: "How to resolve OSError 'Errno 9' Bad file descriptor when deploying a TensorFlow Estimator model on multiple GPUs using MirroredStrategy?"
date: "2025-01-30"
id: "how-to-resolve-oserror-errno-9-bad-file"
---
The `OSError: [Errno 9] Bad file descriptor` during multi-GPU deployment of a TensorFlow Estimator model using `tf.distribute.MirroredStrategy` typically surfaces from an underlying resource contention issue, specifically with how file handles are managed across multiple processes when using the `Estimator` API. This is not an inherent flaw of `MirroredStrategy` itself, but rather a consequence of its interaction with the `Estimator`'s data input pipeline. I've encountered this firsthand during large-scale training runs on a cluster with eight NVIDIA V100s, where early experiments frequently exhibited this error.

The root cause is often that the `Estimator`'s input function, which is executed within each replica process (one per GPU in a `MirroredStrategy` context), opens files independently and without proper coordination. For example, consider the scenario where your training data is read from multiple TFRecord files. Each replica might attempt to open the same file at nearly the same time, leading to a race condition in the operating system’s file descriptor management. While TensorFlow attempts to use its own mechanisms for data prefetching and caching, these aren't always sufficient to prevent this contention, particularly with very large or numerous files. The operating system limits the number of simultaneously opened file descriptors per process; when exceeded, an `OSError [Errno 9]` is raised. This is exacerbated in distributed training because the strategy replicates the input function, thus multiplying the problem.

The initial intuition might lead one to believe the issue is about TensorFlow, but the real problem is with the underlying operating system file descriptor limit. Each process, corresponding to each GPU in `MirroredStrategy`, typically has a hard and soft limit for file handles. When the input function opens files on every replica without any strategy for avoiding contention, the OS limits can easily be surpassed. This is further aggravated by internal TensorFlow file handles used for checkpointing or other internal mechanisms, thus reducing available handles for data I/O. It is worth noting that this problem is more frequent when reading data from disk, as opposed to having data in memory.

Let's illustrate this issue and its solutions with a series of code examples.

**Example 1: Problematic Code - Uncoordinated File Opening**

```python
import tensorflow as tf
import os

def input_fn(params):
    batch_size = params['batch_size']
    file_pattern = "data/training_data_*.tfrecord"  # Assume multiple TFRecord files
    file_list = tf.io.gfile.glob(file_pattern)

    dataset = tf.data.TFRecordDataset(file_list)
    def parse_function(example_proto):
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_png(features['image_raw'], channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        label = features['label']
        return image, label

    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size)
    return dataset

def model_fn(features, labels, mode, params):
    #Simplified model for illustration
    dense = tf.keras.layers.Dense(10, activation=tf.nn.relu)(tf.reshape(features, [-1, 3072]))
    logits = tf.keras.layers.Dense(2)(dense)
    loss = tf.losses.sparse_categorical_crossentropy(labels, logits)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )

# Example usage with MirroredStrategy (Problematic)
strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
    train_distribute=strategy,
    eval_distribute=strategy,
    save_checkpoints_steps=100
)
params = {"batch_size": 64}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=config)

estimator.train(input_fn=lambda: input_fn(params), steps=1000) # This often results in OSError [Errno 9]
```

In this example, each replica will open the TFRecord files at the same time, which can quickly exceed the file descriptor limit. It's not about TensorFlow's usage of datasets but the underlying operating system file limits.

**Example 2: Solution - Using `tf.data.Dataset.interleave`**

```python
import tensorflow as tf
import os

def input_fn(params):
    batch_size = params['batch_size']
    file_pattern = "data/training_data_*.tfrecord"
    file_list = tf.io.gfile.glob(file_pattern)

    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def parse_function(example_proto):
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_png(features['image_raw'], channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        label = features['label']
        return image, label

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def model_fn(features, labels, mode, params):
     #Simplified model for illustration
    dense = tf.keras.layers.Dense(10, activation=tf.nn.relu)(tf.reshape(features, [-1, 3072]))
    logits = tf.keras.layers.Dense(2)(dense)
    loss = tf.losses.sparse_categorical_crossentropy(labels, logits)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )


# Example usage with MirroredStrategy (Solution)
strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
    train_distribute=strategy,
    eval_distribute=strategy,
    save_checkpoints_steps=100
)
params = {"batch_size": 64}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=config)


estimator.train(input_fn=lambda: input_fn(params), steps=1000)
```

The key change here is replacing direct usage of `tf.data.TFRecordDataset(file_list)` with `tf.data.Dataset.from_tensor_slices(file_list)` followed by an `interleave`. `interleave` creates parallel threads that read the files on demand, reducing the chance that all replicas try to open the same set of files simultaneously. The arguments `cycle_length` and `num_parallel_calls` are set to `AUTOTUNE` for automatic optimization by TensorFlow, which usually yields good performance. The `map` step also utilizes `num_parallel_calls` for parallel processing. This effectively parallelizes the loading and processing of the data, distributing the load and reducing file descriptor conflicts.

**Example 3: Further Solution - Sharding within Dataset**

```python
import tensorflow as tf
import os

def input_fn(params):
    batch_size = params['batch_size']
    file_pattern = "data/training_data_*.tfrecord"
    file_list = tf.io.gfile.glob(file_pattern)

    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    num_shards = params.get('num_shards', 1)
    shard_index = params.get('shard_index', 0)

    if num_shards > 1:
        dataset = dataset.shard(num_shards, shard_index)

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    def parse_function(example_proto):
            feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
            features = tf.io.parse_single_example(example_proto, feature_description)
            image = tf.io.decode_png(features['image_raw'], channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            label = features['label']
            return image, label

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def model_fn(features, labels, mode, params):
    #Simplified model for illustration
    dense = tf.keras.layers.Dense(10, activation=tf.nn.relu)(tf.reshape(features, [-1, 3072]))
    logits = tf.keras.layers.Dense(2)(dense)
    loss = tf.losses.sparse_categorical_crossentropy(labels, logits)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )

# Example usage with MirroredStrategy (Solution with sharding)
strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
config = tf.estimator.RunConfig(
    train_distribute=strategy,
    eval_distribute=strategy,
    save_checkpoints_steps=100
)
params = {"batch_size": 64, "num_shards": num_replicas, 'shard_index': strategy.cluster_resolver.task_id}


estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=config)

estimator.train(input_fn=lambda: input_fn(params), steps=1000)
```

Here, `tf.data.Dataset.shard()` is employed. Sharding divides the dataset into non-overlapping pieces, with each replica processing a specific subset. Each replica (GPU) will read different files, avoiding conflict. We query `strategy.num_replicas_in_sync` to know how many shards to create, and also get the replica ID, `strategy.cluster_resolver.task_id` to pass in as the `shard_index` to `input_fn`. This ensures each replica is given a unique set of files to open, minimizing file handle contention. While `interleave` can often be enough, sharding is usually needed with large or numerous files.

For additional learning, I recommend exploring the TensorFlow official documentation on `tf.data` API, particularly focusing on the section about `tf.data.Dataset`, `tf.data.TFRecordDataset`, and techniques for improving data pipeline performance using `interleave` and `shard`. Also, review the usage of `tf.distribute` strategies including `MirroredStrategy`. Understanding how the data flows within the framework is essential. Furthermore, I suggest research into best practices for file system management in distributed computing environments, particularly within the context of Linux systems, since the error originates at the OS level. Consulting resources on operating system level file descriptors will provide important context.
