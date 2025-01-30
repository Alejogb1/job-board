---
title: "How can TensorFlow operations be distributed across GPUs?"
date: "2025-01-30"
id: "how-can-tensorflow-operations-be-distributed-across-gpus"
---
TensorFlow's distributed strategy, particularly for utilizing multiple GPUs, hinges on explicit device placement and data parallelization. I've personally observed significant performance gains when transitioning from single-GPU training to a properly configured multi-GPU setup, especially with large models and datasets. The framework does not automatically distribute operations; rather, it requires developer specification of where computations occur.

A fundamental concept is the use of TensorFlow's `tf.distribute` API, which provides high-level abstractions for distributing training across multiple devices. Specifically, the `tf.distribute.MirroredStrategy` is commonly employed for data parallelism across multiple GPUs on a single machine. This strategy replicates the model on each device, distributing a batch of data across all replicas, and synchronizes gradients during training.

The distribution process starts with creating a `MirroredStrategy` instance, specifying which devices (GPUs) to utilize. This is typically achieved using the `devices` parameter, a list of strings indicating GPU device identifiers. TensorFlow then manages replicating the model's variables and computations to these devices. Crucially, data must be fed into the model in a manner that matches the device allocation strategy. This often involves dividing the input batch among replicas, a task handled transparently by the strategy when paired with a suitable `tf.data.Dataset`.

The core workflow can be described in three stages: First, establish the distribution strategy and associated devices. Second, define and build the model within the strategy's scope, ensuring that all computations occur within the designated device context. Third, prepare the dataset using methods compatible with the distribution strategy. This dataset will then feed data to each replica. Gradients are calculated locally on each replica and are then aggregated before being applied to the model's variables, maintaining consistency across all replicas.

Now, let’s look at some practical examples. Consider a scenario where I'm training a convolutional neural network (CNN) on a dataset of images using two GPUs on a local machine.

```python
import tensorflow as tf

# 1. Define Distribution Strategy
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# 2. Build the Model Within the Strategy's Scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def loss_function(y_true, y_pred):
        return loss_fn(y_true, y_pred)

    def accuracy_function(y_true, y_pred):
        return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

    train_metric = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()

# 3. Prepare Dataset (assume some 'train_images', 'train_labels' are already defined)
# Example using dummy data
train_images = tf.random.normal((64, 28, 28, 1))
train_labels = tf.random.uniform((64,10), minval=0, maxval=2, dtype=tf.int32)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
distributed_dataset = strategy.experimental_distribute_dataset(train_dataset)

# 4. Define Training Step
@tf.function
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_function(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    accuracy = accuracy_function(labels,predictions)

    train_metric.update_state(loss)
    train_accuracy.update_state(accuracy)

# 5. Define Distributed Training
@tf.function
def distributed_train_step(inputs):
    strategy.run(train_step, args=(inputs,))

# 6. Training Loop
epochs = 2
for epoch in range(epochs):
    train_metric.reset_states()
    train_accuracy.reset_states()
    for batch in distributed_dataset:
        distributed_train_step(batch)

    print(f"Epoch {epoch+1} Loss: {train_metric.result():.4f} Accuracy: {train_accuracy.result():.4f}")
```

In this example, a `MirroredStrategy` is initialized targeting `/gpu:0` and `/gpu:1`. The model, optimizer, loss function, and evaluation metrics are instantiated within the `strategy.scope()`, ensuring that all operations are executed across the specified GPUs. I've incorporated a simple loss calculation, and included updates to metrics for training status. Crucially, the dataset is distributed using `strategy.experimental_distribute_dataset`, and each step of training is managed via the `strategy.run` method, which is essential for the operations to execute on the assigned devices. Note I also utilized `tf.function` to compile `train_step` and `distributed_train_step`, a recommended practice to maximize performance when using a strategy.

Another critical aspect when performing distributed training is how one handles stateful objects like metrics. Because the training process is distributed, it's essential that these objects are also updated consistently across different replicas. I've demonstrated this with the metrics here, which are updated within the train step. When training, each replica calculates and updates their own instance of these metrics, and the aggregated result is finally returned at the end of the training loop. The `Mean` metric is a useful tool here since it performs a local mean update per device before a final aggregation, which avoids incorrect results if we only calculate a mean once we have all of the per-replica results.

Next, let's consider the use of a different strategy, specifically `tf.distribute.MultiWorkerMirroredStrategy`. This strategy is tailored for training across multiple machines, each possibly equipped with multiple GPUs.

```python
import tensorflow as tf
import os

# Setup environment variables for multi-worker strategy (for demonstration on a single machine)
os.environ['TF_CONFIG'] = """
{
    "cluster": {
        "worker": ["localhost:12345", "localhost:12346"]
    },
    "task": {"type": "worker", "index": 0}
}
"""

# 1. Define the Multi-Worker Mirrored Strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# 2. Build the Model Within the Strategy's Scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def loss_function(y_true, y_pred):
        return loss_fn(y_true, y_pred)

    def accuracy_function(y_true, y_pred):
        return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

    train_metric = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()

# 3. Prepare Dataset (assume some 'train_features', 'train_labels' are already defined)
train_features = tf.random.normal((64, 10))
train_labels = tf.random.uniform((64, 10), minval=0, maxval=2, dtype=tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
distributed_dataset = strategy.distribute_datasets_from_function(
      lambda input_context:
        train_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id))

# 4. Define Training Step
@tf.function
def train_step(inputs):
    features, labels = inputs
    with tf.GradientTape() as tape:
      predictions = model(features)
      loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy = accuracy_function(labels,predictions)

    train_metric.update_state(loss)
    train_accuracy.update_state(accuracy)

# 5. Define Distributed Training
@tf.function
def distributed_train_step(inputs):
    strategy.run(train_step, args=(inputs,))


# 6. Training Loop
epochs = 2
for epoch in range(epochs):
    train_metric.reset_states()
    train_accuracy.reset_states()
    for batch in distributed_dataset:
        distributed_train_step(batch)
    print(f"Epoch {epoch+1} Loss: {train_metric.result():.4f} Accuracy: {train_accuracy.result():.4f}")

```

This example, while running on a single machine for simplicity, simulates a multi-worker environment by using environment variables to define the cluster configuration. The key here is the `strategy.distribute_datasets_from_function` which shards the dataset across different workers, with the logic for how to shard contained in the lambda function. It is important to note that for true multi-worker setup, each machine must be able to reach the others via the network specified in `TF_CONFIG` and that in that situation, each machine is running a worker. Note again the use of `tf.function` on `train_step` and `distributed_train_step` to maximise performance. Also note the use of the metrics and the requirement to reset them at the start of every epoch.

Finally, let's demonstrate using `tf.distribute.experimental.CentralStorageStrategy`, a strategy that is suited for large models where the memory constraints are an issue.  This strategy stores variables on the CPU, and computation occurs on the GPU, leading to a potential bottleneck in variable movement, but allows larger models.

```python
import tensorflow as tf
import os

# Define Environment variables for central storage, similar to multi-worker.
os.environ['TF_CONFIG'] = """
{
    "cluster": {
        "worker": ["localhost:12345"]
    },
    "task": {"type": "worker", "index": 0}
}
"""

# 1. Define Central Storage Strategy
strategy = tf.distribute.experimental.CentralStorageStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")


# 2. Build Model Within Strategy Scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def loss_function(y_true, y_pred):
        return loss_fn(y_true, y_pred)

    def accuracy_function(y_true, y_pred):
       return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

    train_metric = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()

# 3. Prepare Dataset (using dummy data)
train_features = tf.random.normal((64, 10))
train_labels = tf.random.uniform((64, 10), minval=0, maxval=2, dtype=tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

distributed_dataset = strategy.experimental_distribute_dataset(train_dataset)


# 4. Define Training Step
@tf.function
def train_step(inputs):
    features, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_function(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy = accuracy_function(labels,predictions)

    train_metric.update_state(loss)
    train_accuracy.update_state(accuracy)

# 5. Define Distributed Training
@tf.function
def distributed_train_step(inputs):
  strategy.run(train_step, args=(inputs,))

# 6. Training Loop
epochs = 2
for epoch in range(epochs):
    train_metric.reset_states()
    train_accuracy.reset_states()
    for batch in distributed_dataset:
        distributed_train_step(batch)
    print(f"Epoch {epoch+1} Loss: {train_metric.result():.4f} Accuracy: {train_accuracy.result():.4f}")

```
The key differentiator here is again the chosen strategy. Using this strategy results in variables residing on the CPU, while the computation is handled by the available GPU, requiring explicit management of variable transfer. This results in slower training compared to a more memory-intensive approach. It is important to note that this strategy is an experimental feature, and should be used with care. The `tf.distribute` API continues to evolve, and using more modern or recommended options for distribution are advisable. The use of `tf.function`, the strategy context scope, and the distributed dataset also occur here.

To delve deeper into distributed TensorFlow training, I recommend consulting the official TensorFlow documentation on distributed training strategies. Also, reading literature on parallel computing and data-parallelism will provide valuable background. The TensorFlow tutorials available from its core team are another useful source, as are the examples provided when installing TensorFlow. The various training strategies and their performance characteristics are explained well, along with concrete practical examples.
