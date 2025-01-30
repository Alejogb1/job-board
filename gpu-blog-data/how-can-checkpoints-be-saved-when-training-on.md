---
title: "How can checkpoints be saved when training on a TPU?"
date: "2025-01-30"
id: "how-can-checkpoints-be-saved-when-training-on"
---
Tensor Processing Units (TPUs) offer significant acceleration for deep learning tasks, but their distributed nature necessitates a careful approach to checkpointing. I’ve spent considerable time optimizing model training on these accelerators and the checkpointing process is a critical area for stability and efficiency. Standard checkpointing mechanisms designed for single-GPU training often falter within the TPU environment, requiring a different paradigm that leverages the TPU's distributed architecture and the specific API provided by libraries like TensorFlow.

The fundamental challenge lies in the fact that TPUs operate as a cluster of interconnected devices. Training data and model parameters are distributed across these cores. When saving a traditional checkpoint, we essentially collect all parameters onto a single device, often a CPU, write them to disk, and then later restore this complete state. This collection and aggregation step can be severely inefficient and, in many cases, completely impractical, due to the sheer volume of data involved. Furthermore, the centralized I/O operation can quickly become a bottleneck. On a TPU, we must instead checkpoint in a *distributed* fashion. This involves each core independently saving *its own* subset of the parameters, and later restoring *only* the data relevant to its computations. The infrastructure allows this with an abstracted file system that understands sharding.

The main strategies for distributed checkpointing on TPUs revolve around leveraging TensorFlow's `tf.train.CheckpointManager` and its ability to interact with `tf.distribute.TPUStrategy`. The `TPUStrategy` ensures that all model training and checkpoint operations are aware of the distributed environment. The `CheckpointManager` provides tools for tracking the history of saves, implementing custom saving logic, and handling restoration. The core idea is to couple the model architecture and optimizer to this manager, enabling each TPU core to save its portion of the training state to unique files. When restoring, we effectively reverse this process, assigning the saved parameter shards back to their corresponding TPU cores.

Here are three practical code examples demonstrating the key concepts.

**Example 1: Basic checkpointing with `CheckpointManager`**

This example illustrates the standard pattern for initiating and managing checkpoint saves.

```python
import tensorflow as tf

# Assuming a TPU strategy has been defined elsewhere
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
      tf.keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                            directory='./checkpoints',
                                            max_to_keep=5)

# Example save during training (typically in the training loop)
checkpoint_manager.save()

# To restore
latest_checkpoint = checkpoint_manager.latest_checkpoint
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restored.")
```

**Commentary:** This snippet demonstrates the fundamental elements. The `tf.train.Checkpoint` groups the model and optimizer variables that should be saved together. The `tf.train.CheckpointManager` provides a higher-level abstraction, managing multiple checkpoint files and a save/restore process based on their number and modification time. Crucially, the `TPUStrategy` has been defined, meaning that the save and restore operations will automatically respect the distributed context and store each shard to individual files in the directory `./checkpoints`. The  `max_to_keep` argument ensures that older checkpoints are periodically removed, preventing excessive storage consumption. This is especially important in large experiments where frequent checkpoints are required.

**Example 2: Periodic checkpointing within a training loop**

This demonstrates how checkpointing should be integrated into a typical training regime.

```python
import tensorflow as tf
import numpy as np

# Assuming the strategy is set up as in the previous example

with strategy.scope():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
      tf.keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                             directory='./checkpoints',
                                             max_to_keep=5)

  def train_step(inputs, labels):
     with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.keras.losses.MeanSquaredError()(labels, predictions)
     grads = tape.gradient(loss, model.trainable_variables)
     optimizer.apply_gradients(zip(grads, model.trainable_variables))
     return loss

  @tf.function
  def distributed_train_step(inputs, labels):
    per_replica_losses = strategy.run(train_step, args=(inputs, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


  NUM_EPOCHS = 10
  BATCH_SIZE = 64
  NUM_SAMPLES = 1000
  NUM_FEATURES = 10

  dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(NUM_SAMPLES, NUM_FEATURES).astype(np.float32),
                                                np.random.rand(NUM_SAMPLES, 1).astype(np.float32))).batch(BATCH_SIZE)
  dist_dataset = strategy.experimental_distribute_dataset(dataset)

  for epoch in range(NUM_EPOCHS):
    for inputs, labels in dist_dataset:
        loss = distributed_train_step(inputs, labels)
    print(f"Epoch: {epoch+1}, Loss: {loss.numpy()}")
    if (epoch+1) % 2 == 0: # Save every other epoch
        checkpoint_manager.save()
        print("Checkpoint Saved!")
```

**Commentary:** This example embeds the checkpointing process within a model training loop using a distributed dataset. Notice how `distributed_train_step` is wrapped by the `strategy.run` call, which distributes the training logic across the TPU cores. The save is conditionally executed on every other training epoch using the `checkpoint_manager.save()` method. This illustrates the common practice of periodically storing the model's state. The actual saving operation only occurs once per cycle, due to the nature of the `CheckpointManager`, even though we loop through the TPU cores. This approach prevents needless operations. I should point out that a more realistic example might leverage more sophisticated logic regarding when to perform checkpointing, such as saving when validation metrics have improved.

**Example 3: Custom checkpoint names using step number**

Here, we refine the checkpointing process to incorporate the global step number in the filename.

```python
import tensorflow as tf
import numpy as np
# Assuming strategy is defined from previous examples.

with strategy.scope():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

  global_step = tf.Variable(0, dtype=tf.int64)  # Track training steps
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=global_step)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                           directory='./checkpoints_named',
                                           max_to_keep=5)

  def train_step(inputs, labels):
      with tf.GradientTape() as tape:
          predictions = model(inputs)
          loss = tf.keras.losses.MeanSquaredError()(labels, predictions)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      global_step.assign_add(1)
      return loss

  @tf.function
  def distributed_train_step(inputs, labels):
    per_replica_losses = strategy.run(train_step, args=(inputs, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  NUM_EPOCHS = 10
  BATCH_SIZE = 64
  NUM_SAMPLES = 1000
  NUM_FEATURES = 10
  dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(NUM_SAMPLES, NUM_FEATURES).astype(np.float32),
                                                np.random.rand(NUM_SAMPLES, 1).astype(np.float32))).batch(BATCH_SIZE)
  dist_dataset = strategy.experimental_distribute_dataset(dataset)


  for epoch in range(NUM_EPOCHS):
    for inputs, labels in dist_dataset:
        loss = distributed_train_step(inputs, labels)
    print(f"Epoch: {epoch+1}, Loss: {loss.numpy()}")
    if (epoch+1) % 2 == 0:
        checkpoint_manager.save(checkpoint_number=global_step)
        print(f"Checkpoint Saved at step: {global_step.numpy()}")

```

**Commentary:** Here, a dedicated global step counter (`global_step`) is added to the checkpoint object. This allows the `CheckpointManager` to use the current value of `global_step` when saving. We pass it to `checkpoint_manager.save` as the `checkpoint_number`. By using step numbers, it becomes easier to track training progress. These custom names provide more control over the checkpoint file names and can be very useful when trying to reconstruct a training run from multiple checkpoints. I prefer a step-based system as it gives more granular information than an epoch, particularly with varying batch sizes or data loads.

For further study and practical implementations of these concepts, the official TensorFlow documentation on distributed training, specifically focusing on TPU strategy, is essential. The API documentation for `tf.train.Checkpoint` and `tf.train.CheckpointManager` offer detailed parameters and configurations. Also, studying code examples within the TensorFlow models repository, especially those that involve TPU training, is immensely beneficial. Lastly, researching best practices within cloud environments when dealing with very large scale models and checkpointing, will provide insight into a scalable, robust methodology. In my experience, diligent study of both theoretical frameworks and existing, robust implementations is crucial for success in TPU development.
