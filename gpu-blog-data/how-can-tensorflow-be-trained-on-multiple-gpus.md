---
title: "How can TensorFlow be trained on multiple GPUs with varying memory capacities?"
date: "2025-01-30"
id: "how-can-tensorflow-be-trained-on-multiple-gpus"
---
TensorFlow's capability to distribute training across multiple GPUs, especially those with heterogeneous memory constraints, requires careful configuration and a strategic approach to data management. This scenario is not merely about splitting the workload; it's about optimizing resource utilization to prevent out-of-memory errors on smaller GPUs while still leveraging the collective compute power. My experience building large-scale recommendation systems taught me that naive distribution often leads to performance bottlenecks and instability.

The core challenge stems from the fact that TensorFlow, by default, attempts to allocate memory on each device upfront, typically the maximum available. When GPUs have varying amounts of RAM, this can cripple devices with limited capacity. The solution centers around two key mechanisms: *memory growth* and *distribution strategies*, with further fine-tuning possible via data sharding and custom gradient accumulation techniques.

Firstly, enabling memory growth allows TensorFlow to allocate only what's immediately needed, and it expands memory usage as required. This prevents eager allocation of the entire GPU memory pool. It’s configured using the `tf.config.experimental.set_memory_growth()` function.  If not set, TensorFlow can pre-allocate almost all of the GPU memory, causing issues for the GPUs with lower capacity. I found this essential when initially deploying a model on a mixed-GPU cluster where some were older models with significantly less onboard memory.

Secondly, choosing the correct distribution strategy is crucial. TensorFlow provides several options, but `tf.distribute.MirroredStrategy` is often the first choice for multi-GPU setups within a single machine. However, it operates under the assumption that each GPU will handle a copy of the entire model and a portion of the batch. When GPUs differ significantly in their memory, this poses a problem, especially with larger models. The strategy can be configured to only allocate on GPUs that are enabled, and should be done after enabling memory growth. Further, for heterogeneous memory environments, we can move to strategies that involve data sharding or model parallelism. Data sharding implies that each device receives a unique subset of the data to train, and this eliminates redundant storage of the entire batch. This approach tends to work well but can require more complex coding and management.

Let’s look at the memory growth implementation. The example below illustrates the general principle:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # This is required for virtual devices to be created
        for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set successfully")
    except RuntimeError as e:
        print(e)
        print("Error setting memory growth")

# Example model (placeholder)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
  tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

#Dummy data
import numpy as np
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 10, 100)
y_train = tf.one_hot(y_train, depth = 10)


# Basic training loop example (single epoch)
for i in range(100):
  with tf.GradientTape() as tape:
      predictions = model(x_train)
      loss = loss_fn(y_train, predictions)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Model trained successfully.")
```

In this initial example, the primary focus is enabling memory growth before defining our model, optimizers, loss functions, and dummy data. This is absolutely critical before any model is trained in a mixed GPU system, as otherwise the system is likely to report out of memory issues. The dummy model and data are for illustrative purposes; in practice, one would load a real model, a dataset, and perform more epochs of training. The primary function here is not to showcase a model, but to demonstrate the principle behind proper memory management, which was the major source of my issues in early deployments.

Now let's move on to using a mirrored strategy for a multiple GPU setup.

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set successfully")
    except RuntimeError as e:
        print(e)
        print("Error setting memory growth")


    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
      model = tf.keras.Sequential([
          tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
          tf.keras.layers.Dense(10, activation='softmax')
        ])

      optimizer = tf.keras.optimizers.Adam(0.001)
      loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Dummy Data
    import numpy as np
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 10, 100)
    y_train = tf.one_hot(y_train, depth=10)

    # Training step function
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_fn(y, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Training loop
    for i in range(100):
      loss = strategy.run(train_step, args=(x_train, y_train))
      print("Loss at step {}: {}".format(i, loss))
    print("Training Complete")
```

Here, the major shift is in the inclusion of the MirroredStrategy, a critical part of TensorFlow for multi-GPU distribution. As before, we set the memory growth for all GPUs. Then we encapsulate the model definition and training within the strategy's scope. The `train_step` function is decorated with `tf.function` to improve the training performance, especially in a multi-GPU setting.  Note the use of `strategy.run` to execute the training step. In my experience with mirrored strategies, a substantial performance gain over single GPU training was often achievable, especially for larger model and data sets.

Finally, let's consider data sharding. This example will be more involved, since it will require some dataset preparation. The principle is to split the input data across devices.

```python
import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set successfully")
    except RuntimeError as e:
        print(e)
        print("Error setting memory growth")


    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    num_devices = strategy.num_replicas_in_sync

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(10, activation='softmax')
          ])

        optimizer = tf.keras.optimizers.Adam(0.001)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

    #Dummy Data
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 10, 100)
    y_train = tf.one_hot(y_train, depth=10)

    # Data sharding
    shard_size = len(x_train) // num_devices
    x_train_sharded = [x_train[i * shard_size : (i + 1) * shard_size] for i in range(num_devices)]
    y_train_sharded = [y_train[i * shard_size : (i + 1) * shard_size] for i in range(num_devices)]

    # Training step function, modified to take shards
    @tf.function
    def train_step(x_shard, y_shard):
        with tf.GradientTape() as tape:
            predictions = model(x_shard)
            loss = loss_fn(y_shard, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Training loop
    for epoch in range(100):
        losses = []
        for i in range(num_devices):
            loss = strategy.run(train_step, args=(x_train_sharded[i], y_train_sharded[i]))
            losses.append(loss)
        print(f'Epoch {epoch}, Average loss: {sum(losses)/len(losses)}')

    print("Training Complete")
```

In this last example, I manually split the `x_train` and `y_train` arrays into shards based on the number of GPUs available. The key is that the training function now only processes its local data shard. This is important when the data is so large that it does not fit on the smallest GPU in the system. I was often pushed into using data sharding when working with high-resolution image datasets, where memory allocation became a major bottleneck.  Further this is very useful when you have different generations of GPUs, for example using a mix of older P100s with modern A100s, as the smaller memory footprint of the P100s often is not able to handle a full replica of the data.

For further study and refinement, I recommend exploring TensorFlow documentation on distributed training. Particular attention should be given to the concept of custom training loops for flexibility. Specifically, the official guide for `tf.distribute.Strategy`, and the tutorials on using them for various use cases, are invaluable resources. Additionally, the TensorFlow performance guide and the discussion on memory allocation practices can greatly aid in fine-tuning.  Understanding data pipelines, specifically how `tf.data` can be used to optimize data ingestion is also paramount when dealing with very large datasets. Lastly, investigating the use of custom gradient accumulation, a strategy that can trade training throughput for memory usage, can be beneficial. I have found that practical experiments are always the best teacher, as optimal configurations tend to be highly dependent on the specific hardware setup, model structure, and dataset.
