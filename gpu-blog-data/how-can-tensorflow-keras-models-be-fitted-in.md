---
title: "How can TensorFlow Keras models be fitted in parallel?"
date: "2025-01-30"
id: "how-can-tensorflow-keras-models-be-fitted-in"
---
Fitting TensorFlow Keras models in parallel, particularly across multiple GPUs or machines, requires careful consideration of data distribution and model synchronization. My experience over several years implementing large-scale deep learning systems has shown that naive parallelization can easily lead to inconsistent results or limited performance gains. The core challenge isn’t simply dividing the computation; it's ensuring that model updates, gradients, and data remain coherent and synchronized.

The primary strategy for parallel model training in TensorFlow Keras revolves around `tf.distribute.Strategy`. This abstraction encapsulates how training is distributed across different devices, be they multiple GPUs on a single machine or multiple machines in a cluster. I find that using a strategy is nearly always preferable to manual data splitting and gradient calculation, mainly due to its built-in optimizations and ease of use. TensorFlow provides several pre-built strategies, each tailored for different hardware setups and performance trade-offs.

For scenarios involving multiple GPUs on a single machine, `tf.distribute.MirroredStrategy` is a workhorse. This strategy replicates the model onto each GPU and distributes the incoming training data batches. Gradients computed on each replica are aggregated, and the model is updated based on the averaged result. This synchronous approach ensures consistency across replicas, but the speed increase isn't linear, especially with high device count.

A somewhat less synchronous approach, and useful for scenarios spanning multiple machines, is `tf.distribute.MultiWorkerMirroredStrategy`. This is an extension of the MirroredStrategy, designed to coordinate training across multiple workers or machines. Each worker maintains a model replica, and data distribution and gradient aggregation happen across a network. A critical aspect of this strategy is the correct configuration of the cluster, often relying on environment variables or configuration files to define worker addresses and communication ports. Synchronization here becomes a more complex undertaking, requiring a fault-tolerant communication layer.

The final strategy worth mentioning is `tf.distribute.ParameterServerStrategy`, though I've encountered it less frequently. Parameter servers decouple model parameters from the training workers, allowing more flexibility in distributing model parameters across different machines. This can be advantageous when model size exceeds single machine memory, but it introduces further complexities in management.

Now, let's illustrate these with some practical examples. The following examples show the conceptual approach, not necessarily a copy-paste ready solution for all environments due to variations in setup details.

**Example 1: Using MirroredStrategy (Multiple GPUs on a Single Machine)**

```python
import tensorflow as tf

# Detect available devices.
gpus = tf.config.list_physical_devices('GPU')

# Define distribution strategy
if gpus:
  strategy = tf.distribute.MirroredStrategy(devices=gpus)
else:
  strategy = tf.distribute.get_strategy()


with strategy.scope():
  # Define a Keras model (example: a simple dense model)
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  # Define an optimizer
  optimizer = tf.keras.optimizers.Adam()

  # Define a loss function
  loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Compile the model and train it
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Create a sample dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

model.fit(dataset, epochs=5, validation_data=dataset_test)
```

Here, the model, optimizer, and loss function are instantiated within the `strategy.scope()`. The dataset, already in `tf.data` format, is automatically distributed by the strategy during the fitting process, without requiring explicit batch shuffling across devices. TensorFlow handles the necessary data sharding and gradient aggregation behind the scenes. If GPUs aren't available, the code falls back to using whatever strategy TensorFlow decides on by default.

**Example 2: MultiWorkerMirroredStrategy (Training across multiple machines)**

```python
import tensorflow as tf
import os

# Necessary for multiple worker strategy
os.environ['TF_CONFIG'] = '{"cluster": {"worker": ["10.10.10.1:12345", "10.10.10.2:12345"]}, "task": {"type": "worker", "index": 0}}'

# Specify which strategy to use based on the environment.
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
  # Define a simple Keras model.
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.CategoricalCrossentropy()


model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Sample dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)


model.fit(dataset, epochs=5, validation_data=dataset_test)
```

This second example demonstrates a basic multi-worker setup using `MultiWorkerMirroredStrategy`. In practice, each machine/worker would run this code snippet after correctly setting the environment variable `TF_CONFIG`. Note that the example TF_CONFIG setting should be adapted to the actual IP addresses of the machines being used for training. Further setup involves properly configuring the relevant communication ports which are needed for worker synchronization.

**Example 3: A Custom Training Loop Example**

```python
import tensorflow as tf

# Detect available devices.
gpus = tf.config.list_physical_devices('GPU')

if gpus:
  strategy = tf.distribute.MirroredStrategy(devices=gpus)
else:
  strategy = tf.distribute.get_strategy()

with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.CategoricalCrossentropy()

  def train_step(inputs, labels):
    with tf.GradientTape() as tape:
      predictions = model(inputs)
      loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

  @tf.function
  def distributed_train_step(inputs, labels):
     per_replica_losses = strategy.run(train_step, args=(inputs, labels,))
     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)


# Load a simple dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).repeat()

for inputs, labels in dataset.take(1000):
    loss = distributed_train_step(inputs, labels)
    print(f"loss: {loss}")
```

In this example, we move away from the standard `model.fit` and implement our own custom training loop. While initially more complex, this approach gives fine-grained control. The `distributed_train_step` is decorated with `tf.function`, and the `strategy.run` function orchestrates the training across devices. Then, the losses are aggregated using `strategy.reduce`.

For further exploration, I suggest reviewing TensorFlow’s official documentation on distributed training. It offers exhaustive details on specific strategy nuances, configuration options, and performance optimization techniques. Also, consider resources focused on practical large-scale machine learning implementations; these offer invaluable hands-on experience and often address common pitfalls I've encountered. Books covering advanced deep learning or distributed systems would also be valuable. Finally, exploring example implementations in the TensorFlow GitHub repository is a great way to see how this all fits together in more complicated applications. Using these resources and keeping experimentation based on real world requirements in mind has proven invaluable in my efforts over the past few years.
