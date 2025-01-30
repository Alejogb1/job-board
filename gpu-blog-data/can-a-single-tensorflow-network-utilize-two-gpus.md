---
title: "Can a single TensorFlow network utilize two GPUs effectively?"
date: "2025-01-30"
id: "can-a-single-tensorflow-network-utilize-two-gpus"
---
In my experience architecting distributed training pipelines for large language models, achieving optimal utilization of multiple GPUs with a single TensorFlow network requires a nuanced understanding of data parallelism and model parallelism. Directly leveraging two GPUs effectively is possible, but it's not a simple plug-and-play process; specific strategies must be employed to avoid bottlenecks and ensure balanced workload distribution.

The fundamental challenge stems from the fact that TensorFlow, by default, operates within a single device context. Therefore, naively creating a model and expecting it to magically distribute itself across multiple GPUs is not how it functions. To harness the computational power of two GPUs, or more, we must explicitly instruct TensorFlow how to divide the workload. The primary mechanism for this is data parallelism, where identical copies of the model are deployed to each GPU, and the training data is split and fed to each replica. Gradients are then aggregated to update the model weights.

A secondary, more complex technique is model parallelism. Here, different parts of the model reside on different GPUs. This is typically required when a single model exceeds the memory capacity of a single GPU. Model parallelism can be extremely complex to implement, as careful consideration must be given to how activations and gradients are passed between the distributed parts of the network. While the question does not explicitly ask about model parallelism, the context of GPU utilization suggests it might be a relevant consideration. For this response, however, I will concentrate on the more immediately relevant data parallelism.

The easiest way to enable multi-GPU training in TensorFlow is using the `tf.distribute.MirroredStrategy`. This strategy creates copies of the model on each available GPU and handles the data distribution and gradient aggregation automatically. However, this method works well only for models that fit into the memory of a single GPU, as it replicates the entire model across the devices. It will not address a model that needs model parallelism.

Let's illustrate with a code example. Assume we have a simple convolutional neural network designed for image classification.

```python
import tensorflow as tf
import numpy as np

# Define a basic CNN model
def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  return model

# Generate some dummy data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


# Create a MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Inside the strategy's scope, create and train the model
with strategy.scope():
  model = create_model()
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.CategoricalCrossentropy()
  metrics = ['accuracy']

  model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Train the model
model.fit(x_train, y_train, epochs=2, batch_size=64)
```

In this first example, we create a simple convolutional network. Crucially, the model is created and compiled *inside* the `strategy.scope()`. This is mandatory to ensure TensorFlow properly replicates the model on each available GPU. `MirroredStrategy` then handles the data distribution and gradient aggregation transparently. The training data is split into subsets, each assigned to a GPU. After each mini-batch forward pass, gradients are computed, aggregated across the GPUs, and used to update the model's weights.

Now, consider a scenario where we are dealing with a custom training loop. The steps are similar but require more explicit control over the process. We must also carefully ensure our dataset is properly distributed.

```python
import tensorflow as tf
import numpy as np

# Define a basic CNN model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Generate some dummy data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Create a MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Prepare the dataset
batch_size_per_replica = 64
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(global_batch_size)
distributed_dataset = strategy.experimental_distribute_dataset(dataset)

# Inside the strategy's scope, create and train the model
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_metric = tf.keras.metrics.CategoricalAccuracy()

    # Define train step
    @tf.function
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_metric.update_state(labels, predictions)
        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    # Train the model
    epochs = 2
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_inputs in distributed_dataset:
            loss = distributed_train_step(batch_inputs)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {train_metric.result()}")
        train_metric.reset_states()
```

This example presents a more involved approach, using a custom training loop. The dataset is created, batched, and then distributed using `strategy.experimental_distribute_dataset`. The actual training is encapsulated inside `distributed_train_step` which calls the per-replica `train_step` with the data. It is in the per replica step that the forward and back propagation happens. The `strategy.run` command allows each replica to execute its training steps and returns a reduced value across the distributed devices. We also need to scale our loss and metrics correctly by using reduction. This granular control is necessary for advanced scenarios such as checkpointing and more complex optimization strategies.

Finally, let's examine a scenario where we might have other hardware specific configuration needs and use strategy configuration.

```python
import tensorflow as tf
import numpy as np

# Define a basic CNN model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Generate some dummy data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Configure the strategy
strategy = tf.distribute.MirroredStrategy(
    devices = ["/gpu:0","/gpu:1"], #Explicit device specifications
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce() #Optimized gradient aggregation
)

# Prepare the dataset
batch_size_per_replica = 64
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(global_batch_size)
distributed_dataset = strategy.experimental_distribute_dataset(dataset)

# Inside the strategy's scope, create and train the model
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_metric = tf.keras.metrics.CategoricalAccuracy()

    # Define train step
    @tf.function
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_metric.update_state(labels, predictions)
        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    # Train the model
    epochs = 2
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_inputs in distributed_dataset:
            loss = distributed_train_step(batch_inputs)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {train_metric.result()}")
        train_metric.reset_states()
```

In this last example, the `MirroredStrategy` constructor was configured to select specific devices and a specific cross-device operation type. While it looks similar, the usage of the device selection and reduction algorithm can allow the programmer to have more explicit control over the execution. There are other configurations options available to more complex use cases as well.

In summary, effectively utilizing two GPUs for a single TensorFlow network is achievable primarily through data parallelism. `tf.distribute.MirroredStrategy` offers a streamlined approach. Custom training loops allow for greater control with `strategy.experimental_distribute_dataset` . Advanced configuration of `MirroredStrategy` allows for more granular hardware optimization. For models exceeding a single GPU's memory, model parallelism is necessary, a more complex undertaking not directly addressed in this initial answer.

For further exploration, I suggest consulting the official TensorFlow documentation on distributed training; specifically, the sections on `tf.distribute` strategies and data parallelism. Additionally, review tutorials and examples provided by the TensorFlow team demonstrating the use of multiple GPUs. Finally, consider examining research papers focused on distributed training techniques for neural networks, especially those addressing data parallel architectures.
