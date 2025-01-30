---
title: "How can TensorFlow utilize multiple GPUs?"
date: "2025-01-30"
id: "how-can-tensorflow-utilize-multiple-gpus"
---
TensorFlow’s ability to leverage multiple GPUs is paramount for training large-scale deep learning models efficiently, fundamentally relying on strategies to parallelize computations across these devices. Having spent years optimizing complex architectures in TensorFlow, I’ve found that understanding the nuances of data parallelism and device placement is critical for achieving optimal performance.

The core concept is data parallelism, where training data is divided into batches and each batch is processed independently on a different GPU. Gradient computations generated from each GPU are then combined or averaged before the model’s parameters are updated. TensorFlow facilitates this through several API options, the most relevant being the `tf.distribute.Strategy` API. This API abstracts away the complexities of multi-GPU and distributed training. It encapsulates the strategy for replicating the model and synchronizing the parameter updates across different devices.

There are primarily two distribution strategies relevant for multi-GPU scenarios within a single machine: `tf.distribute.MirroredStrategy` and `tf.distribute.MultiWorkerMirroredStrategy`. `MirroredStrategy` works within a single machine, creating a copy of the model on each GPU, and uses a collective communication approach to aggregate gradients. `MultiWorkerMirroredStrategy`, though also performing synchronous gradient aggregation, is intended for distributed training across multiple machines, each potentially containing one or more GPUs. I primarily use `MirroredStrategy` for situations where all GPUs reside on the same workstation.

The process involves creating an instance of `MirroredStrategy`, defining model and optimizer under its scope and then using it in the standard TensorFlow training loop.

Here's how `MirroredStrategy` works in practice with a simple Keras model using Python.

```python
import tensorflow as tf

# Detect available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Create a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
      # Define the model within the strategy's scope.
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Define optimizer and loss within the strategy's scope
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        # Compile the model.
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Generate dummy data for training
    import numpy as np
    x_train = np.random.random((10000,784))
    y_train = np.random.randint(0,10,size=(10000,1))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    # Train the model.
    model.fit(x_train, y_train, epochs=2, batch_size=32)


else:
    print("No GPUs detected. Cannot demonstrate MirroredStrategy.")
```

This code snippet first checks for the presence of GPUs. If GPUs are available, a `MirroredStrategy` is created. The model, optimizer, and loss function are then defined within the strategy’s `scope()`. This is essential because it instructs TensorFlow to replicate the model's weights across all available GPUs and to synchronize gradients during training. A small dummy dataset is generated and the model is then trained using `model.fit()`. The critical part here is the scoping, directing TensorFlow to parallelize the training automatically. If no GPUs are detected, a message is printed.

Another common challenge I’ve faced involves custom training loops rather than relying on Keras’ built-in `fit()` method. Custom training loops require explicit control over the distribution process, making it crucial to use `strategy.run()`. This function executes a given function on all replicas (GPUs), distributing computations accordingly.

Below is an example of how to achieve custom training with `MirroredStrategy`.

```python
import tensorflow as tf

# Detect available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Create a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Define a data loader.
    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 32
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    import numpy as np
    x_train = np.random.random((10000,784))
    y_train = np.random.randint(0,10,size=(10000,1))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)


    with strategy.scope():

        # Define the model within the strategy's scope.
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # Define optimizer and loss within the strategy's scope
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.CategoricalCrossentropy()


        # Define the training step
        def train_step(inputs):
             images, labels = inputs
             with tf.GradientTape() as tape:
                logits = model(images,training=True)
                loss = loss_fn(labels, logits)

             gradients = tape.gradient(loss, model.trainable_variables)
             optimizer.apply_gradients(zip(gradients, model.trainable_variables))

             return loss
    # Define a distributed training step
    @tf.function
    def distributed_train_step(dataset_inputs):
         per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
         return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # Training loop
    for epoch in range(2):
       for inputs in train_dist_dataset:
            total_loss = distributed_train_step(inputs)
            print("epoch : {} loss : {}".format(epoch,total_loss))


else:
    print("No GPUs detected. Cannot demonstrate MirroredStrategy.")
```
Here, the data is loaded using `tf.data.Dataset`, shuffled and then batched with a batch size appropriate to the number of GPUs, and converted to distributed dataset using `strategy.experimental_distribute_dataset`. Inside the `strategy.scope`, the model, optimizer, and loss function are defined. A custom `train_step` function handles the forward and backward passes for a single batch. `distributed_train_step`, decorated with `@tf.function` to utilize graph compilation for performance, uses `strategy.run()` to execute `train_step` on all GPUs and aggregates the computed losses using `strategy.reduce()`. The core training loop then iterates through the distributed dataset, executing the distributed training step. The reduced loss is printed for each batch. It’s worth noting the explicit handling of dataset distribution and the use of `strategy.reduce`, which are critical for custom training loops across multiple GPUs.

Beyond basic model training, another aspect where I’ve seen multi-GPU usage prove invaluable involves model parameter saving and loading. Saving and loading models when using a `tf.distribute.Strategy` requires additional awareness of how the model's parameters are distributed across GPUs. A common practice is to ensure all GPUs write the model's variables. However, when loading, variables should be re-distributed based on the devices specified during loading.

The following example illustrates how the model from the previous custom training example can be saved and reloaded.

```python
import tensorflow as tf
import os

# Detect available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Create a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    model_path ="saved_model"

    # Model definition and training (as shown in previous example)
    # Reuse data loader, model,optimizer, loss_fn, training steps from previous examples
    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 32
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    import numpy as np
    x_train = np.random.random((10000,784))
    y_train = np.random.randint(0,10,size=(10000,1))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)


    with strategy.scope():

        # Define the model within the strategy's scope.
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # Define optimizer and loss within the strategy's scope
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.CategoricalCrossentropy()


        # Define the training step
        def train_step(inputs):
             images, labels = inputs
             with tf.GradientTape() as tape:
                logits = model(images,training=True)
                loss = loss_fn(labels, logits)

             gradients = tape.gradient(loss, model.trainable_variables)
             optimizer.apply_gradients(zip(gradients, model.trainable_variables))

             return loss
    # Define a distributed training step
    @tf.function
    def distributed_train_step(dataset_inputs):
         per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
         return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # Training loop
    for epoch in range(2):
       for inputs in train_dist_dataset:
            total_loss = distributed_train_step(inputs)
            print("epoch : {} loss : {}".format(epoch,total_loss))
        # Save Model
    if not os.path.exists(model_path):
       os.makedirs(model_path)

    model.save(model_path)


     # Reload model from saved path inside strategy's scope
    with strategy.scope():
      loaded_model = tf.keras.models.load_model(model_path)

    # Verify model by testing on random sample
    test_sample = np.random.random((1,784))
    output = loaded_model(test_sample)
    print ("prediction : {}".format(output))


else:
    print("No GPUs detected. Cannot demonstrate MirroredStrategy.")
```
The code reuses the data loading, training setup from the previous example, trains the model and then saves it using `model.save(model_path)`. After that, the model is reloaded inside a `strategy.scope`. This ensures that when the model is loaded, TensorFlow re-distributes the weights correctly across all the available GPUs. The model is tested with a random sample to verify it is working correctly after reload. The loading process is critical in a multi-GPU setup since simply loading the saved weights might lead to inconsistencies if device placements are not handled properly.

When diving deeper into optimizing multi-GPU setups, several resources are useful. The TensorFlow documentation on distributed training offers a comprehensive overview of different strategies and their usage. The "TensorFlow Performance Guide" contains details on optimizing training pipelines to maximize GPU utilization. Lastly, publications focused on distributed deep learning architectures can provide theoretical understanding, that can further refine the approach to multi-GPU training. These resources, combined with practical experimentation, are invaluable when striving to achieve optimal performance with multi-GPU training in TensorFlow.
