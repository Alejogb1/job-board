---
title: "How can Keras leverage TF datasets with images for multi-GPU training?"
date: "2025-01-30"
id: "how-can-keras-leverage-tf-datasets-with-images"
---
Multi-GPU training with Keras and TensorFlow datasets (tf.data) is fundamentally about efficiently distributing the workload across available GPUs while ensuring data consistency and avoiding bottlenecks. The key lies in understanding how the `tf.distribute.Strategy` API interacts with `tf.data.Dataset` objects to facilitate parallel processing. I’ve spent the last few years refining pipelines for large-scale image recognition models, and the seamless integration of these two components was a crucial step in achieving substantial performance improvements.

The core challenge isn't just about copying data to multiple GPUs. It's about strategically splitting the data, ensuring that each GPU receives a unique, non-overlapping subset, and then aggregating the gradients calculated on each GPU for a unified update to the model's weights. The `tf.distribute.Strategy` handles the data distribution and gradient aggregation aspects.

A `tf.distribute.Strategy` needs to be instantiated before the model is created, dictating how computations will be distributed across devices. TensorFlow provides several strategies, but `tf.distribute.MirroredStrategy` is the most common for multi-GPU training within a single machine. This strategy replicates the model’s weights across all specified GPUs and distributes batches of input data. Each GPU processes a different portion of the batch, and the gradients are then synchronized and aggregated before updating the shared weights. Using `MirroredStrategy` automatically manages the replication of the model's variables and gradients across all the GPUs used. Therefore, only minimal changes are required in the model building and training process compared to single GPU training.

The `tf.data.Dataset` API, on the other hand, is designed for efficiently loading and preprocessing data. When combined with a `tf.distribute.Strategy`, the `tf.data` pipeline needs to be configured to ensure that the data is distributed correctly across the GPUs without introducing any overlap. This commonly involves using the `.distribute()` method to tell the data generator what distribution strategy it needs to follow.

Here is an example of building such a pipeline for multi-GPU training using `MirroredStrategy` and `tf.data`:

```python
import tensorflow as tf
import numpy as np

# Example: Simulate image loading function
def load_image(image_path, label):
    # Assume `image_path` contains paths to images, and labels are associated integers
    # Pretend to load and decode an image.
    # Typically this would involve `tf.io.read_file` followed by image decoding
    image = tf.random.normal((224, 224, 3))
    return image, label

def create_dataset(image_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Assume we have image paths and their corresponding labels
num_images = 1000
image_paths = [f"path/to/image_{i}.jpg" for i in range(num_images)]
labels = np.random.randint(0, 10, num_images) #Example: 10 classes

BATCH_SIZE = 64 #Example of batch size for the dataset. 

# Strategy for multi-gpu training
strategy = tf.distribute.MirroredStrategy()

# Get the number of devices in current scope.
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

# Create the dataset
dataset = create_dataset(image_paths, labels, GLOBAL_BATCH_SIZE)

# Distribute the dataset using the strategy.
distributed_dataset = strategy.experimental_distribute_dataset(dataset)

# Here goes the Model code.
with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation='softmax') # Example of a classification model with 10 classes
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    #Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
        
    # Scale the loss to use it across all GPUs
    scaled_loss = loss / strategy.num_replicas_in_sync
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return scaled_loss

@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs[0], dataset_inputs[1],))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    for batch in distributed_dataset:
        loss = distributed_train_step(batch)
        total_loss += loss
    
    print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(dataset)}, Accuracy: {train_accuracy.result()}')
    train_accuracy.reset_state()
```
In the first example, I define the data loading process within the `create_dataset` function. The key lines here are `.batch(batch_size)` which creates batches, `dataset.shuffle(buffer_size=len(image_paths))` which shuffles the data, and then `dataset.prefetch(tf.data.AUTOTUNE)` which allows asynchronous loading of the data. Crucially, I’ve ensured that the batch size that I use when creating the dataset is equal to `GLOBAL_BATCH_SIZE`, which is the batch size times the number of replicas available. This means, each GPU receives a mini batch of the correct size.

Then the dataset is distributed using `.experimental_distribute_dataset`, which ensures that each device (GPU) is assigned a different part of the batch.  The rest of the code is model training, which is done within the scope of the previously defined strategy. Here, the training step is wrapped in a `tf.function` and then another helper function `distributed_train_step` to distribute the computation across devices using the defined strategy. The `train_step` is where forward and backward passes happen, and also where we compute the loss which is then scaled using the number of replicas.

The loss is reduced and the resulting gradient is applied to the model variables. Note that the reduction of the loss by the number of replicas in the synchronous training case is crucial because if not done, loss will be averaged based on the number of GPUs which means that large datasets won’t need more GPUs since their loss will be reduced by the increase in the number of GPUs.
Next, let's illustrate a more complex scenario involving image augmentations:

```python
import tensorflow as tf
import numpy as np

# Simulate image loading and augmentation function
def load_and_augment_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Assume JPEGs
    image = tf.image.resize(image, [256, 256])
    
    # Data Augmentations
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_flip_left_right(image)
    
    #Resize to target size
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(image_paths, labels, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
  dataset = dataset.map(load_and_augment_image, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.shuffle(buffer_size=len(image_paths))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset

# Example: Simulated image paths and labels
num_images = 2000
image_paths = [f"path/to/image_{i}.jpg" for i in range(num_images)]
labels = np.random.randint(0, 10, num_images) #Example: 10 classes

BATCH_SIZE = 64

# Strategy for multi-gpu training
strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

# Dataset Creation
dataset = create_dataset(image_paths, labels, GLOBAL_BATCH_SIZE)

#Distribute dataset
distributed_dataset = strategy.experimental_distribute_dataset(dataset)


with strategy.scope():
    model = tf.keras.applications.ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = loss_fn(labels, predictions)
  scaled_loss = loss / strategy.num_replicas_in_sync
  gradients = tape.gradient(scaled_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_accuracy.update_state(labels, predictions)
  return scaled_loss

@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs[0], dataset_inputs[1],))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


epochs = 10
for epoch in range(epochs):
  total_loss = 0.0
  for batch in distributed_dataset:
      loss = distributed_train_step(batch)
      total_loss += loss

  print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(dataset)}, Accuracy: {train_accuracy.result()}')
  train_accuracy.reset_state()
```
Here, I've introduced more sophisticated image processing within `load_and_augment_image`, including random brightness, contrast adjustments, and left-right flips before resizing it to the correct input size for a `ResNet50`. I've also made use of `tf.io.read_file` to read file paths from the input, and `tf.image.decode_jpeg` to decode images. This demonstrates that more complex operations can be carried out in this processing step of the dataset without impacting the parallelization. The remaining code is the same as before.

Lastly, let’s look at a case that incorporates distributed validation:

```python
import tensorflow as tf
import numpy as np

# Simulate image loading function
def load_image(image_path, label):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image = tf.cast(image, tf.float32) / 255.0
  return image, label

def create_dataset(image_paths, labels, batch_size, shuffle=True):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
  dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(image_paths))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset

# Example: Simulated image paths and labels
num_train_images = 3000
train_image_paths = [f"path/to/train_image_{i}.jpg" for i in range(num_train_images)]
train_labels = np.random.randint(0, 10, num_train_images) #Example: 10 classes

num_val_images = 1000
val_image_paths = [f"path/to/val_image_{i}.jpg" for i in range(num_val_images)]
val_labels = np.random.randint(0, 10, num_val_images)

BATCH_SIZE = 64

# Strategy for multi-gpu training
strategy = tf.distribute.MirroredStrategy()

GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

# Dataset Creation
train_dataset = create_dataset(train_image_paths, train_labels, GLOBAL_BATCH_SIZE)
val_dataset = create_dataset(val_image_paths, val_labels, GLOBAL_BATCH_SIZE, shuffle=False)


#Distribute the datasets
distributed_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
distributed_val_dataset = strategy.experimental_distribute_dataset(val_dataset)


with strategy.scope():
    model = tf.keras.applications.ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    scaled_loss = loss / strategy.num_replicas_in_sync
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return scaled_loss

@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs[0], dataset_inputs[1],))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def val_step(inputs, labels):
    predictions = model(inputs, training=False)
    val_accuracy.update_state(labels, predictions)

@tf.function
def distributed_val_step(dataset_inputs):
    strategy.run(val_step, args=(dataset_inputs[0], dataset_inputs[1],))

epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    for batch in distributed_train_dataset:
        loss = distributed_train_step(batch)
        total_loss += loss

    for batch in distributed_val_dataset:
        distributed_val_step(batch)

    print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_dataset)}, Train Accuracy: {train_accuracy.result()}, Validation Accuracy: {val_accuracy.result()}')
    train_accuracy.reset_state()
    val_accuracy.reset_state()
```
In this final example, a validation dataset has been added using the same process as the training dataset, but with shuffling turned off to allow consistent validation. The same model and parameters were used for this validation step, and a new `val_step` was added for the validation calculation which is run after each training epoch. The metric is also reset at the end of the epoch.

For further exploration of this topic, I recommend reviewing the official TensorFlow documentation on distributed training and `tf.data` API. The TensorFlow tutorials often contain in-depth examples of different training strategies which can help deepen understanding. Books specializing in deep learning engineering and MLOps also offer substantial practical insights into designing efficient pipelines. Furthermore, I suggest examining publicly available implementations of state-of-the-art image recognition models, often available on GitHub and other platforms, to see how these strategies are applied in practice for large-scale applications.
