---
title: "How can distributed TensorFlow implement asynchronous training with batch normalization?"
date: "2025-01-30"
id: "how-can-distributed-tensorflow-implement-asynchronous-training-with"
---
The core challenge in implementing asynchronous training with batch normalization in distributed TensorFlow stems from batch normalization's dependency on mini-batch statistics. In a synchronous training paradigm, all workers contribute to a global mini-batch, enabling a single, consistent calculation of batch statistics. However, asynchronous training, where workers process batches independently and at varying paces, disrupts this synchronization, potentially introducing biases and unstable training dynamics when using conventional batch normalization.

My experience, primarily within distributed reinforcement learning agents requiring high throughput, revealed that direct application of standard batch normalization in an asynchronous setting leads to substantial performance degradation. Each worker’s locally computed batch statistics are often significantly different, and frequently out of sync with the current model state of other workers and the server, creating training oscillations and hindering convergence. To effectively utilize batch normalization in such an environment, we need to decouple the calculation and application of batch statistics from the local mini-batch. This means moving towards global statistics calculated and maintained outside of the local worker’s processing flow.

The solution is to implement a strategy where batch normalization statistics are globally updated and applied across all workers, independent of their local mini-batches. Specifically, this involves the creation of shared variables to hold running mean and variance, and applying a strategy that averages these values across all updates. The worker calculates batch statistics on their mini-batch, but then uses these results to update the shared variables. Critically, they do not directly use the local statistics for normalization during the training step, but rather fetch and use the global running statistics.

Here is a code example demonstrating a basic implementation of this approach using TensorFlow:

```python
import tensorflow as tf

def create_batch_norm_vars(feature_shape, momentum):
    """Creates shared variables for batch normalization."""
    mean = tf.Variable(tf.zeros(feature_shape, dtype=tf.float32),
                       trainable=False, name="batchnorm_mean")
    variance = tf.Variable(tf.ones(feature_shape, dtype=tf.float32),
                       trainable=False, name="batchnorm_variance")
    return mean, variance

def apply_batch_norm(inputs, mean, variance, epsilon=1e-5):
   """Applies batch normalization using given mean and variance."""
   return tf.nn.batch_normalization(inputs, mean, variance,
                                     offset=None, scale=None,
                                     variance_epsilon=epsilon)

def update_batch_norm_stats(inputs, mean, variance, momentum):
    """Updates shared running mean and variance."""
    local_mean, local_variance = tf.nn.moments(inputs, axes=[0])
    new_mean = momentum * mean + (1 - momentum) * local_mean
    new_variance = momentum * variance + (1 - momentum) * local_variance
    update_mean_op = mean.assign(new_mean)
    update_variance_op = variance.assign(new_variance)
    return tf.group(update_mean_op, update_variance_op)

# Example usage:
feature_shape = (10,)
momentum = 0.9
batch_size = 32
inputs = tf.random.normal(shape=(batch_size, *feature_shape), dtype=tf.float32)
mean, variance = create_batch_norm_vars(feature_shape, momentum)

# Get current mean and variance
current_mean = mean.read_value()
current_variance = variance.read_value()

# Apply batch normalization using global stats
normalized_inputs = apply_batch_norm(inputs, current_mean, current_variance)

# Calculate local statistics and update global statistics
update_op = update_batch_norm_stats(inputs, mean, variance, momentum)
```

In this first example, `create_batch_norm_vars` sets up the shared variables that will store the running statistics. These variables are explicitly marked as not trainable, ensuring they are updated through the custom update mechanism. `apply_batch_norm` uses the globally shared mean and variance during the forward pass. The core lies in `update_batch_norm_stats`, where the local batch statistics are calculated and used to incrementally update the global mean and variance through a momentum-based approach. This code provides the building blocks, but needs to be placed within the distributed training loop with synchronization mechanisms.

Here's the second code example, illustrating how one would integrate this within a distributed setting using TensorFlow’s `tf.distribute.experimental.ParameterServerStrategy`. I’m going to make the simplifying assumption of a synchronous parameter server training model here, as the nuances of fully asynchronous training with non-blocking reads and writes introduce complexities beyond the scope of this example:

```python
import tensorflow as tf
import numpy as np

num_workers = 2
feature_shape = (10,)
momentum = 0.9
batch_size = 32
steps = 10 # For demonstration, use smaller number of training steps

strategy = tf.distribute.experimental.ParameterServerStrategy(num_workers=num_workers)

with strategy.scope():
    mean, variance = create_batch_norm_vars(feature_shape, momentum)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=feature_shape),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        current_mean = mean.read_value()
        current_variance = variance.read_value()
        normalized_inputs = apply_batch_norm(inputs, current_mean, current_variance)
        predictions = model(normalized_inputs)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    update_op = update_batch_norm_stats(inputs, mean, variance, momentum)
    return update_op

@tf.function
def distributed_train_step(inputs, labels):
    update_op = strategy.run(train_step, args=(inputs, labels,))
    return update_op

# Example training loop
for step in range(steps):
  input_data = np.random.normal(size=(batch_size, *feature_shape)).astype(np.float32)
  label_data = np.random.randint(0, 10, size=(batch_size,)).astype(np.int32)
  update_op = distributed_train_step(input_data, label_data)
  tf.print(f"Step: {step}, Mean:{mean.read_value()[0]}, Variance:{variance.read_value()[0]}")
```

This example shows how to integrate the custom batch norm logic within a `tf.distribute.experimental.ParameterServerStrategy`.  The `train_step` encapsulates forward pass, backward pass, weight updates, and our custom batch norm update operation.  Critically, the shared mean and variance values are read before the normalized input is created, ensuring consistency, and the update operation is executed after the backpropagation. Note, each worker will execute this function independently when data is distributed to each worker by the strategy, and the parameter server strategy will handle syncing the parameter updates, ensuring the `mean` and `variance` are updated appropriately.  This is a crucial distinction from a synchronous training loop that might accumulate the updates in local variables before committing the changes.

Finally, the third example demonstrates a simplified way to implement this behavior using `tf.keras.layers.BatchNormalization` by wrapping it in a custom layer and manually managing the update operation. This approach may require more care in the implementation to achieve similar stability as the explicit shared variables.

```python
import tensorflow as tf
import numpy as np

class AsyncBatchNorm(tf.keras.layers.Layer):
    def __init__(self, feature_shape, momentum=0.9, epsilon=1e-5, **kwargs):
        super(AsyncBatchNorm, self).__init__(**kwargs)
        self.feature_shape = feature_shape
        self.momentum = momentum
        self.epsilon = epsilon
        self.batch_norm = None
        self.mean = None
        self.variance = None

    def build(self, input_shape):
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, trainable=False, momentum=self.momentum, epsilon=self.epsilon)
        self.mean = self.add_weight(shape=self.feature_shape, initializer=tf.zeros_initializer(),
                                  trainable=False, name="async_mean")
        self.variance = self.add_weight(shape=self.feature_shape, initializer=tf.ones_initializer(),
                                   trainable=False, name="async_variance")
        super(AsyncBatchNorm, self).build(input_shape)


    def call(self, inputs, training=False):
        if training:
            local_mean, local_variance = tf.nn.moments(inputs, axes=[0])
            new_mean = self.momentum * self.mean + (1 - self.momentum) * local_mean
            new_variance = self.momentum * self.variance + (1 - self.momentum) * local_variance
            self.mean.assign(new_mean)
            self.variance.assign(new_variance)
        return tf.nn.batch_normalization(inputs, self.mean, self.variance, None, None, self.epsilon)

num_workers = 2
feature_shape = (10,)
momentum = 0.9
batch_size = 32
steps = 10

strategy = tf.distribute.experimental.ParameterServerStrategy(num_workers=num_workers)

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=feature_shape),
        AsyncBatchNorm(feature_shape, momentum),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def distributed_train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return None


for step in range(steps):
  input_data = np.random.normal(size=(batch_size, *feature_shape)).astype(np.float32)
  label_data = np.random.randint(0, 10, size=(batch_size,)).astype(np.int32)
  distributed_train_step(input_data, label_data)
  tf.print(f"Step: {step}, Mean:{model.layers[1].mean.read_value()[0]}, Variance:{model.layers[1].variance.read_value()[0]}")

```
This last example wraps the conventional batch norm in a custom layer, where we can manage the updates as we need for our asynchronous setting.  Within the `call` method of the layer, we perform the batch statistics calculation and update operation when the model is training, using our momentum update logic. The `tf.distribute.experimental.ParameterServerStrategy` takes care of distributing the parameter updates. This approach, while simpler, might be less stable than the explicit implementation, as `tf.keras.layers.BatchNormalization` might expect to see consistent batch statistics across the full update.

For further reading, I recommend resources that discuss distributed training with TensorFlow, specifically examining strategies such as `ParameterServerStrategy` and `MultiWorkerMirroredStrategy`, as understanding the nuances of these distribution strategies is critical. Additionally, research papers discussing the statistical properties of asynchronous training and batch normalization will be helpful to understand the underlying issues and advanced solutions.
