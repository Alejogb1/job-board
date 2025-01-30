---
title: "How can two neural networks be normalized using TensorFlow?"
date: "2025-01-30"
id: "how-can-two-neural-networks-be-normalized-using"
---
The critical point for effective training when using multiple neural networks in conjunction is ensuring that their activations and gradients exist within a comparable scale. Failing to do so can lead to instability, poor convergence, and an overall inability for the models to learn meaningfully.  In my experience developing a complex reinforcement learning system composed of separate policy and value networks, I encountered firsthand the severe challenges of training when the networks' outputs diverged substantially. TensorFlow offers several effective techniques to address this, all centered around the concept of normalization. We are not normalizing the weights themselves, but rather the data or the activations passing through the networks.

Normalization involves adjusting the values in a dataset to a standardized range, typically zero mean and unit variance. This is distinct from regularization, which directly penalizes weights. The necessity for this adjustment arises from various factors, including the disparate scales of input data, different activation functions, and the natural tendency of networks to shift the scale of representations as they propagate through layers. Using unnormalized data can cause gradients to either explode or vanish, a condition that hinders the optimization process. There are several ways to perform normalization within TensorFlow, which can be broadly categorized by where they're applied.

The most common form of normalization when dealing with multiple neural networks is *batch normalization*, applied on the *activations* within a single network, rather than between networks. It is also possible and sometimes useful to normalize the inputs to *each network*, but this is considered a preprocessing step, and not "between" networks. It's critical to understand this distinction; normalization is very seldom performed on outputs of different networks when the goal is to combine their outputs. Batch normalization works by normalizing the activations of a layer across the batch dimension, and re-scaling these normalized activations with two learnable parameters, allowing the network to learn the optimal scale for its activations. This is often inserted directly after a convolutional or linear layer and is applicable to various architectures.

Consider the following first scenario of two simple feedforward networks that each process different input data. These networks might feed into a third network at some point, but their outputs are not normalized in any way:

```python
import tensorflow as tf

# Network 1
def build_network_1(input_shape, output_units):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Network 2
def build_network_2(input_shape, output_units):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Example usage
input_shape_1 = (10,)
output_units_1 = 5
network_1 = build_network_1(input_shape_1, output_units_1)

input_shape_2 = (20,)
output_units_2 = 10
network_2 = build_network_2(input_shape_2, output_units_2)

# Generate some dummy input data
data1 = tf.random.normal((32, 10))
data2 = tf.random.normal((32, 20))

# Passing the data through networks
out1 = network_1(data1)
out2 = network_2(data2)

print("Output 1 shape: ", out1.shape) #Shape (32, 5)
print("Output 2 shape: ", out2.shape) #Shape (32, 10)
```

In the above example, we have created two separate networks, but there is no form of normalization occurring within each network itself. This is not incorrect, but it can be improved. Next, consider an example where batch normalization is explicitly added.

```python
import tensorflow as tf

# Network 1 with Batch Normalization
def build_network_1_bn(input_shape, output_units):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)  # Batch normalization after the dense layer
    x = tf.keras.layers.Activation('relu')(x) # Apply activation after normalization
    outputs = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Network 2 with Batch Normalization
def build_network_2_bn(input_shape, output_units):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    outputs = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Example usage with Batch Norm
input_shape_1 = (10,)
output_units_1 = 5
network_1_bn = build_network_1_bn(input_shape_1, output_units_1)

input_shape_2 = (20,)
output_units_2 = 10
network_2_bn = build_network_2_bn(input_shape_2, output_units_2)

# Generate some dummy input data
data1 = tf.random.normal((32, 10))
data2 = tf.random.normal((32, 20))


out1 = network_1_bn(data1)
out2 = network_2_bn(data2)

print("Output 1 shape with batch norm: ", out1.shape) #Shape (32, 5)
print("Output 2 shape with batch norm: ", out2.shape) #Shape (32, 10)
```

Here, `tf.keras.layers.BatchNormalization()` is inserted after each dense layer to ensure the scale of activations remains controlled and consistent. The order of operations is very important. It should be placed after the linear transformation (Dense or Convolutional) but prior to the non-linear activation. This allows the non-linearity to be applied to the normalized output, thereby promoting more effective gradient flow. Batch normalization stabilizes training and often results in faster convergence.

Lastly, let's examine the less common, but still applicable, case of input normalization. Consider that our different networks receive entirely different types of input data which may be wildly out of scale. This step is normally part of a preprocessing pipeline, but is still an important aspect of normalization.

```python
import tensorflow as tf
import numpy as np

# Function to normalize input data using a moving average
class InputNormalizer():
    def __init__(self, input_shape):
      self.mean = tf.Variable(tf.zeros(input_shape, dtype=tf.float32), trainable=False)
      self.variance = tf.Variable(tf.ones(input_shape, dtype=tf.float32), trainable=False)
      self.count = tf.Variable(0.0, trainable=False)
      self.epsilon = 1e-5

    @tf.function
    def update(self, new_data):
      new_count = self.count + tf.cast(tf.shape(new_data)[0], tf.float32)
      new_mean = (self.mean * self.count + tf.reduce_sum(new_data, axis=0)) / new_count
      new_variance = (self.variance * self.count + tf.reduce_sum((new_data - new_mean) ** 2, axis = 0) ) / new_count
      self.mean.assign(new_mean)
      self.variance.assign(new_variance)
      self.count.assign(new_count)

    @tf.function
    def normalize(self, data):
      return (data - self.mean) / tf.sqrt(self.variance + self.epsilon)

# Network 1
def build_network_1_normalized_input(input_shape, output_units, input_normalizer):
    inputs = tf.keras.layers.Input(shape=input_shape)
    normalized_inputs = input_normalizer.normalize(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(normalized_inputs)
    outputs = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Network 2
def build_network_2_normalized_input(input_shape, output_units, input_normalizer):
    inputs = tf.keras.layers.Input(shape=input_shape)
    normalized_inputs = input_normalizer.normalize(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(normalized_inputs)
    outputs = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Example usage
input_shape_1 = (10,)
output_units_1 = 5
input_normalizer_1 = InputNormalizer(input_shape_1)
network_1_normalized = build_network_1_normalized_input(input_shape_1, output_units_1, input_normalizer_1)

input_shape_2 = (20,)
output_units_2 = 10
input_normalizer_2 = InputNormalizer(input_shape_2)
network_2_normalized = build_network_2_normalized_input(input_shape_2, output_units_2, input_normalizer_2)

# Generate some dummy input data
data1 = tf.random.normal((32, 10)) * 100 #Scale of data 1 is 100 times greater
data2 = tf.random.normal((32, 20))

# Update the normalizers
input_normalizer_1.update(data1)
input_normalizer_2.update(data2)

# Passing data through the networks
out1 = network_1_normalized(data1)
out2 = network_2_normalized(data2)

print("Output 1 shape with input normalization: ", out1.shape) #Shape (32, 5)
print("Output 2 shape with input normalization: ", out2.shape) #Shape (32, 10)
```

Here, a custom `InputNormalizer` class is introduced. This class maintains a running estimate of the mean and variance for the input data. This allows us to keep track of the normalization parameters of the dataset while we are using it. This is done because not every network gets to see every datapoint in general. Before each batch passes through the model, the input data is normalized using this class. This normalizer must be updated in an online fashion. While this is not strictly "normalization between neural networks" it allows us to make the inputs of each model be in approximately the same scale.

To further understand and implement these techniques, I would recommend investigating several resources. The official TensorFlow documentation provides extensive guides and tutorials on batch normalization and preprocessing techniques.  Additionally, research papers on batch normalization and related normalization methods provide deeper theoretical and practical insights. Online courses in deep learning, like those from universities, often cover these topics in detail. Finally, delving into the source code of various neural network implementations can offer a pragmatic understanding of how normalization is applied in practice, which can be highly illuminating for implementing custom normalization schemes. It's worth repeating that while it is useful to normalize inputs and layers within a single network, it's rare to normalize the *outputs* of separate networks when they are being used together.
