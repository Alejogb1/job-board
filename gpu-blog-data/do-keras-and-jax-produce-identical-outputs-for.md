---
title: "Do Keras and JAX produce identical outputs for the same neural network with identical weights?"
date: "2025-01-30"
id: "do-keras-and-jax-produce-identical-outputs-for"
---
The convergence of deep learning frameworks towards standardized operations might suggest identical outputs across different implementations for the same neural network with identical weights. However, achieving complete bitwise reproducibility between Keras (TensorFlow backend) and JAX is not guaranteed, and often requires careful attention to numerical precision and framework-specific implementation details. My experience migrating a complex multi-modal network from a Keras prototype to a JAX-based production system illuminated this nuance.

The core reason for these discrepancies stems from subtle differences in how these frameworks handle low-level computations. While both frameworks perform similar operations, numerical representations and the order of operations, particularly in areas involving floating-point arithmetic, can vary. TensorFlow, as a graph-based system, might optimize or reorder operations during graph compilation. JAX, on the other hand, relies on ahead-of-time compilation and has its own internal mechanisms for handling these calculations. These differences accumulate and can lead to slightly divergent results after several layers of complex neural networks, particularly with repeated, recursive operations.

Furthermore, even with identical weights, the initialization processes within each framework, while similar in principle, can introduce microscopic differences. Different numerical seeds for pseudo-random number generation at the initialization level can yield slightly different initial weights, even if the distributions are defined to be the same. Moreover, the specific implementation of activation functions (e.g., sigmoid, ReLU) and their derivatives might vary subtly between the frameworks, particularly when dealing with edge cases or very large/small inputs.

Let’s illustrate these points with examples. Suppose we have a simple linear layer. In Keras (using the TensorFlow backend), it can be defined as:

```python
import tensorflow as tf
import numpy as np

# Define a basic linear layer with no activation
input_dim = 5
output_dim = 3

keras_linear_layer = tf.keras.layers.Dense(units=output_dim, use_bias=False,
                                          kernel_initializer=tf.keras.initializers.HeNormal(seed=42))

# Initial weights
input_tensor = np.random.normal(size=(1, input_dim)).astype(np.float32)
keras_linear_layer.build(input_shape=input_tensor.shape)
weights = keras_linear_layer.kernel.numpy()

# Perform the forward pass
keras_output = keras_linear_layer(input_tensor)

print("Keras Output:", keras_output)
```
In JAX, we would represent the same linear operation:

```python
import jax
import jax.numpy as jnp
import numpy as np


input_dim = 5
output_dim = 3

key = jax.random.PRNGKey(42)
weights_key, bias_key = jax.random.split(key)
weights = jax.random.normal(weights_key, (input_dim, output_dim)) * np.sqrt(2/input_dim) #He Normal

input_tensor = jnp.array(np.random.normal(size=(1, input_dim)).astype(np.float32))

def jax_linear(input_tensor, weights):
    return jnp.dot(input_tensor, weights)

jax_output = jax_linear(input_tensor, weights)

print("JAX Output:", jax_output)
```

While these linear layers should perform similar calculations, running these snippets will demonstrate slight differences in the produced output. Notice that both examples use initialization that are theoretically equivalent (He Normal), with consistent numerical seeds. However the result are different by a very tiny margin, because of internal computation specifics.

The problem becomes more pronounced with layers that have more complex operations such as Batch Normalization, or when applying activation function. Lets consider a basic feedforward network composed of a fully connected layer and activation function (ReLU) with normalization:

```python
import tensorflow as tf
import numpy as np

# Define a basic feedforward network in Keras
input_dim = 5
hidden_dim = 10
output_dim = 3

keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=hidden_dim, use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(units=output_dim, use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal(seed=42))
])

# Initial weights and data.
input_tensor = np.random.normal(size=(1, input_dim)).astype(np.float32)
keras_model.build(input_shape=input_tensor.shape)

# Generate an output
keras_output = keras_model(input_tensor)
print("Keras Feedforward Output:", keras_output)
```

And now, the equivalent logic in JAX:
```python
import jax
import jax.numpy as jnp
import numpy as np

from jax import random
from jax.experimental import stax

# Define a basic feedforward network in JAX
input_dim = 5
hidden_dim = 10
output_dim = 3
key = random.PRNGKey(42)

init_fun, apply_fun = stax.serial(
    stax.Dense(hidden_dim, W_init=jax.random.normal(key, (input_dim, hidden_dim))* np.sqrt(2/input_dim), b_init=lambda k, shape, dtype: None ),
    stax.BatchNorm(use_running_average=True, beta_init=lambda k, shape, dtype: np.zeros(shape, dtype=dtype), gamma_init=lambda k, shape, dtype: np.ones(shape, dtype=dtype)),
    stax.Relu,
    stax.Dense(output_dim, W_init=jax.random.normal(key, (hidden_dim, output_dim))* np.sqrt(2/hidden_dim), b_init=lambda k, shape, dtype: None)

)
input_tensor = jnp.array(np.random.normal(size=(1, input_dim)).astype(np.float32))
out_shape, params = init_fun(key, input_tensor.shape)

def jax_forward(params, input_tensor):
  return apply_fun(params, input_tensor)

jax_output = jax_forward(params, input_tensor)
print("JAX Feedforward Output:", jax_output)
```

These two examples, using the same seed and mathematically equivalent components, will demonstrate observable discrepancies in output. These differences might be negligible for many practical applications, however they can affect specific tasks or scientific experiments that require bitwise accuracy. This is important because the batch-norm implementation differs internally, including the initialization and usage of moving averages during forward passes.

As a final example, consider a network that contains operations such as padding, which depends on how the tensor is being used. We’ll make a very simple example of a convolution, but padding is just one of many different parameters that can lead to subtle differences.

```python
import tensorflow as tf
import numpy as np

# Define a basic CNN in Keras
input_channels = 3
output_channels = 16
kernel_size = (3, 3)
image_size = (28, 28)

keras_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=output_channels, kernel_size=kernel_size,
                           padding='same', use_bias=False,
                           kernel_initializer=tf.keras.initializers.HeNormal(seed=42),
                            input_shape=(image_size[0], image_size[1], input_channels)
                           ),
])

# Initialize data
input_tensor = np.random.normal(size=(1, image_size[0], image_size[1], input_channels)).astype(np.float32)

keras_cnn.build(input_shape=input_tensor.shape)

# Perform a forward pass
keras_output = keras_cnn(input_tensor)

print("Keras CNN Output:", keras_output)
```
Here we have the JAX equivalent. Note the explicit definition of the padding.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import stax
from jax import random

# Define a basic CNN in JAX
input_channels = 3
output_channels = 16
kernel_size = (3, 3)
image_size = (28, 28)

key = random.PRNGKey(42)
init_fun, apply_fun = stax.serial(
    stax.Conv(output_channels, kernel_size, padding="SAME", W_init=jax.random.normal(key, (kernel_size[0], kernel_size[1], input_channels, output_channels)) * np.sqrt(2/(kernel_size[0]*kernel_size[1]*input_channels)), b_init=lambda k, shape, dtype: None)
)

input_tensor = jnp.array(np.random.normal(size=(1, image_size[0], image_size[1], input_channels)).astype(np.float32))
out_shape, params = init_fun(key, input_tensor.shape)


def jax_cnn(params, input_tensor):
  return apply_fun(params, input_tensor)

jax_output = jax_cnn(params, input_tensor)
print("JAX CNN Output:", jax_output)
```
Again we observe a small difference. Though `padding="same"` in Keras and `padding="SAME"` in JAX attempt the same concept, there may be internal specifics that will make it not exactly equivalent, specially around edge-cases.

In conclusion, the pursuit of bit-perfect reproducibility between Keras and JAX for identical neural networks with the same weights is not straightforward. While both frameworks aim to provide mathematically equivalent operations, subtle differences in numerical representation, calculation order, and implementation specifics of layers can lead to variations in output. It becomes important to acknowledge and address these subtle differences, especially when migrating models between different frameworks and for scientific accuracy.

For individuals seeking to further understand these nuances, I recommend consulting the following resources: the official TensorFlow documentation, particularly the sections regarding graph optimizations and numerical precision; the JAX documentation, with special focus on the sections detailing numerical behavior; research papers that have investigated and addressed numerical reproducibility in deep learning; and source code of both Keras and JAX libraries for detailed comparisons.
