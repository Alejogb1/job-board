---
title: "How to assign moving variance and average in TensorFlow batch normalization?"
date: "2025-01-30"
id: "how-to-assign-moving-variance-and-average-in"
---
Batch normalization, particularly when employed during training with mini-batches, requires the maintenance of moving statistics to ensure consistent behavior during inference. A critical aspect of achieving this lies in correctly updating the moving variance and average, not merely utilizing the batch statistics directly. The key challenge is striking a balance between responsiveness to recent batch information and retaining the broader data distribution learned over time. Ignoring this balance can lead to instability during training or discrepancies between training and inference performance.

In my experience, many misunderstandings stem from viewing batch normalization exclusively through the lens of individual mini-batch processing. While the batch mean and variance are indispensable for normalization within each training iteration, their direct application to the subsequent inference stage would result in an inconsistent mapping reliant on batch composition. To remedy this, we employ *moving* or *exponentially weighted* statistics. These statistics effectively create a rolling average of the mean and variance observed across all batches seen during training, thereby offering a more stable representation of the data distribution.

The process typically involves two critical steps: calculating batch statistics and updating moving statistics using a momentum-like parameter. During each mini-batch forward pass, we calculate the mean and variance of the activations *within the batch*. These values, denoted as μ<sub>batch</sub> and σ<sup>2</sup><sub>batch</sub>, are used to normalize the batch activations before they are fed into the subsequent layer. Concurrently, and independently of this normalization step, we also calculate and update the *moving* mean and variance using the batch values, the moving values already existing from prior mini-batches, and a momentum decay parameter.

The mathematical expressions for these updates are given as follows:

μ<sub>moving</sub> = (1 - momentum) * μ<sub>moving</sub> + momentum * μ<sub>batch</sub>
σ<sup>2</sup><sub>moving</sub> = (1 - momentum) * σ<sup>2</sup><sub>moving</sub> + momentum * σ<sup>2</sup><sub>batch</sub>

Here, 'momentum' (often denoted as β) is a hyperparameter between 0 and 1 that determines the influence of each new batch on the moving statistics. A higher momentum (e.g., 0.9 or 0.99) prioritizes the past values, which is common during stable training regimes, whereas a lower value (e.g., 0.1) gives greater weight to recent batches for faster adaptation but may be less stable. Crucially, these moving statistics, μ<sub>moving</sub> and σ<sup>2</sup><sub>moving</sub>, are *not* used during the forward pass of the training phase; they are only updated. During inference (after training), these moving statistics are utilized to perform batch normalization. This ensures consistent behavior independent of the current input size.

Let us consider some TensorFlow code implementations, highlighting both proper and potentially flawed techniques.

**Example 1: Demonstrating Correct Moving Statistics Update**

This example showcases how to properly calculate and update the moving mean and variance during training in a TensorFlow model. This method aligns with the best practice for implementing batch normalization layers.

```python
import tensorflow as tf

def batch_norm_layer(inputs, is_training, momentum=0.99):
    channels = inputs.shape[-1]
    moving_mean = tf.Variable(tf.zeros([channels]), trainable=False)
    moving_variance = tf.Variable(tf.ones([channels]), trainable=False)

    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
        update_moving_mean = tf.assign(moving_mean, moving_mean * (1 - momentum) + mean * momentum)
        update_moving_variance = tf.assign(moving_variance, moving_variance * (1 - momentum) + variance * momentum)

        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            normalized_inputs = tf.nn.batch_normalization(inputs, mean, variance, offset=None, scale=None, variance_epsilon=1e-5)
    else:
        normalized_inputs = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, offset=None, scale=None, variance_epsilon=1e-5)
    
    return normalized_inputs

# Example Usage (Assume 'input_tensor' is a training tensor of shape (batch, height, width, channels))
input_tensor = tf.random.normal((32, 32, 32, 32))
is_training = True # Use False during inference

normalized_tensor = batch_norm_layer(input_tensor, is_training)
```

**Commentary on Example 1:** This code snippet implements a custom batch normalization layer. Crucially, it makes use of `tf.Variable` to store the moving mean and variance. These variables are set as *non-trainable*, so they will not update during backpropagation. During training (`is_training=True`), the `tf.nn.moments` function calculates the batch mean and variance. The critical part is the use of `tf.assign` to update the moving statistics using the momentum value. `tf.control_dependencies` ensures that the update operations occur *before* the batch normalization operation is performed. During inference, when `is_training` is set to `False`, the `moving_mean` and `moving_variance` are used directly in `tf.nn.batch_normalization`. This implements the distinction between training and inference necessary for effective batch normalization.

**Example 2: A Common Pitfall with Incorrect Update Logic**

This example shows what *not* to do when handling moving statistics. It highlights how an incorrect approach can lead to instability during training and inconsistency between training and inference.

```python
import tensorflow as tf

def incorrect_batch_norm_layer(inputs, is_training, momentum=0.99):
    channels = inputs.shape[-1]
    moving_mean = tf.Variable(tf.zeros([channels]), trainable=False)
    moving_variance = tf.Variable(tf.ones([channels]), trainable=False)

    mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)

    if is_training:
      moving_mean = moving_mean * (1 - momentum) + mean * momentum # Incorrect update
      moving_variance = moving_variance * (1 - momentum) + variance * momentum # Incorrect update
      normalized_inputs = tf.nn.batch_normalization(inputs, mean, variance, offset=None, scale=None, variance_epsilon=1e-5)

    else:
      normalized_inputs = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, offset=None, scale=None, variance_epsilon=1e-5)
    
    return normalized_inputs

# Example Usage
input_tensor = tf.random.normal((32, 32, 32, 32))
is_training = True # Use False during inference

normalized_tensor = incorrect_batch_norm_layer(input_tensor, is_training)
```

**Commentary on Example 2:** The core problem here is the direct assignment of updated moving statistics using standard arithmetic operations (`=`). This *does not* modify the underlying TensorFlow `Variable` objects. In TensorFlow, assignments to `tf.Variable` objects should always occur using a method that updates that object's internal data (i.e `tf.assign`). Therefore, the `moving_mean` and `moving_variance` are effectively *shadowed* by newly created tensors, which are subsequently lost and therefore these moving statistics do not actually get updated. This means the `moving_mean` and `moving_variance` remain at their initial values (0s and 1s, respectively), creating a divergence between training and inference phases because in the training phase, the batch statistics (not the moving statistics) are used for normalization, whereas during inference, constant initial values are used.

**Example 3: Illustrating usage of `tf.keras.layers.BatchNormalization` (Recommended)**

TensorFlow already provides a dedicated layer for batch normalization that encapsulates the mechanisms needed to maintain moving statistics, simplifying development. The following example shows this approach.

```python
import tensorflow as tf

def batch_norm_keras_layer(inputs, is_training, momentum=0.99):
    bn_layer = tf.keras.layers.BatchNormalization(momentum=momentum)
    if is_training:
        normalized_inputs = bn_layer(inputs, training=True)
    else:
        normalized_inputs = bn_layer(inputs, training=False)
    return normalized_inputs

# Example Usage
input_tensor = tf.random.normal((32, 32, 32, 32))
is_training = True # Use False during inference

normalized_tensor = batch_norm_keras_layer(input_tensor, is_training)
```

**Commentary on Example 3:** The code leverages the `tf.keras.layers.BatchNormalization` layer. This implementation is straightforward because it abstracts away much of the boilerplate code necessary for maintaining moving statistics. Crucially, the `training` parameter controls whether batch statistics or moving statistics are utilized and whether the moving statistics are updated. This simplifies the management of moving statistics and allows a user to focus on higher-level model architectures. This approach is generally preferred for its conciseness, clarity, and reliable implementation of batch normalization. The `momentum` parameter is passed directly to the `BatchNormalization` class during instantiation, aligning with the previously discussed momentum usage.

For a deeper dive, I suggest reviewing the batch normalization sections in the TensorFlow official documentation and exploring more complex examples that utilize it, such as those found in image classification or natural language processing tasks. Furthermore, studying research papers on adaptive normalization techniques that modify the momentum or overall batch normalization process can enhance a practitioner's understanding of this crucial technique.
