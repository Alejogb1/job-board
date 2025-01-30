---
title: "How does TensorFlow's batch normalization function work?"
date: "2025-01-30"
id: "how-does-tensorflows-batch-normalization-function-work"
---
Batch normalization is, at its core, a technique to address internal covariate shift during neural network training, where the distribution of layer inputs changes as the parameters of previous layers evolve. My experience over the past seven years building computer vision models highlighted its importance; without it, achieving stable and performant training, especially with deep architectures, was significantly more challenging. TensorFlow's `tf.keras.layers.BatchNormalization` function implements this powerful normalization.

The core concept behind batch normalization involves normalizing the activations of a layer within each mini-batch during training. This normalization process shifts and scales the activations to have a mean of zero and a standard deviation of one, effectively stabilizing the input distribution for subsequent layers. This stability, in turn, allows for higher learning rates and accelerates convergence. Crucially, this normalization is performed per feature, not across the entire batch dimension for each feature channel. While normalization is applied during training, during inference, fixed statistics (mean and variance) calculated over the entire training dataset are used instead. The method provides learnable affine transformation parameters (gamma and beta) to maintain network expressive power and allow the network to learn the optimal scale and bias for the activations, if such are required.

Specifically, within each mini-batch, batch normalization performs the following calculations. For each feature or activation channel, denoted by *xᵢ*, we first compute the mini-batch mean, *μ<sub>B</sub>*:

*μ<sub>B</sub>* = (1/m) Σ *xᵢ*  where m is the batch size and the sum is over all examples in the batch for the given channel.

Next, we calculate the mini-batch variance, *σ<sup>2</sup><sub>B</sub>*:

*σ<sup>2</sup><sub>B</sub>* = (1/m) Σ (*xᵢ* - *μ<sub>B</sub>*)<sup>2</sup>

With these batch statistics, we normalize the input activations, producing *x̂ᵢ*:

*x̂ᵢ* = (*xᵢ* - *μ<sub>B</sub>*) / sqrt(*σ<sup>2</sup><sub>B</sub>* + *ϵ*)

Here, *ϵ* is a small constant added for numerical stability, preventing division by zero. TensorFlow defaults to 1e-3.

Finally, the output *yᵢ* of the batch normalization layer is computed via affine transformation:

*yᵢ* = *γ* *x̂ᵢ* + *β*

The trainable parameters, *γ* (gamma) and *β* (beta), allow the network to learn a suitable scale and shift for each feature, undoing the initial standard normalization if the network requires it. This ensures the model does not get forced into a specific activation distribution, preserving the model's representation capacity.

Let’s look at concrete examples using TensorFlow. The first example demonstrates basic usage within a sequential model:

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Generate dummy data for demonstration
import numpy as np
x_train = np.random.rand(1000, 100)
y_train = np.random.randint(0, 10, size=(1000, ))
y_train_onehot = tf.one_hot(y_train, depth=10)

model.fit(x_train, y_train_onehot, epochs=10, batch_size=32)
```
Here, a simple feedforward neural network is created. The `BatchNormalization` layer is inserted after the first Dense layer. During the fit operation, TensorFlow automatically calculates the per-batch means and variances. Note that `BatchNormalization` is parameterizable, with adjustable momentum.  The momentum is used for moving average calculation of mean and variance during training which will be used during inference. The fit call updates trainable parameters such as γ, β and internal statistics such as mean and variance.

A second example illustrates how batch normalization can be applied in a convolutional network, a more common use case:

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Dummy data, representing grayscale images
import numpy as np
x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(0, 10, size=(1000, ))
y_train_onehot = tf.one_hot(y_train, depth=10)


model.fit(x_train, y_train_onehot, epochs=10, batch_size=32)
```

This example adds batch normalization layers after the convolutional layers, which is typically where we see it applied in image-based tasks.  The `BatchNormalization` layer normalizes the feature maps output from the `Conv2D` layer independently for each channel. The batch normalization layer's location after convolution or after an activation function can have a slight impact, where activation is generally preferred.

Finally, let’s illustrate how we can inspect the internal parameters of the layer. This can be helpful for understanding the learned transformations:

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
  tf.keras.layers.BatchNormalization(name='batch_norm_layer'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Generate dummy data
import numpy as np
x_train = np.random.rand(1000, 100)
y_train = np.random.randint(0, 10, size=(1000, ))
y_train_onehot = tf.one_hot(y_train, depth=10)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train_onehot, epochs=2, batch_size=32, verbose=0) # training for params to learn

batch_norm_layer = model.get_layer('batch_norm_layer')
gamma = batch_norm_layer.gamma.numpy()
beta = batch_norm_layer.beta.numpy()
moving_mean = batch_norm_layer.moving_mean.numpy()
moving_variance = batch_norm_layer.moving_variance.numpy()

print("Gamma:", gamma.shape, gamma)
print("Beta:", beta.shape, beta)
print("Moving Mean:", moving_mean.shape, moving_mean)
print("Moving Variance:", moving_variance.shape, moving_variance)
```

In this code, we specifically named the batch normalization layer for retrieval later. After a few epochs of training, the gamma and beta, trainable parameters, and the moving mean and moving variance internal parameters are extracted and printed, revealing the shape and the learned values. This allows us to confirm that the layer is learning what it is intended to learn. When using the model for inference, the moving mean and moving variance parameters are used for normalization instead of batch-level statistics.

Several resources delve into the intricacies of batch normalization. The original research paper is always a valuable source. Publications on more recent improvements and extensions to batch normalization such as layer normalization, and group normalization are worth reviewing. Additionally, review books and articles focusing on the practical aspects of deep learning to gain a deeper understanding of batch normalization within the context of training various neural network architectures.
Understanding batch normalization is essential for effectively training deep neural networks. Its capacity to mitigate the issues caused by covariate shift contributes to more stable, accurate, and efficient training processes. The ability to retrieve the internal parameters of the `tf.keras.layers.BatchNormalization` function, and the ability to configure parameters like momentum and *ϵ* are important aspects of it’s practical use, as demonstrated.
