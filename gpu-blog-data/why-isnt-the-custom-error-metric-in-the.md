---
title: "Why isn't the custom error metric in the autoencoder performing as expected?"
date: "2025-01-30"
id: "why-isnt-the-custom-error-metric-in-the"
---
Specifically, the model's reconstruction loss is decreasing as expected, but my custom error metric is not converging at all, and sometimes seems to increase even as the reconstruction improves.

The discrepancy between a decreasing reconstruction loss and a non-converging, or even increasing, custom error metric in an autoencoder points to a fundamental misalignment between the loss function guiding the network's optimization and the metric used to assess performance. Having spent the better part of a year wrestling with anomaly detection autoencoders for industrial sensor data, I’ve repeatedly encountered this exact issue, and it usually stems from a flawed assumption about the custom metric’s relationship to the underlying data space.

The reconstruction loss, usually mean squared error (MSE) or binary cross-entropy (BCE), forces the autoencoder to learn a compressed representation that can accurately recreate the input. Critically, this optimization is confined to the *encoded* space. A well-trained model in this context learns to minimize the *difference* between the input and its reconstructed version. However, a custom error metric often measures a different aspect of the reconstruction, often one that lies outside of the direct gradient descent performed on the loss function. Let’s break this down with specific examples.

The first scenario involves using a custom metric that's sensitive to the **relative differences** between features, while the reconstruction loss is based on absolute differences. Consider a simple case where we are reconstructing a vector of three features representing vibration frequencies.

```python
import tensorflow as tf
import numpy as np

def custom_relative_error(y_true, y_pred):
  """Calculates the mean relative absolute difference. Not suitable as a loss function."""
  return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-8)))

# Dummy data
input_data = tf.constant(np.array([[10, 2, 0.5], [12, 3, 0.6], [14, 4, 0.7]], dtype=np.float32))

# Reconstruction using simple linear layers (for demonstrative purposes)
encoder = tf.keras.layers.Dense(units=2, activation='relu')
decoder = tf.keras.layers.Dense(units=3, activation='linear')

encoded_data = encoder(input_data)
decoded_data = decoder(encoded_data)

# Example using MSE for the reconstruction loss
mse_loss = tf.keras.losses.MeanSquaredError()(input_data, decoded_data)

# Applying the custom metric to assess reconstruction
custom_error = custom_relative_error(input_data, decoded_data)

print(f"Mean Squared Error Loss: {mse_loss.numpy():.4f}")
print(f"Custom Relative Error: {custom_error.numpy():.4f}")
```

This example defines a custom error function that calculates the mean relative difference, which is not a good candidate for a loss function. The model’s *loss* gradient is based on the MSE which aims for small absolute differences, therefore, it can minimize MSE even when the *relative* error is high. For instance, if the original data point is `[10, 2, 0.5]`, and the reconstruction is `[10.1, 2.1, 0.6]`, the MSE would be relatively small. However, if the model's next update drives the reconstruction to `[10.01, 2.01, 0.59]`, the MSE decreases, but the relative error might not have changed, or, in more extreme cases, it could have slightly increased. If the original features had a magnitude difference like `[100, 1, 0.01]`, the effect of this could be much larger. The point here is the metric is being evaluated in a different space than what is being optimized. The reconstruction loss does not guide the network to minimize the relative error, and therefore you can have scenarios where one decreases and the other stays the same or increase.

A second scenario that can produce these divergent trends involves metrics that are sensitive to **specific patterns** within the data, rather than overall reconstruction quality. Imagine that we want to reconstruct a time-series signal, and our custom metric focuses on the presence of a certain frequency within the signal, while the MSE loss focuses on overall signal reconstruction.

```python
import tensorflow as tf
import numpy as np

def custom_frequency_metric(y_true, y_pred):
  """Measures the difference in the magnitude of a specific frequency using FFT. 
  Not suitable as a loss function."""
  true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
  pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))
  target_frequency_bin = 5  # Example: measuring the 5th frequency component
  return tf.abs(tf.abs(true_fft[..., target_frequency_bin]) - tf.abs(pred_fft[..., target_frequency_bin]))


# Generate a sine wave with a specific frequency
time = np.arange(100)
frequency = 0.05
signal = np.sin(2 * np.pi * frequency * time) + 0.1*np.random.randn(100) # Add some noise
input_signal = tf.constant(signal.reshape(1, 100), dtype=tf.float32)

# Reconstruction with simple linear layers, similar to the previous example
encoder = tf.keras.layers.Dense(units=50, activation='relu')
decoder = tf.keras.layers.Dense(units=100, activation='linear')

encoded_signal = encoder(input_signal)
decoded_signal = decoder(encoded_signal)

# Example using MSE for reconstruction loss
mse_loss = tf.keras.losses.MeanSquaredError()(input_signal, decoded_signal)

# Apply custom frequency metric
custom_metric = custom_frequency_metric(input_signal, decoded_signal)
print(f"Mean Squared Error Loss: {mse_loss.numpy():.4f}")
print(f"Custom Frequency Metric: {custom_metric.numpy():.4f}")
```

In this example, the MSE aims for overall signal approximation but not explicitly to match the *magnitude of the 5th component* of the FFT. During training, the model could be learning to reconstruct the general shape of the signal quite well, reducing the MSE, but the frequency of interest might be represented with an inaccurate amplitude, resulting in a non-converging frequency metric or even an increased difference. In short, the metric is measuring something the loss function is not optimizing.

Finally, a third common source of this problem is when the custom metric is sensitive to **small variations** in the data that the loss function considers inconsequential.

```python
import tensorflow as tf
import numpy as np

def custom_absolute_deviation_metric(y_true, y_pred):
    """Calculates a highly sensitive deviation measure. Not suitable as a loss function"""
    return tf.reduce_sum(tf.abs(y_true - y_pred))

# Dummy data with slight variations
input_data = tf.constant(np.array([[1.001, 2.002, 3.001], [4.002, 5.001, 6.002]], dtype=np.float32))

# Reconstruction using simple linear layers
encoder = tf.keras.layers.Dense(units=2, activation='relu')
decoder = tf.keras.layers.Dense(units=3, activation='linear')

encoded_data = encoder(input_data)
decoded_data = decoder(encoded_data)

# Example using MSE for reconstruction loss
mse_loss = tf.keras.losses.MeanSquaredError()(input_data, decoded_data)

# Apply the custom metric
custom_metric = custom_absolute_deviation_metric(input_data, decoded_data)
print(f"Mean Squared Error Loss: {mse_loss.numpy():.4f}")
print(f"Custom Deviation Metric: {custom_metric.numpy():.4f}")
```

In this scenario, the MSE might consider small differences insignificant, but the custom metric, which simply calculates the sum of absolute differences, can be extremely sensitive to these deviations and might appear inconsistent as the network learns to minimize the MSE. Specifically, the MSE calculates the mean of the *squared* difference, and small variations have a small squared effect, therefore the gradient is small when that effect is small. However, the simple sum of the absolute difference will cause a very large gradient for small variations and can easily misalign with the optimization.

The key takeaway is to understand the space in which both the loss function and the custom metric are operating. If these spaces are fundamentally different, the metrics will likely diverge. A robust approach to resolving such issues includes: 1) **Carefully evaluating the chosen custom metric:** understanding if it's measuring a different aspect of the data than the loss is optimizing, and if it is, considering if it is *necessary* for optimization or just an evaluation metric. If it's necessary for optimization, the loss function should be altered so that it includes that aspect of the data space. 2) **Consider using the custom metric as the loss function:** if the metric is really what you care about, it may be possible to use that or an approximation of it as a differentiable loss function. However, certain metrics may be non-differentiable, or they may cause instability during gradient updates. 3) **Carefully examine the scale and type of error:** is it relative error that matters, or absolute? Does it have to do with specific frequency components? This will enable one to focus the model on the aspects of the data that are important. 4) **Visualization:** Plotting both the loss and the metric in each training step can provide valuable insights into when the divergence appears.

For further information, I would suggest examining resources on loss functions, error metrics, signal processing basics, and general concepts in machine learning. Specifically, resources focused on *metric learning*, such as those that focus on contrastive loss or triplet loss, can provide alternative strategies if you need to optimize with non-standard distance measures. Additionally, more information can be found in introductory books and documentation on Tensorflow/Keras loss functions and optimizers, which can help ensure the custom metric is appropriate for the network. Finally, more practical guidance can be found in the numerous papers on anomaly detection using autoencoders.
