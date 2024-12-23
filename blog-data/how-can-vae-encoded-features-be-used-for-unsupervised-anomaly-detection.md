---
title: "How can VAE encoded features be used for unsupervised anomaly detection?"
date: "2024-12-23"
id: "how-can-vae-encoded-features-be-used-for-unsupervised-anomaly-detection"
---

, let's dive into this. Thinking back to my time at that fintech startup, we had a real need for robust anomaly detection, specifically within user transaction data. We were drowning in high-dimensionality time-series data, and labeling the anomalous ones just wasn't scalable. This led us down the path of leveraging variational autoencoders (vaes) for an unsupervised solution, and I can tell you it was quite the journey.

The core idea behind using vaes for unsupervised anomaly detection hinges on their ability to learn a compressed, latent representation of the input data distribution. Essentially, a vae consists of an encoder network that maps the input data (let’s call it 'x') to a lower-dimensional latent space (represented by 'z'), and a decoder network that attempts to reconstruct the input from this latent representation. This process forces the vae to learn a compact encoding that captures the essential features of the typical, “normal” data.

Now, here's where it gets interesting for anomaly detection. Anomalous data, by definition, deviates significantly from the normal patterns that the vae has learned. When an anomaly is fed into the trained encoder, the latent representation it produces tends to fall outside the typical region of the latent space. As a result, the decoder, which is trained on 'normal' data, will likely struggle to accurately reconstruct this anomalous input. This reconstruction error – the difference between the original input and the reconstructed output – becomes our anomaly score. The higher the reconstruction error, the more likely the input is an anomaly.

There are a few nuances to consider though. The latent space, 'z', isn't just a single point; it's a distribution. The vae learns the parameters (mean and variance) of this distribution for each input. This probabilistic nature is what gives the vae its strength over basic autoencoders (aes), which just learn a deterministic encoding. During the training, the vae optimizes its parameters using a loss function that consists of a reconstruction loss and a kullback-leibler (kl) divergence term. The reconstruction loss forces the decoder to accurately reconstruct the input. The kl divergence term ensures that the latent distributions for each input adhere to a predefined distribution (usually a standard normal distribution), acting as a regularizer which encourages a well-structured and continuous latent space.

To put it concretely, consider a simplified example using Python with TensorFlow/Keras (assuming you’re familiar with these libraries). Let’s create a basic vae implementation:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 2 # Latent space dimension

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the encoder
encoder_inputs = keras.Input(shape=(784,)) # Assuming 28x28 flattened input
x = layers.Dense(512, activation="relu")(encoder_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


# Define the decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(512, activation="relu")(latent_inputs)
decoder_outputs = layers.Dense(784, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Define the VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Build and compile VAE, then train on your normal data
vae = VAE(encoder, decoder)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer=optimizer)

# example training data loading and running
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(x_train.shape[0], 784)

vae.fit(x_train, epochs=10, batch_size=32)


# Anomaly detection
def compute_anomaly_score(vae, x):
  _, _, z = vae.encoder(x)
  reconstructed = vae.decoder(z)
  reconstruction_error = tf.reduce_mean(tf.square(x - reconstructed), axis=1)
  return reconstruction_error.numpy()

# Example with test data that may contain anomalies
(x_test, _), _ = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(x_test.shape[0], 784)

anomaly_scores = compute_anomaly_score(vae, x_test)
threshold = 0.01  # Adjust this based on your data
anomalies = x_test[anomaly_scores > threshold]
print(f"Number of anomalies detected: {len(anomalies)}")

```

In this simplified snippet, we define the encoder, decoder, and then the complete vae model. We then train this model on a training dataset which is assumed to contain only non-anomalous data. The anomaly score, in this case, is simply the squared error between the input and its reconstruction from the vae. You’d typically define a threshold beyond which a sample is flagged as an anomaly. This is very basic; you'd want to adjust network architecture and hyperparameters for your specific dataset.

Now, let’s imagine you have a complex dataset, like time-series data. You might need more sophisticated vae architectures, such as recurrent vaes (rvaes), which can model the temporal dependencies within time series. Here's a conceptual example, showcasing a similar architecture:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 2
time_steps = 10 #sequence length
features = 10 #number of features

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the rvae encoder
encoder_inputs = keras.Input(shape=(time_steps, features))
x = layers.GRU(128, return_sequences=False)(encoder_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Define the rvae decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.RepeatVector(time_steps)(latent_inputs)
x = layers.GRU(128, return_sequences=True)(x)
decoder_outputs = layers.TimeDistributed(layers.Dense(features))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Define the VAE class (using the same loss as before)
class RVAE(keras.Model):
     def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

     @property
     def metrics(self):
         return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

     def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mse(data, reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# Build and compile RVAE
rvae = RVAE(encoder, decoder)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
rvae.compile(optimizer=optimizer)

# Example time series training data:
import numpy as np
x_train_ts = np.random.rand(1000, time_steps, features).astype(np.float32)

rvae.fit(x_train_ts, epochs=10, batch_size=32)


# anomaly detection for time series:
def compute_anomaly_score_rvae(rvae, x):
    _, _, z = rvae.encoder(x)
    reconstructed = rvae.decoder(z)
    reconstruction_error = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstructed), axis=2),axis=1) #mean squared error for time dimension and then the mean over features
    return reconstruction_error.numpy()

# Generate some anomaly samples
x_anomalous = np.random.rand(100, time_steps, features).astype(np.float32) + np.random.normal(scale=0.5, size=(100, time_steps, features))

anomaly_scores = compute_anomaly_score_rvae(rvae, x_anomalous)
threshold = 0.2 #adjust threshold
anomalies = x_anomalous[anomaly_scores > threshold]

print(f"Number of anomalies detected in the time series data: {len(anomalies)}")


```

Finally, beyond just reconstruction error, you can leverage other aspects of the vae outputs for anomaly detection. For example, the variance in the latent space itself can be indicative of an anomaly. If the encoder maps an input to a region of the latent space where there’s low variance across normal samples, a large variance estimate for a given input could also signal an anomaly. I have included this concept within the `Sampling` layer which uses the mean and standard deviation output by the encoder. This is a more advanced application and would require delving further into the probabilistic nature of the latent space and often a good follow-up step if the simple reconstruction approach isn't cutting it. Here is a short example how it is being used in code:

```python
def compute_anomaly_score_latent_space(vae, x):
    z_mean, z_log_var, z = vae.encoder(x)
    latent_variance = tf.math.reduce_mean(tf.math.exp(z_log_var), axis=1)
    return latent_variance.numpy()

latent_variance_scores = compute_anomaly_score_latent_space(rvae, x_anomalous)
threshold_latent = 0.5 # Adjust as needed
latent_anomalies = x_anomalous[latent_variance_scores > threshold_latent]

print(f"Number of anomalies detected by latent space variance: {len(latent_anomalies)}")
```
This approach captures the uncertainty that the vae has when it encodes a particular input.

For a deeper understanding of vaes, I highly recommend reading "Auto-Encoding Variational Bayes" by Kingma and Welling; that's a foundational paper. For more on anomaly detection in general, "Outlier Analysis" by Charu Aggarwal is a very comprehensive reference. And if you need to tackle time-series, "Deep Learning for Time Series Forecasting" by Jason Brownlee is a good practical resource that also discusses vae-based methods. These sources will provide the rigorous theoretical background and practical insights you need to implement this kind of approach.

In practice, fine-tuning your thresholds and potentially utilizing combinations of different anomaly scoring methods can significantly improve detection accuracy. The specific choices will be highly data-dependent, necessitating a good understanding of the characteristics of your dataset and the performance characteristics of the vae you are using.
