---
title: "How can TensorFlow Probability be used for anomaly detection?"
date: "2025-01-30"
id: "how-can-tensorflow-probability-be-used-for-anomaly"
---
TensorFlow Probability (TFP) offers a robust framework for anomaly detection by leveraging its capabilities in probabilistic modeling. Unlike traditional deterministic methods that might flag outliers based on rigid thresholds, TFP allows us to model the underlying distribution of normal data and then identify deviations from that distribution as anomalies. This approach accommodates inherent data variability and provides a more nuanced understanding of what constitutes an outlier. My experience working with complex sensor data in industrial environments has demonstrated the practical utility of this probabilistic method.

The core concept lies in establishing a model for the “normal” behavior of the system or process under observation. We achieve this by representing the data generating process as a probability distribution, parametrized by learnable variables. This distribution encapsulates our understanding of how typical data points are generated, and anomalies are, therefore, data points that are unlikely under this distribution. TFP provides the tools for defining such distributions, estimating their parameters based on training data (which is presumed to contain mostly normal samples), and ultimately scoring new data points by their probability or log probability under the learned distribution. Lower probabilities indicate a greater likelihood of an anomaly.

This approach is particularly beneficial in situations where the data has inherent noise, complex relationships, or temporal dependencies. For instance, a simple Gaussian mixture model can capture multi-modal behavior, whereas a deep autoregressive flow can handle sequential data exhibiting non-trivial temporal patterns. In either case, we focus not on predetermined rules but rather on the probabilistic fit of each observation given the model learned from historical data.

To illustrate with code examples, let's begin with a basic scenario: anomaly detection based on a Gaussian distribution. This is suited for simpler datasets where the assumption of normality holds reasonably well.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Simulate some normal data with a mean and standard deviation
normal_data = tf.random.normal([1000], mean=5.0, stddev=1.0)

# Estimate the parameters of the Gaussian distribution from the normal data
mean_est = tf.reduce_mean(normal_data)
std_est = tf.math.reduce_std(normal_data)
gaussian = tfd.Normal(loc=mean_est, scale=std_est)

# Generate some new test data including anomalies
test_data_normal = tf.random.normal([500], mean=5.0, stddev=1.0)
test_data_anomalies = tf.constant([10.0, 12.0, 1.0, -2.0], dtype=tf.float32)
test_data = tf.concat([test_data_normal, test_data_anomalies], axis=0)

# Evaluate log probability of each data point
log_probs = gaussian.log_prob(test_data)

# Threshold for anomaly detection using log probability
threshold = -3.0  # Example threshold - must be tuned for the problem
anomalies = tf.where(log_probs < threshold)

print("Indices of potential anomalies:", anomalies)
```
This code demonstrates a straightforward anomaly detection system. First, it generates training data that conforms to a Normal distribution, then estimates parameters of the distribution to create a model. A mixture of normal and anomalous test data are then scored. Data points with very low log probabilities (i.e. unlikely given the model) are flagged as anomalies. The log probability threshold is a crucial hyperparameter requiring problem-specific tuning. It should be emphasized that the appropriate value of this threshold will depend heavily on factors including the acceptable false positive and false negative rates.

Next, consider a more complex scenario using a Gaussian mixture model (GMM). This can be effective in datasets that show multimodal behavior, where data clusters around multiple means.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Simulate data coming from two gaussian distributions
mixture_probs = [0.4, 0.6] # Probability of each component
normals = [
    tfd.Normal(loc=2.0, scale=1.0),
    tfd.Normal(loc=8.0, scale=1.0),
]
mixture_distribution = tfd.Mixture(
    cat=tfd.Categorical(probs=mixture_probs),
    components=normals
)
normal_data = mixture_distribution.sample(1000)


# Define the GMM with two components and trainable parameters
num_components = 2
locs = tf.Variable(tf.random.normal([num_components]))
scales = tf.Variable(tf.random.normal([num_components], mean=0.5))
mixture_probs_model = tf.Variable(tf.random.uniform([num_components]), constraint=tf.function(lambda x: tf.clip(tf.nn.softmax(x), 1e-8, 1)))


gmm = tfd.Mixture(
    cat=tfd.Categorical(probs=mixture_probs_model),
    components=[tfd.Normal(loc=locs[0], scale=tf.math.softplus(scales[0])), tfd.Normal(loc=locs[1], scale=tf.math.softplus(scales[1]))],
)

# Training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
      loss = -tf.reduce_mean(gmm.log_prob(data))
    grads = tape.gradient(loss, gmm.trainable_variables)
    optimizer.apply_gradients(zip(grads, gmm.trainable_variables))
    return loss

for i in range(500):
    loss = train_step(normal_data)
    if i % 100 == 0:
      print(f"Loss at step {i}: {loss}")


# Test data and log probs calculation
test_data_normal = mixture_distribution.sample(500)
test_data_anomalies = tf.constant([12.0, 14.0, -1.0, -3.0], dtype=tf.float32)
test_data = tf.concat([test_data_normal, test_data_anomalies], axis=0)
log_probs = gmm.log_prob(test_data)
threshold = -3.0
anomalies = tf.where(log_probs < threshold)
print("Indices of potential anomalies:", anomalies)
```

Here, we've extended the concept to a mixture of two Gaussian distributions. We simulate training data from this distribution, and then initialize a model with trainable parameters representing the GMM. The `train_step` function uses gradient descent to estimate the parameters by minimizing the negative log-likelihood, thereby effectively maximizing the probability of the observed training data. Anomaly detection is performed in the same manner as with the simple gaussian case, but utilizing the learned GMM model.

Finally, consider a scenario involving sequential data, employing a Recurrent Neural Network (RNN) to capture temporal dependencies. This demonstrates how more sophisticated TFP features can be utilized for anomaly detection. We’ll use a variational autoencoder (VAE) for this, which is trained to reconstruct the input sequence, providing a likelihood measure.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Generate a synthetic time series dataset
def create_time_series_data(seq_length, num_sequences, noise_std=0.1):
    times = tf.range(0, seq_length, dtype=tf.float32) / seq_length
    signal = tf.sin(2 * 3.14 * times) + tf.sin(4 * 3.14 * times)
    signals = tf.broadcast_to(signal, (num_sequences, seq_length))
    noise = tf.random.normal((num_sequences, seq_length), stddev=noise_std)
    return signals + noise

seq_length = 20
train_sequences = create_time_series_data(seq_length=seq_length, num_sequences=500)

# Build the VAE model
latent_dim = 4
hidden_units = 32

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, hidden_units):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_units, return_sequences=False),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim), activation=None),
            tfp.layers.MultivariateNormalTriL(latent_dim)
            ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.RepeatVector(seq_length),
            tf.keras.layers.LSTM(hidden_units, return_sequences=True),
            tf.keras.layers.Dense(1)
            ])

    def call(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z.sample())
        return z, reconstruction

model = VAE(latent_dim=latent_dim, hidden_units=hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
      z, reconstruction = model(x)
      reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - tf.expand_dims(x, axis=-1)))
      kl_divergence_loss = tf.reduce_mean(z.kl_divergence(tfd.Normal(0, 1)))
      loss = reconstruction_loss + kl_divergence_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for i in range(1000):
    loss = train_step(train_sequences)
    if i % 100 == 0:
      print(f"Loss at step {i}: {loss}")

# Anomaly Detection
test_sequences_normal = create_time_series_data(seq_length, num_sequences=200)
test_sequences_anomaly = create_time_series_data(seq_length, num_sequences=4, noise_std=1.0) # Anomaly by noise
test_sequences_anomaly = test_sequences_anomaly + 1.0 # Anomaly by increased value

test_sequences = tf.concat([test_sequences_normal, test_sequences_anomaly], axis=0)

latents, reconstruction_out = model(test_sequences)
reconstruction_error = tf.reduce_mean(tf.square(reconstruction_out - tf.expand_dims(test_sequences, axis=-1)), axis=(1,2)) #Error for each series

threshold_vae = 0.4
anomalies = tf.where(reconstruction_error > threshold_vae)
print("Indices of potential anomalies:", anomalies)
```

This VAE code represents a more complex example. Training the VAE minimizes a combination of a reconstruction loss and a KL divergence term, encouraging the latent space to conform to a normal distribution. After training, by generating a reconstructed output and comparing it to the original input, we can identify sequences that do not conform to the model as anomalies (measured through higher reconstruction errors).

In summary, TFP provides the flexible modeling tools necessary to detect anomalies effectively, ranging from basic Gaussian assumptions to complex temporal patterns. Key to successful implementation involves a careful selection of the appropriate probability distribution for the problem, effective training methodologies, and the careful tuning of relevant anomaly detection thresholds. It should be noted that the appropriate thresholds and even the specific model chosen will be highly dependent on both the characteristics of the data being processed and also the specific application requirements regarding trade-offs between false positive rates and false negative rates.

To further your understanding, I would recommend studying resources on Bayesian statistics, particularly focusing on probabilistic modeling and inference.  Material covering time series analysis and the application of neural networks for sequential data processing will also be useful. Examining literature and tutorials on Variational Autoencoders can also provide further clarity.
