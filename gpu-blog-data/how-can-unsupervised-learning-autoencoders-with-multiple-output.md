---
title: "How can unsupervised learning autoencoders with multiple output CNNs effectively extract features from MNIST data?"
date: "2025-01-30"
id: "how-can-unsupervised-learning-autoencoders-with-multiple-output"
---
The core challenge in leveraging autoencoders for feature extraction from MNIST images, particularly when coupled with multiple output convolutional neural networks (CNNs), lies in effectively disentangling latent representations to yield discriminative and reusable features for downstream tasks. This differs from simply reconstructing the input; it requires training the autoencoder to prioritize those latent factors that are relevant to the desired characteristics, and CNNs must then learn to map those latent spaces to task specific outputs. My own experience working with image recognition systems highlights the need to go beyond straightforward image reconstruction to achieve this.

An autoencoder, at its foundational level, consists of two primary components: an encoder and a decoder. The encoder compresses the high-dimensional input data, such as an MNIST image (28x28 pixels), into a lower-dimensional latent space representation. Conversely, the decoder attempts to reconstruct the original input based solely on this latent representation. Unsupervised learning of this architecture centers on the reconstruction error—the discrepancy between the input and reconstructed image. However, for feature extraction, merely reconstructing the input is inadequate. The latent representation must encapsulate the salient features for subsequent usage. This involves moving beyond simple pixel replication.

When we consider coupling this with multiple output CNNs, we introduce additional layers of complexity and opportunity. The CNNs, in this architecture, are no longer operating directly on raw image pixels; rather, they operate on the compressed latent representation output by the encoder. Each output CNN can be viewed as having its own specialization, mapping the same latent space to a different task or classification objective. This enables us to train, for example, separate CNNs to identify particular digit styles, rotation angles, or even more abstract properties if the latent space representation is sufficiently rich and disentangled.

Let’s examine a simplified implementation using Python with TensorFlow/Keras, focusing on how the components interoperate:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Define the encoder
def build_encoder(latent_dim):
    encoder = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(latent_dim, activation=None) # No activation here for latent representation
    ], name='encoder')
    return encoder


# Define the decoder
def build_decoder(latent_dim):
  decoder = models.Sequential([
      layers.Input(shape=(latent_dim,)),
      layers.Dense(7*7*64, activation='relu'),
      layers.Reshape((7,7,64)),
      layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu'),
      layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu'),
      layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid') # Sigmoid for pixel values between 0 and 1
      ], name='decoder')
  return decoder

# Build the Autoencoder
def build_autoencoder(latent_dim):
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    autoencoder = models.Model(inputs=encoder.input, outputs=decoder(encoder.output))
    return autoencoder

latent_dim = 16 # Example latent space dimensionality

autoencoder = build_autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss='mse') # Using Mean Squared Error for Reconstruction

# Load MNIST Data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0  # Normalize to 0-1 range
x_train = np.reshape(x_train, (-1, 28, 28, 1))

# Train Autoencoder
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)


```

In this code snippet, I first define separate `build_encoder` and `build_decoder` functions, utilizing convolutional layers, pooling and transposed convolutional layers to handle spatial information. The important distinction lies in the encoder's final dense layer that produces the latent representation which has no activation function. This allows for a flexible, possibly unbounded, latent space. The decoder mirrors the encoder's structure, upsampling and using transposed convolution to recover image size. The `build_autoencoder` function integrates the encoder and decoder. Critically, mean squared error (MSE) is used as the loss, pushing the network to learn a compressed representation capable of good reconstruction. The MNIST dataset is normalized and used for training. This is the crucial unsupervised learning phase where no labels are used and the network learns to compress and reconstruct the image. The use of sigmoid activation in the last layer of the decoder ensures output is between 0 and 1, like the normalized input.

Now, let’s integrate multiple output CNNs. This involves utilizing the encoder from the previous autoencoder as a feature extractor. I will create two example CNNs, each with a different classification target: the first to classify between odd and even, and the second to categorize the digit by its "curvature," a crude metric defined as if a sum of pixels is above a threshold to be 'rounded' otherwise 'sharp'. Note: This is for demonstration only and does not represent effective categorization of the digits.

```python

# Define feature extraction model
feature_extractor = models.Model(inputs=autoencoder.get_layer('encoder').input, outputs=autoencoder.get_layer('encoder').output)

# Define the CNN for odd/even classification
def build_odd_even_classifier(latent_dim):
    classifier = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid', name='odd_even_output') # Single output for binary classification
    ], name='odd_even_classifier')
    return classifier

# Define the CNN for curvature classification
def build_curvature_classifier(latent_dim):
    classifier = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax', name='curvature_output') # Two outputs for classification of rounded/sharp
    ], name='curvature_classifier')
    return classifier


odd_even_classifier = build_odd_even_classifier(latent_dim)
curvature_classifier = build_curvature_classifier(latent_dim)

# Create the combined model
latent_input = layers.Input(shape=(latent_dim,))
odd_even_output = odd_even_classifier(latent_input)
curvature_output = curvature_classifier(latent_input)
combined_model = models.Model(inputs=latent_input, outputs=[odd_even_output, curvature_output])


#Generate features using encoder
latent_features = feature_extractor.predict(x_train)

# Prepare output labels
odd_even_labels = np.array([i % 2 for i in range(len(x_train))])
#For demonstration, a crude method of curvature labeling
curvature_labels = np.array([1 if np.sum(x_train[i]) > (28*28*0.4) else 0 for i in range(len(x_train))])
curvature_labels = tf.keras.utils.to_categorical(curvature_labels)

# Compile the model
combined_model.compile(optimizer='adam', loss=['binary_crossentropy', 'categorical_crossentropy'], metrics=[['accuracy'], ['accuracy']])

# Train the classifier models using the encoded latent space.
combined_model.fit(latent_features, [odd_even_labels, curvature_labels], epochs=20, batch_size=256, shuffle=True)


```

Here, I extract the encoder part from the previously trained autoencoder and use it as a feature extractor, discarding the decoder. I then define `build_odd_even_classifier` and `build_curvature_classifier` functions to instantiate two different CNN architectures, each tailored to a specific classification task. The classifiers are densely connected networks with an output layer that corresponds to the required number of classes. I then create a combined model that chains together the classifiers with latent features as input. The output of the feature extractor is fed into the two classifier networks, demonstrating how the extracted features can be used for various downstream tasks. Separate categorical crossentropy losses are used for the classifiers, appropriate for the respective classification types. The data are prepared with odd/even and curvature labels and are trained against the combined model.

Finally, to enhance feature disentanglement, a modified loss function during autoencoder training can be helpful. This is a technique called variational autoencoding. Here I introduce a simplified example to give a concept, but actual VAE implementation will require probability distribution modeling and sampling methods.

```python
# Function to add a KL divergence term to the loss (simplified)
def kl_divergence_loss(mean, log_var):
    return -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))

def build_variational_encoder(latent_dim):
    encoder = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(latent_dim * 2) # Now we output mean and log variance
    ], name='variational_encoder')
    return encoder

def build_variational_autoencoder(latent_dim):
    encoder = build_variational_encoder(latent_dim)
    decoder = build_decoder(latent_dim) # Using the previous decoder
    # Function to sample from the latent space using reparameterization trick (simplified)
    def sample(z_mean, z_log_var):
      epsilon = tf.random.normal(shape=tf.shape(z_mean))
      return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Model setup
    encoded_output = encoder(encoder.input)
    mean, log_var = tf.split(encoded_output, num_or_size_splits=2, axis=1)
    sampled_latent = sample(mean, log_var)
    decoded_output = decoder(sampled_latent)
    vae = models.Model(inputs=encoder.input, outputs=decoded_output)


    # Add Custom Loss Function
    def vae_loss(y_true, y_pred):
      reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred))
      kl_loss = kl_divergence_loss(mean, log_var)
      return reconstruction_loss + kl_loss

    vae.add_loss(vae_loss(vae.input, vae.output))
    return vae

variational_autoencoder = build_variational_autoencoder(latent_dim)
variational_autoencoder.compile(optimizer='adam')

# Train the Variational Autoencoder
variational_autoencoder.fit(x_train, epochs=20, batch_size=256, shuffle=True)

```

In this simplified VAE example, the encoder now outputs parameters for a probability distribution (mean and log variance) instead of just the encoded representation. A simplified sampling function is introduced to sample from the learned distribution. The core difference is the addition of a KL divergence loss term. This term encourages the latent space to conform to a standard normal distribution, thus promoting a more structured, disentangled latent space. This is a highly simplified version, actual VAE requires more complex sampling and distribution modelling.  It will encourage the learned representation to better capture meaningful variations across the dataset compared to a standard autoencoder. This, in turn, would improve the usefulness of the features for subsequent tasks.

For further research, I recommend exploring resources that delve deeper into autoencoder architectures, particularly variational autoencoders, and examine the theoretical underpinnings of disentanglement. Publications from the machine learning community focusing on representation learning are also essential. Textbooks on deep learning often provide comprehensive coverage of the mathematical principles at play. Additionally, tutorials and articles on implementing advanced autoencoders with popular deep learning frameworks like TensorFlow and PyTorch can prove valuable. Understanding the subtle nuances of loss function design and optimization is also necessary to fully grasp the potential of these architectures.
