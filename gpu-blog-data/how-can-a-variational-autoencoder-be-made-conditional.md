---
title: "How can a variational autoencoder be made conditional?"
date: "2025-01-30"
id: "how-can-a-variational-autoencoder-be-made-conditional"
---
Variational autoencoders (VAEs), in their standard formulation, are generative models that learn a latent representation of input data and can sample from this latent space to generate new data points similar to the training set. However, this generation process is inherently unsupervised; there is no direct control over the characteristics of the generated samples. Conditioning a VAE, therefore, enables us to guide the generative process by incorporating auxiliary information, achieving more targeted and controllable generation. I have found this to be essential when working on projects involving structured data generation, particularly in areas like image synthesis with specific attributes, or generating sequences based on given context.

The core idea behind conditional variational autoencoders (CVAEs) is to incorporate the conditioning information into both the encoder and the decoder of the VAE. Specifically, the conditioning information is treated as additional input during both the inference (encoding) and generative (decoding) phases. This modification alters the architecture of the networks and consequently the loss function, allowing the model to learn a latent representation that is sensitive to the conditioning variables. Essentially, we're learning a conditional probability distribution p(x|c), where x is the data and c is the conditioning variable.

In a standard VAE, the encoder maps the input data, x, to a probability distribution in latent space, parameterized by mean (μ) and variance (σ²). Samples, z, from this distribution, are then fed into the decoder to reconstruct the original input, ˆx. The loss function, typically the negative Evidence Lower Bound (ELBO), consists of a reconstruction loss term (measuring the similarity between x and ˆx) and a Kullback-Leibler divergence term (encouraging the latent distribution to be similar to a standard normal distribution).

In a CVAE, we modify this procedure. The conditioning information, c, is concatenated with the input x *before* it enters the encoder. The encoder now learns to map this combined vector to a latent distribution which is also influenced by c. Similarly, when decoding, c is concatenated with the sampled latent variable z. The decoder then learns to reconstruct the input *given* both the latent representation and the conditioning information. This can lead to generation of output data that matches the desired condition. This approach requires some attention to the dimensions of the concatenated vectors; incorrect alignment can render the model ineffective.

Let us look at the implementation. Consider a simple example using Python and TensorFlow. I have used this structure for several data generation projects, and it offers a good foundation for more complex tasks.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 2
condition_dim = 5

# Encoder Model
class Encoder(layers.Layer):
    def __init__(self, latent_dim, condition_dim):
        super(Encoder, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x, c = inputs
        combined = tf.concat([x, c], axis=1)
        h1 = self.dense1(combined)
        h2 = self.dense2(h1)
        return self.mean(h2), self.log_var(h2)

# Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

# Decoder Model
class Decoder(layers.Layer):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(output_dim)

    def call(self, inputs):
        z, c = inputs
        combined = tf.concat([z, c], axis=1)
        h1 = self.dense1(combined)
        h2 = self.dense2(h1)
        return self.out(h2)

# CVAE Model
class CVAE(keras.Model):
    def __init__(self, latent_dim, condition_dim, input_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(latent_dim, condition_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(latent_dim, condition_dim, input_dim)

    def call(self, inputs, training=False):
        x, c = inputs
        mean, log_var = self.encoder((x,c))
        z = self.sampling((mean, log_var))
        reconstructed = self.decoder((z,c))
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        return reconstructed, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            x, c = data
        else:
            x,c = data, None
            raise ValueError("Expected condition variable to be included.")
        with tf.GradientTape() as tape:
             reconstructed, kl_loss = self((x,c), training=True)
             reconstruction_loss = tf.reduce_mean(
                 keras.losses.binary_crossentropy(x, reconstructed)
            ) #For simplicity binary crossentropy assumes x is binary.
             loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "reconstruction_loss": reconstruction_loss,"kl_loss": tf.reduce_mean(kl_loss)}
```
This first example shows the basic structure of a CVAE with encoder, sampling, and decoder classes. Notice the concatenation of input and condition at the encoder and decoder input layers. The `train_step` shows how to incorporate the loss terms. I opted for a binary cross entropy loss for reconstruction. The CVAE object manages everything from sampling to training.

Here's a second example, incorporating a slightly different architecture using convolutional layers. I found convolutional layers to be superior for image data, due to their ability to capture local spatial dependencies.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 16
condition_dim = 5
image_dim = (28,28,1)

#Encoder with convolutional layers
class ConvEncoder(layers.Layer):
    def __init__(self, latent_dim, condition_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')
        self.pool1 = layers.MaxPool2D((2,2))
        self.conv2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.pool2 = layers.MaxPool2D((2,2))
        self.flatten = layers.Flatten()
        self.condition_dense = layers.Dense(64, activation='relu')
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x,c = inputs
        h1 = self.conv1(x)
        h2 = self.pool1(h1)
        h3 = self.conv2(h2)
        h4 = self.pool2(h3)
        flat = self.flatten(h4)
        c_encoded = self.condition_dense(c)
        combined = tf.concat([flat, c_encoded], axis=1)
        return self.mean(combined), self.log_var(combined)

#Decoder with deconvolutional layers
class ConvDecoder(layers.Layer):
    def __init__(self, latent_dim, condition_dim, image_dim):
        super(ConvDecoder, self).__init__()
        self.dense = layers.Dense(7*7*64, activation='relu')
        self.reshape = layers.Reshape((7,7,64))
        self.deconv1 = layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same')
        self.upsample1 = layers.UpSampling2D((2,2))
        self.deconv2 = layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same')
        self.upsample2 = layers.UpSampling2D((2,2))
        self.out = layers.Conv2D(image_dim[-1], (3,3), activation='sigmoid', padding='same')

    def call(self, inputs):
      z, c = inputs
      c_encoded = self.dense(c)
      combined = tf.concat([z,c_encoded], axis=1)
      h1 = self.dense(combined)
      h2 = self.reshape(h1)
      h3 = self.deconv1(h2)
      h4 = self.upsample1(h3)
      h5 = self.deconv2(h4)
      h6 = self.upsample2(h5)
      return self.out(h6)

# CVAE Model with convolutional layers
class ConvCVAE(keras.Model):
    def __init__(self, latent_dim, condition_dim, image_dim):
        super(ConvCVAE, self).__init__()
        self.encoder = ConvEncoder(latent_dim, condition_dim)
        self.sampling = Sampling()
        self.decoder = ConvDecoder(latent_dim, condition_dim, image_dim)

    def call(self, inputs, training=False):
        x, c = inputs
        mean, log_var = self.encoder((x,c))
        z = self.sampling((mean, log_var))
        reconstructed = self.decoder((z,c))
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        return reconstructed, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            x, c = data
        else:
            x,c = data, None
            raise ValueError("Expected condition variable to be included.")
        with tf.GradientTape() as tape:
             reconstructed, kl_loss = self((x,c), training=True)
             reconstruction_loss = tf.reduce_mean(
                 keras.losses.binary_crossentropy(x, reconstructed)
            )
             loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "reconstruction_loss": reconstruction_loss,"kl_loss": tf.reduce_mean(kl_loss)}
```

This example uses convolutional and deconvolutional layers in the encoder and decoder, respectively. The conditioning variable is also passed to the encoder after passing through a dense layer. This separation allows for separate learning channels for the image feature and the condition information. The rest of the model definition and training procedure follows the same pattern as the first example.

Finally, here's an example showing how to embed categorical conditioning variables, which I've used frequently when dealing with text or labeled datasets:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 2
num_classes = 10
condition_dim = 16
input_dim = 784

# Encoder Model
class CatEncoder(layers.Layer):
    def __init__(self, latent_dim, condition_dim, num_classes):
        super(CatEncoder, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.embedding = layers.Embedding(num_classes, condition_dim)
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x, c = inputs
        c_embed = self.embedding(c)
        flat_c = layers.Flatten()(c_embed)
        combined = tf.concat([x, flat_c], axis=1)
        h1 = self.dense1(combined)
        h2 = self.dense2(h1)
        return self.mean(h2), self.log_var(h2)

# Decoder Model
class CatDecoder(layers.Layer):
    def __init__(self, latent_dim, condition_dim, output_dim, num_classes):
        super(CatDecoder, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.embedding = layers.Embedding(num_classes, condition_dim)
        self.out = layers.Dense(output_dim)

    def call(self, inputs):
        z, c = inputs
        c_embed = self.embedding(c)
        flat_c = layers.Flatten()(c_embed)
        combined = tf.concat([z, flat_c], axis=1)
        h1 = self.dense1(combined)
        h2 = self.dense2(h1)
        return self.out(h2)

# CVAE Model with Categorical Conditions
class CatCVAE(keras.Model):
    def __init__(self, latent_dim, condition_dim, input_dim, num_classes):
        super(CatCVAE, self).__init__()
        self.encoder = CatEncoder(latent_dim, condition_dim, num_classes)
        self.sampling = Sampling()
        self.decoder = CatDecoder(latent_dim, condition_dim, input_dim, num_classes)

    def call(self, inputs, training=False):
        x, c = inputs
        mean, log_var = self.encoder((x,c))
        z = self.sampling((mean, log_var))
        reconstructed = self.decoder((z,c))
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        return reconstructed, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            x, c = data
        else:
            x,c = data, None
            raise ValueError("Expected condition variable to be included.")
        with tf.GradientTape() as tape:
             reconstructed, kl_loss = self((x,c), training=True)
             reconstruction_loss = tf.reduce_mean(
                 keras.losses.binary_crossentropy(x, reconstructed)
            ) #For simplicity binary crossentropy assumes x is binary.
             loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "reconstruction_loss": reconstruction_loss,"kl_loss": tf.reduce_mean(kl_loss)}
```
This example uses `Embedding` layers to encode the categorical conditioning variables.  Embedding layers are appropriate for learning representations of discrete inputs. Again, the encoder and decoder inputs are modified to incorporate this embedding.

When experimenting with these CVAEs, the choice of architecture and parameters significantly affects performance. Careful consideration of the dataset and conditioning variables is crucial.

For further resources, I'd recommend exploring academic publications on generative modeling and specifically on VAEs.  Textbooks on deep learning also often include detailed explanations of VAEs and their variations.  Numerous open source examples are available on platforms like GitHub, but careful review is needed before using them, as implementation quality can vary.  Finally, the official documentation of deep learning frameworks (such as TensorFlow and PyTorch) offers detailed API references, which are indispensable for implementation. I've found that studying these resources in conjunction with experimentation has been the most effective way to deepen my understanding.
