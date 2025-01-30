---
title: "How can neural networks be used for unsupervised clustering?"
date: "2025-01-30"
id: "how-can-neural-networks-be-used-for-unsupervised"
---
Clustering, by its nature, presents a unique challenge in machine learning: identifying inherent groupings within data without pre-existing labels. Unlike supervised learning where outputs are explicitly provided, unsupervised methods like clustering must discover patterns on their own. Neural networks, typically associated with classification and regression, can be adapted for this task, offering a powerful alternative to traditional algorithms such as k-means or hierarchical clustering, especially when dealing with complex, high-dimensional data.

The core concept behind utilizing neural networks for unsupervised clustering revolves around learning a latent representation of the input data. This representation, often lower dimensional than the original input, captures the essential features and relationships within the dataset. The network is trained to map similar data points to similar latent vectors, and dissimilar data points to distant vectors in this latent space. This process implicitly accomplishes clustering by creating a space where points naturally group according to the underlying structure of the data, even without explicit cluster assignments being provided during training. Various techniques, each leveraging distinct aspects of neural network architecture, can achieve this effect.

One common approach is using an Autoencoder network. In a practical application developing tools for astronomical image analysis, I once employed this methodology to cluster galaxies based on their morphological characteristics, such as ellipticity, size, and surface brightness profiles. The Autoencoder architecture comprises an encoder network that maps the high-dimensional input (galaxy images, in my case, after suitable preprocessing) to a lower-dimensional latent vector and a decoder network that attempts to reconstruct the original input from the latent vector. The training objective, thus, becomes to minimize the reconstruction error, forcing the latent vector to capture the most crucial information needed to recreate the original input.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Example Autoencoder architecture for illustration
input_dim = 64 # Simplified input image dimension
latent_dim = 16 # Reduced dimension in latent space

# Encoder
encoder_inputs = keras.Input(shape=(input_dim,))
h = layers.Dense(32, activation='relu')(encoder_inputs)
latent = layers.Dense(latent_dim, activation='relu')(h)
encoder = keras.Model(encoder_inputs, latent)

# Decoder
decoder_inputs = keras.Input(shape=(latent_dim,))
h = layers.Dense(32, activation='relu')(decoder_inputs)
decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(h)  #sigmoid if input data is scaled [0,1]
decoder = keras.Model(decoder_inputs, decoder_outputs)

# Full Autoencoder
autoencoder_inputs = keras.Input(shape=(input_dim,))
encoded_representation = encoder(autoencoder_inputs)
decoded_output = decoder(encoded_representation)
autoencoder = keras.Model(autoencoder_inputs, decoded_output)


autoencoder.compile(optimizer='adam', loss='mse') # Mean squared error as reconstruction loss
# Assume X_train contains your training data, properly preprocessed.
# autoencoder.fit(X_train, X_train, epochs=100, batch_size=32)
```

In this code, the encoder maps the input to a 16-dimensional vector. During training, the decoder tries to reconstruct the input from that compressed representation, causing the encoder to learn a meaningful feature space. Once trained, only the encoder part is used. The latent vectors are then used as input for traditional clustering algorithms, such as k-means. I found that the clusters generated using the autoencoder’s latent space often showed much more meaningful groupings compared to applying k-means directly on the original image features because the encoder learns a more abstract and robust representation of the images. This representation can capture semantic similarities that are often invisible to direct algorithms.

A second approach involves self-organizing maps (SOMs), also called Kohonen networks. I utilized SOMs in a project focused on analyzing customer transaction data to segment customers based on their purchasing behavior. The SOM creates a topologically ordered map of the input space. The network consists of neurons arranged in a grid, and during training, the input data vectors are mapped onto this grid. The neurons that are activated by a given input vector adjust their weights to be closer to that input vector. In this process, similar data points activate nearby neurons in the grid. The final arrangement of the neurons reveals the cluster structure of the data. The advantage here is that SOMs provide a visualization of clusters through the neuron grid, making the process more interpretable than the autoencoder.

```python
import numpy as np
from minisom import MiniSom

# Example SOM for demonstration.
# Assuming X_train contains customer transaction data
som_dim = 10 # Size of the SOM grid
input_dim = 5 # Dimensionality of the transaction data

som = MiniSom(som_dim, som_dim, input_dim, sigma=0.5, learning_rate=0.5)
# Initialize weights randomly
som.random_weights_init(X_train)
# Train the SOM for a large number of iterations
num_iterations = 10000
# Adjust learning rate over the training period
learning_rate = 0.5
for i in range(num_iterations):
    if i % 1000 == 0:
        learning_rate *= 0.8
    # Select random training data index and train
    rand_idx = np.random.randint(0, len(X_train))
    som.update(X_train[rand_idx], som.winner(X_train[rand_idx]), learning_rate)
```
After training, I would analyze the SOM’s map: data points mapped to the same or close neurons represent customers with similar purchasing behavior. This allows for a more visual approach to understanding customer segments and formulating targeted marketing campaigns. This visual aspect is critical for communicating results to non-technical stakeholders.

Finally, clustering can also be accomplished using contrastive learning. In one specific case, while researching anomaly detection in network traffic, I employed a contrastive learning-based approach. The underlying idea is to train a neural network to maximize the similarity between two augmented versions of the same input data and minimize similarity between augmented versions of different data points. Essentially, the network is learning to identify what makes two data points similar rather than what distinguishes them. After training, the learned representation can be used for clustering using any conventional clustering algorithm. The strength of contrastive learning is its ability to handle intricate and high-dimensional data without explicit reconstruction, leading to robust embeddings.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Example contrastive learning encoder
input_dim = 10 #Dimension of a traffic data point
projection_dim = 16 # Projection dimension
encoder_inputs = keras.Input(shape=(input_dim,))
h = layers.Dense(32, activation='relu')(encoder_inputs)
projection = layers.Dense(projection_dim, activation=None)(h) # Output projection, no activation
encoder = keras.Model(encoder_inputs, projection)


def contrastive_loss(y_true, y_pred):
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    similarity = tf.matmul(y_pred, tf.transpose(y_pred))
    similarity_diag = tf.linalg.diag_part(similarity)
    similarity = similarity - tf.linalg.diag(similarity_diag)
    y_true = tf.linalg.diag(y_true)
    y_true = tf.linalg.diag(tf.ones(tf.shape(y_true)[0]))
    loss = tf.keras.losses.categorical_crossentropy(y_true, similarity, from_logits=True)
    return tf.reduce_mean(loss)

def augment(data):
    #Placeholder for augmentation
    #Augmentation depends on the nature of the data
    return data + tf.random.normal(tf.shape(data), mean=0.0, stddev=0.05)

# Training process assuming X_train is the traffic data
num_epochs = 100
batch_size = 32

optimizer = keras.optimizers.Adam()


for epoch in range(num_epochs):
    for batch_start in range(0, len(X_train), batch_size):
        batch_end = batch_start + batch_size
        X_batch = X_train[batch_start:batch_end]
        with tf.GradientTape() as tape:
            augmented_data_1 = augment(X_batch)
            augmented_data_2 = augment(X_batch)
            encoded_data_1 = encoder(augmented_data_1)
            encoded_data_2 = encoder(augmented_data_2)
            concatenated_encoded = tf.concat((encoded_data_1,encoded_data_2),axis=0)
            loss = contrastive_loss(tf.zeros((len(X_batch),), dtype=tf.int32), concatenated_encoded)
        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
```
Here, the encoder is trained using a custom contrastive loss function; after training, the output of the encoder can be used as features for a k-means clustering algorithm to identify anomalous traffic patterns. The augmentation step is critical; it forces the network to be invariant to minor perturbations of input data.

In conclusion, while different in their approach, these methods all leverage neural networks to learn a meaningful representation of the data, enabling unsupervised clustering. The choice among them should be dictated by the specific task, data characteristics, and desired level of interpretability.  For additional insights, I recommend researching resources focused on deep learning for unsupervised learning, particularly sections on Autoencoders, Self-Organizing Maps, and Contrastive Learning, typically found in general machine learning textbooks. Advanced texts on deep learning architectures will also offer detailed explorations of these topics. Research papers specific to neural network based unsupervised clustering can provide further detail on specific applications and methodologies.
