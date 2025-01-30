---
title: "How to infer unique sequences with TensorFlow RNNs?"
date: "2025-01-30"
id: "how-to-infer-unique-sequences-with-tensorflow-rnns"
---
Identifying unique sequences using Recurrent Neural Networks (RNNs) within TensorFlow presents a multifaceted challenge. While RNNs excel at processing sequential data, directly inferring the uniqueness of a sequence requires a specialized approach beyond basic sequence modeling. I've encountered this exact problem while developing a system to identify novel DNA sequences, where simply predicting the next base pair was insufficient; the system needed to flag previously unseen combinations. The core issue lies in the RNN's inherent focus on predicting the next element in a sequence based on preceding elements, not necessarily on determining the novelty of an entire sequence.

To address this, I utilize an encoder-based approach leveraging a modified loss function rather than focusing on a standard classification task. The process involves training an RNN, typically an LSTM or GRU, as an encoder to transform input sequences into a fixed-length embedding vector. This embedding captures the essential characteristics of the input sequence within a lower-dimensional space. The key to inferring uniqueness lies in how we then use these embeddings. Instead of predicting a subsequent element, I train the encoder to map similar sequences (in terms of content, not necessarily in literal matching) closer together in the embedding space and dissimilar sequences further apart.

This is accomplished using a contrastive loss function. The intuition is that a sequence identical to one in training data should produce an embedding close to that existing representation. Novel sequences, on the other hand, should reside in less populated regions of this embedding space, thus facilitating a measure of uniqueness. We achieve this by constructing pairs of training sequences: one positive pair (two similar sequences) and one negative pair (two dissimilar sequences). The contrastive loss encourages the RNN encoder to minimize the distance between embeddings of positive pairs and maximize the distance between embeddings of negative pairs.

Specifically, the loss function considers the distance between the embeddings (e.g., Euclidean distance) of the two sequences in a pair. If the pair is similar (positive pair), the loss penalizes large distances. If the pair is dissimilar (negative pair), the loss penalizes small distances, up to a margin value. The margin parameter prevents over-packing of dissimilar embeddings by enforcing a minimum distance. This architecture allows us to assess the novelty of a sequence based on its distance to its nearest neighbor in the learned embedding space. Sequences with high minimum distances are more likely to be novel.

Here's a conceptual example using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Sample sequence data (replace with actual data)
def generate_dummy_sequences(num_sequences, sequence_length, vocab_size):
    sequences = np.random.randint(0, vocab_size, size=(num_sequences, sequence_length))
    return sequences

vocab_size = 10 # Example vocabulary size
sequence_length = 20 # Example sequence length
num_sequences = 1000 # Example number of sequences
sequences = generate_dummy_sequences(num_sequences, sequence_length, vocab_size)


# 1. Define the RNN Encoder Model
embedding_dim = 64
lstm_units = 128
encoded_dim = 32

inputs = keras.Input(shape=(sequence_length,))
x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
x = layers.LSTM(lstm_units, return_sequences=False)(x)
encoded = layers.Dense(encoded_dim)(x)
encoder = keras.Model(inputs=inputs, outputs=encoded)

# 2. Define the Contrastive Loss function
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# 3. Create input pairs (simplified example - ideal generation needs care)
def create_pairs(sequences):
    pairs = []
    labels = []
    num = min(100,len(sequences))
    for i in range(num):
        for j in range(num):
          if i ==j:
            pairs.append((sequences[i],sequences[j]))
            labels.append(1)
          else:
             pairs.append((sequences[i],sequences[j]))
             labels.append(0)
    return np.array(pairs),np.array(labels)

pairs, labels = create_pairs(sequences)

# Convert pairs to tensors
sequence_1 = tf.convert_to_tensor(np.array([pair[0] for pair in pairs]))
sequence_2 = tf.convert_to_tensor(np.array([pair[1] for pair in pairs]))

# Calculate distances between encoded vectors
encoded_sequence_1 = encoder(sequence_1)
encoded_sequence_2 = encoder(sequence_2)

distance = tf.reduce_sum(tf.square(encoded_sequence_1 - encoded_sequence_2), axis=1)

labels = labels.astype('float32')
# 4. Train the encoder model (using a dummy optimizer/training)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_values = []
for i in range(100):
  with tf.GradientTape() as tape:
      encoded_sequence_1 = encoder(sequence_1)
      encoded_sequence_2 = encoder(sequence_2)
      distance = tf.reduce_sum(tf.square(encoded_sequence_1 - encoded_sequence_2), axis=1)
      loss_value = contrastive_loss(tf.convert_to_tensor(labels),distance)
  gradients = tape.gradient(loss_value, encoder.trainable_variables)
  optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
  loss_values.append(loss_value)
  if i%10 == 0:
      print(f"loss at step {i}: {loss_value}")

# After training, to infer uniqueness of a new sequence:
new_sequence = generate_dummy_sequences(1, sequence_length, vocab_size)
new_embedding = encoder(tf.convert_to_tensor(new_sequence)).numpy()

# Calculate distance to nearest neighbors
encoded_training_data = encoder(tf.convert_to_tensor(sequences)).numpy()
distances = np.linalg.norm(encoded_training_data - new_embedding, axis=1)
min_distance = np.min(distances)

print(f"Minimum distance of the new sequence in the training space {min_distance}")
```

In this example, I first define an RNN encoder, transforming input sequences to embedding vectors. Then, the `contrastive_loss` function is implemented. Pairs of sequences along with labels (1 for similar, 0 for dissimilar) are made. The model is then trained via backpropagation. After training, the function evaluates the uniqueness of a new sequence by calculating its distance to the existing embeddings in training data, returning the smallest value. Larger distances indicate higher novelty. The `create_pairs` method provides only a simplified example; in reality, generating meaningful similar and dissimilar pairs is crucial for effective training. This would involve techniques such as adding slight noise to known data to create similar sequences and selecting random sequences from training as dissimilar examples.

Another strategy, which I have also successfully implemented, centers around using an autoencoder with a similar approach.

```python
# 1. Define the Autoencoder Model
autoencoder_input = keras.Input(shape=(sequence_length,))
x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(autoencoder_input)
encoded = layers.LSTM(lstm_units, return_sequences=False)(x)
encoded = layers.Dense(encoded_dim, activation = 'relu')(encoded)

x = layers.Dense(lstm_units, activation = 'relu')(encoded)
x = layers.Reshape((1,lstm_units))(x)
decoded = layers.LSTM(lstm_units, return_sequences=True)(x)
decoded = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(decoded)

autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoded)
encoder_only = keras.Model(inputs=autoencoder_input, outputs = encoded)

# 2. Define loss function
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#3 Train the Autoencoder
training_sequences = generate_dummy_sequences(num_sequences, sequence_length, vocab_size)
training_labels = training_sequences
autoencoder.fit(training_sequences,training_labels,epochs=100, verbose = 0)

# After training, to infer uniqueness of a new sequence:
new_sequence = generate_dummy_sequences(1, sequence_length, vocab_size)
new_embedding = encoder_only(tf.convert_to_tensor(new_sequence)).numpy()

# Calculate distance to nearest neighbors
encoded_training_data = encoder_only(tf.convert_to_tensor(sequences)).numpy()
distances = np.linalg.norm(encoded_training_data - new_embedding, axis=1)
min_distance = np.min(distances)
print(f"Minimum distance of the new sequence in the training space {min_distance}")
```

This architecture first compresses the sequence and then decompresses. After training, we use the encoder output as the embedding for downstream distance evaluation. Sequences that are less well reconstructed by the autoencoder are also sequences that reside in underpopulated regions of the encoding space. Therefore, the minimum distance from encoded training samples can again be used to establish novelty.

Finally, an alternative approach involves directly using the reconstruction error itself as an indicator of novelty. In this approach, you first train an RNN autoencoder to minimize reconstruction error on training data. During inference, a new sequence is passed through the trained autoencoder. Higher reconstruction error implies the sequence is less similar to what was seen during training, therefore indicating higher novelty.

```python
# 1. Define the Autoencoder Model
autoencoder_input = keras.Input(shape=(sequence_length,))
x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(autoencoder_input)
encoded = layers.LSTM(lstm_units, return_sequences=False)(x)

x = layers.Reshape((1,lstm_units))(encoded)
decoded = layers.LSTM(lstm_units, return_sequences=True)(x)
decoded = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(decoded)

autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoded)


# 2. Define loss function
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#3 Train the Autoencoder
training_sequences = generate_dummy_sequences(num_sequences, sequence_length, vocab_size)
training_labels = training_sequences
autoencoder.fit(training_sequences,training_labels,epochs=100, verbose = 0)

# After training, to infer uniqueness of a new sequence:
new_sequence = generate_dummy_sequences(1, sequence_length, vocab_size)
reconstructed_sequence = autoencoder.predict(new_sequence)

# Calculate Reconstruction Error
reconstruction_error = np.mean(np.square(new_sequence - np.argmax(reconstructed_sequence,axis = -1)))
print(f"Reconstruction Error for the new sequence is {reconstruction_error}")
```

Here we pass the new sequence through the autoencoder. We then calculate the reconstruction error with the new sequence (converted from the one hot representation to it's one dimensional integer equivalent for comparison). Sequences with higher reconstruction errors are more likely to be unseen during training, thereby demonstrating uniqueness. Note this is a simplified reconstruction error - more robust techniques that consider individual elements can improve accuracy.

For further exploration of these concepts, I recommend researching areas of machine learning such as metric learning, siamese networks, contrastive learning and autoencoders, especially with regards to time-series data. These are foundational topics and readily accessible via textbooks on machine learning and specialized journal papers. Also, studying the TensorFlow and Keras documentation regarding sequence models and custom training loops can be beneficial.
