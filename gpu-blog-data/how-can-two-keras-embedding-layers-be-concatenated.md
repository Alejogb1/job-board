---
title: "How can two Keras embedding layers be concatenated?"
date: "2025-01-30"
id: "how-can-two-keras-embedding-layers-be-concatenated"
---
Embedding layers in Keras are fundamentally lookup tables, converting integer indices into dense vector representations. When concatenating two embedding layers, the critical aspect lies in managing the output shapes and ensuring that the resulting combined representation is meaningful within the context of the downstream model. The process isn't a simple append operation; we must understand the dimension of the embeddings to make it work. I’ve observed that naive concatenation attempts, without careful attention to output tensors' dimensions, often lead to unexpected errors and poorly performing models.

The core issue arises because each embedding layer outputs a tensor of shape `(batch_size, input_length, embedding_dimension)`. Concatenation across the 'embedding\_dimension' axis is what we typically seek, effectively adding new features to the representation rather than creating a sequence of embeddings. Let's consider I have two datasets: one contains text and the other contains user preferences encoded numerically. I’d like to embed both and combine them before passing to a predictive model. The first embedding will map words in the text and the second will map numerical categories for user preferences.

The `keras.layers.concatenate` function provides the functionality to join tensors along a specified axis. In this case, axis -1, representing the feature dimension (i.e. the embedding\_dimension). To utilize this function effectively, the batch_size and input\_length dimensions must match between the outputs of the two embedding layers. If they do not match, you'd have to reshape or transform the tensor. Here, I'll be focusing on the case when they match. Furthermore, each embedding layer has its unique vocabulary size and embedding dimension, and these should not be the same before concatenation to avoid redundancy. This technique allows for richer information to be used in subsequent layers. I will showcase three common situations using code examples to illustrate how you can concatenate.

**Example 1: Two Simple Embeddings with different Vocabularies**

In my first project dealing with a movie recommendation system, I had user IDs and movie titles, both represented by integers. Here, the user IDs would represent one vocabulary and the movie title IDs would represent the other. My intent was to concatenate those embedding vectors and feed them into a dense layer.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define vocabulary sizes and embedding dimensions
vocab_size_user = 1000
embedding_dim_user = 16
vocab_size_movie = 5000
embedding_dim_movie = 32

# Define input layers (batch_size, input_length)
input_user = keras.Input(shape=(1,), name="user_input")
input_movie = keras.Input(shape=(1,), name="movie_input")

# Embedding layers
embedding_user = layers.Embedding(input_dim=vocab_size_user, output_dim=embedding_dim_user, name="user_embedding")(input_user)
embedding_movie = layers.Embedding(input_dim=vocab_size_movie, output_dim=embedding_dim_movie, name="movie_embedding")(input_movie)

# Flatten the embeddings (due to single input length)
flat_user = layers.Flatten()(embedding_user)
flat_movie = layers.Flatten()(embedding_movie)

# Concatenate the embeddings along the last axis
concatenated = layers.concatenate([flat_user, flat_movie], axis=-1, name='concatenated_embeddings')

# Subsequent layers
dense_1 = layers.Dense(64, activation='relu')(concatenated)
output = layers.Dense(1, activation='sigmoid')(dense_1)

# Model Definition
model = keras.Model(inputs=[input_user, input_movie], outputs=output)

# Example of usage
user_data = tf.constant([[10], [20], [30]])
movie_data = tf.constant([[100], [200], [300]])
prediction = model([user_data, movie_data])
print(prediction)
```

In this example, I define two input layers, user\_input and movie\_input. Each input represents a single integer index. Correspondingly, two embedding layers are created. Critically, I have to flatten the embeddings before concatenation because the output from an embedding layer is a 3D tensor and we want to concatenate vectors. The flattened tensors are concatenated along the last axis (`axis=-1`), effectively creating a new vector with a length equal to the sum of the individual embedding dimensions. A dense layer and an output layer then follow. The model output will be the score. Notice that I'm passing two inputs to the model, one for the users and another for movies. This structure is commonly used in recommendation systems or other use cases that combine information from different sources. The print statement will demonstrate the prediction.

**Example 2: Sequential Embeddings for Text Data**

Another instance I faced was with text data. Here, I needed to combine two different aspects of the text using two separate embedding layers. This required padding the sequences to have the same input length. In my particular task, the goal was to predict the sentiment of the sentence.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define vocabulary sizes and embedding dimensions
vocab_size_prefix = 200
embedding_dim_prefix = 8
vocab_size_suffix = 300
embedding_dim_suffix = 12
max_len = 10

# Input layer for sequence of words
input_sequence = keras.Input(shape=(max_len,), name="sequence_input")

# Embedding layers for prefixes and suffixes of each word
embedding_prefix = layers.Embedding(input_dim=vocab_size_prefix, output_dim=embedding_dim_prefix, name="prefix_embedding")(input_sequence)
embedding_suffix = layers.Embedding(input_dim=vocab_size_suffix, output_dim=embedding_dim_suffix, name="suffix_embedding")(input_sequence)

# Concatenate the embeddings along the last axis
concatenated = layers.concatenate([embedding_prefix, embedding_suffix], axis=-1, name='concatenated_embeddings')

# Subsequent layers
lstm = layers.LSTM(32)(concatenated)
output = layers.Dense(1, activation='sigmoid')(lstm)

# Model Definition
model = keras.Model(inputs=input_sequence, outputs=output)

# Example of usage
example_sequence = tf.constant([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0],[4, 5, 6, 7, 8, 9, 0, 0, 0, 0]]) # Padded sequence
prediction = model(example_sequence)
print(prediction)
```

Here, I define a single input layer designed to receive sequences of word indices, with a maximum length of `max_len`. Two separate embedding layers are then used, one for the prefix and another for the suffix of each word. The concatenated output represents an enriched feature for each word.  A LSTM layer follows the concatenated embedding layer and an output dense layer produces the sentiment score. Notice that both embeddings have `max_len` in their output as that's their input length. They have different vocab and embedding dimensions. The example input, `example_sequence`, is a padded tensor with two sample sentences, making the input a batch. The output will be the predicted sentiment score for each sentence.

**Example 3: Concatenating Embeddings with Pre-Computed Weights**

Occasionally, I've needed to utilize pre-computed embeddings, like GloVe or Word2Vec. In this situation, I load the pretrained embeddings and use that as initial weights for the embedding layer.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load pre-trained embedding matrices (replace with your data loading)
embedding_matrix_pre = np.random.rand(250, 10).astype(np.float32)
embedding_matrix_post = np.random.rand(150, 20).astype(np.float32)

# Define vocabulary sizes and embedding dimensions based on pre-trained matrices
vocab_size_pre = embedding_matrix_pre.shape[0]
embedding_dim_pre = embedding_matrix_pre.shape[1]
vocab_size_post = embedding_matrix_post.shape[0]
embedding_dim_post = embedding_matrix_post.shape[1]
max_len = 15

# Input layer
input_sequence = keras.Input(shape=(max_len,), name="sequence_input")

# Embedding layer with pre-computed weights and set trainable to false
embedding_pre = layers.Embedding(input_dim=vocab_size_pre, output_dim=embedding_dim_pre,
                                 weights=[embedding_matrix_pre], trainable=False, name="pre_embedding")(input_sequence)

# Embedding layer with pre-computed weights and set trainable to false
embedding_post = layers.Embedding(input_dim=vocab_size_post, output_dim=embedding_dim_post,
                                  weights=[embedding_matrix_post], trainable=False, name="post_embedding")(input_sequence)

# Concatenate the embeddings along the last axis
concatenated = layers.concatenate([embedding_pre, embedding_post], axis=-1, name='concatenated_embeddings')

# Subsequent layers
gru = layers.GRU(32)(concatenated)
output = layers.Dense(1, activation='sigmoid')(gru)

# Model Definition
model = keras.Model(inputs=input_sequence, outputs=output)

# Example of usage
example_sequence = tf.constant([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0]]) # Padded sequence
prediction = model(example_sequence)
print(prediction)
```

Here, instead of defining vocabulary sizes and allowing the embedding layers to train from scratch, I load pre-computed embedding matrices. It’s important to initialize the layers' weights using these matrices during the construction. I set `trainable=False` to prevent the pre-trained embeddings from being modified during training, effectively freezing them. You may also keep it `True` if you want them to be trainable. After freezing the embeddings, I follow the same pattern of concatenation.  The code uses random matrices to demonstrate the initialization. In practice, they should be replaced with pre-trained embeddings. The rest of the architecture follows the typical pattern.

**Resource Recommendations**

To further explore the intricacies of Keras embedding layers and tensor concatenation, I would suggest several resources. Consult the Keras documentation directly for the latest details on the `Embedding` and `concatenate` layers. Several online courses provide tutorials and demonstrations on embedding layers, particularly in the context of natural language processing. The Keras official website provides a clear explanation of layers and functional API usage. Furthermore, a good book on deep learning with Python or Tensorflow will provide more detailed explanations and further examples of such tasks. Exploring open-source projects on Github which use embedding and concatenation layers could provide a broader understanding. These resources, while not specific links, have consistently helped me better understand and apply these concepts in my projects.
