---
title: "Does Keras learn embeddings for out-of-vocabulary words?"
date: "2025-01-30"
id: "does-keras-learn-embeddings-for-out-of-vocabulary-words"
---
The Keras Embedding layer, by default, does not inherently learn or assign vector representations for out-of-vocabulary (OOV) words during the training process. This is a crucial detail that impacts the performance of natural language processing models, and one I have had to address in multiple projects ranging from sentiment analysis of product reviews to medical text classification. When an input text contains a token not seen in the training data's vocabulary, the Keras Embedding layer, lacking any other explicit instruction, generally treats it as a zero-vector. This behavior, while straightforward, often leads to significant information loss.

The Keras Embedding layer operates by maintaining a lookup table; each unique word in the vocabulary is associated with a unique dense vector of a specific dimensionality, initialized randomly or with pre-trained weights, and learned through backpropagation. During the training phase, the embedding vectors corresponding to input words are looked up, fed into the subsequent layers, and the gradients flowing back adjust these vector representations to minimize the model's loss function. These adjustments are solely within the confines of the known vocabulary. Therefore, if a word does not have a corresponding entry in this lookup table, it cannot benefit from this learning process.

In practical terms, this means that during inference, encountering an OOV word results in the model essentially ignoring or treating it as meaningless. The degree of negative impact is dependent on the frequency of OOV words in the data, but it is almost always a detrimental factor.

To address this, we can employ a number of strategies in conjunction with the embedding layer. One fundamental step involves preprocessing the text and mapping all OOV words to a specific, reserved vocabulary index. This allows us to have a learnable embedding vector even for tokens not explicitly in the core vocabulary. Often, we define a special token (e.g., “<UNK>” or “[UNK]”) that is used for all such instances. The embedding vector for this placeholder token is, consequently, learned alongside embeddings for in-vocabulary tokens. This addresses, in part, the problem, as the model has at least a trainable representation rather than complete ignorance of the input, although it still merges the potential meaning of various unknown words to a single embedding.

Furthermore, considering subword units, via techniques like Byte Pair Encoding (BPE) or WordPiece, can mitigate the OOV problem significantly. By breaking down words into smaller, often reusable, units, we increase the likelihood that novel words during inference are built from units present in the vocabulary. These methods, however, are more complex to implement and are beyond the core workings of the Keras Embedding Layer itself but integrate well with it as preprocessing steps.

Here are three code examples that illustrate the default behavior of the embedding layer and a mitigation strategy:

**Example 1: The Default Behavior with OOV Words**

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
import numpy as np

# Define vocabulary and an OOV word
vocabulary = ["the", "quick", "brown", "fox"]
word_to_index = {word: index for index, word in enumerate(vocabulary)}
oov_word = "jumps"
oov_index = len(vocabulary) # Index for oov token if we wanted one

# Create input with both in-vocabulary and OOV words
input_sequence = [word_to_index["the"], word_to_index["quick"], oov_index] #Include OOV index at the end.
input_sequence_np = np.array([input_sequence])

# Define the embedding layer
embedding_dim = 4
embedding_layer = Embedding(input_dim=len(vocabulary) + 1, output_dim=embedding_dim) # Add 1 for potential OOV
model = Sequential([embedding_layer])


# Perform embedding lookup
embeddings_result = model.predict(input_sequence_np)


# Print the results
print("Embedding for 'the':", embeddings_result[0, 0])
print("Embedding for 'quick':", embeddings_result[0, 1])
print("Embedding for 'OOV' :", embeddings_result[0, 2])
```

*Commentary:* This example showcases that, even when the OOV word’s index is present in the input, the initial embedding for the OOV word is randomly initialized and stays that way, because it is never part of the original vocabulary lookup table. It does not contribute to the training, and even a new model instance will not provide a different representation for this, or any other word outside the initial vocabulary list, on lookup. The vector for ‘the’ and ‘quick’ will change during training while the last token in the example’s list will remain static.

**Example 2: Using a Placeholder Token for OOV Words**

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
import numpy as np

# Define vocabulary, and a placeholder token for OOV words
vocabulary = ["the", "quick", "brown", "fox", "<UNK>"]
word_to_index = {word: index for index, word in enumerate(vocabulary)}
oov_word = "jumps" # the actual word itself is never used during training or look up

# Create input with in-vocabulary words and map OOV words to the placeholder token
input_sequence = [word_to_index["the"], word_to_index["quick"], word_to_index["<UNK>"]]
input_sequence_np = np.array([input_sequence])

# Define the embedding layer
embedding_dim = 4
embedding_layer = Embedding(input_dim=len(vocabulary), output_dim=embedding_dim)
model = Sequential([embedding_layer])

# Perform embedding lookup and print the results.
embeddings_result = model.predict(input_sequence_np)

print("Embedding for 'the':", embeddings_result[0, 0])
print("Embedding for 'quick':", embeddings_result[0, 1])
print("Embedding for '<UNK>':", embeddings_result[0, 2])
```

*Commentary:* Here, we have created a new placeholder token `<UNK>` within our vocabulary. The index for the OOV word is never directly used in the training process. Instead, any word not found during the encoding process is mapped to the `<UNK>` placeholder. The model during its training phase learns a vector representation for this token which will be the representation used for all out-of-vocabulary words encountered. This ensures that at the very least, each OOV word is mapped to the same, learned embedding vector, as opposed to a zero or random vector during lookup. However, all OOV terms will have the same vector representation.

**Example 3: Training with Placeholder and Observing Vector Adjustment**

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np

# Define vocabulary, placeholder token, and training data
vocabulary = ["the", "quick", "brown", "fox", "<UNK>"]
word_to_index = {word: index for index, word in enumerate(vocabulary)}
training_data = [
  [word_to_index["the"], word_to_index["quick"], word_to_index["brown"], 1],
  [word_to_index["the"], word_to_index["<UNK>"], word_to_index["brown"], 0],
  [word_to_index["fox"], word_to_index["quick"], word_to_index["<UNK>"], 1],
  ]
training_data_np = np.array([data[:-1] for data in training_data])
labels_np = np.array([data[-1] for data in training_data])

# Define the embedding layer and a simple classification model
embedding_dim = 4
embedding_layer = Embedding(input_dim=len(vocabulary), output_dim=embedding_dim)
model = Sequential([
    embedding_layer,
    Flatten(),
    Dense(units=1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(training_data_np, labels_np, epochs=100, verbose=0) # Train silently


# Perform embedding lookup
embeddings_result = model.predict(training_data_np)

# Print the results
print("Embedding after training for 'the':", model.layers[0].get_weights()[0][word_to_index["the"]])
print("Embedding after training for 'quick':", model.layers[0].get_weights()[0][word_to_index["quick"]])
print("Embedding after training for '<UNK>':", model.layers[0].get_weights()[0][word_to_index["<UNK>"]])
```

*Commentary:* This example extends the previous one by including model training. Here, we observe the changes in the embedding vector for the `<UNK>` token after the model undergoes the training process. Importantly, we can now see that the embedding of the unknown token is no longer static. The embeddings of in-vocabulary words are also impacted by the training process. However, the key learning here is that our `<UNK>` placeholder, while not representing any specific in-vocabulary word, learns a generalized representation useful for the model. This is a basic illustration of how the embedding vectors are modified to minimize a loss function during the training phase. This shows that when you use a placeholder token during training, you provide an opportunity for OOV words to have at least a representative representation within the model, whereas without the placeholder, the model would simply ignore the presence of unknown tokens.

For further reading on this topic, consider exploring the documentation on Keras Embedding layer and the related text preprocessing utilities in the Keras/TensorFlow documentation. Additionally, resources on Byte Pair Encoding and subword tokenization strategies provide detailed explanations and implementations of these techniques. Works discussing advanced word embedding models, beyond basic embedding lookups, can enhance comprehension and provide alternative solutions to the OOV problem, such as contextual word embeddings. While these do not directly modify the Embedding layer's learning, understanding them gives a complete view of handling such issues in NLP. The key understanding is that the fundamental Keras Embedding layer works purely based on indices assigned during training, and if the index is not present, no learning will be possible.
