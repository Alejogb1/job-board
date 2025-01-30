---
title: "How can TensorFlow generate text?"
date: "2025-01-30"
id: "how-can-tensorflow-generate-text"
---
Text generation using TensorFlow leverages the power of recurrent neural networks (RNNs), specifically Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures, to model sequential dependencies within textual data. I've spent considerable time fine-tuning these models for natural language processing tasks, including creative text generation, and the core principle involves training a network to predict the probability of the next character or word, given the preceding sequence.

Fundamentally, a text generation model in TensorFlow is trained on a large corpus of text. The text is first converted into a numerical representation, typically using either character-level or word-level embeddings. Character-level embeddings treat each unique character in the training text as a separate unit, while word-level embeddings map each unique word to a vector representation. Choosing between the two depends on the specific needs of the project: character-level models tend to be more flexible in terms of the vocabulary but require more computation and data, whereas word-level models are generally more efficient and may capture semantic meaning more readily but are limited by their vocabulary.

The training process involves feeding sequences from the text corpus into the RNN, which processes them step-by-step. At each step, the RNN receives the current input (either a character or word embedding) and the hidden state from the previous step. The hidden state represents the model’s memory of what it has seen so far. The output of the RNN at each step is passed through a fully connected layer, followed by a softmax activation function. The softmax layer produces a probability distribution over the entire vocabulary, indicating the likelihood of each possible character or word being the next in the sequence. The training objective is to minimize the difference between the predicted distribution and the actual next character or word using a loss function, such as categorical cross-entropy.

Once trained, the text generation process begins with an initial input sequence (often called a seed). This sequence is fed into the RNN, and the model predicts the next token based on the learned probability distribution. The predicted token is then appended to the input sequence, and the process repeats. This iterative generation continues until a predetermined length is reached or a special end-of-sequence token is generated. Temperature sampling can be employed to control the creativity of the output by adjusting the softmax probabilities: lower temperatures result in more deterministic outputs (selecting the most probable token), whereas higher temperatures lead to more varied outputs by giving less probable tokens a higher chance of being sampled.

Here are three code examples illustrating different approaches to text generation with TensorFlow:

**Example 1: Character-Level LSTM Text Generation**

```python
import tensorflow as tf
import numpy as np

# Dummy training text and vocabulary
text = "the quick brown fox jumps over the lazy dog"
chars = sorted(list(set(text)))
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

# Convert text to numerical sequences
seq_length = 10
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, 1):
  seq_in = text[i:i + seq_length]
  seq_out = text[i + seq_length]
  dataX.append([char_to_idx[char] for char in seq_in])
  dataY.append(char_to_idx[seq_out])
X = np.reshape(dataX, (len(dataX), seq_length, 1))
X = X / float(vocab_size)
Y = tf.keras.utils.to_categorical(dataY)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model (for demonstration purposes, training is short)
model.fit(X, Y, epochs=10, verbose=0)

# Generate text
seed = dataX[0]  # Use first input sequence as the seed
generated_text = [idx_to_char[i] for i in seed]
for _ in range(50):
    x = np.reshape(seed, (1, len(seed), 1))
    x = x / float(vocab_size)
    prediction = model.predict(x, verbose=0)
    predicted_idx = np.argmax(prediction)
    generated_text.append(idx_to_char[predicted_idx])
    seed.append(predicted_idx)
    seed = seed[1:] # Shift the seed
print(''.join(generated_text))
```
This example demonstrates a rudimentary character-level LSTM for text generation.  A vocabulary of unique characters is extracted, each character converted to a numerical index.  Input sequences (`dataX`) and corresponding target characters (`dataY`) are created, then reshaped for input into the LSTM layer. A softmax layer is used for predicting the next character's index. The seed text provides the initial context for text generation, with the output being appended iteratively and used as the next input.  Notably, this is a vastly simplified example and would not produce compelling text, given the extremely limited training data and epochs, but effectively demonstrates the general process.

**Example 2: Word-Level Text Generation Using Word Embeddings**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Dummy text
text = "this is the first sentence and this is the second sentence the sentences are different"
sentences = text.split(' ')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1

# Create input sequences
seq_length = 5
sequences = []
for i in range(1, len(sentences)):
  seq = sentences[i-seq_length:i+1]
  sequences.append(seq)

dataX = []
dataY = []
for seq in sequences:
  dataX.append(seq[:-1]) # input all words except the last one
  dataY.append(seq[-1]) # output the last word
X = tokenizer.texts_to_sequences(dataX)
X = pad_sequences(X, maxlen=seq_length)

Y = np.array([tokenizer.word_index[word] for word in dataY])
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 100, input_length=seq_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model (very short training for example)
model.fit(X, Y, epochs=10, verbose=0)

# Generate text
seed_text = ["this", "is", "the", "first", "sentence"]
generated_text = seed_text
for _ in range(10):
    encoded_seed = tokenizer.texts_to_sequences([seed_text])
    padded_seed = pad_sequences(encoded_seed, maxlen=seq_length)
    prediction = model.predict(padded_seed, verbose=0)
    predicted_idx = np.argmax(prediction)
    predicted_word = tokenizer.index_word.get(predicted_idx, '<unk>') # Handle unknown words
    generated_text.append(predicted_word)
    seed_text = seed_text[1:] + [predicted_word]
print(' '.join(generated_text))

```

This example shifts to a word-level approach, using Keras’ Tokenizer to convert text into word indices.  Padding is applied to ensure that all input sequences have equal length. An Embedding layer maps each word index to a dense vector. The text is generated word-by-word, with the generated word appended to a growing sequence. Again, the limited training dataset means this output will not be high-quality, but showcases the core mechanics involved in word-level generation.

**Example 3: Using GRU and Temperature Sampling**

```python
import tensorflow as tf
import numpy as np

# Dummy training text and vocabulary
text = "the quick brown fox jumps over the lazy dog"
chars = sorted(list(set(text)))
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

# Convert text to numerical sequences
seq_length = 10
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, 1):
  seq_in = text[i:i + seq_length]
  seq_out = text[i + seq_length]
  dataX.append([char_to_idx[char] for char in seq_in])
  dataY.append(char_to_idx[seq_out])
X = np.reshape(dataX, (len(dataX), seq_length, 1))
X = X / float(vocab_size)
Y = tf.keras.utils.to_categorical(dataY)


# Model Definition using GRU
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(128, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model (for demonstration purposes)
model.fit(X, Y, epochs=10, verbose=0)

# Generate text with temperature sampling
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


seed = dataX[0]
generated_text = [idx_to_char[i] for i in seed]
for _ in range(50):
    x = np.reshape(seed, (1, len(seed), 1))
    x = x / float(vocab_size)
    prediction = model.predict(x, verbose=0)[0]
    next_index = sample(prediction, temperature=0.8)
    generated_text.append(idx_to_char[next_index])
    seed.append(next_index)
    seed = seed[1:]

print(''.join(generated_text))
```

This example introduces a GRU layer as an alternative to LSTM and demonstrates temperature sampling during the generation process. The `sample` function applies a temperature factor to the predicted probabilities from the softmax layer. Adjusting the temperature alters the randomness of the sampling and influences the generated text.  This example, while simplistic, indicates how to add control and variation to the generation of textual data.

For those seeking more advanced understanding, I recommend exploring resources that discuss recurrent neural network architectures in detail, including the mathematical formulations of LSTM and GRU units. Additionally, familiarize yourself with the specifics of word embeddings techniques, particularly Word2Vec and GloVe. Further, research advanced sampling strategies beyond temperature, such as top-k and nucleus sampling, which have been demonstrated to enhance the quality and coherence of generated text significantly.  Documentation for TensorFlow's Keras API for RNNs and related preprocessing modules is essential.  Lastly, examining existing pre-trained models and code examples in the field of Natural Language Processing can offer valuable insights.
