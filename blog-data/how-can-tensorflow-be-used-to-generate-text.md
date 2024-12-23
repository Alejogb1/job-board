---
title: "How can TensorFlow be used to generate text?"
date: "2024-12-23"
id: "how-can-tensorflow-be-used-to-generate-text"
---

, let's talk about text generation with TensorFlow. I've seen quite a few iterations of this over the years, from rudimentary character-level models to the more sophisticated transformer-based architectures we use today. It’s a fascinating area because it bridges the gap between pure number crunching and the complex world of language. What we're fundamentally doing is training a model to predict the next word (or character, depending on the granularity) in a sequence, given the preceding sequence. This can be applied to everything from generating poetry to coding snippets or even synthetic dialogues.

My own journey with this began, if memory serves, with a particularly challenging project where we were tasked with generating training data for a speech recognition system. Hand-labeling everything was out of the question, so we explored generative models. Initially, we used simple recurrent neural networks (RNNs), specifically LSTMs (Long Short-Term Memory networks), as they are well-suited for sequential data. While they are now considered a more established choice compared to the current state-of-the-art, they still provide a clear picture of the fundamental ideas involved in the text generation process.

Essentially, we feed a sequence of text to the model. During training, the model learns the statistical relationships between the words (or characters). After training, we feed an initial seed text, or a start sequence, to the model and instruct it to generate the next word based on what it learned during training. This predicted word is then appended to the sequence, and the process repeats. Here’s the core principle: the model is iteratively predicting the most probable next word based on the previously generated text.

Let’s break this down into a practical illustration using code snippets. First, I will demonstrate a character-level generation example using an LSTM. This illustrates the core concept very clearly. Note this will be a simplified version for the sake of clarity and we use keras API since it handles the backend Tensorflow intricacies.

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np


# sample text (replace with any corpus)
text = "the quick brown fox jumps over the lazy dog."
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

seq_length = 10
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

X = np.zeros((len(sentences), seq_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=100, verbose=0)


def generate_text(model, seed, length):
    generated = ''
    sentence = seed
    generated += seed
    for i in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


print(generate_text(model, "the quick ", 50))

```

This code establishes a basic character-level LSTM model. The model learns to predict the next character given the previous characters, demonstrating text generation principles using characters. The output will not be fluent or coherent because of the basic model, data size, and epochs, but this illustrates the core concept of learning sequential patterns.

Moving towards more sophisticated models, let’s look at a word-level approach using embeddings to better capture word semantics, which we encountered when building systems that required more structured outputs. Using embeddings allows the model to learn relationships between words in a continuous vector space, unlike one-hot encoding. This can lead to the model capturing semantic meaning better than a character based model.

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

text = "the quick brown fox jumps over the lazy dog. the dog is lazy. the fox is quick. the brown dog is lazy".lower()
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in [text]:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X, labels = input_sequences[:,:-1],input_sequences[:,-1]
y = to_categorical(labels, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=100, verbose=0)

def generate_text_word_level(seed_text, model, tokenizer, max_sequence_len, n_words):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

print(generate_text_word_level("the quick brown", model, tokenizer, max_sequence_len, 10))
```

This second example demonstrates a basic word-level language model, using the same principle as the character-level model, but with sequences of words, instead of characters. This approach can lead to better text structure and can model grammatical rules more effectively. Again, the output would not be perfect, given limited training data, but would be more reasonable than the character-level model due to the usage of word embeddings and a better defined model granularity.

While LSTMs have been foundational, they have a limitation in their ability to handle long-range dependencies effectively. This led to the rise of transformer models, which use a mechanism called "attention" to process sequences. This attention mechanism allows the model to focus on relevant words in the sequence, regardless of their position. This results in significantly improved text generation quality, although requiring more computational resources. I won't provide full transformer code within this limited space, as it is significantly more complex, but I will touch on the key idea using the same concept. The main idea, whether we use an LSTM, a GRU or a Transformer, remains: train a model to predict the next word in a sequence, based on what it has learnt during training on a given dataset.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Layer, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ScaledDotProductAttention(Layer):
    def __init__(self, d_k, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.d_k = d_k

    def call(self, query, key, value, mask=None):
        attention_weights = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))
        attention_weights = tf.divide(attention_weights, tf.math.sqrt(tf.cast(self.d_k, tf.float32)))
        if mask is not None:
            attention_weights = tf.where(mask, -1e9, attention_weights)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_k": self.d_k
        })
        return config

text = "the quick brown fox jumps over the lazy dog. the dog is lazy. the fox is quick. the brown dog is lazy".lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in [text]:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X, labels = input_sequences[:,:-1],input_sequences[:,-1]
y = to_categorical(labels, num_classes=total_words)

d_model = 64
d_k = d_model // 8

input_layer = Input(shape=(max_sequence_len - 1,))
embed_layer = Embedding(total_words, d_model)(input_layer)

query = Dense(d_k)(embed_layer)
key = Dense(d_k)(embed_layer)
value = Dense(d_model)(embed_layer)

attention_output = ScaledDotProductAttention(d_k)([query, key, value])
flattened_output = tf.keras.layers.Flatten()(attention_output)
output_layer = Dense(total_words, activation='softmax')(flattened_output)

transformer_model = Model(inputs=input_layer, outputs=output_layer)
transformer_model.compile(loss='categorical_crossentropy', optimizer='adam')
transformer_model.fit(X, y, epochs=100, verbose=0)

def generate_text_transformer(seed_text, model, tokenizer, max_sequence_len, n_words):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


print(generate_text_transformer("the quick brown", transformer_model, tokenizer, max_sequence_len, 10))
```

This final code snippet introduces a simplified transformer-like model using self-attention, demonstrating the use of this mechanism for capturing dependencies within the input sequence, even though it remains basic to enable its presentation. While a full transformer model involves additional complexities such as positional encoding and multi-head attention, this simplified version illustrates how attention allows the model to focus on different parts of the input sequence when generating text, which is a core characteristic of these architectures and a fundamental step in the evolution of language models.

For deeper understanding, I strongly recommend starting with "Speech and Language Processing" by Daniel Jurafsky and James H. Martin for the broader NLP context. Additionally, "Attention is All You Need", the original transformer paper by Vaswani et al. (2017), provides an explanation of transformers themselves. And for more technical details on building models, the TensorFlow documentation is crucial, especially the sections on text processing, recurrent layers and attention mechanisms. Keep in mind that text generation is an active area of research, and there are ongoing advancements and new approaches to investigate regularly.
