---
title: "How can I use legacy_seq2seq in TensorFlow 2?"
date: "2025-01-30"
id: "how-can-i-use-legacyseq2seq-in-tensorflow-2"
---
TensorFlow 2’s focus on eager execution and the Keras API significantly alters how sequence-to-sequence (seq2seq) models, typically handled by `tf.contrib.legacy_seq2seq` in TensorFlow 1, are implemented. The removal of `tf.contrib` and the shift away from static graph construction necessitate a reimagining of these architectures using either the Keras API or TensorFlow's lower-level operations compatible with eager execution. I’ve personally encountered this transition while migrating several older NLP models, and it's crucial to understand that direct usage of `legacy_seq2seq` is simply no longer an option in TensorFlow 2. Instead, we must construct equivalent functionality using the available building blocks.

The core challenge lies in replacing the abstractions offered by `legacy_seq2seq`, specifically the `basic_rnn_seq2seq`, `embedding_rnn_seq2seq`, and `attention_seq2seq` functions. These typically handled the intricate looping and unrolling of recurrent neural networks, along with embedding lookups, all within a static graph. In TensorFlow 2, this process is largely handled explicitly within custom layers or Keras models, leveraging the dynamic nature of eager execution. Below, I will demonstrate how to recreate common seq2seq patterns using the Keras API, specifically the `tf.keras.layers.RNN`, `tf.keras.layers.Embedding`, and the attention mechanisms available through `tf.keras.layers.Attention` and related classes.

The fundamental component of any seq2seq model is the RNN layer. Here, we use `tf.keras.layers.GRU`, a gated recurrent unit, though you could readily substitute `LSTM` or a custom RNN cell. A major shift from `legacy_seq2seq` is that we now have to handle the decoding process with explicit looping within the model, instead of relying on `legacy_seq2seq`'s hidden abstractions. This shift is not a setback; it provides greater control and flexibility over the model architecture.

**Example 1: Simple Encoder-Decoder (without Attention)**

Here’s a simplified encoder-decoder without attention to show the basic structure:

```python
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=False, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return state

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
       x = self.embedding(x)
       output, state = self.gru(x, initial_state=hidden)
       output = self.fc(output)
       return output, state

def initialize_hidden_state(units):
  return tf.zeros((1,units))

# Example Usage
vocab_size_enc = 1000
vocab_size_dec = 1000
embedding_dim = 256
units = 1024

encoder = Encoder(vocab_size_enc, embedding_dim, units)
decoder = Decoder(vocab_size_dec, embedding_dim, units)

input_data = tf.random.uniform((32, 10), minval=0, maxval=vocab_size_enc, dtype=tf.int32)
target_data = tf.random.uniform((32, 12), minval=0, maxval=vocab_size_dec, dtype=tf.int32)

hidden = initialize_hidden_state(units)
encoder_state = encoder(input_data, hidden)

decoder_input = tf.random.uniform((32, 1), minval=0, maxval=vocab_size_dec, dtype=tf.int32)  # Initial decoder input (e.g. START token)

for t in range(target_data.shape[1]):
    decoder_output, decoder_state = decoder(decoder_input, encoder_state)
    decoder_input = tf.argmax(decoder_output, axis=-1) # Use argmax for next input in this example, typically the teacher forcing technique will be used in training
```

This example showcases a simple seq2seq structure. The `Encoder` takes input sequences, embeds them, and uses a GRU to obtain a context vector, represented by the final hidden state. The `Decoder` takes the context vector, along with previously generated tokens and performs recurrent decoding step-by-step to generate the target sequence. Crucially, the decoder logic is explicit, and we manually perform the iteration over the target sequence length using for loop, unlike `legacy_seq2seq`, where this is handled implicitly. Note the initialization of the hidden state with zeros. I’ve found that proper initialization is often overlooked, but it’s essential for correct model functioning.

**Example 2:  Seq2Seq with Attention**

Adding attention requires more explicit computation, but provides a very robust result. Here we demonstrate a Bahdanau-style attention mechanism:

```python
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
      super(BahdanauAttention, self).__init__()
      self.W1 = tf.keras.layers.Dense(units)
      self.W2 = tf.keras.layers.Dense(units)
      self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) # concatenate context to input
        output, state = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.fc(output)
        return output, state, attention_weights

def initialize_hidden_state(units):
  return tf.zeros((1,units))


# Example Usage
vocab_size_enc = 1000
vocab_size_dec = 1000
embedding_dim = 256
units = 1024

encoder = Encoder(vocab_size_enc, embedding_dim, units)
decoder = Decoder(vocab_size_dec, embedding_dim, units)

input_data = tf.random.uniform((32, 10), minval=0, maxval=vocab_size_enc, dtype=tf.int32)
target_data = tf.random.uniform((32, 12), minval=0, maxval=vocab_size_dec, dtype=tf.int32)

hidden = initialize_hidden_state(units)
enc_output, encoder_state = encoder(input_data, hidden)

decoder_input = tf.random.uniform((32, 1), minval=0, maxval=vocab_size_dec, dtype=tf.int32)

for t in range(target_data.shape[1]):
    decoder_output, decoder_state, attention_weights = decoder(decoder_input, encoder_state, enc_output)
    decoder_input = tf.argmax(decoder_output, axis=-1)
```

In this augmented example, the `BahdanauAttention` class implements the attention mechanism. The encoder now returns *all* hidden states of the GRU and the final state. The decoder uses the `BahdanauAttention` class to calculate a context vector for each decoding step, which is then concatenated to the input for the decoder GRU. The key change here is that the decoder takes into account all encoder output states to compute its current output, which adds substantial sophistication to the model. The usage of `tf.expand_dims` and `tf.reduce_sum` might be slightly verbose, but that's the explicit tensor manipulation we need to work with in TensorFlow 2.

**Example 3: Using Keras Attention Layer**

For added convenience, we can also use the attention layers provided within Keras:

```python
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.attention = tf.keras.layers.Attention()
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        query_with_time_axis = tf.expand_dims(hidden, 1)
        context_vector = self.attention([query_with_time_axis, enc_output]) # key, value from encoder
        x = tf.concat([context_vector, x], axis=-1)
        output, state = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.fc(output)
        return output, state

def initialize_hidden_state(units):
  return tf.zeros((1,units))


# Example Usage
vocab_size_enc = 1000
vocab_size_dec = 1000
embedding_dim = 256
units = 1024

encoder = Encoder(vocab_size_enc, embedding_dim, units)
decoder = Decoder(vocab_size_dec, embedding_dim, units)

input_data = tf.random.uniform((32, 10), minval=0, maxval=vocab_size_enc, dtype=tf.int32)
target_data = tf.random.uniform((32, 12), minval=0, maxval=vocab_size_dec, dtype=tf.int32)

hidden = initialize_hidden_state(units)
enc_output, encoder_state = encoder(input_data, hidden)

decoder_input = tf.random.uniform((32, 1), minval=0, maxval=vocab_size_dec, dtype=tf.int32)

for t in range(target_data.shape[1]):
    decoder_output, decoder_state = decoder(decoder_input, encoder_state, enc_output)
    decoder_input = tf.argmax(decoder_output, axis=-1)
```

This variant utilizes the built-in `tf.keras.layers.Attention` layer. It's simpler to use, but as the documentation will tell you, the default attention implementation is scaled dot-product attention. I have found this difference leads to slight variations in behavior compared to a handcrafted attention, as shown in example 2. You'll notice there is no explicit attention weights returned in this example, but that is the nature of Keras API: things are abstracted and easier to implement.

To further solidify your understanding of seq2seq models in TensorFlow 2, I recommend focusing on the official TensorFlow documentation for Keras, especially the sections on recurrent layers, embeddings, and attention. The "Text classification with an RNN" tutorial can provide a solid foundation. In addition, many educational resources such as university courses in Natural Language Processing provide valuable explanations of these concepts. Experimenting with various model configurations and datasets is also key to mastering the nuances of seq2seq implementation. Furthermore, review the code associated with research papers on seq2seq models, which provides real-world examples of using TensorFlow 2 for complex seq2seq architectures. You’ll discover how research-level models are built directly upon the APIs provided, similar to what I have outlined. By diving into these resources, you'll effectively transition from relying on `legacy_seq2seq` to wielding the power and flexibility of modern TensorFlow.
