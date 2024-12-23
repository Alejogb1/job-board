---
title: "How can LSTMs with multiple embeddings and numerical data effectively model time series?"
date: "2024-12-23"
id: "how-can-lstms-with-multiple-embeddings-and-numerical-data-effectively-model-time-series"
---

,  I've actually spent a fair bit of time building forecasting models, and incorporating multi-modal data into lstms is something I've consistently found quite valuable. It's not just about throwing everything at the network; there's definitely a nuanced approach to get it working effectively.

The core issue here revolves around the heterogeneous nature of your input. We're not dealing with a single, uniform data type. You have text represented as embeddings, alongside numerical time-series data. The challenge lies in ensuring that these disparate inputs contribute meaningfully to the model's understanding of temporal patterns. A naive approach of just concatenating everything and feeding it to an lstm won't cut it; we need a more structured methodology.

First off, let's consider why we use embeddings for text. Words, or more accurately, tokens, don't inherently possess numerical relationships usable by a neural network. Embeddings, usually dense vectors, provide a way to represent these discrete tokens as points in a continuous space. The idea is that words with similar meanings or contexts will have embeddings closer together. This is crucial for the lstm, as it allows the network to learn relationships between textual data in a way that raw text strings or one-hot encodings simply can't. We're effectively translating semantics into numeric representations.

Now, how do we handle combining this with numerical data? The simplest method, the aforementioned concatenation, often underperforms. The issue is that the magnitude of values in embedding spaces and numerical time-series data can vary dramatically, and the model can become biased towards the higher magnitude data. In my experience, scaling becomes absolutely vital. Before anything else, standardize your numerical data to have zero mean and unit variance or use min-max scaling, ensuring that their numerical ranges align better with the embedding values. Also, it's not enough to merely scale the inputs, but also the final concatenated embedding layer before passing through the lstm cell.

Moreover, rather than just direct concatenation, a more effective strategy often involves processing the inputs *separately* through distinct layers and then merging those processed representations. Think about it: we're likely asking the lstm to understand different types of information. An lstm handling numerical data will learn temporal dependencies within that numerical domain, whereas another layer processing embeddings will learn from the semantic relationships contained within the text. The key lies in letting each specialize. After each lstm layer does its work, we then combine the two layers through another layer (this layer should not be an lstm as it is not concerned with temporal relations, think dense or convolutional).

Here’s a schematic idea and code snippet to demonstrate this concept using keras (with tensorflow backend):

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_multimodal_lstm(vocab_size, embedding_dim, numerical_input_shape, lstm_units=64, dense_units=32):
    # Numerical Input branch
    numerical_input = layers.Input(shape=(numerical_input_shape,1))
    numerical_lstm = layers.LSTM(lstm_units, return_sequences=False)(numerical_input)

    # Text input branch
    text_input = layers.Input(shape=(None,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    text_lstm = layers.LSTM(lstm_units, return_sequences=False)(embedding_layer)

    # Merge the layers
    merged_layers = layers.concatenate([numerical_lstm, text_lstm])

    # Additional dense layers and final output
    dense1 = layers.Dense(dense_units, activation='relu')(merged_layers)
    output_layer = layers.Dense(1)(dense1) # assuming a single numeric output as a prediction


    model = keras.Model(inputs=[numerical_input, text_input], outputs=output_layer)
    return model


# Example Usage
vocab_size = 1000 # Assume a vocabulary of 1000 words
embedding_dim = 50
numerical_input_shape = 20 # length of the numerical time series
model = build_multimodal_lstm(vocab_size, embedding_dim, numerical_input_shape)

# generate some random data
numerical_data = np.random.rand(100, numerical_input_shape, 1)
text_data = np.random.randint(0, vocab_size, size=(100, 10)) # each text sequence can be of length 10 (it will be padded automatically)
target = np.random.rand(100,1)

model.compile(optimizer='adam', loss='mse')
model.fit([numerical_data, text_data], target, epochs=5)
```

This code illustrates a setup where numerical data and text data follow different paths up to the merging layer. Note that each branch has its own lstm, and that text input branch goes through an embedding layer first before reaching the lstm. After each branch processes the information, the merged layers are fed to a series of dense layers before the final output. The specific architecture can and should be adjusted for your particular problem, for example, by adding more dense layers or lstm layers to the merge layer.

Another very important technique to consider is the use of attention mechanisms. Attention can help the lstm selectively focus on the most relevant parts of the input sequence. This is often crucial when the text component is quite long or when not every part of the text is equally important for the prediction. Adding an attention layer *before* merging can be very effective to filter through the embeddings effectively.

Let’s consider an example with attention over the embedding branch. This modification involves adding a attention layer to the embedding branch:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # calculate attention weights
        e = tf.nn.tanh(tf.matmul(x, self.W)+self.b)
        a = tf.nn.softmax(e, axis=1)
        # apply attention weights to the input
        output = x * a
        return tf.reduce_sum(output, axis=1)


def build_multimodal_lstm_with_attention(vocab_size, embedding_dim, numerical_input_shape, lstm_units=64, dense_units=32):
    # Numerical Input branch
    numerical_input = layers.Input(shape=(numerical_input_shape,1))
    numerical_lstm = layers.LSTM(lstm_units, return_sequences=False)(numerical_input)

    # Text input branch
    text_input = layers.Input(shape=(None,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    text_lstm = layers.LSTM(lstm_units, return_sequences=True)(embedding_layer) # keep sequences
    attention_layer = Attention()(text_lstm)


    # Merge the layers
    merged_layers = layers.concatenate([numerical_lstm, attention_layer])

    # Additional dense layers and final output
    dense1 = layers.Dense(dense_units, activation='relu')(merged_layers)
    output_layer = layers.Dense(1)(dense1) # assuming a single numeric output as a prediction

    model = keras.Model(inputs=[numerical_input, text_input], outputs=output_layer)
    return model

# Example Usage
vocab_size = 1000 # Assume a vocabulary of 1000 words
embedding_dim = 50
numerical_input_shape = 20 # length of the numerical time series
model = build_multimodal_lstm_with_attention(vocab_size, embedding_dim, numerical_input_shape)

# generate some random data
numerical_data = np.random.rand(100, numerical_input_shape, 1)
text_data = np.random.randint(0, vocab_size, size=(100, 10))
target = np.random.rand(100,1)

model.compile(optimizer='adam', loss='mse')
model.fit([numerical_data, text_data], target, epochs=5)
```

Here, we modify the previous code by including a custom attention layer. Notice that the text lstm layer is set to `return_sequences=True` to provide the attention layer with the sequence, rather than just the last state. Then attention is performed and the attention layer output is reduced to a vector and then concatenated with the numerical lstm. Again, this is a generic example, and the attention should be applied based on the context of your data.

Finally, let's discuss training. When training a multi-modal model like this, consider techniques like early stopping. Because we're combining different types of input, it's often helpful to train using a validation set, stopping the training once the validation loss begins to increase, as this can prevent overfitting and improve the generalisation ability of the model. Also, carefully consider your loss function; Mean Squared Error might be suitable for numerical outputs, but consider other options if you're dealing with classification tasks or other output types.

In summary, effective modelling of time series data with lstms using multiple embeddings and numerical data involves: proper scaling, specialized processing branches for each type of data, attention layers (often with text) to improve focus, careful validation and the selection of appropriate loss functions and optimisation methods. I've found that a structured approach, rather than simply throwing everything at the lstm, always yields the most robust and reliable results. If you're looking for further reading on some of these techniques, I'd recommend delving into papers on recurrent neural networks, particularly those focusing on multi-modal data fusion (many of which are presented in top tier conferences such as neurips, icml or acl) and for broader theory about sequential neural networks look into "Deep Learning" by Goodfellow, Bengio, and Courville. I’ve found these resources incredibly valuable in my own work.
