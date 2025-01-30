---
title: "How can I implement a bidirectional LSTM?"
date: "2025-01-30"
id: "how-can-i-implement-a-bidirectional-lstm"
---
Implementing a bidirectional Long Short-Term Memory (LSTM) network requires careful attention to how sequential data is processed in both forward and backward directions, thereby capturing contextual information from the entire sequence rather than just preceding elements. I've found this technique particularly powerful when dealing with problems involving natural language understanding and time series analysis.

The core concept of a bidirectional LSTM is to present the same input sequence to two distinct LSTM layers, one processing the sequence in its original order and the other processing it in reverse. This allows the network to learn representations that integrate information from both past and future contexts at each time step. In effect, two hidden states, each representing a distinct interpretation, are then concatenated or combined in some other manner to form a richer, more context-aware representation. This combined representation is then used for subsequent processing such as classification or sequence prediction.

A standard, unidirectional LSTM cell, at any given timestep *t*, only considers input up to *t*. In contrast, a bidirectional approach allows the model to consider input up to *t*, and input after *t*, when making predictions or inferences at time *t*. This bidirectional processing is critical for tasks where future context is as important as past context. For instance, in a sentence like "The cat sat on the mat but...", knowing what word comes after "but" ("...was not comfortable.") is crucial to accurately understand the meaning of "but" and its effect on the sentiment of the entire sentence. A unidirectional LSTM, by its nature, will have no information about words that follow.

The key to implementing this structure lies in correctly instantiating the two LSTM layers and subsequently combining their outputs. Most deep learning frameworks provide convenient abstractions that simplify the construction of bidirectional LSTMs.

Here are three code examples, constructed based on experiences using different deep learning libraries, to illustrate implementations:

**Example 1: PyTorch**

```python
import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size) # *2 because outputs are concatenated.

    def forward(self, x):
        # x.shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # out.shape: (batch_size, sequence_length, hidden_size * 2) - bidirectional output is concatenated

        out = self.fc(out)
        # out.shape: (batch_size, sequence_length, output_size)
        return out

# Example usage:
input_size = 10 # Size of input feature vectors
hidden_size = 20
num_layers = 2
output_size = 5 # For example: 5 classes
batch_size = 32
sequence_length = 40

model = BidirectionalLSTM(input_size, hidden_size, num_layers, output_size)

input_data = torch.randn(batch_size, sequence_length, input_size)
output = model(input_data)

print(output.shape) # Output: torch.Size([32, 40, 5])
```

In this PyTorch example, the `nn.LSTM` module is configured with `bidirectional=True`. This single line instructs the LSTM to operate in both forward and reverse directions. Notably, both the initial hidden state and cell state are initialized with double the layer size to account for the bi-directional aspect. The forward pass includes a simple linear layer ( `nn.Linear`) that transforms the concatenated hidden representations of both directions to the final output. Crucially, the final linear layer maps the concatenated representation (`hidden_size*2`) to the desired `output_size`.

**Example 2: TensorFlow/Keras**

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.models import Model

def create_bidirectional_lstm(input_size, hidden_size, num_layers, output_size):
    inputs = Input(shape=(None, input_size))

    lstm_layers = Bidirectional(LSTM(hidden_size, return_sequences=True))(inputs)

    for i in range(num_layers - 1): #Additional Layers
       lstm_layers = Bidirectional(LSTM(hidden_size, return_sequences=True))(lstm_layers)

    outputs = Dense(output_size)(lstm_layers)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5

model = create_bidirectional_lstm(input_size, hidden_size, num_layers, output_size)

batch_size = 32
sequence_length = 40
input_data = tf.random.normal(shape=(batch_size, sequence_length, input_size))

output = model(input_data)
print(output.shape)  # Output: (32, 40, 5)
```

The TensorFlow/Keras example uses the `Bidirectional` wrapper to transform a single standard `LSTM` layer into a bidirectional layer. Here, I've chosen to demonstrate a multi-layered Bi-LSTM using for loop. The `return_sequences=True` parameter ensures that the outputs from all time steps are returned for use by the subsequent layer, which can be another Bi-LSTM layer or a dense layer for classification. In this implementation, the Bi-LSTM has an input of shape (None, input\_size), where None implies it can accept inputs of variable sequence length.

**Example 3: Custom Implementation (Conceptual)**

This final example presents a conceptual breakdown of how a BiLSTM might work, avoiding specific library implementations:

```python
import numpy as np

def lstm_cell(x_t, h_prev, c_prev, Wx, Wh, b):
    # Forget Gate
    f_t = sigmoid(np.dot(Wx[0], x_t) + np.dot(Wh[0], h_prev) + b[0])
    # Input Gate
    i_t = sigmoid(np.dot(Wx[1], x_t) + np.dot(Wh[1], h_prev) + b[1])
    # Update Cell State
    c_t_candidate = np.tanh(np.dot(Wx[2], x_t) + np.dot(Wh[2], h_prev) + b[2])
    c_t = f_t * c_prev + i_t * c_t_candidate
    # Output Gate
    o_t = sigmoid(np.dot(Wx[3], x_t) + np.dot(Wh[3], h_prev) + b[3])
    # Hidden State
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t

def bidirectional_lstm(X, Wx, Wh, b, initial_h, initial_c):
    # X is a sequence of vectors [x_1, x_2,... x_T]
    T = len(X)
    h_forward = []
    h_backward = []
    h_t = initial_h
    c_t = initial_c

    # Forward Pass
    for t in range(T):
      h_t, c_t = lstm_cell(X[t], h_t, c_t, Wx[0], Wh[0], b[0])
      h_forward.append(h_t)

    # Reverse Pass
    h_t = initial_h
    c_t = initial_c
    for t in reversed(range(T)):
      h_t, c_t = lstm_cell(X[t], h_t, c_t, Wx[1], Wh[1], b[1])
      h_backward.insert(0, h_t) # Inserting from start to make output time-aligned.

    # Concatenate the forward and backward states at each timestep
    h_bidirectional = []
    for i in range(T):
      h_bidirectional.append(np.concatenate((h_forward[i], h_backward[i])))
    return h_bidirectional

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Example
input_size = 10
hidden_size = 20
sequence_length = 40

Wx = np.random.rand(2,4,hidden_size, input_size) # Two for forward, backward. 4 for gates.
Wh = np.random.rand(2,4,hidden_size, hidden_size)
b = np.random.rand(2,4,hidden_size)
initial_h = np.zeros(hidden_size)
initial_c = np.zeros(hidden_size)

input_sequence = [np.random.rand(input_size) for _ in range(sequence_length)] # Random vectors

output = bidirectional_lstm(input_sequence, Wx, Wh, b, initial_h, initial_c)

print(len(output), len(output[0])) # Output: (40,40)
```

This conceptual implementation demonstrates the core calculations of a bidirectional LSTM cell. Note that the simplified `lstm_cell` function showcases the fundamental mathematical operations for a single LSTM cell, while the `bidirectional_lstm` function applies this cell to process a sequence in both directions. This is, of course, very simplified and does not consider mini-batching. In practice, libraries handle all of these low-level operations in an optimized and more abstract way, as shown in the previous examples.

Regarding additional learning resources, I recommend delving into documentation for any deep learning library one intends to use, such as PyTorch or TensorFlow, as they typically provide comprehensive explanations and examples. Additionally, academic texts and papers on recurrent neural networks and sequence modeling are invaluable for understanding the underlying principles and potential applications. Specifically, researching variations of LSTMs, such as gated recurrent units (GRUs), can offer valuable insights. Further exploration of sequence-to-sequence models and attention mechanisms can help in understanding how bidirectional LSTMs fit into more complex neural architectures. Consulting online educational platforms that offer courses on deep learning and recurrent neural networks is also highly beneficial. These resources offer a balance of theoretical understanding and practical implementation guidance.
