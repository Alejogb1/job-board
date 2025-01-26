---
title: "How can I implement a multi-layered RNN for sentiment analysis in PyTorch?"
date: "2025-01-26"
id: "how-can-i-implement-a-multi-layered-rnn-for-sentiment-analysis-in-pytorch"
---

The efficacy of recurrent neural networks (RNNs) for sequence-based tasks, such as sentiment analysis, can be significantly enhanced by employing multiple stacked layers. This approach allows the model to learn hierarchical representations of the input sequence, potentially capturing more intricate relationships between words and phrases. Single-layer RNNs often struggle with long-range dependencies; multi-layered architectures mitigate this by allowing each layer to operate on the transformed output of the previous layer. I’ve found this significantly improves performance in cases where context spans multiple sentences or even paragraphs.

To implement a multi-layered RNN in PyTorch, we’ll leverage the `torch.nn.RNN` or `torch.nn.LSTM` classes, or their more modern counterpart, `torch.nn.GRU`. While `RNN` represents the basic recurrent cell, `LSTM` and `GRU` offer mechanisms to handle vanishing gradients and are generally preferred for more complex sequence modeling. The key modification compared to a single-layer setup is passing `num_layers` to the constructor of these classes. This integer specifies the number of recurrent layers to be stacked on top of each other. PyTorch takes care of the internal mechanisms involved in passing hidden states from one layer to the next.

Here's a breakdown of how a multi-layered RNN works conceptually in this scenario. First, the input sequence, which may be encoded as a sequence of word embeddings, is fed into the first RNN layer. This layer processes the input and produces an output sequence *and* a hidden state. The output sequence from the first layer becomes the input sequence for the second layer. The second layer processes this new sequence and produces its own output and hidden state. This cascading process continues for all the stacked layers. The final output from the last layer is typically used as a basis for sentiment classification by passing it through a linear layer and a softmax or sigmoid activation function.

Let me illustrate with a few code examples:

**Example 1: Basic Multi-Layered RNN with `nn.RNN`**

```python
import torch
import torch.nn as nn

class MultiLayerRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(MultiLayerRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) # shape: (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(embedded) # output shape: (batch_size, seq_len, hidden_dim); hidden shape: (num_layers, batch_size, hidden_dim)
        # We're only using the output of the last time step
        out = output[:, -1, :] # shape: (batch_size, hidden_dim)
        out = self.fc(out) # shape: (batch_size, output_dim)
        return out

# Example usage:
vocab_size = 1000  # Size of vocabulary
embedding_dim = 100
hidden_dim = 256
num_layers = 2
output_dim = 2 # Binary sentiment
batch_size = 32
seq_len = 20

model = MultiLayerRNN(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim)
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(dummy_input)
print(output.shape)
```
In this example, the `MultiLayerRNN` class initializes an embedding layer, a multi-layered `nn.RNN` with the `num_layers` parameter, and a fully connected layer for the final classification. The `forward` method demonstrates how the sequence flows through these layers. The embedding layer transforms the input word indices into dense vector representations, which are then fed into the multi-layered RNN. The output of the last time step of the final RNN layer is then passed to a fully connected layer. This is done because we are performing classification rather than sequence generation so we are not interested in the output of every time step but the overall output. The final output shape reflects the number of classes in our sentiment classification problem (2 here for binary).
    
**Example 2:  Multi-Layered LSTM with masking**

```python
import torch
import torch.nn as nn

class MultiLayerLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, pad_idx):
        super(MultiLayerLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
       embedded = self.embedding(x)
       packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False) # lengths need to be on CPU for this operation
       packed_output, (hidden, cell) = self.lstm(packed_embedded)
       output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
       out = output[torch.arange(output.shape[0]), lengths-1, :]
       out = self.fc(out)
       return out


vocab_size = 1000
embedding_dim = 100
hidden_dim = 256
num_layers = 2
output_dim = 2
batch_size = 32
seq_len = 20
pad_idx = 0

# Example usage:
model = MultiLayerLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, pad_idx)
# Variable length inputs with padding
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
lengths = torch.randint(1,seq_len+1, (batch_size,))
# Zero out sequences where the length is less than the maximal sequence length
for i in range(batch_size):
    if lengths[i] < seq_len:
        dummy_input[i, lengths[i]:] = pad_idx

output = model(dummy_input, lengths)
print(output.shape)
```

This second example enhances the previous one by incorporating sequence padding and masking, which are essential when working with variable length input sequences, which is often the case in practical applications. The `nn.Embedding` layer is initialized with a `padding_idx` to signify what value is used for padding sequences so that its gradients do not affect the model. The `nn.utils.rnn.pack_padded_sequence` function masks the padding so that it does not affect the RNN output and `nn.utils.rnn.pad_packed_sequence` is used to convert back to a tensor after the forward pass. Then, since each sequence has a different length the last output is taken from each sequence at their given sequence lengths. Finally, it passes through the linear layer. The `enforce_sorted=False` allows for unsorted lengths when using the packed padded sequence function. 

**Example 3: Multi-layered GRU with dropout**

```python
import torch
import torch.nn as nn

class MultiLayerGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(MultiLayerGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        out = self.dropout(output[:, -1, :])
        out = self.fc(out)
        return out

# Example usage:
vocab_size = 1000
embedding_dim = 100
hidden_dim = 256
num_layers = 2
output_dim = 2
dropout_rate = 0.5
batch_size = 32
seq_len = 20

model = MultiLayerGRU(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout_rate)
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(dummy_input)
print(output.shape)
```
This final example demonstrates how to use the `GRU` cell, which is a computationally cheaper alternative to `LSTM` and performs comparably well in many tasks. The code also includes dropout in the RNN layers, which is added through the `dropout` parameter during the initialization of the `GRU`. This regularization technique helps prevent overfitting. Furthermore, the dropout layer is applied *after* the last time step of the RNN, not within the recurrent computation itself.

For further exploration, I would recommend consulting several resources. The official PyTorch documentation provides exhaustive information on the `nn.RNN`, `nn.LSTM`, and `nn.GRU` modules. Textbooks and online courses on deep learning, particularly those focusing on recurrent networks, offer valuable insights into the underlying theory and practical applications. Publications on natural language processing often present advanced techniques for optimizing multi-layered RNNs for sentiment analysis. Additionally, tutorials and code repositories on platforms like Github often showcase complete implementations, which can be useful references.
