---
title: "How can I implement a multi-layered RNN for sentiment analysis in PyTorch?"
date: "2024-12-23"
id: "how-can-i-implement-a-multi-layered-rnn-for-sentiment-analysis-in-pytorch"
---

Alright, let’s tackle this. I recall a project from my earlier days where we were attempting to discern customer sentiment from highly nuanced product reviews—a real trial by fire that definitely solidified my understanding of multi-layered recurrent neural networks (RNNs). The challenge wasn’t just about getting *any* sentiment analysis, but rather achieving high precision and capturing the subtle interplay of context within the text. So, let's unpack how we can implement such a system effectively using PyTorch.

First, it's crucial to understand what we mean by a 'multi-layered' RNN. Essentially, we're stacking multiple RNN cells on top of each other. The output from one layer becomes the input to the next. This architecture allows the network to learn hierarchical representations of the data, which is particularly beneficial for tasks like sentiment analysis where the meaning can be quite dependent on the overall structure of the sentence or paragraph. Single-layered RNNs, while adequate for simpler tasks, often struggle to capture these higher-order dependencies.

The intuition behind this is that the initial layer might learn to detect simple features – maybe word order or the presence of specific keywords. The layer above that, receiving information about these features, learns to combine them into more complex concepts, like phrases or clauses. Finally, the topmost layers build up an understanding of the entire sentiment or topic. In essence, the network builds its understanding of the text in stages.

Now, let’s talk code. PyTorch makes it surprisingly approachable, even with the complexities involved. The core construct for an RNN in PyTorch is the `nn.RNN`, `nn.LSTM`, or `nn.GRU` module. For the examples, I'll primarily be using `nn.LSTM` due to its effectiveness at handling long-range dependencies, although the same principles apply to `nn.RNN` and `nn.GRU`.

Here's our first snippet, setting up the basic architecture:

```python
import torch
import torch.nn as nn

class MultiLayerRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(MultiLayerRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        # we only want the final time step output
        prediction = self.fc(hidden[-1,:,:])
        return prediction

# Example usage:
vocab_size = 10000  # size of your vocabulary
embedding_dim = 100  # dimensionality of word embeddings
hidden_dim = 256   # dimensionality of the hidden state
num_layers = 2      # number of LSTM layers
output_dim = 3      # number of output classes (e.g., positive, negative, neutral)


model = MultiLayerRNN(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim)

dummy_input = torch.randint(0, vocab_size, (32, 50)) # batch size 32, sequence length 50
output = model(dummy_input)
print(output.shape) # expected output: torch.Size([32, 3])
```

This snippet shows the basic setup. The `__init__` method defines our layers: an embedding layer, the multi-layered LSTM, and a fully connected layer for classification. Crucially, we're passing `num_layers` to the `nn.LSTM` constructor. In the `forward` method, we pass the embedded text to the LSTM, then extract the final hidden state of the last layer (`hidden[-1,:,:]`) and feed it to the fully connected layer, producing our prediction. The `batch_first=True` argument in the LSTM constructor specifies that the input and output tensors will have batch size as the first dimension, which simplifies the batch handling in our code.

But, what about handling padded sequences of differing lengths? A common issue when working with text is the variability in sentence length. We need to make sure that our recurrent layers operate on valid parts of the sequence only. Here's how we might deal with that:

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MultiLayerRNNWithPadding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(MultiLayerRNNWithPadding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, text, text_lengths):
      embedded = self.embedding(text)
      packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
      packed_output, (hidden, cell) = self.lstm(packed_embedded)

      # hidden is the last hidden state of each layer, we want the last layer
      prediction = self.fc(hidden[-1,:,:])
      return prediction

# Example usage:
vocab_size = 10000  # size of your vocabulary
embedding_dim = 100  # dimensionality of word embeddings
hidden_dim = 256   # dimensionality of the hidden state
num_layers = 2      # number of LSTM layers
output_dim = 3      # number of output classes (e.g., positive, negative, neutral)


model_pad = MultiLayerRNNWithPadding(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim)

# Sample input with varying lengths
dummy_input = torch.randint(0, vocab_size, (32, 50))
lengths = torch.randint(10,50,(32,)) # random lengths for each sequence
output = model_pad(dummy_input, lengths)

print(output.shape)
```

Here, we’ve incorporated `pack_padded_sequence` and `pad_packed_sequence`. `pack_padded_sequence` masks out padded tokens during the LSTM calculation, preventing them from interfering with the learning process. We pass the lengths of each sentence as a tensor. We pack the padded input, pass it to the LSTM, and retrieve the final hidden state. Crucially, the model now handles variable length sequences without leaking padding effects into the hidden state computation.

Finally, let's consider a slight modification, using a bidirectional LSTM. Bidirectional LSTMs are useful because they can capture information from both past and future context, which can improve the model's performance. This is another technique I deployed in that past project.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiMultiLayerRNNWithPadding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(BiMultiLayerRNNWithPadding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # multiply by 2 as we have forward and backward hidden states


    def forward(self, text, text_lengths):
      embedded = self.embedding(text)
      packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
      packed_output, (hidden, cell) = self.lstm(packed_embedded)

      # hidden has dimensions (num_layers*num_directions, batch, hidden_size) we want the last hidden state from each direction from last layer
      hidden_forward = hidden[-2,:,:] # last layer forward hidden state
      hidden_backward = hidden[-1,:,:]# last layer backward hidden state

      prediction = self.fc(torch.cat((hidden_forward, hidden_backward), dim=1)) # concat forward and backward hidden states then pass to fc
      return prediction


# Example usage:
vocab_size = 10000  # size of your vocabulary
embedding_dim = 100  # dimensionality of word embeddings
hidden_dim = 256   # dimensionality of the hidden state
num_layers = 2      # number of LSTM layers
output_dim = 3      # number of output classes (e.g., positive, negative, neutral)


model_bi = BiMultiLayerRNNWithPadding(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim)

# Sample input with varying lengths
dummy_input = torch.randint(0, vocab_size, (32, 50))
lengths = torch.randint(10,50,(32,)) # random lengths for each sequence
output = model_bi(dummy_input, lengths)
print(output.shape) # expected: torch.Size([32, 3])
```

Here, we set `bidirectional=True` in the LSTM constructor, which doubles the size of the output hidden states, so the output of our linear layer needs to be adjusted accordingly. The forward hidden states from the last layer are accessible with `hidden[-2,:,:]`, and the backward with `hidden[-1,:,:]`, where we concatenate them before passing to the fully connected layer.

For further study, I would suggest looking into papers like "Long Short-Term Memory" by Hochreiter and Schmidhuber, which will provide a strong understanding of the underlying mechanics of the LSTM. Additionally, the book “Deep Learning” by Goodfellow, Bengio, and Courville contains thorough coverage on recurrent networks and sequence models. Experimentation is key. Try various hidden layer dimensions, numbers of layers, and embedding sizes. The optimal settings are often task-specific. This should be a solid foundation for your implementation of multi-layered RNNs for sentiment analysis.
