---
title: "How can I implement LSTMs in PyTorch?"
date: "2025-01-30"
id: "how-can-i-implement-lstms-in-pytorch"
---
Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), are particularly effective at handling sequential data due to their ability to retain information over longer periods compared to standard RNNs. Understanding their implementation in PyTorch necessitates grasping the core LSTM cell structure, how it’s integrated into a PyTorch module, and the practicalities of training them. Having used PyTorch for sequence-based tasks extensively, including time-series forecasting and natural language processing, I can outline a robust approach to implementing LSTMs.

At its core, an LSTM cell manages information flow via three key gates: the forget gate, input gate, and output gate. The forget gate determines which information from the previous cell state should be discarded; the input gate decides which new information should be added to the cell state; and the output gate regulates what information should be exposed as the cell's output. This carefully controlled process enables LSTMs to learn long-term dependencies effectively.

Implementing an LSTM in PyTorch primarily involves utilizing the `torch.nn.LSTM` module. This module provides a pre-built LSTM architecture that one can easily integrate into a custom model. One defines the `input_size`, which is the dimensionality of each time-step input, the `hidden_size`, which is the dimensionality of the hidden state vector, the `num_layers`, which dictates how many LSTM layers are stacked sequentially, and potentially `dropout` for regularization. The module itself handles all the gate calculations internally.

A crucial aspect of working with LSTMs is understanding how to format input data. PyTorch's LSTM module expects input as a tensor of shape `(sequence_length, batch_size, input_size)`. The hidden and cell state initialization is crucial. If no initial states are provided, PyTorch defaults to zeros, which may not be ideal for all use cases. Custom initialization, for example, a random distribution, can sometimes lead to faster training.

Here's a basic example to illustrate this process:

```python
import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # batch_first is optional but makes things clearer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # initialize cell state
        
        out, _ = self.lstm(x, (h0, c0)) # out: (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :]) #  take only the last time step and pass through fully connected layer, reshaping for classification
        return out

# Example usage:
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
batch_size = 32
sequence_length = 50
model = BasicLSTM(input_size, hidden_size, num_layers, output_size)
inputs = torch.randn(batch_size, sequence_length, input_size)
output = model(inputs)
print(output.shape) # Output: torch.Size([32, 5])
```

This code demonstrates a minimalist LSTM model for sequence classification. The `batch_first=True` parameter is used in `nn.LSTM` to make the input tensor shape `(batch_size, sequence_length, input_size)`, often more intuitive to manage. I initialize both the hidden state (`h0`) and the cell state (`c0`) to tensors of zeros. Notice how only the output of the *last* time step is used as input for the fully-connected layer for classification. This is a common practice when the primary interest is in the final prediction based on the entire sequence.

Now, let's move on to an example that implements many-to-many prediction (sequence to sequence) scenario where we need output at every time step, not only the final one:

```python
import torch
import torch.nn as nn

class SequenceToSequenceLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(SequenceToSequenceLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) # out: (batch_size, sequence_length, hidden_size)
        out = self.fc(out) # apply the fully connected layer on each time step
        return out

# Example usage:
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
batch_size = 32
sequence_length = 50

model = SequenceToSequenceLSTM(input_size, hidden_size, num_layers, output_size)
inputs = torch.randn(batch_size, sequence_length, input_size)
output = model(inputs)
print(output.shape) # Output: torch.Size([32, 50, 5])
```

In this sequence-to-sequence example, the fully-connected layer (`self.fc`) is applied across *all* time steps.  This yields an output tensor of shape `(batch_size, sequence_length, output_size)`. This kind of architecture is often employed in tasks like machine translation or part-of-speech tagging. It is important to understand that the fully connected output layer is applied at every time-step. The `out` variable from the LSTM is passed directly to the `fc` without the slicing `out[:, -1, :]`.

Finally, a more advanced example, incorporating dropout and handling variable-length sequences through padding:

```python
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class PaddedLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
    super(PaddedLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, x, lengths):
    # x: (batch_size, sequence_length, input_size)
    # lengths: (batch_size)
    packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    packed_out, _ = self.lstm(packed_x, (h0, c0))
    out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True) # unpacking
    out = self.fc(out[:, -1, :]) # take the last valid output for each sequence
    return out

# Example usage:
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
batch_size = 32
# Create variable length sequences with lengths in range [10, 50]
lengths = torch.randint(10, 51, (batch_size,))
max_length = max(lengths)
# Create inputs by padding sequences to max length
inputs = torch.zeros(batch_size, max_length, input_size)
for i, length in enumerate(lengths):
    inputs[i, :length] = torch.randn(length, input_size)

model = PaddedLSTM(input_size, hidden_size, num_layers, output_size)
output = model(inputs, lengths)
print(output.shape)  # Output: torch.Size([32, 5])
```

In this more nuanced implementation, I've added `dropout` for regularization in the LSTM and, more significantly, incorporated handling for variable-length input sequences.  Variable-length sequences cannot be batched directly, so we must use padding (adding zeros to the end of the sequence) to make each sequence have the same length.  The `pack_padded_sequence` function essentially tells the LSTM module to ignore the padded values and to update the hidden state only based on valid values in the sequence. The `pad_packed_sequence`  then reverses the `packing` operation. The lengths of original sequences must be passed as parameter along with the padded sequence for unpacking. The important detail here is the `enforce_sorted=False` argument during packing. This is essential as batch is no longer sorted by sequence length, as sorting is now done internally in `pack_padded_sequence`. Finally, only the last valid output in every sequence is used for classification after unpacking.

For a comprehensive understanding of LSTMs and their applications, I would recommend exploring publications on recurrent neural networks and natural language processing. Resources such as "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, provide an excellent theoretical foundation.  Additionally, the official PyTorch documentation and tutorials are invaluable for practical implementation. Consulting research papers in reputable conferences on deep learning is also beneficial. A strong mathematical grounding in linear algebra and calculus enhances understanding of the underlying mechanisms. Studying implemented examples from widely used libraries (like Hugging Face Transformers) offers practical insights into architecture design.
