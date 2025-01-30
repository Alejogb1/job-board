---
title: "How can a bidirectional RNN be implemented manually in PyTorch?"
date: "2025-01-30"
id: "how-can-a-bidirectional-rnn-be-implemented-manually"
---
Implementing a bidirectional recurrent neural network (RNN) manually in PyTorch offers a deeper understanding of the underlying mechanics compared to relying on pre-built layers.  My experience optimizing sequence-to-sequence models for natural language processing tasks has highlighted the importance of this understanding, especially when dealing with less common architectures or needing fine-grained control over the computational graph.  Crucially, understanding the forward and backward passes independently is key to a successful manual implementation.

The core concept lies in processing the input sequence in two directions – forward and backward – and concatenating the hidden states at each time step.  This allows the network to capture context from both preceding and succeeding elements in the sequence, leading to improved performance, particularly in tasks requiring long-range dependencies.  A naive approach would simply involve running two separate RNNs, but a more efficient method involves leveraging PyTorch's computational graph capabilities for optimization.

**1.  Clear Explanation:**

A bidirectional RNN consists of two independent RNNs: one processing the input sequence in the forward direction (from time step 1 to T), and another processing it in the backward direction (from T to 1).  Let's denote the forward RNN's hidden state at time step *t* as  `h_forward(t)` and the backward RNN's hidden state as `h_backward(t)`. The final hidden state at each time step is the concatenation of these two: `h(t) = [h_forward(t); h_backward(t)]`.  This concatenated hidden state can then be used to predict the output.

The forward pass involves iteratively computing `h_forward(t)` using the input at time step *t* and the previous forward hidden state `h_forward(t-1)`. Similarly, the backward pass calculates `h_backward(t)` using the input at time step *t* and the previous backward hidden state `h_backward(t+1)`.  Note that the backward RNN requires the entire input sequence before it can begin computation, hence the name.  Crucially, both RNNs share the same weights, but operate independently on different temporal directions. The output layer then receives the concatenated hidden state `h(t)` to produce the prediction at time *t*.  This contrasts with the simpler unidirectional RNN which only considers past context.

**2. Code Examples with Commentary:**

**Example 1: Basic Bidirectional RNN with tanh activation:**

```python
import torch
import torch.nn as nn

class ManualBidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualBidirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.forward_rnn = nn.RNN(input_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.backward_rnn = nn.RNN(input_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Forward pass
        forward_output, _ = self.forward_rnn(x)

        # Backward pass (requires reversing the input sequence)
        reversed_x = torch.flip(x, [1]) #Reverse along the sequence dimension
        backward_output, _ = self.backward_rnn(reversed_x)
        backward_output = torch.flip(backward_output, [1]) # Reverse back to original order

        # Concatenate hidden states
        combined_output = torch.cat((forward_output, backward_output), dim=2)

        # Output layer
        output = self.output_layer(combined_output)
        return output

# Example Usage:
input_size = 10
hidden_size = 20
output_size = 5
seq_len = 30
batch_size = 64

model = ManualBidirectionalRNN(input_size, hidden_size, output_size)
input_data = torch.randn(batch_size, seq_len, input_size)
output = model(input_data)  #output shape (batch_size, seq_len, output_size)
print(output.shape)
```

This example utilizes two separate `nn.RNN` layers.  The input sequence is reversed for the backward pass and then reversed again to match the original order. This maintains correct temporal alignment for concatenation.  The `batch_first=True` argument ensures the batch dimension is the first dimension, aligning with common PyTorch conventions.


**Example 2:  Bidirectional RNN with LSTM:**

```python
import torch
import torch.nn as nn

class ManualBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualBidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.forward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Forward pass
        forward_output, _ = self.forward_lstm(x)

        # Backward pass
        reversed_x = torch.flip(x, [1])
        backward_output, _ = self.backward_lstm(reversed_x)
        backward_output = torch.flip(backward_output, [1])

        # Concatenate hidden states
        combined_output = torch.cat((forward_output, backward_output), dim=2)

        # Output Layer
        output = self.output_layer(combined_output)
        return output

# Example usage (same as Example 1, just change the model instantiation)
model = ManualBidirectionalLSTM(input_size, hidden_size, output_size)
output = model(input_data)
print(output.shape)
```

This example replaces the `nn.RNN` with `nn.LSTM`, demonstrating the adaptability of the manual implementation to different RNN cell types.  The core structure remains identical, highlighting the flexibility of the approach.


**Example 3:  Handling Variable-Length Sequences:**

```python
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class ManualBidirectionalRNNVariableLength(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualBidirectionalRNNVariableLength, self).__init__()
        self.hidden_size = hidden_size
        self.forward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.backward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x, seq_lengths):
        # x shape: (batch_size, max_seq_len, input_size)
        # seq_lengths: (batch_size) tensor of sequence lengths
        
        # Pack padded sequences
        packed_x = rnn_utils.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Forward pass on packed sequences
        forward_output, _ = self.forward_rnn(packed_x)

        # Unpack sequences
        forward_output, _ = rnn_utils.pad_packed_sequence(forward_output, batch_first=True)


        # Backward pass requires additional steps for variable lengths

        packed_x_reversed = rnn_utils.pack_padded_sequence(torch.flip(x,[1]), seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        backward_output, _ = self.backward_rnn(packed_x_reversed)
        backward_output, _ = rnn_utils.pad_packed_sequence(backward_output, batch_first=True)
        backward_output = torch.flip(backward_output,[1])



        # Concatenate hidden states
        combined_output = torch.cat((forward_output, backward_output), dim=2)

        # Output layer
        output = self.output_layer(combined_output)
        return output


# Example Usage with variable length sequences:
batch_size = 64
max_seq_len = 50
input_size = 10
hidden_size = 20
output_size = 5


seq_lengths = torch.randint(1, max_seq_len, (batch_size,))
input_data = torch.randn(batch_size, max_seq_len, input_size)
input_data = rnn_utils.pad_packed_sequence(rnn_utils.pack_padded_sequence(input_data, seq_lengths.cpu(), batch_first=True, enforce_sorted=False), batch_first=True)[0]
model = ManualBidirectionalRNNVariableLength(input_size, hidden_size, output_size)
output = model(input_data, seq_lengths)
print(output.shape)

```

This example addresses the practical scenario of variable-length input sequences, using `torch.nn.utils.rnn.pack_padded_sequence` to efficiently handle padding.  This is crucial for real-world applications where sequences are rarely of uniform length. Note that the backward pass also requires careful handling of padding during packing and unpacking.


**3. Resource Recommendations:**

"Deep Learning" by Goodfellow, Bengio, and Courville;  "Neural Network and Deep Learning" by Michael Nielsen;  PyTorch documentation.  A solid grasp of linear algebra and calculus is also highly beneficial.
