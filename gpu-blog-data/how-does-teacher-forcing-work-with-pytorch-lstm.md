---
title: "How does teacher forcing work with PyTorch LSTM cells?"
date: "2025-01-30"
id: "how-does-teacher-forcing-work-with-pytorch-lstm"
---
Teacher forcing, within the context of recurrent neural networks (RNNs), specifically those employing Long Short-Term Memory (LSTM) cells in PyTorch, is a training technique impacting the input sequence fed to the network during each timestep.  My experience building sequence-to-sequence models for natural language processing tasks highlighted its crucial role in stabilizing training and improving model performance, particularly during early epochs.  Crucially, it involves providing the *ground truth* output from the previous timestep as input at the current timestep, rather than relying on the model's own previous prediction. This contrasts with free-running inference, where the model’s prediction at time *t-1* informs its prediction at time *t*.

The core mechanism hinges on the understanding of how LSTMs process sequential data.  An LSTM cell receives an input at each timestep, typically alongside the previous hidden state. This hidden state encapsulates information from previous timesteps.  In a typical sequence-to-sequence scenario (e.g., machine translation), the LSTM attempts to predict the next element in the output sequence.  During training, without teacher forcing, the input at timestep *t* is the model's prediction from timestep *t-1*.  Errors accumulate, as the model learns on a sequence of potentially inaccurate predictions, leading to instability and hindering the learning process, especially for long sequences.

Teacher forcing mitigates this instability by presenting the *correct* output at each timestep as input. This allows the LSTM to focus on learning the mapping between input and output sequences without compounding errors.  The model is effectively “taught” the correct sequence step-by-step.  However, this introduces a discrepancy between training and inference, since during inference, the model operates in a free-running mode. This discrepancy, often referred to as the *exposure bias*, can lead to performance degradation during deployment.  Addressing this is a topic requiring careful consideration of techniques like scheduled sampling.


**Explanation:**

The implementation of teacher forcing in PyTorch is straightforward.  It involves modifying the training loop to selectively feed the network either the model’s previous prediction (free-running) or the actual ground truth.  This is commonly controlled by a boolean flag or a probability, allowing for scheduled sampling techniques. The core modification focuses on how the input sequence is generated during each iteration of the training loop.

The following code examples demonstrate teacher forcing in PyTorch using an LSTM for a sequence-to-sequence task.

**Code Example 1: Basic Teacher Forcing**

```python
import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, teacher_forcing=True):
        output, hidden = self.lstm(input, hidden)
        output = self.fc(output)
        return output, hidden

# Training loop with teacher forcing
input_size = 10
hidden_size = 20
output_size = 10
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

input_seq = torch.randn(seq_len, batch_size, input_size) #Example input sequence
target_seq = torch.randn(seq_len, batch_size, output_size) #Example target sequence

hidden = (torch.zeros(1, batch_size, hidden_size), torch.zeros(1, batch_size, hidden_size))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output, hidden = model(input_seq[0,:,:], hidden, teacher_forcing=True) # First timestep uses initial hidden state
    loss = criterion(output, target_seq[0,:,:])
    loss.backward()
    optimizer.step()
    hidden = (hidden[0].detach(), hidden[1].detach()) # detach hidden states to prevent backpropagation through the entire sequence at once

    for t in range(1, seq_len):
      if teacher_forcing:
          input = target_seq[t-1,:,:]
      else:
          input = output
      output, hidden = model(input, hidden)
      loss = criterion(output, target_seq[t,:,:])
      loss.backward()
      optimizer.step()
      hidden = (hidden[0].detach(), hidden[1].detach())


```

This example demonstrates a simple implementation. The `teacher_forcing` flag controls the input selection. The `detach()` call is crucial for preventing exploding gradients.


**Code Example 2: Scheduled Sampling**

```python
import torch
import torch.nn as nn
import random

# ... (LSTM model definition from Example 1) ...

# Training loop with scheduled sampling
teacher_forcing_ratio = 0.5 #Probability of using teacher forcing.

for epoch in range(num_epochs):
    # ... (Initialization as in Example 1) ...

    for t in range(1, seq_len):
        if random.random() < teacher_forcing_ratio:
            input = target_seq[t-1,:,:]  # Teacher forcing
        else:
            input = output              # Free running
        output, hidden = model(input, hidden)
        loss = criterion(output, target_seq[t,:,:])
        loss.backward()
        optimizer.step()
        hidden = (hidden[0].detach(), hidden[1].detach())

```

This refines the previous example by introducing scheduled sampling, where the probability of teacher forcing decreases over time, gradually transitioning to free-running inference.


**Code Example 3:  Handling Variable Sequence Lengths**

```python
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# ... (LSTM model definition, similar to Example 1 but potentially with packing/padding) ...

# Training loop with teacher forcing and packing for variable-length sequences
sequences = [torch.randn(length, input_size) for length in lengths] # Example with variable lengths
targets = [torch.randn(length, output_size) for length in lengths]

packed_sequences = rnn_utils.pack_sequence(sequences)
packed_targets = rnn_utils.pack_sequence(targets)

hidden = (torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)) # Added num_layers

# The forward pass needs to accommodate the packed sequences
packed_output, hidden = model(packed_sequences, hidden)
# unpack the sequences to calculate the loss
output, _ = rnn_utils.pad_packed_sequence(packed_output)

loss = criterion(output, packed_targets.data) # Accessing packed_targets.data for loss calculation

loss.backward()
optimizer.step()
```

This example demonstrates handling variable-length sequences, a common scenario in real-world applications, using PyTorch’s `pack_sequence` and `pad_packed_sequence` functions.  This is vital for efficient processing of sequences with varying lengths, a common characteristic of many datasets.


**Resource Recommendations:**

The PyTorch documentation, specifically the sections on RNNs and LSTMs, are essential.  A thorough understanding of the underlying mathematical principles governing LSTMs is also crucial.  Consider consulting textbooks on deep learning and natural language processing that delve into RNN architectures and training techniques.  Reviewing research papers focusing on sequence-to-sequence models and addressing the exposure bias would further enhance understanding and provide advanced techniques.
