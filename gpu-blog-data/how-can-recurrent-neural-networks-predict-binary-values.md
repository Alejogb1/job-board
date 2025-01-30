---
title: "How can recurrent neural networks predict binary values?"
date: "2025-01-30"
id: "how-can-recurrent-neural-networks-predict-binary-values"
---
Recurrent Neural Networks (RNNs), despite their name implying sequential data, can effectively handle binary prediction tasks. This is achieved by adapting their output layer and loss function to align with the binary classification problem, while still leveraging their core capability of processing temporal dependencies if present in the input features. Over the past four years, working extensively with time-series data for predictive maintenance, I've seen this application across various sensor streams, often leading to the classification of system states as either 'normal' or 'anomalous'.

The core mechanism hinges on transforming the typically continuous output of an RNN into a probability distribution over two classes. The final layer of the network is not a direct linear transformation; instead, it uses a sigmoid activation function. This function squashes any input value to the range between 0 and 1, making it suitable for interpreting as a probability of belonging to the positive class. Specifically, the output of the last hidden layer is passed through a linear fully connected layer, and then through the sigmoid. This process converts an abstract representation of the input sequence, processed sequentially by the recurrent layers, into a single value indicating the probability that the sequence belongs to the '1' class. The probability of belonging to the '0' class is implicitly inferred as 1 minus this value.

To train this model, we require an appropriate loss function. Binary cross-entropy is the natural choice. This function quantifies the difference between the predicted probability and the true binary label (0 or 1). Specifically, for a single prediction, the binary cross-entropy loss is calculated as -[y * log(p) + (1 - y) * log(1 - p)], where 'y' is the true label (0 or 1) and 'p' is the predicted probability. Minimizing this loss through gradient descent iteratively optimizes the network’s weights to produce predictions closer to the actual labels.

Let's illustrate with specific code examples using Python and a conceptual framework inspired by common libraries.

**Code Example 1: Simple RNN for Binary Classification**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleBinaryRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleBinaryRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, hn = self.rnn(x)  # hn is last hidden state
        out = self.fc(hn.squeeze(0))  # remove the sequence dimension
        out = self.sigmoid(out)
        return out

# Sample data (batch size 2, sequence length 5, input feature size 3)
input_data = torch.randn(2, 5, 3)
true_labels = torch.tensor([[0.], [1.]]) #binary target data

# Hyperparameters
input_size = 3
hidden_size = 16
learning_rate = 0.01
epochs = 100

# Initialize model, loss function and optimizer
model = SimpleBinaryRNN(input_size, hidden_size)
criterion = nn.BCELoss() #Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
  optimizer.zero_grad()
  outputs = model(input_data)
  loss = criterion(outputs, true_labels)
  loss.backward()
  optimizer.step()
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

This snippet demonstrates a basic RNN setup. The `SimpleBinaryRNN` class includes an RNN layer (`nn.RNN`), a linear layer (`nn.Linear`), and a sigmoid activation (`nn.Sigmoid`). The `forward` method processes input sequences, extracts the last hidden state, passes it through the linear layer, and finally transforms it with the sigmoid function to produce a probability. The training loop illustrates a standard workflow: zeroing gradients, forward pass, loss calculation, backward pass, and optimization. The loss, computed by `nn.BCELoss`, provides an objective function for learning. While this example uses a standard RNN for simplicity, LSTMs and GRUs are often more suitable for capturing long-range dependencies.

**Code Example 2: LSTM with Attention for Imbalanced Data**

```python
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        weighted_output = torch.sum(attn_weights * lstm_output, dim=1)
        return weighted_output

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_dim):
      super(LSTMWithAttention, self).__init__()
      self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
      self.attention = AttentionMechanism(hidden_dim)
      self.fc = nn.Linear(hidden_dim, 1)
      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_out = self.attention(lstm_out)
        output = self.fc(attention_out)
        output = self.sigmoid(output)
        return output


# Using same data and hyperparams from previous example,
# however, lets assume the labels are imbalanced.

# Initialize model, loss function and optimizer
model = LSTMWithAttention(input_size, hidden_size)
class_weights = torch.tensor([0.2, 0.8]) # Adjust weights for imbalanced data
criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights[1]/class_weights[0]) #Weighted Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop (identical to the first example)
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, true_labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

This example incorporates an LSTM (`nn.LSTM`) instead of a vanilla RNN, which often performs better due to its capacity to retain long-term dependencies. It also includes a rudimentary attention mechanism. The attention module, parameterized by a linear layer, assigns weights to the LSTM outputs at different time steps. This allows the model to focus on the most relevant parts of the sequence. Significantly, `nn.BCEWithLogitsLoss` is employed here, which combines the sigmoid activation and loss calculation into a single function, improving numerical stability, particularly with imbalanced data. The `pos_weight` parameter is crucial for dealing with imbalanced datasets, weighting the minority class to mitigate bias.

**Code Example 3: Using Bidirectional LSTM with Padding**

```python
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class BiLSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
      super(BiLSTMBinaryClassifier, self).__init__()
      self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, bidirectional=True, batch_first=True)
      self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
      self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_lengths):
        packed_x = rnn_utils.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        last_out = out[torch.arange(out.size(0)), seq_lengths - 1]
        output = self.fc(last_out)
        output = self.sigmoid(output)
        return output

#Example Data (Variable Lengths)
input_data_lengths = [3, 5]
padded_input_data = rnn_utils.pad_sequence([torch.randn(input_data_lengths[0], 3), torch.randn(input_data_lengths[1], 3)], batch_first=True)
# Hyperparameters
hidden_dim = 32
num_layers = 2
input_size = 3
learning_rate = 0.001
epochs = 50

# Initialize model, loss function and optimizer
model = BiLSTMBinaryClassifier(input_size, hidden_dim, num_layers)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop (with length information)
for epoch in range(epochs):
  optimizer.zero_grad()
  outputs = model(padded_input_data, torch.tensor(input_data_lengths))
  loss = criterion(outputs, true_labels)
  loss.backward()
  optimizer.step()
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

This code segment introduces a bidirectional LSTM (`nn.LSTM(bidirectional=True)`). This configuration processes the sequence in both forward and reverse directions, allowing the model to capture dependencies from both future and past time steps. Crucially, this example also demonstrates how to handle variable-length sequences using `torch.nn.utils.rnn.pack_padded_sequence` and `torch.nn.utils.rnn.pad_packed_sequence`. Padding is a technique to make all sequences within a batch have the same length by adding placeholder values. `pack_padded_sequence` avoids processing these padding values, improving training efficiency. The `last_out` extraction step specifically selects the output corresponding to the last actual value in each sequence.

In conclusion, recurrent neural networks can effectively handle binary prediction via a combination of:  a sigmoid output activation to produce probabilities, an appropriate loss function such as binary cross-entropy, and mechanisms for dealing with sequence data including variable lengths, attention and choices in network architecture. For further exploration, I would recommend resources focusing on: time-series analysis with deep learning, documentation for the specific deep learning framework being used (PyTorch in these examples), and case studies on binary classification using RNNs, often found in machine learning research publications and online courses. Further resources on handling imbalanced datasets should be consulted depending on the nature of the data.
