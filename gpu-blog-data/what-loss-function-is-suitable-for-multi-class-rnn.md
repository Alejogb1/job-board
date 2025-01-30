---
title: "What loss function is suitable for multi-class RNN sequence classification?"
date: "2025-01-30"
id: "what-loss-function-is-suitable-for-multi-class-rnn"
---
Recurrent Neural Networks (RNNs) for multi-class sequence classification present a specific challenge: they predict a sequence label at each time step, with the target label belonging to one of several distinct classes. Consequently, the loss function must adequately capture the discrepancies between predicted probabilities and the actual class labels across the entire sequence. The most appropriate choice is Categorical Cross-Entropy loss, which, in the context of sequence data, is often implemented as a sequence-aware variant.

Categorical Cross-Entropy measures the difference between two probability distributions: the predicted probability distribution over classes and the true, one-hot encoded probability distribution of the target class at each time step. It is crucial that the output layer of the RNN is configured to produce probability values, which is typically achieved with a Softmax activation. This activation ensures the output vector sums to 1, representing a probability distribution over the possible classes.

The standard cross-entropy, applied to individual data points, is given by:

$L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$

where:

*   $C$ is the number of classes.
*   $y_c$ is the true probability for class c (1 for the true class, 0 for others in one-hot encoding).
*   $\hat{y}_c$ is the predicted probability for class c.

However, when dealing with sequences, this loss is typically applied at each time step in the sequence, and the loss for the entire sequence is aggregated, usually by averaging. This is essential to account for the temporal dependencies captured by the RNN. Thus, the aggregated loss across the sequence becomes:

$L_{sequence} = \frac{1}{T} \sum_{t=1}^{T} L_t$

Where:
* $T$ is the length of the sequence.
* $L_t$ is the Categorical Cross-Entropy loss at time step *t*.

This approach facilitates backpropagation through time, allowing the network to learn from gradients propagated from each time step. Without considering the entire sequence, individual time step outputs would not contribute to adjustments affecting preceding timesteps.

Here are three illustrative examples, demonstrating how this might be implemented and the types of challenges encountered:

**Example 1: Basic Multi-Class Sequence Classification**

This example assumes a text categorization scenario with three classes (e.g., 'positive', 'negative', 'neutral'). The input sequences are sequences of word indices.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        # Use the output of the last time step for classification
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        return logits

# Example usage
vocab_size = 100
embedding_dim = 32
hidden_dim = 64
num_classes = 3
sequence_length = 20
batch_size = 16

model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()  # Categorical Cross-Entropy built-in in PyTorch
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example dummy training data
input_sequences = torch.randint(0, vocab_size, (batch_size, sequence_length))
target_labels = torch.randint(0, num_classes, (batch_size,))


for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(input_sequences) #Outputs size (batch_size, num_classes)
    loss = criterion(outputs, target_labels) # CrossEntropyLoss expects outputs of (batch_size, num_classes) and target labels of (batch_size)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
      print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')
```

In this example, `nn.CrossEntropyLoss()` directly handles the Softmax operation and calculation of categorical cross-entropy, thus simplifying the overall code. Note that outputs are obtained only from the last time step. This is common for a many-to-one RNN sequence classification task.

**Example 2: Sequence Labelling with Per-Timestep Classification**

This example deals with a sequence labelling scenario (e.g., Part-of-Speech tagging). Here, we need to generate a label at each time step.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SequenceLabellingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SequenceLabellingRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output) # logits shape is (batch_size, sequence_length, num_classes)
        return logits

# Example usage
vocab_size = 100
embedding_dim = 32
hidden_dim = 64
num_classes = 5
sequence_length = 20
batch_size = 16

model = SequenceLabellingRNN(vocab_size, embedding_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()  # Categorical Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example dummy training data
input_sequences = torch.randint(0, vocab_size, (batch_size, sequence_length))
target_labels = torch.randint(0, num_classes, (batch_size, sequence_length))


for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(input_sequences) #Outputs size (batch_size, sequence_length, num_classes)

    # CrossEntropyLoss requires logits to be of shape (N, C, H, W) when the target has shape (N, H, W)
    # In our case target is (batch_size, sequence_length) and outputs are (batch_size, sequence_length, num_classes) so we need to re-arrange

    loss = criterion(outputs.transpose(1,2), target_labels) # Transpose the dimension for criterion input
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
      print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')

```

Here, we classify each time step, generating a sequence of predictions corresponding to each input. The `CrossEntropyLoss` expects the logits to be of shape (batch\_size, num\_classes, sequence\_length), thus it is needed to transpose the dimensions of the model outputs.

**Example 3: Masked Loss with Padding**

Often, input sequences have varying lengths, requiring padding to form a uniform batch. This can lead to padded elements erroneously contributing to the loss. A mask ensures padded values do not affect the calculation.

```python
import torch
import torch.nn as nn
import torch.optim as optim


class MaskedRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(MaskedRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits


# Example usage
vocab_size = 100
embedding_dim = 32
hidden_dim = 64
num_classes = 5
batch_size = 16
max_seq_length = 20


model = MaskedRNN(vocab_size, embedding_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss(reduction='none')  # Calculate loss per example, not aggregated

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example of sequence data and lengths
seq_lengths = torch.randint(5, max_seq_length + 1, (batch_size,))
input_sequences = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
target_labels = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

for i in range(batch_size):
    length = seq_lengths[i]
    input_sequences[i, :length] = torch.randint(0, vocab_size, (length,))
    target_labels[i, :length] = torch.randint(0, num_classes, (length,))

# Create a mask for padding
mask = torch.arange(max_seq_length).unsqueeze(0) < seq_lengths.unsqueeze(1) # (batch_size, seq_len)


for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(input_sequences) #Outputs size (batch_size, sequence_length, num_classes)

    loss_values = criterion(outputs.transpose(1, 2), target_labels) # Loss for each time step of each sequence
    masked_loss = (loss_values * mask).sum() / mask.sum() # Average over valid timesteps only
    masked_loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
      print(f'Epoch [{epoch+1}/200], Loss: {masked_loss.item():.4f}')

```
In this example, `reduction='none'` allows us to obtain a loss value per time step and instance. The `mask` tensor is then applied to zero-out loss contributions from the padded parts of the sequence.  The masked loss is then averaged only over valid timesteps.

For further exploration and a deeper understanding of this topic, I would recommend the following resources: textbooks on neural networks and deep learning, covering RNN architectures and loss functions; comprehensive online course materials on sequence modeling and recurrent networks; and the official documentation for deep learning libraries such as PyTorch and TensorFlow. Specific emphasis should be given to sequence classification, cross-entropy variants, and backpropagation through time. These resources, while not providing specific solutions, equip one with the foundational knowledge to approach various sequence modeling problems in the future.
