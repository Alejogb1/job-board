---
title: "How can LSTM layers be used as encoders in PyTorch?"
date: "2024-12-23"
id: "how-can-lstm-layers-be-used-as-encoders-in-pytorch"
---

Alright, let's talk about using lstm layers as encoders. I've spent a fair bit of time mucking around with recurrent neural networks, and specifically lstms, in various projects over the years, from natural language processing tasks to time-series analysis, and have definitely explored this particular pattern a number of times. It’s actually a quite versatile approach once you grasp the nuances.

The core idea behind using an lstm as an encoder is to take a sequential input – a sentence, a time series of sensor readings, or anything similar – and transform it into a fixed-length vector representation that captures the essence of that sequence. This vector is then used by the downstream model, often another neural network, for tasks like classification, regression, or generating new sequences (think sequence-to-sequence models). The final hidden state, or sometimes both the final hidden and cell states, produced by the lstm layer effectively summarizes the input sequence.

Now, the mechanics of doing this in pytorch are fairly straightforward, but it's useful to be explicit. We’re going to feed a sequence through the lstm and extract either the last hidden state or both the last hidden and cell states, depending on the requirements of the downstream task. Let's look at an example where we just want the last hidden state for a classification problem.

```python
import torch
import torch.nn as nn

class LSTMEncoderClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMEncoderClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
      # x is expected to be of shape (batch_size, seq_len, input_size)
      _, (hidden, _) = self.lstm(x)  # _ is for the cell state, we're ignoring it
      # hidden shape: (num_layers, batch_size, hidden_size)
      last_hidden = hidden[-1, :, :] # shape: (batch_size, hidden_size)
      output = self.fc(last_hidden)
      return output

# Example usage
input_size = 10 # Size of each vector in the input sequence
hidden_size = 20
num_layers = 2
num_classes = 5
batch_size = 32
sequence_length = 50

model = LSTMEncoderClassifier(input_size, hidden_size, num_layers, num_classes)
sample_input = torch.randn(batch_size, sequence_length, input_size)
output = model(sample_input)
print(f"Output shape: {output.shape}") # Output shape: torch.Size([32, 5])
```

Here, the `LSTMEncoderClassifier` takes input sequences and produces a classification over `num_classes`. The key part is in the forward pass where `_, (hidden, _ ) = self.lstm(x)`. This runs the input through the lstm, and we extract only the last hidden state by indexing with `hidden[-1, :, :]`. The cell state is not used in this case. This extracted `last_hidden` vector is a representation of the input sequence, that now undergoes linear transformation to fit the output dimension.

Now, let’s consider a case where you’d like to utilize both the hidden and cell states, as often happens when feeding into a decoder in sequence-to-sequence models.

```python
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
      _, (hidden, cell) = self.lstm(x) # hidden and cell shape are (num_layers, batch_size, hidden_size)
      return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        # x is of shape (batch_size, seq_len_decoder, input_size) in training and (batch_size, 1, input_size) during inference
        output, _ = self.lstm(x,(hidden_state,cell_state))
        # output shape: (batch_size, seq_len_decoder/1, hidden_size)
        output = self.fc(output) # shape: (batch_size, seq_len_decoder/1, output_size)
        return output


# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
batch_size = 32
sequence_length_encoder = 50
sequence_length_decoder = 20
encoder = LSTMEncoder(input_size, hidden_size, num_layers)
decoder = Decoder(input_size, hidden_size, num_layers,output_size)
sample_encoder_input = torch.randn(batch_size, sequence_length_encoder, input_size)
hidden, cell = encoder(sample_encoder_input)

# The decoder initial hidden and cell states are the encoder's last states.
# Example using a tensor of zeros to showcase decoder training
sample_decoder_input = torch.randn(batch_size, sequence_length_decoder, input_size)
output = decoder(sample_decoder_input, hidden, cell)
print(f"Decoder output shape: {output.shape}") # Output shape: torch.Size([32, 20, 5])
```

Here, we have a paired encoder and decoder. The `LSTMEncoder` extracts both the final hidden state and cell state. These states are then used to initialize the `LSTM` of the `Decoder` module in its forward method. This initialization allows the decoder to start its processing using the encoded information, which is critical for sequence-to-sequence tasks such as machine translation.

Finally, you can also create a bidirectional lstm based encoder for more intricate sequence modeling.

```python
import torch
import torch.nn as nn

class BidirectionalLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BidirectionalLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        # hidden shape will be (num_layers * 2, batch_size, hidden_size)
        # since it is bidirectional, we need to concatenate the forward and backward hidden states
        forward_last_hidden = hidden[-2, :, :] # last layer of the forward network
        backward_last_hidden = hidden[-1,:, :] # last layer of the backward network
        encoded_vector = torch.cat((forward_last_hidden, backward_last_hidden), dim = 1) #Shape: (batch_size, 2*hidden_size)
        return encoded_vector

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
batch_size = 32
sequence_length = 50

model = BidirectionalLSTMEncoder(input_size, hidden_size, num_layers)
sample_input = torch.randn(batch_size, sequence_length, input_size)
encoded_vector = model(sample_input)
print(f"Encoded vector shape: {encoded_vector.shape}")  # Output shape: torch.Size([32, 40])
```

This time, we initialize our lstm layer with `bidirectional = True`. Because we have a bidirectional lstm, at each time step the lstm propagates forward and backward in the sequence, meaning the output states returned are actually two sets of states, one for the forward pass and one for the backward pass. So in the forward pass we extract the last hidden state from each direction and concatenate them together. This vector now contains information from both the start and end of the sequence and may help capture more complex long range dependencies within sequences.

For deeper understanding of lstms and sequence modelling, I’d strongly recommend reading ‘Understanding LSTM Networks’ by Christopher Olah, which is an excellent blog post, and ‘Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow’ by Aurélien Géron, which provides practical coverage of neural networks. For a more theoretical look, you could explore ‘Deep Learning’ by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. These resources provide a solid foundation for anyone working with recurrent neural networks and lstms in particular.

These snippets should give you a solid starting point. Just keep in mind that choice between these approaches, i.e., whether you use only hidden state or both hidden and cell, or whether you implement a bidirectional lstm, will heavily depend on the specific task and characteristics of your data.
