---
title: "Can LSTMs predict sine waves using PyTorch?"
date: "2025-01-30"
id: "can-lstms-predict-sine-waves-using-pytorch"
---
The inherent sequential nature of a sine wave makes it a suitable candidate for prediction with Long Short-Term Memory (LSTM) networks, a type of recurrent neural network. LSTMs are particularly adept at capturing temporal dependencies in data, a crucial element when forecasting future values based on a time series. I've found that while seemingly straightforward, achieving accurate sine wave prediction with an LSTM in PyTorch requires careful consideration of hyperparameters and data preprocessing.

Let's break this down: An LSTM processes a sequence of inputs, maintaining an internal state that evolves with each new observation. This state acts as a memory, allowing the network to understand the context of the current input within the sequence's history. In the context of a sine wave, this memory enables the LSTM to learn the repeating pattern of peaks and troughs. Fundamentally, we are training the model to predict the next value in the sequence, *t+1*, given a history of values from *t-n* to *t*, where *n* is the length of the input sequence or 'lookback'.

The process in PyTorch involves several key steps: defining the LSTM architecture, preparing the data, defining the loss function and optimizer, training the model, and finally, evaluating its predictions. The sine wave data itself, while simple in form, needs to be organized into sequences suitable for LSTM input. Here's how I typically set up the model.

First, consider the architecture. A basic LSTM for this task might consist of an LSTM layer followed by a linear layer to project the LSTM's hidden state into a single output, which represents the predicted sine wave value. Here's the PyTorch code for this simple architecture:

```python
import torch
import torch.nn as nn

class SineWaveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SineWaveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # use the last timestep's hidden state for prediction
        output = self.linear(lstm_out[:, -1, :])
        return output

# Example Usage:
input_size = 1  # single feature
hidden_size = 50
output_size = 1
model = SineWaveLSTM(input_size, hidden_size, output_size)
```

This `SineWaveLSTM` class defines an LSTM network with an `input_size`, which is typically 1 since we're using single values from the sine wave. The `hidden_size` determines the complexity of the internal memory. The `output_size` is also 1, corresponding to the predicted next sine wave value. Note that `batch_first=True` allows us to use the convention of (batch, sequence length, input dimension) when passing inputs. The `forward` method processes the sequence, and then selects the output of the last time step for prediction using `lstm_out[:, -1, :]`, and then pushes it through the linear layer.

The second crucial element is data preparation. We first generate the sine wave, and then we split it into sequences of a specific length. This length determines how many previous time steps the LSTM considers when predicting the next one. Here's a basic method for creating sequences:

```python
import numpy as np
import torch

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return torch.tensor(np.array(xs), dtype=torch.float).unsqueeze(-1), torch.tensor(np.array(ys), dtype=torch.float).unsqueeze(-1)


# Example Usage:
time = np.arange(0, 100, 0.1)
data = np.sin(time)
seq_length = 20
X, y = create_sequences(data, seq_length)

```

The `create_sequences` function iterates through the sine wave data, creating pairs of input sequences (`x`) and corresponding target values (`y`). The output `X` contains sequences and `y` contains the target. Crucially, the `.unsqueeze(-1)` method adds a new dimension at the end making it (batch, sequence, feature) ready for the LSTM layer.

Finally, here’s the training loop and loss function selection, typically using Mean Squared Error and an optimizer such as Adam:

```python
import torch.optim as optim
import torch.nn as nn

# Model and data already initialized

learning_rate = 0.01
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_function(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# After training, prediction can be done
with torch.no_grad():
  test_input = torch.tensor(data[-seq_length:].reshape(1, seq_length, 1), dtype=torch.float)
  future_prediction = model(test_input)
  print(f'Prediction: {future_prediction.item():.4f}')

```

This segment demonstrates a basic training loop. The optimization process includes resetting the gradients, calculating output via `model(X)`, calculating the loss using `loss_function(outputs, y)`, performing backpropagation with `loss.backward()`, and updating parameters with `optimizer.step()`. Finally, it shows how to create a future prediction given a sequence of previous values.

It's crucial to note that the initial conditions, like `hidden_size`, `seq_length` and learning rates greatly impact the model performance. Through experimentation, I've found that longer sequence lengths tend to improve predictions as they allow the LSTM to learn from a longer history. Choosing appropriate values is crucial, often based on the observed period in the signal. Also, normalization of the signal can also increase stability during training.

For further study of recurrent neural networks I recommend the book “Deep Learning” by Goodfellow, Bengio, and Courville. Textbooks on time series analysis provide the necessary background in dealing with sequential data. Finally, the official PyTorch documentation is invaluable for understanding how each component of the code interacts with the library.
