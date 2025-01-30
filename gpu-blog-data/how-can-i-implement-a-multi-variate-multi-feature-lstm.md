---
title: "How can I implement a multi-variate, multi-feature LSTM in PyTorch?"
date: "2025-01-30"
id: "how-can-i-implement-a-multi-variate-multi-feature-lstm"
---
Implementing a multivariate, multi-feature Long Short-Term Memory (LSTM) network in PyTorch necessitates a careful understanding of data preparation, model architecture, and training considerations. I’ve spent considerable time optimizing these networks for time-series forecasting in industrial control systems, and the nuances involved often go beyond basic tutorials. The core challenge lies in properly shaping the input data to accommodate multiple variables, each potentially having numerous features, and feeding that into the LSTM layer.

**Explanation**

The primary distinction between a univariate and multivariate LSTM lies in the input shape. A univariate LSTM typically receives a sequence of single values over time. For instance, stock price fluctuations are a single value over a specific time window. A multivariate LSTM, however, takes multiple time series as input. Each time series can itself have multiple features associated with it. Imagine monitoring a machine where you observe temperature, pressure, and vibration – three time series, each with potential features such as average, standard deviation, and peak-to-peak value over a small interval. This results in a multi-dimensional input for the LSTM.

In PyTorch, the LSTM layer expects input of the shape `(sequence_length, batch_size, input_size)`.  Let’s break this down for the multi-variate, multi-feature scenario:

*   **`sequence_length`**: This is the number of time steps in the input sequence. If you're predicting future values based on the past 20 time units, your sequence length is 20.
*   **`batch_size`**: This is the number of independent input sequences processed concurrently. Using batching accelerates training via parallel computation.
*   **`input_size`**: This is where the complexity arises in the multi-feature, multi-variate scenario. The `input_size` is equal to the total number of features across all the variables you wish to incorporate into your network for each time step.  If you have three variables (temperature, pressure, vibration), and each variable is represented by three features (average, standard deviation, peak), your `input_size` becomes 3 variables * 3 features/variable = 9. Each time step therefore contains an array of nine values.

The preparation stage is critical. The data must first be arranged into individual sequences, each representing one time series window. Each sequence needs to contain the correct ordering of features. Second, these individual sequences must be batched to a shape acceptable to the LSTM layer. Consequently, a considerable portion of the initial development involves transforming raw data into a format suitable for the PyTorch framework.

After processing through the LSTM layer, which has internal states that capture sequential dependencies, you typically append a fully connected (linear) layer. This linear layer takes the output of the LSTM and maps it to the desired output size for the task at hand (for example, predicting the next value in a series or a classification label).

**Code Examples with Commentary**

**Example 1: Data Preparation**

This example demonstrates how to create data that mimics a 3-variate (temperature, pressure, vibration) scenario, each with two features (average, peak). It also shows how data must be reshaped into the desired input format before being input into the LSTM model.

```python
import torch
import numpy as np

# Create dummy time series data.
sequence_length = 5
batch_size = 2
num_variables = 3
features_per_variable = 2

# 2 batches of 3-variable data with 2 features each and a sequence length of 5
dummy_data = np.random.rand(batch_size, sequence_length, num_variables, features_per_variable)

# Reshape for LSTM input (seq_len, batch, input_size)
input_data = torch.tensor(dummy_data, dtype=torch.float32)

# Calculate input size (number of variables * number of features per variable)
input_size = num_variables * features_per_variable

# Change to (seq_len, batch_size, input_size) format
input_data = input_data.permute(1, 0, 2, 3).reshape(sequence_length, batch_size, input_size)

print("Input shape after reshaping:", input_data.shape) # torch.Size([5, 2, 6])
```

This first example outlines the initial data manipulation. The use of `permute` is crucial to rearrange dimensions in the desired order prior to reshaping. The output shows how the final shape corresponds to sequence length, batch size, and finally the aggregated number of input features.

**Example 2: Building the LSTM Model**

Here, the LSTM model is defined with the number of features, hidden size, and the number of layers. A forward method is included which takes the input tensor, passes it through the LSTM layer, and finally applies a linear layer for the output.

```python
import torch.nn as nn

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultivariateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False) # keep batch as second dim
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      # x has shape (seq_len, batch_size, input_size)
      out, _ = self.lstm(x)
      # out has shape (seq_len, batch_size, hidden_size)
      out = self.fc(out[-1, :, :]) # take last output of the sequence for prediction purposes
      # out has shape (batch_size, output_size)
      return out


# Example initialization: 6 input features, 64 hidden neurons, 2 layers, 1 output
input_size = 6
hidden_size = 64
num_layers = 2
output_size = 1

model = MultivariateLSTM(input_size, hidden_size, num_layers, output_size)
```

This second example demonstrates the architecture of the LSTM network. Notice the `batch_first=False` parameter for the LSTM layer, aligning with the shape of the input we created in the previous example where batches are in the second dimension. Finally, the output of the LSTM is passed through a fully connected layer. I used only the last output of the sequence for prediction purposes in this instance.

**Example 3: Training Loop and Inference**

This snippet illustrates a basic training loop, including a loss function, optimizer, and example inference. Crucially, the output of the model is compared to a corresponding target and backpropagated using a Mean Squared Error loss.

```python
import torch.optim as optim

# Assume input_data from previous examples
# Set hyperparameters for demonstration purposes
learning_rate = 0.01
epochs = 100
output_size = 1 # prediction is one value


# Initialize the model and optimizer
model = MultivariateLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate dummy target data (shape: batch_size, output_size)
target = torch.randn(batch_size, output_size)

# Training loop
for epoch in range(epochs):
  optimizer.zero_grad() # clear the accumulated gradients
  output = model(input_data)
  loss = criterion(output, target)
  loss.backward()
  optimizer.step()
  if (epoch+1)% 20 == 0:
      print (f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')


# Inference
with torch.no_grad():
    predicted = model(input_data)

print("Predicted Output Shape:", predicted.shape)
```

The final example showcases a straightforward training loop with forward and backward passes. Loss is computed via MSE, and gradients are updated via the Adam optimizer. The final print statement shows the predicted shape, confirming that it outputs a value per batch.  A key aspect to note is `model.eval()` or `torch.no_grad()` during the inference. This ensures the model is in evaluation mode, disabling operations like dropout.

**Resource Recommendations**

For a deeper understanding of LSTMs and recurrent neural networks, explore literature from Yoshua Bengio's group. I suggest looking into their foundational research papers on sequence learning. Furthermore, research the work on industrial applications of deep learning in time-series forecasting by researchers at MIT. Regarding PyTorch specifically, familiarize yourself with the official tutorials and documentation, particularly the sections detailing the `nn.LSTM` and `optim` modules. Finally, reading blog posts from applied deep learning groups, which often contain helpful code snippets and real-world applications, can be beneficial. These are sources I've found crucial during my own projects.
