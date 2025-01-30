---
title: "How do I correctly connect the final linear layer in a PyTorch LSTM model for multivariate time series forecasting?"
date: "2025-01-30"
id: "how-do-i-correctly-connect-the-final-linear"
---
The final linear layer in a PyTorch LSTM model for multivariate time series forecasting requires careful consideration of the output shape from the LSTM and the desired shape of the prediction. Specifically, ensuring the linear layer maps the LSTM's hidden state or output at the final time step to the correct number of prediction features is paramount. Overlooking this leads to shape mismatches and incorrect training behavior.

When designing an LSTM for multivariate time series forecasting, I often see the confusion stem from how PyTorch handles sequences and how the LSTM internally operates. The LSTM processes sequences, but for forecasting, often, we are interested in a single prediction for the next time step (or a window of future steps). This means we need to carefully select which part of the LSTM's output we use and transform it with the final linear layer.

Let me first delineate the common setup. Typically, your input will be a tensor of shape `(batch_size, seq_length, num_features)`, where `batch_size` is the number of independent samples in your mini-batch, `seq_length` represents the length of the input sequence or the "lookback" window and `num_features` is the number of input variables within each time step. The LSTM's hidden states and cell states are themselves tensors and managed internally. An LSTM layer outputs a tensor of shape `(batch_size, seq_length, hidden_size)` if `batch_first=True` in your LSTM initialization, where `hidden_size` represents the dimensionality of the hidden state. Otherwise the output is `(seq_length, batch_size, hidden_size)`.

For many forecasting use cases, we are interested in using the last time step of the LSTM output as our input to the final linear layer which would be `(batch_size, hidden_size)`. This is because, given we fed an input sequence through the LSTM, the hidden state at the last timestep is a condensed representation of the entire sequence. This last step hidden state can therefore be used as the input for the linear layer, which will project it to the desired prediction output size.

The linear layer should have an input feature size equal to the `hidden_size` of the LSTM. The output feature size of this final layer is critical. It should match the number of features in the forecasting output. If you're predicting a single value for each of your input variables, the output feature size is same as the number of input features, `num_features`. if you want to predict multiple steps into the future and for multiple variables, your output size should match `num_features * forecast_horizon`. Let us assume for now that you are predicting single value for each of your input variables, so your final output has the same number of features as your input.

Here are some code examples with detailed explanations:

**Example 1: Predicting a Single Step Ahead (Matching input feature size)**

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        # x.shape = (batch_size, seq_length, num_features)
        lstm_out, _ = self.lstm(x)
        # lstm_out.shape = (batch_size, seq_length, hidden_size)
        last_step_output = lstm_out[:, -1, :]
        # last_step_output.shape = (batch_size, hidden_size)
        output = self.linear(last_step_output)
        # output.shape = (batch_size, num_features)
        return output

# Example Usage
num_features = 5
hidden_size = 64
batch_size = 32
seq_length = 20

model = LSTMForecaster(num_features, hidden_size)
input_tensor = torch.randn(batch_size, seq_length, num_features)
output_tensor = model(input_tensor)

print(output_tensor.shape) # Output: torch.Size([32, 5])
```

In this first example, the `LSTMForecaster` class initializes an LSTM layer with `batch_first=True`, meaning the batch dimension is the first. The forward method passes the input through the LSTM. It then extracts the output at the final time step using the slicing operation `lstm_out[:, -1, :]` ensuring we work with the shape `(batch_size, hidden_size)`. This last step's hidden state is passed to the final linear layer to project it to the desired output with the shape `(batch_size, num_features)`. Critically the linear layer maps the `hidden_size` to `num_features`, because in this example, we are predicting a single value for each input variable.

**Example 2: Predicting Multiple Steps Ahead (Flattened)**

```python
import torch
import torch.nn as nn

class LSTMForecasterMultiStepFlattened(nn.Module):
    def __init__(self, num_features, hidden_size, forecast_horizon):
        super(LSTMForecasterMultiStepFlattened, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_features * forecast_horizon)
        self.forecast_horizon = forecast_horizon


    def forward(self, x):
         # x.shape = (batch_size, seq_length, num_features)
        lstm_out, _ = self.lstm(x)
         # lstm_out.shape = (batch_size, seq_length, hidden_size)
        last_step_output = lstm_out[:, -1, :]
        # last_step_output.shape = (batch_size, hidden_size)
        output = self.linear(last_step_output)
        # output.shape = (batch_size, num_features * forecast_horizon)
        return output

# Example Usage
num_features = 5
hidden_size = 64
batch_size = 32
seq_length = 20
forecast_horizon = 3

model = LSTMForecasterMultiStepFlattened(num_features, hidden_size, forecast_horizon)
input_tensor = torch.randn(batch_size, seq_length, num_features)
output_tensor = model(input_tensor)

print(output_tensor.shape) # Output: torch.Size([32, 15])
```

This example demonstrates predicting multiple steps into the future, which is often referred to as a forecast horizon. Here, the linear layer is changed to project the hidden state to a flattened output of size `num_features * forecast_horizon`, where `forecast_horizon` represents how far ahead we want to predict. The linear layer's output dimension, in this case, is a flattened representation of the predicted future values. In a subsequent step you may reshape this.

**Example 3: Predicting Multiple Steps Ahead (Separate Linear Layer)**

```python
import torch
import torch.nn as nn

class LSTMForecasterMultiStepSeparate(nn.Module):
    def __init__(self, num_features, hidden_size, forecast_horizon):
        super(LSTMForecasterMultiStepSeparate, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, num_features)
                                        for _ in range(forecast_horizon)])
        self.forecast_horizon = forecast_horizon


    def forward(self, x):
        # x.shape = (batch_size, seq_length, num_features)
        lstm_out, _ = self.lstm(x)
        # lstm_out.shape = (batch_size, seq_length, hidden_size)
        last_step_output = lstm_out[:, -1, :]
        # last_step_output.shape = (batch_size, hidden_size)
        outputs = []
        for layer in self.linear_layers:
            outputs.append(layer(last_step_output))
        # outputs is list of length forecast_horizon, each element (batch_size, num_features)
        output = torch.stack(outputs, dim=1)
        # output.shape = (batch_size, forecast_horizon, num_features)

        return output

# Example Usage
num_features = 5
hidden_size = 64
batch_size = 32
seq_length = 20
forecast_horizon = 3

model = LSTMForecasterMultiStepSeparate(num_features, hidden_size, forecast_horizon)
input_tensor = torch.randn(batch_size, seq_length, num_features)
output_tensor = model(input_tensor)

print(output_tensor.shape) # Output: torch.Size([32, 3, 5])
```

In this third variation, we create a list of linear layers using `nn.ModuleList`, which is the recommended way for storing lists of modules. Each linear layer in this list maps the hidden state from the LSTM (`hidden_size`) to a prediction vector equal to the number of input features (`num_features`). We apply all these linear layers to the same final hidden state from the LSTM, stacking the resulting predictions using `torch.stack`. As a result, the final output is a tensor with shape `(batch_size, forecast_horizon, num_features)`, providing a clear structure for multi-step forecasts where each step maintains the shape of the input variables.

In terms of resources, thoroughly reviewing the official PyTorch documentation on `nn.LSTM` and `nn.Linear` is indispensable. Furthermore, articles and tutorials discussing sequence-to-sequence modeling with attention mechanisms (even if not directly used) can provide better context into the underlying concepts around sequence processing. Examining implementations of similar models within popular time series forecasting libraries can be another invaluable learning approach. Finally, understanding the concepts of recurrent neural networks at a more theoretical level can also aid in avoiding common mistakes, this would involve research papers on LSTMs and their application to timeseries forecasting.
