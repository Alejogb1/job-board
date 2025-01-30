---
title: "How can I build a multivariate time series prediction model using PyTorch?"
date: "2025-01-30"
id: "how-can-i-build-a-multivariate-time-series"
---
Multivariate time series prediction, unlike univariate, involves forecasting multiple time-dependent variables simultaneously, acknowledging their potential interdependencies. This complexity requires models capable of capturing temporal patterns across several input series and outputting corresponding predictions for each. In my experience, building such models in PyTorch benefits from a deliberate approach that leverages its flexibility for defining custom architectures and loss functions.

The core challenge lies in encoding historical information from multiple series and utilizing it for future projections. The most common method involves recurrent neural networks (RNNs), particularly Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures, due to their proven efficacy in handling sequential data. These networks maintain an internal state that summarizes past inputs, enabling them to capture long-range dependencies. For multivariate data, this requires adapting input layers to handle multiple series, which I typically accomplish by combining multiple input features at each time step rather than processing them independently. The model predicts subsequent time steps for all variables concurrently, rather than forecasting them sequentially.

Beyond basic RNNs, transformers, with their self-attention mechanism, provide a powerful alternative. They do not process sequentially, allowing for parallelization, and excel at capturing global dependencies in the data. This is especially useful when interactions between different time series are complex and not strictly time-dependent. I've found that a combination of CNN layers to initially extract local patterns from each series, followed by transformer layers to combine and predict, can often achieve better results than using either RNNs or transformers alone. Additionally, the ability to create attention masks provides a fine-grained mechanism to control which input features each output will focus on.

The selection of the loss function is another critical factor. While mean squared error (MSE) is frequently used, I often prefer mean absolute error (MAE) for its robustness to outliers. Custom loss functions tailored to the specific prediction task can also significantly improve model performance, particularly when dealing with specific constraints or business rules. For instance, if a large underprediction is more costly than a large overprediction for a specific variable, a custom weighted loss function that penalizes underpredictions more heavily can be used to align model behavior with business needs.

Here are examples illustrating several approaches:

**Example 1: LSTM-Based Multivariate Model**

This model uses an LSTM to predict multiple output variables simultaneously based on multiple input variables.

```python
import torch
import torch.nn as nn

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Using the last time step
        # out shape: (batch_size, output_size)

        return out


input_size = 3  # Number of input variables
hidden_size = 64
output_size = 2 # Number of output variables
num_layers = 2
model = MultivariateLSTM(input_size, hidden_size, output_size, num_layers)
# Example usage
batch_size = 32
seq_length = 20
input_data = torch.randn(batch_size, seq_length, input_size) # Simulated input
output = model(input_data)
print(output.shape) # Expects torch.Size([32, 2])
```

This example initializes an LSTM with a given input size, hidden size, output size, and number of layers. It processes input tensors of the shape `(batch_size, sequence_length, input_size)`. The hidden and cell states are initialized to zero at the start. The output from the LSTM is fed into a linear layer, which projects the output to the desired output size. The model returns a tensor of shape `(batch_size, output_size)`, representing a forecast for each output variable across the batch for the next single time step. I opted for the output of the last time step from the LSTM before the linear layer as opposed to processing the output across all sequence lengths. This was based on performance evaluation of the particular project.

**Example 2: Transformer-Based Multivariate Model**

This implementation uses a Transformer Encoder for predicting multiple output variables.

```python
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultivariateTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers):
        super(MultivariateTransformer, self).__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        encoder_layers = TransformerEncoderLayer(hidden_size, num_heads, hidden_size)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_linear = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = self.input_linear(x)
        # x shape: (batch_size, seq_length, hidden_size)
        x = x.permute(1,0,2)  # Change shape to: (seq_length, batch_size, hidden_size) for transformer input
        out = self.transformer_encoder(x)
        # out shape: (seq_length, batch_size, hidden_size)
        out = out.permute(1,0,2) # change back to (batch_size, seq_length, hidden_size)
        out = self.output_linear(out[:, -1, :]) #using the last time step
        # out shape: (batch_size, output_size)

        return out

input_size = 3
hidden_size = 128
output_size = 2
num_heads = 4
num_layers = 2

model = MultivariateTransformer(input_size, hidden_size, output_size, num_heads, num_layers)

# Example usage
batch_size = 32
seq_length = 20
input_data = torch.randn(batch_size, seq_length, input_size)
output = model(input_data)
print(output.shape) # Expects torch.Size([32, 2])
```

In this example, an input linear layer transforms the input to a larger dimension before passing it to a TransformerEncoder. I am using `TransformerEncoderLayer` from `torch.nn` for modularity. The output of the transformer is passed into a second linear layer to produce an output in the shape of `(batch_size, output_size)`. The shape manipulation, using `permute` methods, before passing to the transformer encoder and after obtaining the result is crucial since the transformer modules take time dimension first. I found the transformer approach to be particularly effective for datasets exhibiting non-sequential dependencies between time series as the self-attention mechanism can capture these.

**Example 3: CNN + Transformer Hybrid Model**

This example combines CNN layers for local feature extraction with a Transformer for global dependencies.

```python
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CNNTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers, num_filters=16, kernel_size=3):
        super(CNNTransformer, self).__init__()
        self.conv1d = nn.Conv1d(input_size, num_filters, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.input_linear = nn.Linear(num_filters, hidden_size)
        encoder_layers = TransformerEncoderLayer(hidden_size, num_heads, hidden_size)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
         # x shape: (batch_size, seq_length, input_size)
        x = x.permute(0,2,1)  # Change to: (batch_size, input_size, seq_length) for CNN1D input
        x = self.conv1d(x) #CNN layer
        # x shape: (batch_size, num_filters, seq_length)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  #change to (batch_size, seq_length, num_filters)

        x = self.input_linear(x)
        # x shape: (batch_size, seq_length, hidden_size)
        x = x.permute(1,0,2)
        out = self.transformer_encoder(x)
         # out shape: (seq_length, batch_size, hidden_size)
        out = out.permute(1,0,2)
        out = self.output_linear(out[:, -1, :])
         # out shape: (batch_size, output_size)
        return out


input_size = 3
hidden_size = 128
output_size = 2
num_heads = 4
num_layers = 2
num_filters = 16
kernel_size = 3

model = CNNTransformer(input_size, hidden_size, output_size, num_heads, num_layers, num_filters, kernel_size)
# Example usage
batch_size = 32
seq_length = 20
input_data = torch.randn(batch_size, seq_length, input_size)
output = model(input_data)
print(output.shape) # Expects torch.Size([32, 2])
```

This hybrid architecture first processes each input variable using a 1D convolution, to capture localized feature patterns. Afterwards, I transform the output and utilize a transformer encoder for global analysis and prediction. By extracting temporal features locally first, this approach enhances the transformer encoder's ability to capture complex long range dependencies in the processed data. I found this approach particularly advantageous for large input sequences where the interactions between series are both local and global in nature.

For further study, I would recommend exploring resources detailing time series forecasting, recurrent neural networks, transformers, and the specifics of PyTorch’s implementation of these. Texts on deep learning more broadly can also be beneficial, especially those containing information on backpropagation and optimization algorithms, which are important for model training. Articles focusing on advanced techniques in time series analysis, including topics like attention mechanisms, and the use of convolutions for processing time series data, would also be a valuable resource. Finally, case studies involving real-world multivariate time series problems would provide practical guidance on applying these techniques effectively.
