---
title: "How can I process individual time steps through a feedforward network before an RNN?"
date: "2025-01-30"
id: "how-can-i-process-individual-time-steps-through"
---
I've frequently encountered scenarios where a feedforward network's output needs to be fed into an RNN, particularly when pre-processing complex, high-dimensional input data or deriving crucial intermediate features prior to sequential modeling. This approach allows for a more modular and potentially powerful architecture, leveraging the strengths of both network types. The challenge often lies in ensuring that the data is reshaped correctly between these distinct model components and that the relevant time-step information is preserved before entering the RNN.

Essentially, the process involves applying a feedforward network independently to each time step within a sequence. This creates a new sequence of processed data that can then be consumed by an RNN. Crucially, this is *not* the same as flattening the entire sequence and pushing it through a feedforward network, which would lose the temporal relationships. The goal is to maintain the sequence structure, ensuring that the RNN can learn from the temporal dependencies in the data. We want the feedforward network to act as a feature extractor at each time step.

Consider a dataset with time-series data, where each time step is represented as a vector of features. Our feedforward network acts upon each of these individual time-step vectors, producing a new representation of that time step. This new representation then becomes the input for our recurrent network. This decoupling allows for complex, nonlinear feature mapping via the feedforward network, which can then inform the RNN's sequential analysis. The feedforward network is trained concurrently with the RNN as a single model, allowing both parts to optimize for the final task. The entire pipeline is designed to improve the RNN's ability to capture patterns in the sequential data by providing it with more suitable features.

Let's illustrate this with a few practical code examples, primarily using Python with libraries like PyTorch, which I typically use for such projects.

**Example 1: A Basic Time-Step Processing Approach**

This example demonstrates the fundamental concept of processing each time step independently with a feedforward network and then feeding the results into a basic RNN:

```python
import torch
import torch.nn as nn

class TimeStepProcessor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, ff_hidden_size):
        super(TimeStepProcessor, self).__init__()
        self.ff_network = nn.Sequential(
            nn.Linear(input_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, hidden_size) # output dimension of ff network equals input dimension of RNN
        )
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1) # example output layer for regression


    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        batch_size, seq_length, input_size = x.shape
        
        processed_seq = []
        for t in range(seq_length):
            timestep_data = x[:, t, :] # shape: (batch_size, input_size)
            processed_timestep = self.ff_network(timestep_data) # shape: (batch_size, hidden_size)
            processed_seq.append(processed_timestep)
        
        processed_seq = torch.stack(processed_seq, dim=1) # shape (batch_size, seq_length, hidden_size)

        rnn_out, _ = self.rnn(processed_seq)  # rnn_out: (batch_size, seq_length, hidden_size)
        output = self.output_layer(rnn_out[:, -1, :]) # Example: output only from the last timestep
        return output
    
# Example Usage
input_size = 10
hidden_size = 20
num_layers = 1
ff_hidden_size = 30

model = TimeStepProcessor(input_size, hidden_size, num_layers, ff_hidden_size)
batch_size = 32
seq_length = 5
input_data = torch.randn(batch_size, seq_length, input_size)

output = model(input_data)
print("Output Shape:", output.shape) # Expected: Output Shape: torch.Size([32, 1])

```

In this example, the `TimeStepProcessor` class encapsulates both the feedforward and recurrent networks. Inside the `forward` method, we iterate through each time step of the input tensor `x`. At each iteration, the corresponding time slice is passed through the `ff_network`, which performs a non-linear transformation to generate a `hidden_size` vector representation. Finally, we stack the transformed time steps and use the new sequence as input for the RNN. The final layer produces the output of the whole network. The sequential processing is handled with a simple `for` loop, emphasizing the clarity and directness of implementation, which in practice can be inefficient for extremely long sequences.

**Example 2: Optimized Time-Step Processing using `torch.vmap`**

Here, we leverage PyTorch's `vmap` for vectorized processing across time steps, improving the execution efficiency. I have personally found this method to provide significant speedup for larger sequences.

```python
import torch
import torch.nn as nn
from torch.func import vmap

class TimeStepProcessorOptimized(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, ff_hidden_size):
        super(TimeStepProcessorOptimized, self).__init__()
        self.ff_network = nn.Sequential(
            nn.Linear(input_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, hidden_size) # output dimension of ff network equals input dimension of RNN
        )
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1) # example output layer for regression
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        
        # vmap over seq_length dimension to avoid explicit loops
        processed_seq = vmap(self.ff_network, in_dims=1, out_dims=1)(x) # Shape: (batch_size, seq_length, hidden_size)

        rnn_out, _ = self.rnn(processed_seq) # rnn_out: (batch_size, seq_length, hidden_size)
        output = self.output_layer(rnn_out[:, -1, :]) # Example: output only from the last timestep

        return output
    
# Example Usage
input_size = 10
hidden_size = 20
num_layers = 1
ff_hidden_size = 30

model = TimeStepProcessorOptimized(input_size, hidden_size, num_layers, ff_hidden_size)
batch_size = 32
seq_length = 5
input_data = torch.randn(batch_size, seq_length, input_size)

output = model(input_data)
print("Output Shape:", output.shape) # Expected: Output Shape: torch.Size([32, 1])

```

The key modification here is the use of `vmap`. It applies the feedforward network to each time step in a vectorized way, which is generally much faster than an explicit loop in pure python. This version achieves the same result as Example 1 but with improved efficiency, especially for larger datasets, where the use of loops would significantly slow down computation. Note that the `in_dims=1` and `out_dims=1` arguments to `vmap` specify that the input and output tensors should be vectorized along the second dimension (index 1), which corresponds to `seq_length` dimension.

**Example 3: Using a Convolutional Layer Instead of a Fully Connected Layer**

Sometimes a convolutional approach can be beneficial. I often use temporal convolutions as an alternative to fully connected feedforward networks before RNN processing, especially with time series where local patterns are important within a single time step:

```python
import torch
import torch.nn as nn

class TimeStepProcessorConv(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, conv_channels):
        super(TimeStepProcessorConv, self).__init__()
        self.conv_network = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1), # padding preserves sequence length
            nn.ReLU(),
            nn.Conv1d(conv_channels, hidden_size, kernel_size=3, padding=1)
        )
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1) # example output layer for regression

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)

        # convert to N, C, L format expected by Conv1D layers
        x_conv = x.permute(0, 2, 1) # (batch_size, input_size, seq_length)
        processed_seq = self.conv_network(x_conv) # shape: (batch_size, hidden_size, seq_length)

        # convert back to N, L, C expected by the RNN
        processed_seq = processed_seq.permute(0, 2, 1) # shape: (batch_size, seq_length, hidden_size)


        rnn_out, _ = self.rnn(processed_seq) # rnn_out: (batch_size, seq_length, hidden_size)
        output = self.output_layer(rnn_out[:, -1, :]) # Example: output only from the last timestep
        return output

# Example Usage
input_size = 10
hidden_size = 20
num_layers = 1
conv_channels = 16

model = TimeStepProcessorConv(input_size, hidden_size, num_layers, conv_channels)
batch_size = 32
seq_length = 5
input_data = torch.randn(batch_size, seq_length, input_size)

output = model(input_data)
print("Output Shape:", output.shape) # Expected: Output Shape: torch.Size([32, 1])
```

In this example, we substitute the fully connected network with `nn.Conv1d` layers. It uses 1D convolution to extract feature at each time step which can be more efficient for structured inputs. The key part of this example is the data permutation needed to be compliant with PyTorch's convention of the input data format of a 1D convolutional layer as (N, C, L), where N is batch size, C is number of channels, and L is sequence length. Note that the output of the convolutional block is also permuted before being fed into the RNN block. The padding ensures that the sequence length remains the same after passing through the convolutional block.

For further understanding, delving into literature concerning hybrid neural network architectures would be helpful. Examining the practical implementations of signal processing techniques integrated with deep learning would also deepen one's understanding. The PyTorch documentation, particularly regarding `vmap` and the different neural network layers, is invaluable. Reading research papers focusing on sequence modeling and time-series analysis can also provide insights into different architectural choices and their respective benefits. Exploring tutorials dedicated to both feedforward networks and recurrent networks can provide a more holistic understanding of the specific mechanics.
