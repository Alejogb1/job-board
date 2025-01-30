---
title: "How many FLOPs and parameters does a 1D CNN have in PyTorch?"
date: "2025-01-30"
id: "how-many-flops-and-parameters-does-a-1d"
---
The computational cost of a 1D Convolutional Neural Network (CNN) in PyTorch, quantified by FLOPs (floating-point operations) and the number of parameters, is not a simple function of the layer dimensions alone. It depends critically on the specifics of the architecture: kernel size, number of channels in input and output layers, padding, stride, and the presence of other layers like fully connected layers and activation functions.  My experience optimizing such networks for embedded systems has highlighted this complexity repeatedly.  Precise computation requires careful analysis of each layer's operations.


**1.  A Clear Explanation of FLOP and Parameter Calculation**

Let's consider a single 1D convolutional layer. The number of parameters is determined by the kernel size (K), the number of input channels (C_in), and the number of output channels (C_out). Each filter in the output channel requires K * C_in weights plus a bias term. Since there are C_out output channels, the total number of parameters is:

`Parameters = C_out * (K * C_in + 1)`


Calculating FLOPs is more involved.  For a single convolution operation on a single input channel, a kernel of size K requires K multiplications and K-1 additions.  Therefore, for a single output channel, the number of Multiply-Accumulate (MAC) operations per input data point is K * C_in.  Consider an input sequence of length L.  After considering the effect of padding (P) and stride (S), the output sequence length L_out is given by:


`L_out = floor((L + 2P - K) / S) + 1`


The total number of MAC operations for a single output channel across the entire input sequence is then approximately L_out * K * C_in.  Extending this to all output channels, we get:


`MACs (per layer) ≈ C_out * L_out * K * C_in`


The approximation arises from ignoring operations involved in bias addition and activation functions.  In reality, the activation function (e.g., ReLU) adds a comparable number of operations, while bias addition adds a relatively small number of operations compared to the convolution itself.  A more precise count would necessitate analyzing the specific activation function's computational overhead. Furthermore, this calculation doesn't encompass operations in subsequent layers or fully connected layers.  For a complete network, we need to sum the FLOPs of all layers.


**2. Code Examples with Commentary**

The following examples illustrate the computation for different architectures, highlighting the influence of architectural choices. I've used simplified approaches for FLOP estimations in these examples, focusing on the dominant MAC operations.


**Example 1: A Simple 1D CNN**

```python
import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)

    def forward(self, x):
        return self.conv1(x)

# Example parameters
input_channels = 3
output_channels = 16
kernel_size = 5
input_length = 100

model = Simple1DCNN(input_channels, output_channels, kernel_size)

# Parameter count
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {params}")

# FLOP estimation (simplified) - assumes no padding or stride
input_tensor = torch.randn(1, input_channels, input_length)
output_tensor = model(input_tensor)
flops_approx = output_tensor.numel() * kernel_size * input_channels * output_channels
print(f"Approximate FLOPs: {flops_approx}")
```


This code provides a basic 1D CNN with one convolutional layer.  The parameter count is directly calculated.  The FLOP estimation is simplified and ignores padding and stride effects.  The `numel()` function gives the number of elements in the tensor, which is used as a proxy for the output length.  Remember that this is a very rough estimate.


**Example 2:  1D CNN with Pooling and Fully Connected Layer**

```python
import torch
import torch.nn as nn

class Extended1DCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, pool_kernel_size, fc_size):
        super(Extended1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.pool = nn.MaxPool1d(pool_kernel_size)
        self.fc = nn.Linear(output_channels * (input_length - kernel_size + 1)//pool_kernel_size , fc_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(-1, self.fc.in_features) # Flatten
        x = self.fc(x)
        return x

# Example Parameters
input_channels = 1
output_channels = 32
kernel_size = 3
pool_kernel_size = 2
fc_size = 10
input_length = 100

model = Extended1DCNN(input_channels, output_channels, kernel_size, pool_kernel_size, fc_size)

#Parameter count
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {params}")

#FLOP estimation (simplified) - still ignoring some complexities
input_tensor = torch.randn(1,input_channels, input_length)
output_tensor = model(input_tensor)
#Rudimentary FLOP approximation (ignores pooling, focuses on conv and fc)
conv_flops = output_tensor.numel() * kernel_size * input_channels * output_channels
fc_flops = model.fc.in_features * fc_size + fc_size #Simplified linear layer FLOPS
flops_approx = conv_flops + fc_flops
print(f"Approximate FLOPs: {flops_approx}")

```

This adds a max-pooling layer and a fully connected layer, significantly increasing complexity. The FLOP estimation becomes even more approximate, neglecting the pooling layer's operations almost entirely.   Precise FLOP accounting for pooling requires careful consideration of its implementation details.  Also note the calculation for the fully connected layer FLOPS, using a simplification common in practice.


**Example 3: Utilizing `torchinfo` for More Accurate Estimation**

```python
import torch
import torch.nn as nn
from torchinfo import summary

# (Define the model as in Example 1 or 2)

input_size = (1, input_channels, input_length)
summary(model, input_size=input_size)
```

The `torchinfo` library provides a more comprehensive model summary, including estimates for the number of parameters and FLOPs. While still an approximation, `torchinfo` is considerably more accurate than manual calculations, especially for complex networks. This requires installation (`pip install torchinfo`).


**3. Resource Recommendations**

For more in-depth understanding of CNN architectures and computational analysis, I would recommend studying standard textbooks on deep learning and exploring research papers on efficient CNN implementations for resource-constrained environments. Examining the source code of established deep learning libraries such as PyTorch and TensorFlow, paying particular attention to the implementation of convolutional layers, will prove invaluable.  Understanding numerical linear algebra will improve your comprehension of the underlying mathematical operations.  Finally, studying compiler optimization techniques can provide insights into how operations are actually executed.
