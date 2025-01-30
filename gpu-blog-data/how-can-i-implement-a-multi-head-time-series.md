---
title: "How can I implement a multi-head time series model?"
date: "2025-01-30"
id: "how-can-i-implement-a-multi-head-time-series"
---
Implementing a multi-head time series model necessitates addressing the inherent challenge of capturing diverse temporal patterns within a single sequence. The core idea revolves around training multiple independent attention mechanisms, each focused on extracting distinct features from the input time series. These features are subsequently aggregated to form a comprehensive representation suitable for downstream tasks, such as forecasting or classification.

**Explanation of Multi-Head Attention for Time Series:**

Unlike single-head attention which processes input through a single set of query, key, and value projections, multi-head attention utilizes multiple such sets in parallel. Each "head" learns a separate projection of the input, effectively allowing the model to attend to different aspects of the time series concurrently. Specifically, the input sequence undergoes linear transformations to generate Q (query), K (key), and V (value) matrices. These transformations are unique for each head and can be represented as:

```
Q_i = X * W_i^Q
K_i = X * W_i^K
V_i = X * W_i^V
```

Here, `X` is the input sequence, and `W_i^Q`, `W_i^K`, `W_i^V` are the learned weight matrices for the i-th head.

Within each head, an attention score is calculated:

```
Attention_i = softmax((Q_i * K_i^T) / sqrt(d_k)) * V_i
```

The term `d_k` represents the dimensionality of the key and query vectors within each head, which serves as a scaling factor. This scaled dot-product attention allows for the appropriate weighting of each value based on its relevance to the query.

The outputs of all heads are then concatenated and passed through another linear transformation to produce the final output:

```
MultiHeadAttentionOutput =  Concat(Attention_1, Attention_2, ..., Attention_h) * W^O
```

Where `Concat` represents the concatenation operation, `h` is the number of heads, and `W^O` is a learned weight matrix responsible for combining the head outputs. This final linear transformation reduces the dimensionality back to a desired output space.

The benefit of this architecture is its ability to capture different forms of dependencies within the sequence. For instance, one head might focus on identifying long-range dependencies, whereas another might specialize in capturing short-term trends or seasonality. This leads to more robust representations of complex time series patterns.

When integrating this mechanism into a time series model, you often find it encapsulated within encoder and decoder blocks if dealing with sequence-to-sequence tasks like forecasting, or within a feature extraction portion of a classification network.

**Code Example 1: Basic Multi-Head Attention Implementation (PyTorch)**

This code snippet illustrates the core multi-head attention mechanism without the surrounding model architecture:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
      attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
      if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
      attn_probs = F.softmax(attn_scores, dim=-1)
      return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.combine_heads(attn_output)

        output = self.W_O(attn_output)
        return output


# Example Usage
d_model = 512
num_heads = 8
seq_len = 20
batch_size = 32

multihead_attn = MultiHeadAttention(d_model, num_heads)
input_sequence = torch.randn(batch_size, seq_len, d_model)
output_sequence = multihead_attn(input_sequence)

print("Input Shape:", input_sequence.shape)
print("Output Shape:", output_sequence.shape)
```

**Commentary:** This code defines a `MultiHeadAttention` class. The `forward` method first applies linear transformations to obtain query, key, and value matrices, then splits them into multiple heads. It computes the scaled dot-product attention for each head individually, combines the head outputs and then feeds it through another linear layer. The example usage demonstrates the correct shape transformation of the input.

**Code Example 2: Multi-Head Attention Integration into a Time Series Encoder (PyTorch)**

Here, the multi-head attention is wrapped into a basic transformer encoder layer suitable for time series processing:

```python
class TimeSeriesEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TimeSeriesEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + attn_output) #Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output) #Add & Norm
        return x

class TimeSeriesEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
       super(TimeSeriesEncoder, self).__init__()
       self.layers = nn.ModuleList([TimeSeriesEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, mask=None):
      for layer in self.layers:
        x = layer(x, mask)
      return x

# Example Usage
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
seq_len = 20
batch_size = 32

encoder = TimeSeriesEncoder(d_model, num_heads, d_ff, num_layers)
input_sequence = torch.randn(batch_size, seq_len, d_model)
encoded_sequence = encoder(input_sequence)

print("Input Shape:", input_sequence.shape)
print("Encoded Shape:", encoded_sequence.shape)
```

**Commentary:** This snippet defines an encoder layer including the multi-head attention from the previous example and a feed-forward network. Layer normalization and residual connections are added for better training stability. The `TimeSeriesEncoder` class consists of multiple such layers which operate sequentially. This encapsulates the multi-head attention within a more complete time-series oriented model.

**Code Example 3: Multi-Head Attention in a Temporal Convolutional Network (TCN) Context (PyTorch)**

Multi-head attention can also be integrated with temporal convolutional networks (TCN) which are another popular architecture for time series:

```python
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        if self.residual_conv is not None:
          residual = self.residual_conv(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.norm(x + residual)
        return x

class TCNWithAttention(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, dilations, d_model, num_heads):
       super(TCNWithAttention, self).__init__()
       self.tcn_blocks = nn.ModuleList([
           TCNBlock(in_channels, channels[0], kernel_size, dilations[0]),
           TCNBlock(channels[0], channels[1], kernel_size, dilations[1]),
           TCNBlock(channels[1], d_model, kernel_size, dilations[2])
       ])
       self.attn = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
       for block in self.tcn_blocks:
         x = block(x)
       x = x.transpose(1,2) # Change to (batch, seq, d_model) for attention
       x = self.attn(x)
       return x.transpose(1,2) #Change back to (batch, d_model, seq)

# Example usage
in_channels = 1 # Single feature time series
channels = [32, 64]
kernel_size = 3
dilations = [1, 2, 4]
d_model = 512
num_heads = 8
seq_len = 100
batch_size = 32

model = TCNWithAttention(in_channels, channels, kernel_size, dilations, d_model, num_heads)
input_series = torch.randn(batch_size, in_channels, seq_len)
output_series = model(input_series)
print("Input Shape:", input_series.shape)
print("Output Shape:", output_series.shape)
```

**Commentary:** This example showcases integrating multi-head attention after several layers of TCN. The TCN blocks extract local features, and then the attention mechanism captures global dependencies over the output of the TCN. The sequence shape is transposed before and after attention application because TCN expects input as (batch, channel, seq), while attention requires (batch, seq, d_model). This example highlights the versatility of multi-head attention and its potential integration with different time series architectures.

**Resource Recommendations:**

For further study, I would recommend focusing on research papers detailing the transformer architecture and its applications to time series. Studying examples of successful implementations of multi-head attention in open-source libraries related to deep learning for time series is also highly beneficial. Additionally, explore literature covering the specifics of sequence-to-sequence models and temporal convolutional networks, particularly with a focus on how attention mechanisms augment their capabilities for time series data. Pay close attention to the hyperparameters associated with multi-head attention, such as the number of heads and the dimension of the model, and how these are selected in practical scenarios for time series.
