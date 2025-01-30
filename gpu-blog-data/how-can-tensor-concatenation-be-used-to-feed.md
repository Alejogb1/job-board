---
title: "How can tensor concatenation be used to feed an LSTM layer?"
date: "2025-01-30"
id: "how-can-tensor-concatenation-be-used-to-feed"
---
Tensor concatenation, particularly when preparing input for a Long Short-Term Memory (LSTM) layer, is fundamental for tasks that require combining distinct data streams or augmenting input features over time. The LSTM, inherently designed to process sequential data, benefits from a structured input format where different dimensions represent time steps or feature vectors. Concatenation enables the creation of such inputs by merging multiple tensors along a specified axis. This operation is critical for building complex architectures where the input data isn't singular but rather a combination of various, potentially heterogeneous, information sources.

My experience building time-series forecasting models, especially those incorporating external factors like holidays or promotional events, has made me acutely aware of the necessity of well-managed concatenation. LSTMs often require not just the time series itself, but also metadata or other relevant contextual information. These different inputs need to be combined into a single tensor before being passed to the LSTM. Incorrect or inefficient concatenation can lead to training instability, gradient vanishing, or even model failure due to improper alignment of time steps.

To understand the use case, consider a scenario where you have a univariate time series and a corresponding set of features that change with each time step, such as temperature or customer demographics. The time series will be of shape `(batch_size, timesteps, 1)`, and features `(batch_size, timesteps, features_size)`. An LSTM layer, assuming it is the initial layer, will require an input tensor of `(batch_size, timesteps, input_size)`. To achieve this, the tensors need to be concatenated along the `input_size` axis. This process is often applied at each step, not just once as a pre-processing task. For instance, let's consider time steps of 20, features of size 3, and an initial univariate time series.

```python
import torch

# Dummy data: batch size 4, 20 timesteps
batch_size = 4
timesteps = 20
features_size = 3

# Univariate time series (batch_size, timesteps, 1)
time_series = torch.randn(batch_size, timesteps, 1)

# Features data (batch_size, timesteps, features_size)
features = torch.randn(batch_size, timesteps, features_size)

# Concatenate along the last axis (input feature axis)
combined_input = torch.cat((time_series, features), dim=2)

print("Shape of time series:", time_series.shape)
print("Shape of features:", features.shape)
print("Shape of combined input:", combined_input.shape)

```

In this initial example, we use PyTorch to perform the tensor concatenation. The `torch.cat()` function combines the `time_series` and `features` tensors along the third dimension (axis 2). This results in a tensor with a shape of `(4, 20, 4)`, effectively merging the univariate time series feature with the 3-dimensional feature vector to give a combined vector of 4 dimensions. This result can now serve as input to the initial LSTM layer, where each time step has a vector of size 4 that is processed over time. It's critical that the concatenation is carried out on the correct dimension; if you had mistakenly concatenated on the time-step dimension, the results would be incorrect for processing by an LSTM. The proper axis for concatenation, in this case, is the feature axis.

Now consider a scenario where I am trying to combine multiple time series to predict a single time series output. We might be combining stock prices of competing companies to try to predict the price of a target company. In this case, each stock time series could be of shape `(batch_size, timesteps, 1)`. Concatenation allows merging these time series into a single input. The next example will showcase this, using three different time series, all with the same length.

```python
import torch

batch_size = 4
timesteps = 20

# Three time series: each with (batch_size, timesteps, 1)
ts1 = torch.randn(batch_size, timesteps, 1)
ts2 = torch.randn(batch_size, timesteps, 1)
ts3 = torch.randn(batch_size, timesteps, 1)

# Concatenate along feature axis (axis 2)
combined_ts = torch.cat((ts1, ts2, ts3), dim=2)

print("Shape of ts1:", ts1.shape)
print("Shape of ts2:", ts2.shape)
print("Shape of ts3:", ts3.shape)
print("Shape of combined time series:", combined_ts.shape)

```

This example builds on the first one, but instead of combining a time series with a set of features, it combines three different time series together along the feature axis. The output is now a tensor of shape `(4, 20, 3)`, which can be passed to an LSTM network. Again, the crucial point is the choice of axis. Concatenating along axis 1 would combine the time steps of different series, which is usually not the intended outcome. This also showcases how the input size of the LSTM layer is determined by the number of combined features, which in this case is 3.

Finally, let's consider how concatenation can be used to create inputs with varying lengths for each data point in a batch. This is often required when the length of the sequential data varies across samples. Although LSTMs usually operate on sequences of fixed lengths, masking allows us to pad sequences with zeros to a uniform length and then ignore the padded sections. This can be combined with tensor concatenation to ensure all data points within a batch are the same shape.

```python
import torch
import torch.nn as nn

batch_size = 2
max_len = 25

# Variable length time series (lengths = [15, 25])
lengths = [15, 25]
ts1 = torch.randn(1, lengths[0], 1)
ts2 = torch.randn(1, lengths[1], 1)

# Pad time series to max length
padded_ts1 = nn.functional.pad(ts1, (0, 0, 0, max_len - lengths[0]))
padded_ts2 = ts2
padded_ts = torch.cat((padded_ts1,padded_ts2), dim=0)

# Create corresponding feature for each timesteps
features_1 = torch.randn(1, lengths[0], 2)
features_2 = torch.randn(1, lengths[1], 2)
padded_features_1 = nn.functional.pad(features_1, (0, 0, 0, max_len - lengths[0]))
padded_features_2 = features_2

padded_features = torch.cat((padded_features_1, padded_features_2), dim = 0)


combined_input = torch.cat((padded_ts, padded_features), dim=2)

print("Shape of padded time series:", padded_ts.shape)
print("Shape of padded features:", padded_features.shape)
print("Shape of combined input:", combined_input.shape)
```
In this final example, we are dealing with time series of different lengths. First, we pad the smaller time series to be the same length as the longest, which is 25 in this case. Then, we concatenate along the 0 axis, so that each tensor is stacked into a batch of shape `(batch_size, max_len, 1)`. Finally, we create corresponding features, pad them, and stack the tensors together before concatenating along the feature axis to combine the time series and features for every batch sample and timestep. We now have input data in the shape of `(2, 25, 3)` suitable for an LSTM with an input size of 3. Note, that masking would be needed to ignore the padded inputs during training.

In summary, tensor concatenation is not a standalone solution, but rather a crucial part of data preparation pipelines when building complex LSTM models. It allows the creation of multi-feature input vectors where different time series and feature data are combined into one input. Without proper understanding and implementation of this operation, models may suffer from performance issues, so it is crucial to understand the shape of your data and to perform concatenation along the intended axis.

For further study, I recommend exploring resources covering deep learning with recurrent neural networks and focusing on practical use cases of LSTMs for time series analysis and feature fusion. Materials related to PyTorch's tensor operations will also be invaluable, as well as material on variable length sequences and masking in RNNs. Consulting documentation on working with sequential data will also provide more insights.
