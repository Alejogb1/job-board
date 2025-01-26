---
title: "How can Fastai be used for time series regression?"
date: "2025-01-26"
id: "how-can-fastai-be-used-for-time-series-regression"
---

Time series regression with Fastai leverages its powerful deep learning framework, primarily designed for image and natural language tasks, by adapting input data and model architectures. Unlike standard regression, time series data possesses inherent temporal dependencies, requiring specific handling to ensure model performance. I've found that careful preprocessing, selection of suitable network structures, and appropriate evaluation metrics are critical when applying Fastai in this context.

The core challenge lies in representing temporal sequences to a neural network in a way that preserves their order and relationships. Fastai's `DataLoaders` and `Transforms` facilitate this through custom data processing. For a straightforward time series, converting the sequence into a matrix, where each row represents a contiguous window of historical values, is a practical first step. Consider a time series `X = [x1, x2, x3, x4, x5, x6, x7, x8, x9]`. Using a window size of 3 and a stride of 1 will generate the following input matrix:

```
[[x1, x2, x3],
 [x2, x3, x4],
 [x3, x4, x5],
 [x4, x5, x6],
 [x5, x6, x7],
 [x6, x7, x8],
 [x7, x8, x9]]
```

Each row here is considered a sample, and the corresponding target is the value immediately following the last observation in the window. We then use Fastai’s `DataBlock` to load data with this structure. This data representation, while seemingly simplistic, enables the use of standard regression techniques.

The selection of a suitable model architecture is also important. While fully connected neural networks can approximate functions, they don’t inherently capture temporal dependencies. Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs), are designed for sequential data. For most time series regression problems I have faced, a relatively simple LSTM proves sufficient. However, temporal convolutional networks (TCNs) offer advantages in parallel processing and can be more performant for longer sequences, though I’ve found they sometimes require more meticulous hyperparameter tuning.

Here's a Python code example using Fastai to predict the next value in a univariate time series using an LSTM:

```python
from fastai.tabular.all import *
import torch
import numpy as np

# Generate a synthetic time series for demonstration
np.random.seed(42)
time_series = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.normal(0, 0.1, 200)

# Function to create sliding window data
def create_sliding_windows(data, window_size, stride=1):
  windows = []
  targets = []
  for i in range(0, len(data) - window_size, stride):
    windows.append(data[i:i+window_size])
    targets.append(data[i+window_size])
  return np.array(windows), np.array(targets)

window_size = 20
windows, targets = create_sliding_windows(time_series, window_size)

# Define the DataBlock
dblock = DataBlock(
    blocks=(RegressionBlock(), RegressionBlock()),
    get_items=lambda x: list(range(len(windows))),
    get_x=lambda i: windows[i].astype(np.float32),
    get_y=lambda i: targets[i].astype(np.float32),
    splitter=RandomSplitter(valid_pct=0.2, seed=42)
)

dls = dblock.dataloaders(list(range(len(windows))), bs=32)

# Define the Model
class LSTMModel(Module):
  def __init__(self, input_size, hidden_size, output_size):
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])  # Use the last output of the sequence
    return out


input_size = window_size
hidden_size = 64
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# Create the Learner
learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse)

# Train the Model
learn.fit_one_cycle(10, lr_max=1e-3)

# Make a prediction on a new sequence
new_sequence = time_series[-window_size:].astype(np.float32).reshape(1,-1)
prediction = learn.model(torch.tensor(new_sequence)).detach().numpy()
print(f"Prediction: {prediction[0][0]:.4f}, Actual Value: {time_series[-1]}")
```
In this code, `create_sliding_windows` creates lagged features. The `LSTMModel` comprises an LSTM layer and a linear layer. The model predicts the subsequent value using only the last output from the LSTM sequence. I use `MSELossFlat` for the loss function.

For more complex time series or multivariate problems, embedding categorical features, if present, becomes important. Consider a situation with daily sales data and corresponding day of the week as a categorical variable. Here is a second example including embeddings:

```python
from fastai.tabular.all import *
import torch
import numpy as np
import pandas as pd

# Generate synthetic sales data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=200)
sales = np.sin(np.linspace(0, 10*np.pi, 200)) * 100 + np.random.normal(0, 20, 200) + 500
df = pd.DataFrame({'date': dates, 'sales': sales})
df['dayofweek'] = df['date'].dt.dayofweek
df['dayofweek'] = df['dayofweek'].astype('category')

# Prepare data for the model
window_size = 15

def create_multivariate_windows(df, window_size, sales_col='sales', cat_col='dayofweek'):
    windows = []
    targets = []
    for i in range(0, len(df) - window_size):
        window_sales = df[sales_col].iloc[i:i+window_size].values
        window_cat = df[cat_col].iloc[i:i+window_size].values
        windows.append([window_sales, window_cat])
        targets.append(df[sales_col].iloc[i+window_size])
    return np.array(windows), np.array(targets)

windows, targets = create_multivariate_windows(df, window_size)

def get_x_multi(i):
    sales, cat = windows[i]
    return sales.astype(np.float32), cat.astype(np.int64)

def get_y_multi(i):
    return targets[i].astype(np.float32)

# Define the DataBlock for Multivariate time series
dblock = DataBlock(
    blocks=(
      (RegressionBlock(), CategoryBlock),
      RegressionBlock(),
      ),
    get_items=lambda x: list(range(len(windows))),
    get_x=get_x_multi,
    get_y=get_y_multi,
    splitter=RandomSplitter(valid_pct=0.2, seed=42)
)

dls = dblock.dataloaders(list(range(len(windows))), bs=32)

# Define the model
class MultivariateLSTM(Module):
  def __init__(self, input_size, cat_cardinality, embed_dim, hidden_size, output_size):
      self.embedding = nn.Embedding(cat_cardinality, embed_dim)
      self.lstm = nn.LSTM(input_size + embed_dim, hidden_size, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, sales_input, cat_input):
        embedded_cat = self.embedding(cat_input)
        combined_input = torch.cat((torch.tensor(sales_input).unsqueeze(-1), embedded_cat), dim=2)
        out, _ = self.lstm(combined_input)
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
embed_dim = 5
hidden_size = 64
output_size = 1
cat_cardinality = len(df['dayofweek'].cat.categories)

model = MultivariateLSTM(input_size, cat_cardinality, embed_dim, hidden_size, output_size)

# Learner
learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse)

# Training
learn.fit_one_cycle(10, lr_max=1e-3)

# Prediction
new_sales = df['sales'].values[-window_size:].astype(np.float32)
new_cat = df['dayofweek'].cat.codes.values[-window_size:].astype(np.int64)

with torch.no_grad():
  prediction = learn.model(new_sales, new_cat)

print(f"Prediction: {prediction.item():.4f}, Actual Value: {df['sales'].values[-1]:.4f}")
```
In this second example, I used a `CategoryBlock` for the 'dayofweek' feature, and the `MultivariateLSTM` has an embedding layer to process the categorical data before feeding it to the LSTM. I reshaped the numerical data to have a single input feature.

Furthermore, the use of sequence-to-sequence models is appropriate when the output is also a sequence of values. Such a model might be necessary if the goal were to predict a future time window instead of just one value.

Lastly, I have found that proper model evaluation requires more than just root mean square error (RMSE). Metrics such as mean absolute percentage error (MAPE) can provide insights into the relative error, and visualization techniques like comparing the model's forecast with actual values are crucial to validate model behavior.

Here is a final example demonstrating a TCN:
```python
from fastai.tabular.all import *
import torch
import numpy as np
from torch import nn
from torch.nn.utils import weight_norm

# Generate a synthetic time series for demonstration
np.random.seed(42)
time_series = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.normal(0, 0.1, 200)

# Function to create sliding window data
def create_sliding_windows(data, window_size, stride=1):
    windows = []
    targets = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(windows), np.array(targets)

window_size = 20
windows, targets = create_sliding_windows(time_series, window_size)


# Define the DataBlock
dblock = DataBlock(
    blocks=(RegressionBlock(), RegressionBlock()),
    get_items=lambda x: list(range(len(windows))),
    get_x=lambda i: windows[i].astype(np.float32),
    get_y=lambda i: targets[i].astype(np.float32),
    splitter=RandomSplitter(valid_pct=0.2, seed=42)
)

dls = dblock.dataloaders(list(range(len(windows))), bs=32)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    super(TemporalBlock, self).__init__()
    self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
    self.chomp1 = Chomp1d(padding)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(dropout)

    self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
    self.chomp2 = Chomp1d(padding)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout(dropout)

    self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                             self.conv2, self.chomp2, self.relu2, self.dropout2)
    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    self.relu = nn.ReLU()
    self.init_weights()

  def init_weights(self):
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv2.weight.data.normal_(0, 0.01)
    if self.downsample is not None:
        self.downsample.weight.data.normal_(0, 0.01)

  def forward(self, x):
    out = self.net(x)
    res = x if self.downsample is None else self.downsample(x)
    return self.relu(out + res)


class TCN(nn.Module):
  def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
      super(TCN, self).__init__()
      layers = []
      num_levels = len(num_channels)
      for i in range(num_levels):
          dilation_size = 2 ** i
          in_channels = input_size if i == 0 else num_channels[i-1]
          out_channels = num_channels[i]
          layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]
      self.network = nn.Sequential(*layers)
      self.fc = nn.Linear(num_channels[-1], 1)

  def forward(self, x):
    x = x.unsqueeze(1)
    out = self.network(x)
    out = out[:,:,-1].squeeze(1)
    out = self.fc(out)
    return out

num_channels = [32, 64, 128]
model = TCN(input_size=1, num_channels=num_channels, kernel_size=2, dropout=0.2)

# Create the Learner
learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse)

# Train the Model
learn.fit_one_cycle(10, lr_max=1e-3)

# Make a prediction on a new sequence
new_sequence = time_series[-window_size:].astype(np.float32).reshape(1,-1)
prediction = learn.model(torch.tensor(new_sequence)).detach().numpy()
print(f"Prediction: {prediction[0][0]:.4f}, Actual Value: {time_series[-1]}")
```
This example demonstrates a TCN for the same time series. The `TemporalBlock` and `TCN` classes build a temporal convolutional network. I found the TCN to sometimes give superior results to an LSTM for very long sequences.

For further study, I suggest focusing on "Deep Learning with Python" by Chollet for fundamental neural network concepts and a deep dive into PyTorch documentation for better understanding custom model implementations. For time series specific insights, “Time Series Analysis” by James D. Hamilton is a good resource. Also, studying publications on time series forecasting using deep learning can give insight into research trends and best practices. I’ve personally found that a combination of these resources has been most beneficial when developing my own time-series models.
