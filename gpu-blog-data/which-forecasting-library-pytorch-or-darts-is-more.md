---
title: "Which forecasting library, PyTorch or Darts, is more effective?"
date: "2025-01-30"
id: "which-forecasting-library-pytorch-or-darts-is-more"
---
Having spent the last three years developing time-series forecasting models for a large-scale logistics network, I've found that "more effective" is heavily dependent on the specific context and requirements. There isn't a universally superior library between PyTorch and Darts; they cater to different needs and offer varying levels of abstraction. My experience suggests that PyTorch provides more granular control and flexibility, while Darts streamlines the process, offering rapid prototyping with a wide range of established models.

PyTorch, at its core, is a deep learning framework. When forecasting with PyTorch, you are fundamentally building and training neural networks. This means complete customization, allowing for intricate architectures like recurrent neural networks (RNNs), Transformers, or hybrid models that combine aspects of both. The benefit is unparalleled flexibility; you have direct control over the optimization process, loss functions, and network structure. However, this power comes with a steeper learning curve and a significant investment in development time. You must meticulously define each layer, handle data preprocessing manually, and implement training loops. PyTorch does not provide out-of-the-box solutions specifically optimized for time series. Thus, achieving state-of-the-art results often requires a strong foundation in deep learning principles.

Darts, conversely, is specifically designed for time series forecasting. It's built atop other popular frameworks, including PyTorch and scikit-learn, providing a higher level of abstraction. Darts comes equipped with implementations of classic statistical models (like ARIMA and Exponential Smoothing) and numerous deep learning models (e.g., RNNs, Transformers) ready to be used with minimal configuration. This enables rapid experimentation and prototyping. Data preprocessing and evaluation are also simplified with its built-in functionalities for handling temporal data structures and performance metrics. The focus is on rapid deployment and ease of use, allowing data scientists to quickly iterate through different model architectures without delving into low-level implementation details. However, customization is limited compared to PyTorch. Complex, bespoke model architectures or highly unconventional training procedures may not be readily achievable or require a workaround.

Here are some examples to illustrate the practical differences:

**Example 1: Simple RNN with PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Dummy time series data
time_series = np.random.rand(100, 1).astype(np.float32)
seq_length = 20 # Input sequence length
X = []
y = []
for i in range(len(time_series) - seq_length):
    X.append(time_series[i:i+seq_length])
    y.append(time_series[i+seq_length])

X = torch.tensor(np.array(X))
y = torch.tensor(np.array(y))

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) # Use only last timestep output
        return out

model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop (truncated for brevity)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction
with torch.no_grad():
    test_input = X[-1].unsqueeze(0)  # Use last sequence as test input
    forecast = model(test_input).item()
    print(f"Forecast: {forecast:.4f}")
```

This example demonstrates the manual implementation required using PyTorch. Defining the RNN class, setting up the loss function, optimizer, and writing the training loop is necessary. I must also take care to ensure the data is properly shaped and passed through the layers correctly.

**Example 2: Simple RNN with Darts**

```python
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.utils.timeseries_generation import linear_timeseries

# Dummy data
data = linear_timeseries(length=100, start_value=0, end_value=10).values
series = TimeSeries.from_values(data)

# Split into train/test
train, val = series.split_before(80)

# Define and train the model
model = RNNModel(
    model="RNN",
    hidden_size=16,
    n_epochs=100,
    random_state=42,
    batch_size=16,
)

model.fit(train, val_series=val, verbose=False)

# Forecast
forecast = model.predict(n=20)
print(f"Forecast (first 5): {forecast.values()[:5]}")
```

Here, Darts greatly simplifies the process. I initialize a `TimeSeries` object, split the data, initialize a pre-built `RNNModel`, and then fit and forecast with a few lines of code. There is no explicit need to define the network structure or worry about backpropagation, as those are handled internally by Darts.

**Example 3: Custom Loss Function with PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Assuming X and y data from previous PyTorch example

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    def forward(self, predicted, target):
        # Asymmetric Loss - Penalizing under-prediction more than over-prediction
         error = predicted - target
         loss = torch.mean(torch.where(error > 0, 0.5 * error**2, 2 * error**2))
         return loss

# Define model (Using SimpleRNN from previous example)
model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)
criterion = CustomLoss()  #Using CustomLoss here
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#Prediction
with torch.no_grad():
    test_input = X[-1].unsqueeze(0)  # Use last sequence as test input
    forecast = model(test_input).item()
    print(f"Forecast: {forecast:.4f}")
```

This demonstrates a key advantage of PyTorch, namely the ability to incorporate custom logic. Here, I have defined a custom loss function, which is easily integrated into the PyTorch training loop. This allows for specialized modeling considerations and control over the specific aspects of error that we care about penalizing. Darts models offer configurable loss functions, but do not provide this level of customization.

When choosing between these libraries, I consider my priorities. For projects requiring maximal flexibility, the capability to explore unique model architectures, or a need for specialized loss functions, PyTorch is the stronger choice, although demanding more coding investment. Projects with tight timelines where rapid experimentation and deployment of common time series models are paramount, Darts proves more efficient. For highly complex, custom deep learning tasks, the flexibility afforded by PyTorch is crucial, while the ready-made, out-of-the-box features of Darts shine in the context of rapid prototyping and testing.

For further learning, I recommend exploring the foundational texts on deep learning, such as those covering neural networks and time series analysis. Also, consulting the documentation of both PyTorch and Darts is highly beneficial. There are also numerous online courses and blog posts which can provide practical implementation insights. Examining specific use-case examples online can help solidify understanding of the relative strengths and weaknesses of each library.
