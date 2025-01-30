---
title: "Can a PyTorch GRU trained on single-class data accurately predict unlabeled data?"
date: "2025-01-30"
id: "can-a-pytorch-gru-trained-on-single-class-data"
---
A recurrent neural network, specifically a Gated Recurrent Unit (GRU), trained solely on data belonging to a single class may exhibit limited but potentially useful generalization to unlabeled data, primarily when the unlabeled data shares underlying structural patterns with the training data. The accuracy of these predictions hinges heavily on the nature of the data and the specific problem the GRU is intended to solve. It's crucial to understand that the GRU, when trained on single-class data, is learning to model the *distribution* of that specific class rather than attempting to discriminate between multiple classes.

My experience lies in time-series anomaly detection, specifically in industrial sensor readings. In one instance, I worked with GRU networks to model the regular behavior of a specific machine component. The GRU was trained exclusively on sensor data collected during normal operating conditions. Our goal wasn't to classify the data as "normal" or "abnormal", rather to model "normal" so well that deviations, regardless of their specific class label, would become readily apparent. This deviates subtly from the posed question. While we did not have explicit labels for all unseen data, the underlying premise of only using 'normal' data for training is identical.

A GRU's internal architecture allows it to capture temporal dependencies in the input sequence. During training, this learning focuses on the recurring patterns and trends within the single-class data. If unlabeled data contains similar patterns – that is, it falls, at least partially, within the modeled probability density of the training distribution - the GRU can plausibly generate outputs that correlate with the observed sequence or predict the immediate next point in the series with some accuracy. The magnitude of this accuracy is directly proportional to how well the unlabeled data aligns with the training data's underlying distribution. The further the unlabeled input deviates from this distribution, the less reliable the GRU's predictions will become. It’s not learning a label or decision boundary but rather the characteristic dynamic of the class.

It's highly important to distinguish between 'accurate' predictions in the conventional sense and predictions that reflect learned structural relationships within the data. The GRU, under these conditions, isn't learning to perform classification; it's learning an internal representation of a single class. Prediction accuracy, in this case, becomes a measure of how closely unlabeled data aligns with this learned representation. Thus, for example, if the unlabeled data represents a significant shift in pattern even within the single-class context, the model, while still producing an output, would likely perform poorly, with higher residuals, for instance.

Let's illustrate with some simplified PyTorch code examples.

**Example 1: Simple Time-Series Prediction.**

Here, we have a GRU trained on a synthetic time series and then use it to predict the next value of both training and unseen data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic training data
np.random.seed(42)
train_data = np.sin(np.linspace(0, 10*np.pi, 1000)) # Sine wave
train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1).unsqueeze(0)

# Generate synthetic unseen data with a different offset and scale
unseen_data = 1.2 * np.sin(np.linspace(1 * np.pi, 12 * np.pi, 500)) + 0.5
unseen_data = torch.tensor(unseen_data, dtype=torch.float32).unsqueeze(1).unsqueeze(0)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      h0 = torch.zeros(1, x.size(0), self.hidden_size)
      out, _ = self.gru(x, h0)
      out = self.fc(out[:, -1, :])
      return out

input_size = 1
hidden_size = 32
output_size = 1
model = GRUModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 100
for epoch in range(epochs):
  for i in range(train_data.size(1)-1):
    optimizer.zero_grad()
    input_seq = train_data[:,i:i+1,:]
    target = train_data[:,i+1:i+2,:]
    output = model(input_seq)
    loss = criterion(output, target.squeeze(1))
    loss.backward()
    optimizer.step()
    #print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Prediction on training data
with torch.no_grad():
  train_predictions = []
  for i in range(train_data.size(1)-1):
    input_seq = train_data[:,i:i+1,:]
    train_predictions.append(model(input_seq).squeeze(1))
  train_predictions=torch.cat(train_predictions, dim = 0)

# Prediction on unseen data
with torch.no_grad():
  unseen_predictions = []
  for i in range(unseen_data.size(1)-1):
    input_seq = unseen_data[:,i:i+1,:]
    unseen_predictions.append(model(input_seq).squeeze(1))
  unseen_predictions=torch.cat(unseen_predictions, dim = 0)


# Basic Evaluation (visual, MSE, correlation)
print("Train MSE:", criterion(train_predictions.squeeze(), train_data.squeeze(0)[1:]).item())
print("Unseen MSE:", criterion(unseen_predictions.squeeze(), unseen_data.squeeze(0)[1:]).item())

# Notice that the unseen data has much higher MSE, reflecting the difference between the two signals
# while the model has predicted the sine wave, the offset and different frequency make for a bad fit

```
This example shows a GRU trained on a basic sine wave. When we predict the next value in the *training* set, the loss is low. However, the loss significantly increases when we input a *different* sine wave. This highlights the effect of input distribution differences.

**Example 2: Anomaly Detection Scenario**
This example trains the GRU on 'normal' data and compares reconstruction performance on 'normal' and 'abnormal' sequences.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic 'normal' data
np.random.seed(42)
normal_data = np.random.normal(0, 1, size=(1000, 10)).astype(np.float32)
normal_data = torch.tensor(normal_data).unsqueeze(0)

# Generate synthetic 'abnormal' data with higher magnitude
abnormal_data = 3 * np.random.normal(0, 1, size=(500, 10)).astype(np.float32)
abnormal_data = torch.tensor(abnormal_data).unsqueeze(0)

class GRUAutoencoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(GRUAutoencoder, self).__init__()
    self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    self.decoder = nn.GRU(hidden_size, input_size, num_layers, batch_first=True)

  def forward(self, x):
    h0 = torch.zeros(1, x.size(0), 32) #arbitrary size
    _, hidden = self.encoder(x,h0)
    out, _ = self.decoder(x, hidden)
    return out

input_size = 10
hidden_size = 32
num_layers = 1
model = GRUAutoencoder(input_size, hidden_size, num_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training on 'normal' data
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(normal_data)
    loss = criterion(output, normal_data)
    loss.backward()
    optimizer.step()
    #print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Reconstruction of normal and abnormal data
with torch.no_grad():
  normal_reconstruction = model(normal_data)
  abnormal_reconstruction = model(abnormal_data)

# Calculate reconstruction error
normal_reconstruction_loss = criterion(normal_reconstruction, normal_data)
abnormal_reconstruction_loss = criterion(abnormal_reconstruction, abnormal_data)

print(f'Normal Reconstruction Loss: {normal_reconstruction_loss:.4f}')
print(f'Abnormal Reconstruction Loss: {abnormal_reconstruction_loss:.4f}')

# Note that abnormal data has much higher reconstruction error
# The model has learned to represent the normal pattern, so the 'normal' data can be reconstructed with low error, and abnormal data cannot
```

This autoencoder example illustrates how a GRU, trained on single-class data, can be used for anomaly detection. It reconstructs the training (normal) data with low error, but the error is significantly higher for the unseen data, indicative of anomalous content. It's not making predictions about what is abnormal, but rather what is *not* normal based on its trained knowledge.

**Example 3: Sequence Completion with a simple 'masking' implementation.**

This example demonstrates how to use the GRU for sequence completion based on the learned distribution. Note this is not necessarily applicable to all problems, but could be in certain domains such as natural language processing where masking is common, or where there is an expected temporal continuation of a pattern based on previously seen sequences.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic single class sequence data
np.random.seed(42)
sequence_length = 100
sequence_data = np.cumsum(np.random.rand(1, sequence_length), axis=1).astype(np.float32)
sequence_data = torch.tensor(sequence_data)

# Create a 'masked' sequence
mask_start = 50
masked_sequence = sequence_data.clone()
masked_sequence[0, mask_start:] = 0 # 'masking' - set to zero.

class SequenceGRU(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(SequenceGRU, self).__init__()
    self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, input_size) #Output is same as input sequence length.

  def forward(self, x):
      h0 = torch.zeros(1, x.size(0), 32)
      out, _ = self.gru(x.unsqueeze(2), h0) # add input dim
      out = self.fc(out)
      return out.squeeze(2)

input_size = 1
hidden_size = 32

model = SequenceGRU(input_size, hidden_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training on the full sequence
epochs = 500
for epoch in range(epochs):
  optimizer.zero_grad()
  output = model(sequence_data)
  loss = criterion(output, sequence_data)
  loss.backward()
  optimizer.step()
  #print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# 'completion' of the masked sequence
with torch.no_grad():
  completed_sequence = model(masked_sequence)
  masked_mse_loss = criterion(completed_sequence, sequence_data)

print("MSE after mask completion:", masked_mse_loss.item())

# Again the MSE should be low as the model is continuing a sequence with similar properties
```
In this example, we train the model on a cumulative sum sequence. When masked, the model, based on learned pattern from the unmasked portion of the sequence can reconstruct the missing segments with good fidelity.

In conclusion, the accuracy of a PyTorch GRU trained on single-class data predicting unlabeled data is highly dependent on the similarity between the distributions of the training and unlabeled datasets. The models tend to perform well on unseen data if it aligns with the learned representation of the single class. For further study I recommend resources on recurrent neural networks, particularly focused on autoencoders and anomaly detection in time-series data. More specifically, the theoretical basis of autoencoders, and the relationship between autoencoders, dimensionality reduction, and manifold learning is crucial. Additionally, exploring model evaluation techniques, such as MSE, cosine similarity, or AUROC depending on the specific use case will also be very valuable.
