---
title: "How do I plot training loss from Polyphony RNN training run in Python?"
date: "2024-12-23"
id: "how-do-i-plot-training-loss-from-polyphony-rnn-training-run-in-python"
---

, let's talk about extracting and visualizing that training loss from your Polyphony RNN. It’s a common sticking point, and I've certainly spent my fair share of time debugging similar scenarios back when I was deep in a music generation project a few years ago. The core challenge often isn't in running the model, but in making sense of its internal states and, more specifically, its progress during training.

The process revolves around three main stages: data collection, data processing, and finally, visualization. I’ve seen folks stumble at each one of these, so let’s break it down systematically.

First, you need to ensure you're actually collecting the loss values during training. Most high-level libraries, like those commonly used for RNNs, offer hooks or mechanisms to access this. Usually, it involves saving the loss at each epoch or even after a specified number of iterations within an epoch, depending on your setup. When I first tackled this problem, we were working with a custom-built RNN framework, which meant we had to explicitly write code to capture the loss each time we calculated it within the backpropagation loop. A mistake there, and your loss data would be skewed or missing completely.

Once you have that loss data, the next step is preparing it for visualization. This usually implies transforming the raw data—potentially stored as a list or a numpy array—into a format readily consumed by plotting libraries. It might involve some basic processing, such as averaging loss over multiple iterations if you've been capturing very granular updates, or even smoothing to reduce the noise and highlight the overall trends. It’s important to note here that raw data is rarely directly useful for insights.

Lastly, the visualization part is where we convert our prepared data into meaningful visuals. Python offers great libraries like matplotlib or seaborn for this. Typically, a simple line plot suffices to display the evolution of the training loss across epochs, but sometimes more advanced techniques, like log scaling or adding shaded areas to represent validation loss, can help to understand better training dynamics.

Now, let's get to some code examples. I will provide these examples under the assumption you're working with TensorFlow or PyTorch, as those are the frameworks most often used for RNN training these days. I’ll start with the PyTorch version, then show a version for TensorFlow, and finish with a generalized example of data preparation.

**Example 1: Extracting and Plotting Loss with PyTorch**

This example assumes you've already defined and are training your Polyphony RNN. This illustrates the training loop and loss collection using a mock trainer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Mock RNN Model and Training Setup (replace with your actual implementation)
class MockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MockRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Constants
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 0.001
epochs = 100
batch_size = 32

model = MockRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Mock Data
input_tensor = torch.rand(1000, 5, input_size) #Batch size of 5 for the example
target_tensor = torch.randint(0, output_size, (1000, ))

# Training Loop and Loss Collection
training_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(input_tensor), batch_size):
      input_batch = input_tensor[i:i+batch_size]
      target_batch = target_tensor[i:i+batch_size]
      optimizer.zero_grad()
      outputs = model(input_batch)
      loss = criterion(outputs, target_batch)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item() * len(input_batch)
    average_epoch_loss = epoch_loss / len(input_tensor)
    training_losses.append(average_epoch_loss)
    print(f"Epoch: {epoch+1}, Loss: {average_epoch_loss:.4f}")

# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
```

**Example 2: Extracting and Plotting Loss with TensorFlow/Keras**

Here is a TensorFlow/Keras version demonstrating loss collection within the training process.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Mock RNN Model (replace with your actual implementation)
class MockRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
      super(MockRNN, self).__init__()
      self.rnn = tf.keras.layers.SimpleRNN(hidden_size, return_sequences=False)
      self.fc = tf.keras.layers.Dense(output_size)

    def call(self, x):
      out = self.rnn(x)
      out = self.fc(out)
      return out

# Constants
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 0.001
epochs = 100
batch_size = 32

model = MockRNN(input_size, hidden_size, output_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Mock Data
input_tensor = np.random.rand(1000, 5, input_size).astype(np.float32)
target_tensor = np.random.randint(0, output_size, (1000,)).astype(np.int32)

# Training Loop and Loss Collection
training_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(input_tensor), batch_size):
        input_batch = input_tensor[i:i+batch_size]
        target_batch = target_tensor[i:i+batch_size]
        with tf.GradientTape() as tape:
            outputs = model(input_batch)
            loss = loss_fn(target_batch, outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss += loss.numpy() * len(input_batch)
    average_epoch_loss = epoch_loss / len(input_tensor)
    training_losses.append(average_epoch_loss)
    print(f"Epoch: {epoch+1}, Loss: {average_epoch_loss:.4f}")

# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
```

**Example 3: Generalized Data Processing**

This example focuses on smoothing and prepares data obtained from a hypothetical training process.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def smooth_loss(loss_values, window_size):
  """Applies a moving average to a series of loss values."""
  if window_size <= 1 or len(loss_values) < window_size:
        return loss_values
  window = np.ones(window_size) / window_size
  smoothed = signal.convolve(loss_values, window, mode='valid')
  return smoothed

# Example loss values from training
raw_losses = np.random.uniform(0.1, 1.0, 1000) + np.linspace(0.5,0.0,1000) # Simulated loss data
epochs = np.arange(len(raw_losses))

# Apply Smoothing
window_size = 50
smoothed_losses = smooth_loss(raw_losses, window_size)
smoothed_epochs = epochs[window_size -1:]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, raw_losses, label='Raw Loss', alpha=0.4) # Raw loss in light color
plt.plot(smoothed_epochs, smoothed_losses, label='Smoothed Loss', color='darkorange')  # Smoothed loss in vibrant color
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Raw and Smoothed Loss')
plt.legend()
plt.grid(True)
plt.show()
```

This final example illustrates a simple moving average filter, which can be really useful to get a clearer picture of the trends, especially when the training loss fluctuates a lot.

For further reading, I recommend delving into *Deep Learning* by Goodfellow, Bengio, and Courville, particularly the sections that cover training and optimization. For more specifics on RNNs, *Sequence to Sequence Learning with Neural Networks* by Sutskever et al. is foundational. The official documentation of your specific deep learning framework, whether that's PyTorch or TensorFlow, also provides detailed examples and best practices.

From my experience, a systematic approach to data logging and visualization pays dividends in debugging and understanding the training behavior of neural networks. It’s a vital step to go beyond just running the model to actually understanding how it learns.
