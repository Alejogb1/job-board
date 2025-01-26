---
title: "What is the lottery ticket hypothesis?"
date: "2025-01-26"
id: "what-is-the-lottery-ticket-hypothesis"
---

The lottery ticket hypothesis, as formulated by Frankle and Carbin (2018), posits that within a randomly initialized, dense neural network exists a subnetwork (a "winning ticket") that, when trained in isolation, can achieve comparable or even superior performance to the original, larger network. This subnetwork is identified through iterative magnitude pruning during training. I've personally explored this concept extensively while optimizing large language models (LLMs) and convolutional neural networks (CNNs) for edge deployment, and the practical implications are significant for model compression and efficiency.

Essentially, the hypothesis challenges the notion that the sheer size of a neural network is directly proportional to its effectiveness. Instead, it suggests that overparameterization might inadvertently create a vast search space, within which a highly efficient and performant subnetwork lies hidden. Identifying and extracting this "winning ticket" has substantial implications for reducing the computational demands and memory footprints of deep learning models, particularly in resource-constrained environments.

The process typically involves three main stages: training the initial dense network, iteratively pruning connections (typically based on the magnitude of their weights), and finally, retraining the pruned subnetwork to its optimal performance level. Pruning, in this context, isn't random. Connections with smaller weights are generally considered less significant and are therefore candidates for removal. The iterative aspect is crucial, because by progressively pruning and retraining, the network can adapt and find a winning ticket, which might not be found with a single round of pruning.

Here’s how I’ve practically applied this in a few different contexts, demonstrating the iterative pruning method:

**Example 1: Pruning a Simple Feedforward Network in PyTorch**

This code snippet illustrates the fundamental concepts of iterative magnitude pruning in PyTorch. I’ve deliberately kept the network simple to focus on the core algorithm.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prune_by_magnitude(model, prune_rate):
  parameters_to_prune = []
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

  for module, name in parameters_to_prune:
    weights = module.weight.data.abs()
    num_pruned = int(weights.numel() * prune_rate)
    threshold = torch.kthvalue(weights.view(-1), num_pruned)[0]
    mask = weights.ge(threshold).float()
    module.weight.data *= mask
  return model


def train(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
      for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

# Parameters
input_size = 10
hidden_size = 50
output_size = 2
epochs = 5
learning_rate = 0.01
prune_rate = 0.2
num_rounds = 5

# Dataset and DataLoader (mocked)
train_data = torch.randn(100, input_size)
train_labels = torch.randint(0, output_size, (100,))
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

# Initialize model, loss and optimizer
model = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Iterative Pruning
for round in range(num_rounds):
  print(f"Pruning round {round+1}")
  train(model, train_loader, criterion, optimizer, epochs)
  model = prune_by_magnitude(model, prune_rate)


```
This script sets up a basic feedforward network and implements the `prune_by_magnitude` function. It then iterates through several pruning rounds, retraining the model after each. Each round, 20% of the weights with the smallest magnitudes are zeroed out, simulating the selection of a "winning ticket." The `train` function simulates training, and demonstrates that the model after pruning is retrained. Crucially, the code only prunes weights, leaving the architectural connections intact, which adheres to the Lottery Ticket Hypothesis approach.

**Example 2: Pruning a Convolutional Layer in TensorFlow**

This example shifts focus to pruning convolutional layers, which are foundational in image processing networks. The same magnitude-based pruning technique is applied, but adapted to the structure of convolution kernels.

```python
import tensorflow as tf
import numpy as np

class SimpleCNN(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def prune_by_magnitude_tf(model, prune_rate):
  for layer in model.layers:
      if isinstance(layer, tf.keras.layers.Conv2D):
        weights = layer.kernel.numpy().copy()
        abs_weights = np.abs(weights)
        num_pruned = int(weights.size * prune_rate)
        flat_weights = abs_weights.flatten()
        threshold = np.sort(flat_weights)[num_pruned]

        mask = np.where(abs_weights >= threshold, 1, 0)
        layer.kernel.assign(weights * mask)

  return model


def train_tf(model, train_dataset, optimizer, loss_fn, epochs):
  for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
      with tf.GradientTape() as tape:
          y_pred = model(x_batch)
          loss = loss_fn(y_batch, y_pred)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Model parameters
input_shape = (32, 32, 3) # Example input image shape
num_classes = 10 # Example number of output classes
epochs = 5
learning_rate = 0.001
prune_rate = 0.2
num_rounds = 5

# Dataset (mocked)
train_images = np.random.rand(100, *input_shape).astype(np.float32)
train_labels = np.random.randint(0, num_classes, 100).astype(np.int64)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)


# Initialize model, loss and optimizer
model = SimpleCNN(input_shape, num_classes)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Iterative Pruning
for round in range(num_rounds):
  print(f"Pruning round {round+1}")
  train_tf(model, train_dataset, optimizer, loss_fn, epochs)
  model = prune_by_magnitude_tf(model, prune_rate)
```

In this TensorFlow implementation, I've created a `SimpleCNN` class and adapted the `prune_by_magnitude_tf` function to correctly handle convolutional layer kernels. The process is analogous to the PyTorch example: weights are sorted, a threshold is determined based on the desired pruning rate, and weights below the threshold are set to zero. Again, the key here is that we're pruning the weight *values*, not the connections themselves, and that this is done iteratively. This demonstrates how the principle extends beyond simple fully connected networks.

**Example 3: Maintaining the Initial Weights**

This Python code example focuses on what I have found most useful for applying the lottery ticket hypothesis, which is to mask weights instead of directly zeroing them. When we want to retrain the winning ticket we have to reset them to their initial values which is commonly performed by preserving a mask.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_masks(model, prune_rate):
  masks = {}
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
      weights = module.weight.data.abs()
      num_pruned = int(weights.numel() * prune_rate)
      threshold = torch.kthvalue(weights.view(-1), num_pruned)[0]
      mask = weights.ge(threshold).float()
      masks[name] = mask
  return masks

def apply_mask(model, masks):
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data *= masks[name]
    return model

def reset_weights(model, initial_state_dict, mask, retain_mask):
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data = initial_state_dict[name] * mask[name.split(".weight")[0]] if retain_mask else initial_state_dict[name]
    return model


def train(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
      for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

# Parameters
input_size = 10
hidden_size = 50
output_size = 2
epochs = 5
learning_rate = 0.01
prune_rate = 0.2
num_rounds = 5

# Dataset and DataLoader (mocked)
train_data = torch.randn(100, input_size)
train_labels = torch.randint(0, output_size, (100,))
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

# Initialize model, loss and optimizer
model = SimpleNet(input_size, hidden_size, output_size)
initial_state = copy.deepcopy(model.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
masks = {}


# Iterative Pruning
for round in range(num_rounds):
  print(f"Pruning round {round+1}")
  train(model, train_loader, criterion, optimizer, epochs)
  if round == 0:
    masks = create_masks(model, prune_rate)
  model = apply_mask(model, masks)


# Retrain the "winning ticket"
print("Retraining the 'winning ticket'")
model = reset_weights(model,initial_state, masks, True)
train(model, train_loader, criterion, optimizer, epochs*5)

```

This final example includes code that stores the initial weights and a generated mask during the pruning phase. After the mask is determined, the function `reset_weights` uses the initial weights and masks the relevant weights. This allows one to train the pruned weights from initialization which is typically performed in lottery ticket hypothesis research.

Based on my experiences and literature, here's a comparative breakdown of common pruning techniques and the lottery ticket hypothesis.

| Name                      | Functionality                                                              | Performance                                                     | Use Case Examples                                                           | Trade-offs                                                                                                                                             |
| :------------------------ | :-------------------------------------------------------------------------- | :-------------------------------------------------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Magnitude Pruning (General)** | Removes weights based on their absolute magnitude after training.      | Moderate reduction in model size, typically good performance.   | Basic model compression, speed-up of inference                                | May require iterative pruning, can sometimes lead to more instability if pruned too aggressively.                                                               |
| **Random Pruning**        | Removes weights randomly without a criteria                              | Very little model reduction and typically bad performance.        | Benchmarking for comparison to other methods                                   |  Random performance degradation, no benefits to model size or efficiency                                                                       |
| **Lottery Ticket Hypothesis** | Iterative magnitude pruning and retraining the subnetwork.               | High model compression, often preserves/improves accuracy.    | Efficient edge deployment, low-resource model training, energy-efficient networks | Complex implementation, can be more time-consuming, requires initial training of the full network; requires memory to save full weights.        |
| **Quantization**           | Reduces the precision of weights (e.g., from 32-bit to 8-bit) or binary values   | Significant size reduction, slight decrease in accuracy.        | Mobile and embedded deployment, efficient hardware implementation            | Can introduce accuracy loss depending on degree of quantization, requires specialized hardware for optimal performance.        |

While all methods aim at model compression, the lottery ticket hypothesis offers a distinct advantage in identifying and exploiting the inherent structure within overparameterized networks.

In conclusion, for most deployment scenarios requiring model compression without significant performance degradation, I find that the Lottery Ticket Hypothesis and iterative magnitude pruning are the superior options, even if it comes at the cost of a higher initial training investment. The ability to extract a "winning ticket" from an initial model makes it far more efficient. Quantization, which I also frequently use in combination with pruning, is more suitable when the goal is to reduce memory footprint as much as possible, typically at a small accuracy cost and requiring hardware considerations.
   
For researchers, starting with the "Lottery Ticket Hypothesis" original paper is key. Additionally, I recommend reviewing papers on iterative pruning techniques and their effects on various network architectures. Research into efficient model pruning, in general, has been very useful in my work.  The open-source implementations of these methods available on GitHub are also a great resource.
