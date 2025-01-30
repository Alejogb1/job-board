---
title: "Why are gradients vanishing despite using Kaiming initialization?"
date: "2025-01-30"
id: "why-are-gradients-vanishing-despite-using-kaiming-initialization"
---
Vanishing gradients, even with Kaiming initialization, often indicate deeper architectural or training-related issues beyond simply setting initial weights appropriately. Kaiming initialization primarily addresses the challenge of maintaining signal magnitude during forward propagation, mitigating one aspect of vanishing gradients but not the entire problem space. In my experience building complex convolutional networks for image processing, I've encountered situations where initially sensible networks, using Kaiming initialization, would still struggle to learn, demonstrating a clear vanishing gradient issue. This usually points to a more nuanced explanation than just improper weight initialization.

The fundamental problem arises from the repeated application of non-linear activation functions within deep networks. During backpropagation, gradients are multiplied together layer by layer; if these gradients are consistently smaller than 1, their cumulative product diminishes exponentially as you propagate backward through the network. Even with Kaiming initialization ensuring that weights don't start exceedingly small, gradients can become close to zero after just a few layers if the activations themselves contribute to this shrinkage. Activation functions like Sigmoid and Tanh are especially prone to causing this due to their derivative properties; their maximum derivatives are less than or equal to 1, and these peak derivatives are limited to small regions of their domain, frequently leading to near-zero gradients.

While Kaiming initialization, specifically variance scaling, ensures that the variance of the input to a layer is approximately equal to the variance of the output in the initial forward pass (at initialization), it does not compensate for the non-linearity’s effect on gradients during backward propagation. Specifically, Kaiming initialization aims to keep the variance of the *activations* roughly stable, but doesn't directly address the *gradients* of those activations. If the activations end up in parts of their range where the derivative is small (such as the extremes for sigmoid or tanh), the gradient will still vanish regardless of the initial weight scale.

Another less obvious contributing factor is the architecture of the network. Very deep networks, particularly those without residual connections or other shortcut mechanisms, force the gradient to travel long paths. Each layer's computation is susceptible to the issue of small derivatives, which quickly compounds across layers. A very deep, purely sequential architecture will likely struggle with vanishing gradients even with optimal initialization. Even in networks where activation functions aren't the primary cause, the depth itself can contribute to vanishing gradients because the chain rule repeatedly multiplies gradients, which can lead to the product approaching zero as the chain gets longer.

Furthermore, mini-batch stochastic gradient descent (SGD) combined with the optimization landscape's topology adds another layer of complexity. Even if the theoretical gradient computed from a large batch were favorable, the gradients estimated from small minibatches will be noisy, which can further impede effective learning if the signal in the lower layers has already started to vanish.

Let’s consider a few practical examples.

**Example 1: Simple Deep Neural Network with Sigmoid Activations**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DeepNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Sigmoid())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)

input_size = 10
hidden_size = 128
num_layers = 10
output_size = 1

model = DeepNet(input_size, hidden_size, num_layers, output_size)

# Kaiming initialization is the default for linear layers in PyTorch, so no additional work is needed here

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data for training
inputs = torch.randn(100, input_size)
targets = torch.randn(100, output_size)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
```

In this example, even with the default Kaiming initialization applied to the linear layers, the repeated use of Sigmoid activations in a moderately deep network quickly results in minimal gradients in the lower layers. This is because the sigmoid saturates and contributes negligible derivatives in the lower layers, leading to slow convergence, or even no learning. You will likely observe that the loss either stagnates or decreases slowly.

**Example 2: Deep Network with ReLU and Residual Connections**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
      super(ResidualBlock, self).__init__()
      self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
      self.relu = nn.ReLU()
      self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
      self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
      residual = x
      x = self.relu(self.conv1(x))
      x = self.conv2(x)
      x += self.shortcut(residual)
      return self.relu(x)

class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, output_size):
        super(ResNet, self).__init__()
        self.conv_in = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        blocks = []
        for _ in range(num_blocks):
          blocks.append(ResidualBlock(hidden_size, hidden_size))
        self.blocks = nn.Sequential(*blocks)
        self.conv_out = nn.Conv1d(hidden_size, output_size, kernel_size=1)


    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        return x

input_size = 3
hidden_size = 64
num_blocks = 10
output_size = 1
model = ResNet(input_size, hidden_size, num_blocks, output_size)

# Kaiming initialization is the default for conv layers in PyTorch, so no additional work is needed here

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data for training
inputs = torch.randn(100, input_size, 10) # Added a channel dimension for conv
targets = torch.randn(100, output_size, 10) # added channel dimensions for conv

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

```
This example shows a ResNet architecture, using ReLU activations and skip connections. Even though the network is deep, the residual connections help preserve the gradient flow, and the use of ReLU instead of Sigmoid prevents severe gradient saturation. Because ReLU does not saturate, gradients do not shrink as quickly. Therefore this network will learn much faster and more reliably than the first example.

**Example 3: Impact of Batch Size and Learning Rate**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ShallowNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


input_size = 10
hidden_size = 128
output_size = 1

model = ShallowNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()

# Test with a large learning rate and small batch size

optimizer1 = optim.Adam(model.parameters(), lr=0.05) # A Large learning rate
batch_size1 = 16 # A small batch size

# Test with a small learning rate and large batch size

optimizer2 = optim.Adam(model.parameters(), lr=0.001) # A smaller learning rate
batch_size2 = 128 # A larger batch size

# Dummy data for training
inputs = torch.randn(1000, input_size)
targets = torch.randn(1000, output_size)

# Function to train with specific hyper-parameters
def train(model, optimizer, batch_size, inputs, targets, epochs=100, name=""):
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
      for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
      if epoch % 20 == 0:
        print(f'{name} - Epoch: {epoch}, Loss: {loss.item():.4f}')


train(model, optimizer1, batch_size1, inputs, targets, name="Large LR, Small Batch")
train(model, optimizer2, batch_size2, inputs, targets, name="Small LR, Large Batch")


```
This code example highlights how the interplay between the learning rate and the batch size can also impact the training process. Here, the small batch size combined with a high learning rate will result in erratic updates, making convergence less smooth and potentially worsening the vanishing gradient problem. Conversely, a smaller learning rate and larger batch size leads to smoother gradient updates and more stable learning dynamics, which often mitigates, but does not solve, the underlying problem of vanishing gradients. Even a shallow network can struggle to learn with poor hyperparameters.

To address the issue of vanishing gradients effectively, beyond Kaiming initialization, I recommend considering the following: First, explore alternative activation functions like ReLU and its variations (Leaky ReLU, ELU) which mitigate gradient saturation better than Sigmoid or Tanh. Second, incorporate residual connections to create skip connections that allow gradients to flow more directly across layers. Third, Experiment with different batch sizes and learning rates, use adaptive optimizers (like Adam, or RMSprop) which can automatically adjust gradients to improve convergence. Finally, consider architecture changes like reducing network depth or adding batch normalization layers. These will often prove to be very effective in tackling vanishing gradients. Good documentation can be found by referring to papers detailing the ReLU family of activations, along with papers on residual networks and related concepts. Consulting introductory texts on deep learning that also give a theoretical treatment to these aspects is also highly beneficial.
