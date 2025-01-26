---
title: "How can a two-layer neural network be built using PyTorch?"
date: "2025-01-26"
id: "how-can-a-two-layer-neural-network-be-built-using-pytorch"
---

Building a two-layer neural network in PyTorch involves defining the network architecture, specifying the loss function and optimizer, and then training the model using a dataset. The core principle resides in composing linear transformations followed by non-linear activation functions, a fundamental pattern in neural network design. I've implemented and fine-tuned many such networks during my time developing machine learning solutions for time-series analysis, which has provided me with a practical understanding of these processes.

A two-layer network, in its simplest form, consists of an input layer, a hidden layer, and an output layer. The first layer performs a linear transformation of the input data, followed by a non-linear activation, such as ReLU or sigmoid. The output of this hidden layer then serves as input to a second linear transformation, generating the final output of the network. This structure allows the network to learn non-linear decision boundaries crucial for handling complex datasets.

Let's dive into the specifics. We'll be using the `torch.nn` module in PyTorch to define our network layers and `torch.optim` for the optimization algorithm.

First, I’ll define the network architecture using the `nn.Module` class, a PyTorch class that provides the necessary framework for creating neural network modules.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

In this example, `__init__` is where we define the network's architecture. The `nn.Linear` layers perform the linear transformations, and `nn.ReLU` is our non-linear activation function.  The `forward` method dictates how the input data flows through the network layers. It takes an input tensor `x` and returns the output of the network. This separation of concerns is a crucial aspect of how PyTorch organizes neural network code. I've found this modular design exceptionally helpful for managing complex models involving numerous layers and operations.

Next, we need to instantiate the model, choose a loss function, and configure the optimizer. For demonstration purposes, I will use a mean squared error (MSE) loss and stochastic gradient descent (SGD) optimizer.

```python
# Example Usage
input_size = 10
hidden_size = 5
output_size = 1
model = TwoLayerNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Here, I create an instance of the `TwoLayerNet` with `input_size`, `hidden_size`, and `output_size` values defined as 10, 5 and 1, respectively.  We also specify a loss function `nn.MSELoss` suitable for regression tasks, and an optimizer using the `optim.SGD` with a learning rate of 0.01. I selected these hyperparameters for demonstrative purposes and, in practice, would likely use more sophisticated optimizers, such as Adam, and fine-tune the learning rate. I've found that the choice of optimizer and learning rate often has a considerable impact on training speed and model performance.

Now, I'll generate some sample training data and train the network.

```python
# Sample data for demonstration
X_train = torch.randn(100, input_size)
y_train = torch.randn(100, output_size)


num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad() #Clear gradients from previous step
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

This section generates random input data `X_train` and corresponding target output `y_train`. I iterate through epochs, performing a forward pass through the network, calculating the loss using our `criterion`, propagating gradients backward (`loss.backward()`) and updating model parameters using the optimizer (`optimizer.step()`). Critically, we use `optimizer.zero_grad()` at the beginning of each training step to clear gradients from previous epochs, a common pitfall if overlooked. The `loss.item()` extracts the loss value from the tensor for a clearer printout. This step-by-step execution helps understand the process by which gradients are computed and used to update weights in the network.

The code above illustrates the fundamental workflow for training a two-layer neural network. The training process involves calculating the error between the network’s output and the desired output, and then using this error to adjust network weights, iteratively improving the model. In actual practice, the data would be split into training, validation, and test sets, and evaluation metrics such as mean absolute error or R-squared would be used to assess performance.

Furthermore, we can extend this example to incorporate variations such as using a sigmoid activation function or a different number of nodes in the hidden layer.  Here is an example using sigmoid:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TwoLayerNetSigmoid(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNetSigmoid, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out


input_size = 10
hidden_size = 5
output_size = 1
model_sigmoid = TwoLayerNetSigmoid(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer_sigmoid = optim.SGD(model_sigmoid.parameters(), lr=0.01)

X_train = torch.randn(100, input_size)
y_train = torch.randn(100, output_size)

num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_sigmoid(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer_sigmoid.zero_grad()
    loss.backward()
    optimizer_sigmoid.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Sigmoid Loss: {loss.item():.4f}')
```
This code snippet demonstrates a similar two-layer network but replaces ReLU with the sigmoid activation function.  Sigmoid outputs a value between 0 and 1, making it particularly suited for applications such as binary classification, or when the output layer represents a probability. I have utilized both ReLU and Sigmoid activations extensively, and the choice often depends heavily on the characteristics of the data and problem at hand.

Finally, consider the incorporation of batch training. This often results in more stable training compared to updating weights based on individual samples:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class TwoLayerNetBatch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNetBatch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 10
hidden_size = 5
output_size = 1
model_batch = TwoLayerNetBatch(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer_batch = optim.SGD(model_batch.parameters(), lr=0.01)
batch_size = 32


X_train = torch.randn(100, input_size)
y_train = torch.randn(100, output_size)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model_batch(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer_batch.zero_grad()
        loss.backward()
        optimizer_batch.step()

    if (epoch+1) % 10 == 0:
         print(f'Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}')
```

In this modified code, I've introduced `TensorDataset` and `DataLoader` from `torch.utils.data`.  The `DataLoader` efficiently loads data in batches, which is then processed in the training loop.  Batch processing improves training stability by providing more generalized gradient estimates and often reduces computational requirements compared to processing individual data points. This code also highlights the modularity of PyTorch where data loading strategies can be seamlessly incorporated.

For resources, I would suggest examining official PyTorch documentation. In particular, the introductory tutorials on neural network construction are particularly valuable. Books covering deep learning and PyTorch offer additional depth and context on these concepts. Furthermore, numerous online courses specializing in deep learning provide further exposure and practice. Reviewing research papers on model architectures and training techniques is also highly recommended for advanced understanding and practical proficiency.
