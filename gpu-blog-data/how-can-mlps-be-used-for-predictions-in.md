---
title: "How can MLPs be used for predictions in PyTorch?"
date: "2025-01-30"
id: "how-can-mlps-be-used-for-predictions-in"
---
Multilayer Perceptrons (MLPs), despite their relative simplicity compared to more complex neural network architectures, remain a foundational tool for prediction tasks in PyTorch. My experience working on a time-series forecasting project highlighted their effectiveness when appropriate feature engineering was performed beforehand. The key lies in understanding that MLPs learn non-linear relationships between input features and a target variable. In PyTorch, this translates to defining a network structure of fully connected layers, often interspersed with activation functions, and then training it through backpropagation. The model's predictive power is directly tied to the quality and representativeness of the input data and the tuning of the network’s hyperparameters.

An MLP, at its core, maps an input vector to an output vector through a series of linear transformations followed by non-linear activation functions. Each layer in the MLP consists of interconnected nodes (neurons), where the output of each node is determined by the weighted sum of its inputs plus a bias term, which is then passed through the activation function. The backpropagation algorithm adjusts these weights and biases based on a loss function to minimize the prediction error, iteratively refining the model's ability to capture complex patterns in the data. In PyTorch, the `nn.Linear` module constructs the linear transformation, while modules like `nn.ReLU`, `nn.Sigmoid`, or `nn.Tanh` provide common activation functions.

The architecture of the MLP, specifically the number of layers and neurons per layer, often demands experimentation. Overly complex networks with too many parameters may lead to overfitting, where the model memorizes the training data but generalizes poorly to unseen examples. Conversely, a network with too few parameters might lack the capacity to accurately model the underlying data patterns, resulting in underfitting. I typically begin with a smaller architecture and gradually increase complexity only if the performance on a validation set indicates a need. The choice of activation function is another critical aspect. ReLU is popular due to its computational efficiency and reduced likelihood of vanishing gradient problems, but other functions such as Sigmoid and Tanh may be suitable depending on specific application requirements.

Below are examples demonstrating how MLPs are used for predictions in PyTorch, focusing on different aspects like a simple binary classification, a regression task, and the inclusion of batch normalization and dropout for regularization:

**Example 1: Binary Classification**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model for binary classification
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Output layer, 1 neuron for binary class
        self.sigmoid = nn.Sigmoid() # Sigmoid activation for probability output

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Generate some sample data
input_size = 10
hidden_size = 32
num_samples = 100
X = torch.randn(num_samples, input_size)
y = torch.randint(0, 2, (num_samples, 1)).float() # Binary labels (0 or 1)

# Instantiate model, loss function, and optimizer
model = BinaryClassifier(input_size, hidden_size)
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# Example usage for prediction
with torch.no_grad():
    test_input = torch.randn(1, input_size)
    prediction = model(test_input)
    predicted_class = (prediction > 0.5).float()
    print(f"Prediction Probability: {prediction.item():.4f}, Predicted Class: {predicted_class.item():.0f}")
```

This first example showcases a simple MLP for binary classification. The `BinaryClassifier` class defines a two-layer network with ReLU activation after the first linear transformation and sigmoid after the second, as is standard for binary classification, which outputs a probability between 0 and 1.  `nn.BCELoss()` is chosen as the loss function, designed for binary outputs, and the `optimizer.Adam` facilitates the weight updates through backpropagation.  The training data is synthetically generated and the example ends by taking a sample and performing a prediction. Note the usage of `torch.no_grad()` to perform inference, as no weight updates are needed when performing a forward pass to obtain the final output and its associated probability.

**Example 2: Regression Task**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model for regression
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1) # Output layer, 1 neuron for regression

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate some sample data
input_size = 5
hidden_size = 20
num_samples = 100
X = torch.randn(num_samples, input_size)
y = torch.randn(num_samples, 1)  # Continuous target variable

# Instantiate model, loss function, and optimizer
model = RegressionModel(input_size, hidden_size)
criterion = nn.MSELoss() # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")


# Example usage for prediction
with torch.no_grad():
    test_input = torch.randn(1, input_size)
    prediction = model(test_input)
    print(f"Prediction Value: {prediction.item():.4f}")
```
This example constructs an MLP for a regression problem. The `RegressionModel` is structurally similar to the classification model, but has no output activation function. The output is a continuous value, not a probability, and Mean Squared Error `nn.MSELoss()` is used to calculate the loss between the prediction and the true values. The rest of the workflow, such as training data generation and the optimizer, remains largely consistent, but no output sigmoid function is applied. The example shows prediction on a new sample as well.

**Example 3: Regularization with Batch Normalization and Dropout**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP with Batch Normalization and Dropout
class RegularizedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegularizedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Sample data generation
input_size = 20
hidden_size = 64
output_size = 1
num_samples = 200
X = torch.randn(num_samples, input_size)
y = torch.randn(num_samples, output_size)


# Instantiate model, loss function, and optimizer
model = RegularizedMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
      print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")


# Example usage for prediction
with torch.no_grad():
    test_input = torch.randn(1, input_size)
    prediction = model(test_input)
    print(f"Prediction Value: {prediction.item():.4f}")
```
This final example incorporates batch normalization (`nn.BatchNorm1d`) and dropout (`nn.Dropout`) into the MLP. Batch normalization helps stabilize training and can accelerate convergence, while dropout reduces overfitting by randomly deactivating neurons during training. This makes the model more robust. The position of the batchnorm layer before the activation function, and the dropout layer afterward, is a common and generally effective practice.

For further exploration of MLPs and related concepts within PyTorch, I strongly recommend consulting the official PyTorch documentation, which provides comprehensive explanations of all modules used, as well as practical tutorials covering a wide range of tasks. "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann offers detailed instruction on the nuances of designing and training neural networks. Finally, exploring research papers in areas such as data pre-processing, hyperparameter optimization, and loss functions provides valuable contextual understanding for maximizing the effectiveness of MLPs in various prediction scenarios.
