---
title: "Which loss function best suits binary classification in a CNN with float labels?"
date: "2025-01-30"
id: "which-loss-function-best-suits-binary-classification-in"
---
Binary classification tasks often leverage the sigmoid activation function in the final layer of a convolutional neural network (CNN) to produce a probability score between 0 and 1.  When dealing with **float labels** rather than discrete 0 or 1 values, the standard binary cross-entropy loss, while functional, isn't necessarily the most theoretically sound nor practically effective approach. The discrepancy between a sigmoid output which ideally reflects probability, and a non-binary target variable leads to issues with interpretability and training effectiveness. My experience building medical imaging classifiers, where labels often represent likelihood scores generated from expert evaluations (e.g., presence of tumor scaled between 0 and 1), has underscored this point.

The issue arises from the fundamental assumption of binary cross-entropy (BCE). BCE evaluates how well the predicted probability *p* corresponds to a *true* binary label. It measures the negative log-likelihood of the observed outcome.  However, if the label, *y*, is not binary, say, 0.75, the BCE loss will still treat it as representing a full positive class, contributing to a loss as if *p* should have been 1 rather than something closer to 0.75. We are implicitly imposing a hard binarization on what is actually a continuous score. This is not ideal.

In such scenarios, mean squared error (MSE) becomes a more appropriate loss function. MSE calculates the average of the squares of the differences between predicted and true values. Unlike BCE which measures the mismatch between predicted probabilities and categorical outcomes, MSE directly measures the magnitude of the error, making it directly applicable to float-based labels and more aligned to the underlying objective.

Let's examine why MSE provides a better fit. Consider a scenario where the sigmoid output is 0.65. If the true label is 1, BCE loss would be relatively high as the model had a probability of a positive class, but missed completely. However, if the true label is 0.75, the same BCE loss treats that as if it missed the mark completely. MSE, in contrast, considers the magnitude of the difference. If the label is 1, error is 0.35, and for 0.75, error would be just 0.1, thereby acknowledging that a predicted output of 0.65 was ‘closer’ to 0.75 than to 1 or 0.

The core advantage of using MSE with float labels is that it naturally facilitates a regression-like problem formulation within the confines of a binary classification setup. We are training the network to predict a continuous score within the range [0, 1] that corresponds to the continuous label value. This continuous score can then still be interpreted as a probability (provided the sigmoid output) and allows for finer gradient signals during training, promoting smoother convergence and potentially better calibration.

Let’s explore this with code examples, using PyTorch:

**Example 1: Baseline BCE with float labels**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming y_true is a float between 0 and 1
y_true = torch.tensor([0.2, 0.8, 0.5, 0.9], dtype=torch.float32).reshape(-1, 1)
y_pred = torch.tensor([0.4, 0.7, 0.6, 0.85], dtype=torch.float32).reshape(-1, 1)

# Sigmoid is implicit in BCELoss
loss_function = nn.BCELoss()

loss = loss_function(y_pred, y_true)
print(f"BCE Loss: {loss.item()}")

# This demonstrates how BCELoss evaluates the predicted probabilities given non-binary labels
```
This example highlights that BCELoss, while technically capable of computing a loss from float labels, does not inherently leverage the information contained in the continuous nature of the target variable, it is treating it as a probability with a binary outcome. The resulting loss is not directly a measure of the error between the predicted values and the floating labels.

**Example 2: MSE loss with float labels**

```python
import torch
import torch.nn as nn
import torch.optim as optim

y_true = torch.tensor([0.2, 0.8, 0.5, 0.9], dtype=torch.float32).reshape(-1, 1)
y_pred = torch.tensor([0.4, 0.7, 0.6, 0.85], dtype=torch.float32).reshape(-1, 1)

loss_function = nn.MSELoss()

loss = loss_function(y_pred, y_true)
print(f"MSE Loss: {loss.item()}")

# This showcases how MSE directly measures the squared differences between predictions and labels.
```

Here, the loss value accurately reflects the magnitude of the discrepancy between predicted values and the floating labels, making it a much better fit compared to BCE. MSE effectively treats the problem as a regression task within the [0,1] space, resulting in an output with a more meaningful loss value.

**Example 3: Training loop using MSE with float labels**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(16 * 7 * 7, 1) # Assumes input of 28 x 28

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = torch.sigmoid(self.fc(x)) # Sigmoid for probability output
        return x

# Sample training data
input_data = torch.randn(100, 1, 28, 28)
float_labels = torch.rand(100, 1)

model = SimpleCNN()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
  optimizer.zero_grad()
  output = model(input_data)
  loss = loss_function(output, float_labels)
  loss.backward()
  optimizer.step()
  if (epoch+1)%10 == 0:
    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Demonstrating training with MSE and float labels, showcasing smooth loss decrease during training
```

This example demonstrates a basic training loop using MSE loss.  Crucially, we maintain a sigmoid at the output for interpretability as probabilities but use MSE for the backpropagation of gradient signals. The loss value decreases with the epochs, indicating that the network is learning a continuous mapping from the input data to the labels. The use of MSE here better aligns with the nature of the data and the objective, i.e. predicting a probability-like outcome.

Selecting the right loss function is a balance of theoretical fit and empirical performance. MSE's ability to directly measure the magnitude of errors with float labels makes it a pragmatic and effective choice. In my experience with these types of scenarios, I've observed that MSE loss usually results in more stable training and improved final prediction quality when compared to the naive usage of BCE on non-binary labels.

For further exploration into loss functions and their applications I suggest reviewing literature on regression analysis using neural networks. Specifically, pay attention to discussions of metrics like mean absolute error (MAE), as well as the nuances of optimization algorithms for loss functions within deep learning, and material on calibration analysis when utilizing a sigmoid to yield probability estimates. It's also useful to research how to optimize for both regression and classification on datasets with continuous labels.
