---
title: "How does BatchNorm layer backpropagation differ in PyTorch's training and evaluation modes?"
date: "2025-01-30"
id: "how-does-batchnorm-layer-backpropagation-differ-in-pytorchs"
---
The core difference in BatchNorm layer backpropagation between PyTorch's training and evaluation modes lies in how the layer calculates and applies its running statistics, specifically the mean and variance. During training, these statistics are computed per batch, and are used to normalize the batch data as well as update running averages for inference. In evaluation, these running averages are used directly for normalization, without updating. This distinction significantly impacts backpropagation, as it alters the gradient flow through the BatchNorm layer.

During training, the BatchNorm layer performs the following operations on an input batch *x*:
1. **Batch Mean Calculation:** The mean of the batch, *μ<sub>B</sub>*, is calculated: *μ<sub>B</sub>* = (1/m) Σ *x<sub>i</sub>*, where *m* is the batch size.
2. **Batch Variance Calculation:** The variance of the batch, *σ<sup>2</sup><sub>B</sub>*, is calculated: *σ<sup>2</sup><sub>B</sub>* = (1/m) Σ (*x<sub>i</sub>* - *μ<sub>B</sub>*)<sup>2</sup>.
3. **Normalization:** The batch is normalized: *x<sub>norm</sub>* = (*x* - *μ<sub>B</sub>*) / √(*σ<sup>2</sup><sub>B</sub>* + *ε*), where *ε* is a small constant to prevent division by zero.
4. **Scale and Shift:** The normalized data is scaled and shifted: *y* = *γ* *x<sub>norm</sub>* + *β*, where *γ* and *β* are learnable parameters.
5. **Running Statistics Update:** Running averages of the mean *μ<sub>running</sub>* and variance *σ<sup>2</sup><sub>running</sub>* are updated using an exponential moving average:
   *μ<sub>running</sub>* = (1 - *momentum*) *μ<sub>running</sub>* + *momentum* *μ<sub>B</sub>*
   *σ<sup>2</sup><sub>running</sub>* = (1 - *momentum*) *σ<sup>2</sup><sub>running</sub>* + *momentum* *σ<sup>2</sup><sub>B</sub>*

During backpropagation in training mode, the gradients are calculated and passed through each step: gradients are computed with respect to the output *y*, then backpropagated to the learnable parameters *γ* and *β*, the normalized data *x<sub>norm</sub>*, and further back to the batch statistics *μ<sub>B</sub>* and *σ<sup>2</sup><sub>B</sub>*. This means the backpropagation algorithm needs to consider the dependency of *x<sub>norm</sub>* on the batch statistics as well as the original input *x*. Consequently, gradients flow through not just the scaling and shifting parameters, but also indirectly via the batch statistics used to normalize the data. The overall effect is that the model adapts to the statistics of the current batch as well as learn suitable scale and shift parameters.

In contrast, during evaluation, the following occurs:
1. **Normalization:** The input is normalized using the running mean *μ<sub>running</sub>* and running variance *σ<sup>2</sup><sub>running</sub>*: *x<sub>norm</sub>* = (*x* - *μ<sub>running</sub>*) / √(*σ<sup>2</sup><sub>running</sub>* + *ε*). Note that these are *not* the batch statistics, but precalculated averages.
2. **Scale and Shift:** As in training, *y* = *γ* *x<sub>norm</sub>* + *β*, using the learned parameters.

The crucial difference is that the running mean and variance are *not* updated using batch statistics and are treated as constants during the forward pass, effectively disabling the calculation of batch-specific statistics. During backpropagation, gradients are calculated and passed through *y* back to *x<sub>norm</sub>* and further back through the scaling and shifting parameters, as with training. Crucially, they *do not* backpropagate through the previously computed running statistics because these are treated as constants in evaluation mode. This simplifies the backpropagation calculations, and the gradient flow doesn't involve any implicit dependencies on the current input batch. The model simply uses the learned statistics to scale the incoming data. This ensures that the output is consistent across all evaluation runs for the same input.

Here are three code examples in PyTorch illustrating the described behavior:

**Example 1: Training Mode Backpropagation**

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Define a simple network with BatchNorm
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        return x

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training data
input_tensor = torch.randn(16, 10)
target = torch.randn(16, 1)

# Forward and backward pass
optimizer.zero_grad()
output = model(input_tensor)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Examine the updated running statistics
print("Running Mean after training:", model.bn.running_mean)
print("Running Var after training:", model.bn.running_var)

# Backprop gradients to BatchNorm (not directly shown with 'grad' attributes on output.grad, but is implicit)
for name, param in model.named_parameters():
    if 'bn' in name:
      if param.grad is not None:
        print(f"{name} gradient norms:{torch.norm(param.grad).item()}")
```

In this example, the model is in training mode, and running mean and variance are updated. The backpropagation calculates gradients for both the BatchNorm layer parameters and statistics implicitly, with gradients flowing through the batch normalization to update the weights of the model. This demonstrates that during training, gradients affect not only the trainable parameters *γ* and *β* but also, via the intermediate normalized data, impact the underlying statistics for each layer.

**Example 2: Evaluation Mode Backpropagation**

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Define a simple network with BatchNorm
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
      x = self.bn(x)
      x = self.fc(x)
      return x


model = Net()
model.eval()  # Set to evaluation mode

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training data
input_tensor = torch.randn(16, 10)
target = torch.randn(16, 1)

# Forward and backward pass
optimizer.zero_grad()
output = model(input_tensor)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Examine that running statistics remain unchanged
print("Running Mean after evaluation:", model.bn.running_mean)
print("Running Var after evaluation:", model.bn.running_var)

# Backprop gradients to BatchNorm
for name, param in model.named_parameters():
    if 'bn' in name:
      if param.grad is not None:
        print(f"{name} gradient norms:{torch.norm(param.grad).item()}")
```

This example demonstrates evaluation mode. The running statistics are not modified in the forward pass and are treated as constants in the backward pass. Note how, despite a backward pass, the running mean and variance remain the same and only the trainable parameters of the BatchNorm layer are updated. This is because the batch statistics are no longer contributing to the overall loss, therefore not providing gradients to be used in the update.

**Example 3: Observing Statistics Differ Across Modes**
```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Define a simple network with BatchNorm
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
      x = self.bn(x)
      x = self.fc(x)
      return x


model = Net()
input_tensor = torch.randn(16, 10)

# Get batch statistics while training
model.train()
model(input_tensor)

batch_mean_training = model.bn.running_mean.clone()
batch_var_training = model.bn.running_var.clone()

# Get batch statistics while evaluation
model.eval()
model(input_tensor)


batch_mean_eval = model.bn.running_mean.clone()
batch_var_eval = model.bn.running_var.clone()


print("Training running_mean: ", batch_mean_training)
print("Evaluation running_mean: ", batch_mean_eval)

print("Training running_var: ", batch_var_training)
print("Evaluation running_var: ", batch_var_eval)
```

This final example highlights the critical difference in running statistics before and after moving between training and evaluation modes, reinforcing the point about how the statistics are updated differently. When we call the forward pass during the training phase, our running means/vars are updated with the batch statistics, but when we call it during the evaluation phase, the running means/vars are not.

For further study and reference, I recommend exploring the following: deep learning textbooks that cover normalization techniques, specifically sections related to batch normalization; PyTorch’s official documentation, focusing on `torch.nn.BatchNorm1d`, `torch.nn.BatchNorm2d`, and `torch.nn.BatchNorm3d`; and research papers detailing the original BatchNorm technique and its variants. Additionally, inspecting the PyTorch source code related to these classes provides a detailed understanding of the internal workings.
