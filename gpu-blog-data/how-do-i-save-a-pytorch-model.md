---
title: "How do I save a PyTorch model?"
date: "2025-01-30"
id: "how-do-i-save-a-pytorch-model"
---
PyTorch model saving is fundamentally about capturing the learned parameters and architectural structure required for subsequent inference, not merely the model's in-memory state. This distinction is critical because directly pickling a PyTorch model, while sometimes functional, is unreliable across diverse hardware and software configurations. My experience migrating models between development environments, specifically between Linux servers and local Windows workstations, highlighted the necessity of a more robust methodology. Therefore, when saving a PyTorch model, the focus should be on persisting the `state_dict` – a Python dictionary mapping layer names to their learned parameter tensors – and the model’s architecture definition separately. This ensures portability and reproducibility.

The primary mechanism for saving a model involves the `torch.save()` function. However, its versatility means we need to understand its appropriate usage contexts. I’ve observed that newcomers often gravitate toward saving the entire `nn.Module` object directly. While `torch.save(model)` will work in simple cases, this can introduce issues upon loading, primarily due to subtle variations in environment configurations, or the class definition's code undergoing modification. This method of directly saving the model includes not just weights but the entire class definition, which could be problematic if the class structure is changed.

The proper method involves saving the model's `state_dict`. This dictionary contains only the parameters, not the code used to define the model. Upon loading, we need the original model class definition (or a functionally equivalent one) and then load the `state_dict` into an instance of that class. This two-step process is key to robust model persistence. The `state_dict` can be saved in a variety of ways including but not limited to formats with the '.pt' or '.pth' extension and others. These formats are implicitly binary. The primary advantage of saving the `state_dict` is that only the learned weights are saved. The architecture of the network can be easily replicated on different machines by having the necessary code of the model class and load in the saved weights. This approach to model saving makes it much more likely to be compatible with different environments.

Here's an illustration with a simple feedforward neural network:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model, optimizer, and dummy data
input_size = 10
hidden_size = 5
output_size = 2
model = SimpleNet(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
dummy_input = torch.randn(1, input_size)
dummy_target = torch.randn(1, output_size)

# Perform a dummy training step to populate model weights
output = model(dummy_input)
loss_fn = nn.MSELoss()
loss = loss_fn(output, dummy_target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Save the model's state_dict
torch.save(model.state_dict(), 'simple_net.pth')
print("Model state_dict saved to simple_net.pth")

```
This code initializes a `SimpleNet` model, performs one training step to establish the weights, and then stores the `state_dict` using `torch.save()`. I've found it is good practice to save in the .pth file format for a model. It serves as a common convention. The saved file, `simple_net.pth`, now contains the numerical values of the network's weights. Notice, the architecture of the network is not saved. It is only the weights of the network.

The loading process is equally critical. The model’s architecture has to be defined again. Only then can we load the `state_dict`.

```python
import torch
import torch.nn as nn

# Re-define the model architecture (must be identical to the saved model definition)
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model with the same architecture
input_size = 10
hidden_size = 5
output_size = 2
model = SimpleNet(input_size, hidden_size, output_size)

# Load the saved state_dict
model.load_state_dict(torch.load('simple_net.pth'))
print("Model state_dict loaded from simple_net.pth")

# Verify by evaluating on a dummy input.
model.eval() # Set the model in evaluation mode.
dummy_input = torch.randn(1, input_size)
with torch.no_grad():
    output = model(dummy_input)
    print(f"Output of the loaded model: {output}")

```

This snippet demonstrates how to load the saved parameters. The key thing to recognize is that the model architecture is redefined in this script. Then, the `state_dict` is loaded into the created model instance using the `load_state_dict` method. The `.eval()` and the `with torch.no_grad():` statements are not essential for loading the model, but are best practices when evaluating inference on the model so that gradient computation is disabled. I've encountered situations where forgetting to explicitly put the model in the `eval()` mode led to inconsistent inference behavior, especially when layers like dropout or batch normalization are present.

A more complete example might require saving additional information, such as the optimizer state or epoch numbers, particularly when resuming training from a checkpoint. For that, a composite dictionary can be constructed. This is more robust than saving the model state and optimizer state separately, since the saving process is atomic.

```python
import torch
import torch.nn as nn
import torch.optim as optim
# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model, optimizer, and dummy data
input_size = 10
hidden_size = 5
output_size = 2
model = SimpleNet(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
dummy_input = torch.randn(1, input_size)
dummy_target = torch.randn(1, output_size)
epoch = 10
# Perform a dummy training step to populate model weights
output = model(dummy_input)
loss_fn = nn.MSELoss()
loss = loss_fn(output, dummy_target)
optimizer.zero_grad()
loss.backward()
optimizer.step()


checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

torch.save(checkpoint, 'checkpoint.pth')
print("Checkpoint saved to checkpoint.pth")

#Loading checkpoint

input_size = 10
hidden_size = 5
output_size = 2
model = SimpleNet(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)

checkpoint = torch.load('checkpoint.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

print(f"Checkpoint loaded, epoch: {epoch}")

```

This code saves not just the model's state, but also the state of the optimizer and the training epoch within a single dictionary. When loading a checkpoint for a model, it's crucial to load all parts of the checkpoint to maintain the training history if you plan to continue training. In the loading code, the `load_state_dict()` method is invoked on each relevant object.

When working with exceptionally large models, I’ve found saving the model's state dictionary on a CPU device more reliable. Even if training occurs on a GPU, it is a good practice to copy the model to CPU before saving. This reduces device specific issues. This can be done as follows:
`torch.save(model.cpu().state_dict(), "model.pth")`. When loading the saved model, it should be loaded onto the desired device, typically in the loading script.

For more in-depth exploration of model saving practices, the official PyTorch documentation on saving and loading models is paramount. In addition, research resources focused on reproducible deep learning workflows are beneficial. Finally, studying examples from prominent PyTorch repositories on GitHub can expose practical implementations and diverse scenarios. These resources cover topics such as exporting models for deployment, handling multiple GPUs and saving various types of model related files.
