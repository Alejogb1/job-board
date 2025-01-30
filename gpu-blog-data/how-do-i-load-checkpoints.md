---
title: "How do I load checkpoints?"
date: "2025-01-30"
id: "how-do-i-load-checkpoints"
---
The core challenge in loading checkpoints lies in ensuring state consistency between the saved data and the model infrastructure. Checkpoints, in essence, are serialized snapshots of a model’s parameters, optimizer states, and other relevant training information at a specific point in time. I’ve encountered various complexities in managing these snapshots across different frameworks and project demands, necessitating a robust and adaptable methodology. Loading them correctly is critical not only for restoring training progress but also for deployment and inference.

The process generally involves three distinct stages: locating the checkpoint file, deserializing the data, and then mapping the deserialized information back to the correct components of your model and training loop. Frameworks like TensorFlow and PyTorch offer utilities to streamline this, but a fundamental understanding of what's happening beneath these layers is beneficial for debugging and customization, especially with bespoke training pipelines. For instance, the naming conventions of checkpoint files can introduce ambiguity if not standardized, and discrepancies between the saving and loading environments (e.g., different versions of dependencies or hardware configurations) can invalidate the process entirely.

In my experience, the most prevalent error source is a mismatch in model architecture between the saving and loading phases. If the model structure changes even slightly, for example, a layer is added or removed, the dimensions of the saved tensors will not align with the current model. This will result in errors during the parameter loading. In some instances, a framework will allow partial loading, which is useful if you have intentionally changed the model architecture, but you need to make sure the layer names are consistent.

Here's how I typically approach this in practice, specifically considering a PyTorch scenario:

**Example 1: Basic Checkpoint Loading**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume a simple model and optimizer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 1. Simulating a saved checkpoint
checkpoint_data = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
torch.save(checkpoint_data, 'checkpoint.pth')

# 2. Loading the checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 3. Accessing specific checkpoint information.
loaded_epoch = checkpoint['epoch']
print(f"Loaded model state from epoch: {loaded_epoch}")

# Model and optimizer are now loaded with the saved states.
# Verification (e.g. inference) can be done here
```

This first example demonstrates the foundational process of loading a checkpoint in PyTorch. I structure the save into a dictionary that includes the model’s state (parameter weights and biases), the optimizer’s state (momentum terms etc.), and a useful indicator of training progress, such as the current epoch. The `torch.load` function deserializes the saved file back into a Python dictionary. Subsequently, the `load_state_dict` method on the model and optimizer instances transfers the saved parameter values and optimizer states, respectively, completing the restoration. The inclusion of additional metadata, such as the epoch number, aids with continued training or tracking loaded checkpoints. I frequently use this to log the loaded epoch in my training runs, preventing accidental overrides.

**Example 2: Handling Potential Key Mismatches**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume an original model and optimizer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 1. Simulating a saved checkpoint (from model version 1)
checkpoint_data = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
torch.save(checkpoint_data, 'checkpoint.pth')

# 2. Assume model architecture changed, a new layer is added
class ModifiedModel(nn.Module):
    def __init__(self):
        super(ModifiedModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
       x = self.linear1(x)
       return self.linear2(x)

new_model = ModifiedModel()

# 3. Trying to load the old state dictionary.
checkpoint = torch.load('checkpoint.pth')
try:
   new_model.load_state_dict(checkpoint['model_state_dict'])
except RuntimeError as e:
    print(f"Error loading checkpoint: {e}")

#4. Attempt partial loading to keep weights of the common parts of the model
new_model_state_dict = new_model.state_dict()
saved_state_dict = checkpoint['model_state_dict']

filtered_state_dict = {k: v for k, v in saved_state_dict.items() if k in new_model_state_dict}
new_model_state_dict.update(filtered_state_dict)
new_model.load_state_dict(new_model_state_dict)
print(f"Checkpoint loaded partially, using weights of the first layer.")

#Verification of the loading process
print(f"Weights from the saved checkpoint: {list(saved_state_dict.items())[0][1]}")
print(f"Weights from the modified model: {list(new_model.state_dict().items())[0][1]}")
```

This second example showcases a critical scenario I often encounter: modifications to the model’s architecture between saving and loading. When the structure of the model changes, directly loading the old state dictionary generates a `RuntimeError` because of shape mismatches. I've found that a flexible way to mitigate this is to first load the current model's state, and then selectively update it with the saved weights which are present in the old model and the current one. This allows me to retain partial model state, like the original `linear` layer weights, while training the new `linear2` layer from scratch. A more sophisticated approach, for example in the case of adding new layers, would involve initializing the weights using custom methods to avoid random initialization. I find it best practice to explicitly log all loading errors and implement a way to avoid crashing because of a bad checkpoint.

**Example 3: Loading from different devices (CPU/GPU)**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume a simple model and optimizer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 1. Simulate training on a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Simulate saving checkpoint on GPU
if torch.cuda.is_available():
    checkpoint_data = {
        'epoch': 10,
        'model_state_dict': model.cpu().state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint_data, 'checkpoint.pth')

    # 3. Loading the checkpoint on CPU
    checkpoint = torch.load('checkpoint.pth',map_location=torch.device('cpu'))

    new_model = SimpleModel()
    new_model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Loading the checkpoint on a different GPU
    if torch.cuda.device_count() > 1 :
        checkpoint = torch.load('checkpoint.pth',map_location=torch.device('cuda:1'))
        new_model_cuda = SimpleModel()
        new_model_cuda.load_state_dict(checkpoint['model_state_dict'])
        new_model_cuda.to(torch.device("cuda:1"))

    else:
        print("Single GPU detected. Loading weights on the same device.")

    print("Model loaded on CPU/GPU successfully.")

else:
    # 3. Loading the checkpoint on CPU
    checkpoint_data = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint_data, 'checkpoint.pth')
    checkpoint = torch.load('checkpoint.pth')
    new_model = SimpleModel()
    new_model.load_state_dict(checkpoint['model_state_dict'])
    print("CPU only loading.")
```

My final example illustrates the necessity of handling device placement during loading. Specifically, if a model is trained on a GPU, the saved parameter tensors will reside on that device. When loading on a CPU, without specifying `map_location` to `torch.load`, the tensors won't be copied correctly and this can lead to unexpected memory errors or incorrect inferences. Using `map_location` allows me to selectively transfer the tensors when loading. Furthermore, it can be used to load a checkpoint to a specific GPU if multiple GPUs are available. It's critical to explicitly specify this, as inconsistencies can lead to very hard to debug errors in distributed training scenarios.

For further study, I suggest exploring the official framework documentation, such as the PyTorch tutorial on saving and loading models. I also recommend looking at documentation for deep learning model architectures commonly used to understand the state dictionaries associated with various architectures. I have found that examining code from open-source implementations has also been extremely helpful in understanding best practices when loading checkpoints. Furthermore, research papers that detail specific model architectures often mention nuances in loading weights, which can help you prepare for specific use cases.
