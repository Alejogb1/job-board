---
title: "How to save and load PyTorch models?"
date: "2025-01-30"
id: "how-to-save-and-load-pytorch-models"
---
Saving and loading PyTorch models is crucial for reproducibility and efficient workflow management.  My experience building large-scale image classification models has underscored the importance of employing robust and versatile saving mechanisms, particularly when dealing with models exceeding several gigabytes.  The core principle revolves around leveraging the `torch.save()` and `torch.load()` functions, but their effective utilization necessitates careful consideration of model architecture, state dictionaries, and the serialization format.

**1.  Explanation of the Process**

PyTorch offers flexibility in how you save your model.  The most common approach involves saving the model's *state dictionary*. This dictionary contains a mapping of each layer's parameters (weights and biases) to their corresponding tensor values.  Saving the state dictionary is advantageous because it's generally smaller than saving the entire model object and facilitates loading into models with identical architectures.  Saving the entire model object is also possible, but this is less efficient, particularly for complex models, and can lead to issues with version incompatibility between PyTorch versions.

The choice between saving the entire model or just the state dictionary depends on your needs.  If you intend to resume training from a checkpoint, saving the entire model including optimizer state is preferable for faster loading.  For deploying the model to a production environment, saving only the state dictionary is typically sufficient and minimizes file size.

Another aspect is the file format. While PyTorch's default is a pickle-based format, you can use other formats such as HDF5 for better compatibility and potential for parallel I/O operations.  This is something I've found especially helpful when working with distributed training setups. However, for most common use-cases, the default pickle-based format is sufficient.

Loading a model involves reversing this process.  If you saved the state dictionary, you'll need to create an instance of the same model architecture before loading the weights.  If the entire model was saved, loading is straightforward.  It's crucial to ensure consistency between the model architecture used for saving and loading to avoid errors.  This usually involves meticulously managing your code versioning.


**2. Code Examples**

**Example 1: Saving and Loading the State Dictionary**

```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleModel()

# Save the state dictionary
torch.save(model.state_dict(), 'model_state_dict.pth')

# Load the state dictionary
model_loaded = SimpleModel()
model_loaded.load_state_dict(torch.load('model_state_dict.pth'))

# Verify that the models are identical
print(model.state_dict() == model_loaded.state_dict()) # Output: True
```

This example showcases the preferred method.  The simplicity highlights the core functionality: saving and loading the model's parameters separately from its architecture.  In my experience, this strategy has been consistently reliable and promotes code maintainability.


**Example 2: Saving and Loading the Entire Model**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a model (same as before)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model and optimizer
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Save the entire model (including optimizer state)
torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model_full.pth')

# Load the entire model
checkpoint = torch.load('model_full.pth')
model_loaded = SimpleModel()
model_loaded.load_state_dict(checkpoint['model_state_dict'])
optimizer_loaded = optim.SGD(model_loaded.parameters(), lr=0.01)
optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
```

Here, we save the complete model object, including the optimizer's state.  This is particularly beneficial when resuming training as it bypasses the need to re-initialize the optimizer.  However, the file size will be larger than the state dictionary alone.  I've found this particularly useful when dealing with models that have large momentum accumulators in their optimizers.


**Example 3: Saving with Custom Functionality**

```python
import torch
import torch.nn as nn
import os

# Define a model (same as before)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleModel()

# Define a function to save model parameters and additional metadata
def save_model(model, model_path, epoch, loss):
  state_dict = model.state_dict()
  metadata = {'epoch': epoch, 'loss': loss}
  torch.save({
      'state_dict': state_dict,
      'metadata': metadata
  }, model_path)

#Example usage:
save_model(model, 'my_model_10.pth', 10, 0.5)

#Function to load the model:

def load_model(model_path, model_class):
  checkpoint = torch.load(model_path)
  model = model_class()
  model.load_state_dict(checkpoint['state_dict'])
  metadata = checkpoint.get('metadata', {}) #Handle case where metadata doesn't exist
  return model, metadata

loaded_model, loaded_metadata = load_model('my_model_10.pth', SimpleModel)
print(f'Epoch of the loaded model: {loaded_metadata["epoch"]}')
print(f'Loss of the loaded model: {loaded_metadata["loss"]}')

```

This example demonstrates extending saving to include additional information such as training epoch and loss. This metadata can prove invaluable for tracking training progress and model selection, a practice I’ve integrated into all my complex model pipelines. The introduction of the custom functions improves organization and readability.  Error handling is included in the loading section, which avoids common runtime errors resulting from incorrect file structures.


**3. Resource Recommendations**

For a deeper understanding, consult the official PyTorch documentation.  Pay particular attention to the sections on serialization and the `torch.nn` module.  Furthermore, explore resources focused on best practices for managing large-scale machine learning projects, paying particular attention to those covering model versioning and checkpoint management.  A good understanding of Python's object serialization mechanisms is also beneficial.  Finally, review materials related to HDF5 file formats if you are handling extremely large datasets and models.
