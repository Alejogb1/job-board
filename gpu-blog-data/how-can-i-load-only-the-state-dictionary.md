---
title: "How can I load only the state dictionary of a PyTorch model saved with `torch.save()`?"
date: "2025-01-30"
id: "how-can-i-load-only-the-state-dictionary"
---
PyTorch’s `torch.save()` function, when used without specifying the target save type, persists the entire model object. This encompasses not just the model's parameters (weights and biases) but also its architecture definition and any additional attributes associated with it. Consequently, directly loading this full object via `torch.load()` results in an unnecessary memory overhead when the sole interest is in the model’s state dictionary. I've encountered situations where loading numerous large models just for their parameters consumed disproportionate memory, prompting me to devise methods for selectively loading only the state dictionary.

The state dictionary, accessible via the `model.state_dict()` method, is a Python dictionary that maps each layer or parameter name to its corresponding tensor. It essentially represents the learnable components of the model. The primary reason one might want to load *only* this dictionary is to transfer learned weights to a different instance of the same model architecture. This is often necessary in scenarios like: (1) loading pre-trained weights onto a model that was constructed differently in code, (2) fine-tuning a model that might have been loaded from a checkpoint that contained additional information, or (3) sharing parameters between models that utilize the same architecture but are instantiated independently. Attempting to apply `torch.load()` and then extracting the state dictionary will still load the complete model, so more efficient approaches are needed.

The optimal way to load only the state dictionary is by directly specifying the desired load format when using `torch.load()`. The function's `map_location` argument allows you to remap devices on load. More significantly for this specific situation, `torch.load()` can also directly load a dictionary or a tensor, if that's what was saved in the first place. If, during the saving phase, `torch.save(model.state_dict(), 'model_parameters.pt')` was used, the subsequent loading procedure becomes straightforward. We would use `torch.load('model_parameters.pt')` directly, obtaining just the state dictionary without the overhead of the model itself. However, we're usually not in control of the saving procedure and therefore, must cope with having a full model persisted, requiring more careful techniques to load just the state_dict.

We can approach this in a few ways. One approach is using `torch.load()` with no modification and then immediately retrieving the `state_dict`. This, however, is less efficient since it first loads the entire model. The crucial step for efficient loading is to modify how `torch.load` behaves. This is achievable by passing in a custom `map_location` function. This function is invoked during the deserialization process and enables us to filter out all the attributes except the state dictionary. Using a lambda function we can achieve the same result but often the lambda function is more difficult to debug, so we are going to demonstrate with a normal function instead. This is going to enable us to discard all of the overhead associated with the model object itself and solely load the state dictionary.

Let's examine specific code examples to illustrate this:

**Example 1: Basic Loading (Inefficient)**

```python
import torch
import torch.nn as nn

# Assume a simple model
class ExampleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create and save a model (full model save)
model_full = ExampleModel(10, 20, 2)
torch.save(model_full, 'full_model.pt')

# Inefficient Load (loads entire model first)
loaded_model = torch.load('full_model.pt')
state_dict = loaded_model.state_dict() #Extract the state dictionary

print(f"State dict keys after inefficient load: {list(state_dict.keys())[:2]} ...")
```

In this first example, I save the full model. Then, during loading, I utilize `torch.load`, which returns the model, and I then extract the state_dict. This is inefficient since the full model object is loaded unnecessarily. The output here confirms that the state dictionary is successfully loaded, but only after loading the full model object into memory.

**Example 2: Efficient Loading using a Custom `map_location` Function**

```python
import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_state_dict_only(storage, location):
    loaded = torch.load(storage, map_location=location)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]
    elif hasattr(loaded, "state_dict"):
         return loaded.state_dict()
    return loaded

# Create and save a model (full model save)
model_full = ExampleModel(10, 20, 2)
torch.save(model_full, 'full_model.pt')

# Load only the state dictionary
state_dict = torch.load('full_model.pt', map_location=load_state_dict_only)

print(f"State dict keys after efficient load: {list(state_dict.keys())[:2]} ...")

# Example use with a fresh model
new_model = ExampleModel(10, 20, 2)
new_model.load_state_dict(state_dict)
```
This example demonstrates the primary technique for loading just the state dictionary. The `load_state_dict_only` function will catch both instances when a full model is provided or when the original checkpoint has a dictionary object with the state dict, a useful property when using techniques such as distributed training. We pass this function as the `map_location` argument to `torch.load`. Instead of deserializing all attributes of the model, the function intercepts the loading process and only returns the `state_dict`, which is then immediately loaded by PyTorch. Lastly, I show how to load the state_dict into a new model, ensuring its parameters are properly populated.

**Example 3: A more robust version of the mapping function**
```python
import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def robust_load_state_dict_only(storage, location):
    def internal_load(storage, location):
        loaded = torch.load(storage, map_location=location)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            return loaded["state_dict"]
        elif hasattr(loaded, "state_dict"):
            return loaded.state_dict()
        return loaded
    try:
      return internal_load(storage, location)
    except Exception as e:
      print(f"Failed with an exception {e}, using default load")
      return torch.load(storage, map_location=location).state_dict()

# Create and save a model (full model save)
model_full = ExampleModel(10, 20, 2)
torch.save(model_full, 'full_model.pt')

# Load only the state dictionary
state_dict = torch.load('full_model.pt', map_location=robust_load_state_dict_only)

print(f"State dict keys after robust load: {list(state_dict.keys())[:2]} ...")

# Example use with a fresh model
new_model = ExampleModel(10, 20, 2)
new_model.load_state_dict(state_dict)
```

This third example builds on the previous by adding error handling. We wrap the internal loading function in a try/except and if the load fails, we revert to the default loading behavior, extract the `state_dict`, and continue.

For further information and a comprehensive understanding of PyTorch model saving and loading mechanisms, I'd recommend the official PyTorch documentation, particularly the sections covering model persistence and the `torch.save` and `torch.load` functions. Additionally, exploring the PyTorch tutorials related to transfer learning can provide practical insights into scenarios where loading only the state dictionary is crucial. Finally, consulting the PyTorch forums and community discussions will provide perspectives from other practitioners who have encountered similar challenges. By experimenting with variations of these loading procedures and referring to the documentation, one can effectively manage the memory requirements and model loading behavior in their projects.
