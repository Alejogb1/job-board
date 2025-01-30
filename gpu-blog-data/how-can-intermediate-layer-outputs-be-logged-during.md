---
title: "How can intermediate layer outputs be logged during inference?"
date: "2025-01-30"
id: "how-can-intermediate-layer-outputs-be-logged-during"
---
Intermediate layer outputs during inference provide valuable insights into the internal state of a neural network and are critical for debugging, visualization, and understanding model behavior. I've found that accessing and logging these outputs involves strategic hooks within the model's forward pass, utilizing the framework's inherent capabilities for computation and data management. The primary challenge lies in doing this efficiently, without modifying the core inference logic or significantly impacting performance.

The core concept involves intercepting the activations—the output of a given layer—as they are computed during the forward pass. This can be accomplished using tools provided by deep learning frameworks, usually in the form of forward hooks, which are functions that are executed immediately after a specified layer computes its output. These hooks allow us to retrieve the layer's output tensor, which can then be processed as needed, such as logging to disk or saving to memory for later analysis. The activation tensors are typically high dimensional with significant memory impact, so I tend to log a statistically representative sample or compute aggregated descriptive measures instead.

Implementing this correctly avoids modifying the core model itself, maintains its original inference process, and facilitates the ability to easily turn the logging procedure on or off. This method contrasts with more intrusive approaches like modifying the forward method or adding extra output paths in the model structure, which tend to be more cumbersome to revert and can inadvertently alter the intended network execution.

Consider a scenario where I am working with a convolutional neural network written using a popular deep learning framework (I will abstract the framework’s specific syntax). Let’s imagine a simple CNN for image classification. The model can be described, for this purpose, abstractly as follows: `input -> Conv2d_1 -> ReLU_1 -> MaxPool_1 -> Conv2d_2 -> ReLU_2 -> MaxPool_2 -> FC_1 -> ReLU_3 -> FC_2 -> output`. I want to log the outputs of `ReLU_1`, `MaxPool_2`, and `FC_1`.

**Example 1: Logging intermediate outputs using forward hooks**

Here is an illustrative code example on how we would implement this functionality. We create a dictionary to store hook functions and then define the hook logic. After the layers have been defined (assuming, as I usually do, a modular approach), we register the hooks onto the relevant layers of the network.

```python
import numpy as np  # Using numpy for demonstration purposes

# Assuming a conceptual model structure
class Conv2d:
    def __init__(self, filters, kernel_size, stride):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
    def __call__(self, input):
        return np.random.rand(input.shape[0], self.filters, (input.shape[2] - self.kernel_size) // self.stride + 1, (input.shape[3] - self.kernel_size) // self.stride + 1)

class ReLU:
    def __init__(self):
        pass
    def __call__(self, input):
      return np.maximum(0, input)

class MaxPool:
    def __init__(self, kernel_size, stride):
      self.kernel_size = kernel_size
      self.stride = stride
    def __call__(self, input):
        return np.random.rand(input.shape[0], input.shape[1], (input.shape[2] - self.kernel_size) // self.stride + 1, (input.shape[3] - self.kernel_size) // self.stride + 1)

class FC:
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, input):
        input_flat = np.reshape(input, (input.shape[0], -1))
        return np.random.rand(input.shape[0], self.output_size)

class CNN:
  def __init__(self, input_channels = 3, num_classes = 10):
    self.conv1 = Conv2d(32, 3, 1)
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool(2, 2)
    self.conv2 = Conv2d(64, 3, 1)
    self.relu2 = ReLU()
    self.maxpool2 = MaxPool(2, 2)
    self.fc1 = FC(128)
    self.relu3 = ReLU()
    self.fc2 = FC(num_classes)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)
    x = self.fc1(x)
    x = self.relu3(x)
    x = self.fc2(x)
    return x

def create_forward_hook(layer_name, logged_activations):
    def hook(input, output):
      logged_activations[layer_name] = output # Save the layer output
    return hook

def register_hooks(model, hook_names):
    logged_activations = {}
    hooks = {}
    for name in hook_names:
      hook_fn = create_forward_hook(name, logged_activations)
      if name == "ReLU_1":
        hooks["ReLU_1"] = model.relu1
      elif name == "MaxPool_2":
        hooks["MaxPool_2"] = model.maxpool2
      elif name == "FC_1":
        hooks["FC_1"] = model.fc1
      hooks[name].forward_hook = hook_fn  # Assign the hook to the layer's attribute
    return logged_activations

# Instantiating and using the model
model = CNN()
hook_names = ["ReLU_1", "MaxPool_2", "FC_1"]
logged_activations = register_hooks(model, hook_names)

# Dummy input mimicking a batch of 3 images of shape 28x28 with 3 channels
dummy_input = np.random.rand(3, 3, 28, 28)
output = model.forward(dummy_input) # Executing the forward pass with hooks enabled

# Print shape of logged activations
for name, activation in logged_activations.items():
    print(f"Shape of {name} activation: {activation.shape}")
```

This approach cleanly registers the hooks and saves the activations in the `logged_activations` dictionary. Note that the specific implementation of hooks varies across frameworks but the concept remains consistent: the hook intercepts the output of a layer and allows us to access and modify it.

**Example 2: Aggregating statistics of the activations**

Directly logging every activation can be excessively memory-intensive for large networks or datasets. A more efficient approach is to calculate descriptive statistics of the activations, such as the mean, standard deviation, or percentiles, and log these instead. This approach reduces the amount of data stored while still capturing relevant information about the layer's behavior.

```python
def create_stats_hook(layer_name, logged_stats):
    def hook(input, output):
      logged_stats[layer_name] = {
        "mean": np.mean(output),
        "std": np.std(output),
        "min": np.min(output),
        "max": np.max(output)
      }
    return hook

def register_stats_hooks(model, hook_names):
    logged_stats = {}
    hooks = {}
    for name in hook_names:
      hook_fn = create_stats_hook(name, logged_stats)
      if name == "ReLU_1":
        hooks["ReLU_1"] = model.relu1
      elif name == "MaxPool_2":
        hooks["MaxPool_2"] = model.maxpool2
      elif name == "FC_1":
        hooks["FC_1"] = model.fc1
      hooks[name].forward_hook = hook_fn
    return logged_stats


# Instantiating and using the model
model = CNN()
hook_names = ["ReLU_1", "MaxPool_2", "FC_1"]
logged_stats = register_stats_hooks(model, hook_names)

# Dummy input mimicking a batch of 3 images of shape 28x28 with 3 channels
dummy_input = np.random.rand(3, 3, 28, 28)
output = model.forward(dummy_input) # Executing the forward pass with hooks enabled

# Print statistics for each layer's activations
for name, stats in logged_stats.items():
    print(f"Statistics for {name} activation:")
    for stat_name, stat_value in stats.items():
      print(f"  {stat_name}: {stat_value}")
```

This modification avoids storing the raw activation tensors, making the process more memory-efficient for large models.

**Example 3: Conditional logging based on batch sample**

In many cases, inspecting activations of every sample within a large batch is unnecessary. I usually perform logging only on a subset of the input, based on index in the batch. This can significantly reduce the computation load and disk space needed to maintain output logs. This allows the logging to focus on a representative sample of inputs.

```python
def create_conditional_hook(layer_name, logged_activations, sample_indices):
    def hook(input, output):
        logged_activations[layer_name] = output[sample_indices] # Save activations only for given indexes
    return hook

def register_conditional_hooks(model, hook_names, sample_indices):
  logged_activations = {}
  hooks = {}
  for name in hook_names:
    hook_fn = create_conditional_hook(name, logged_activations, sample_indices)
    if name == "ReLU_1":
      hooks["ReLU_1"] = model.relu1
    elif name == "MaxPool_2":
      hooks["MaxPool_2"] = model.maxpool2
    elif name == "FC_1":
      hooks["FC_1"] = model.fc1
    hooks[name].forward_hook = hook_fn
  return logged_activations


# Instantiating and using the model
model = CNN()
hook_names = ["ReLU_1", "MaxPool_2", "FC_1"]
sample_indices = [0, 2]  # Log for first and third input
logged_activations = register_conditional_hooks(model, hook_names, sample_indices)

# Dummy input mimicking a batch of 3 images of shape 28x28 with 3 channels
dummy_input = np.random.rand(3, 3, 28, 28)
output = model.forward(dummy_input)

# Print shapes of logged activations
for name, activation in logged_activations.items():
    print(f"Shape of {name} activation: {activation.shape}")
```

By implementing conditional logging, the computational and memory overheads can be further minimized. The shape of the logged activation now reflects the batch sample indices that were selected.

The use of hooks allows for a flexible method to observe model behavior without modifying its internal structure. The three examples above illustrate different approaches that can be taken to balance the logging need with computational and memory constraints. These methods should be adapted depending on the model complexity and the amount of information needed to effectively evaluate or debug a model.

For further information about forward hooks, I recommend consulting the documentation and tutorials associated with your specific deep learning framework. These resources usually provide more in-depth explanations and examples specific to their API, and are more effective than secondary references that can miss relevant implementation details. Additionally, consider researching resources related to data analysis and visualization techniques for neural network activations, since interpreting them is equally as important as acquiring the data.
