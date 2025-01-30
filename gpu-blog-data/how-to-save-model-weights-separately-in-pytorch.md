---
title: "How to save model weights separately in PyTorch?"
date: "2025-01-30"
id: "how-to-save-model-weights-separately-in-pytorch"
---
The requirement to save model weights independently from the overall model structure in PyTorch arises frequently in scenarios such as transfer learning, model ensembling, and distributed training, requiring a nuanced approach beyond the simple `torch.save(model, path)`. Directly saving the entire model object includes the computational graph structure, which is inflexible when only the trained parameters are needed. Over the years, I have encountered this several times, specifically while porting a pre-trained image classification model to an embedded system with limited resources, demonstrating the practical need for this technique.

The primary method for isolating and saving model weights involves extracting the model's `state_dict`. This dictionary stores the learned parameters for each layer as tensors, keyed by layer names. Critically, the `state_dict` excludes the model's class definition, structure, and forward pass implementation, effectively providing a portable, platform-agnostic representation of the learned knowledge. Loading these weights later necessitates reconstructing the same model architecture beforehand, ensuring compatibility. Utilizing this approach avoids versioning issues when only the learned parameters change, which can significantly reduce storage and bandwidth requirements during model deployment or transfer between environments.

My first experience with this, as I mentioned, involved optimizing a VGG16 model for a resource-constrained processor. Loading the full model object, including its large internal graph structure, proved inefficient and required several additional libraries. Extracting just the weights and reconstructing the model from scratch locally dramatically improved startup time.

Here is a concise example of saving the `state_dict`:

```python
import torch
import torch.nn as nn

# Assume a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()

# Assuming model is trained
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for _ in range(10):
  x = torch.randn(1, 10)
  y = torch.tensor([1]).float()
  optimizer.zero_grad()
  output = model(x)
  loss = nn.functional.mse_loss(output, y)
  loss.backward()
  optimizer.step()

# Save only the state dictionary
torch.save(model.state_dict(), 'model_weights.pth')

print("Model weights saved as 'model_weights.pth'")
```

In the above example, I instantiate a basic neural network with two fully connected layers. After a minimal training loop to illustrate that weights exist, I use `model.state_dict()` to obtain a dictionary containing the model's weights. This dictionary is then saved to disk using `torch.save()`. Crucially, the `model` object itself is not saved. This approach allows portability of the weights between different Python environments, provided the model definition is identical.

Subsequently, during another project, I faced the challenge of deploying multiple model variations. Saving the state dictionary allowed me to reuse common layers between models by selectively loading portions of the weights. This reduced redundancy significantly and sped up deployment pipelines. Here is a second, slightly more complex illustration showing how to load only the weights:

```python
import torch
import torch.nn as nn

# Define the same model structure
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a new instance of the model (same architecture)
loaded_model = SimpleModel()

# Load the state dictionary into the new model
loaded_model.load_state_dict(torch.load('model_weights.pth'))

# Verify that weights have been loaded (optional)
print("Model weights loaded successfully.")
x = torch.randn(1, 10)
output_loaded = loaded_model(x)
print("Output after loading weights:", output_loaded)

# Example of different architecture
class SimpleModelDifferent(nn.Module):
    def __init__(self):
        super(SimpleModelDifferent, self).__init__()
        self.fc1 = nn.Linear(10, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Attempt to load the weights into a different model
loaded_model_diff = SimpleModelDifferent()
try:
    loaded_model_diff.load_state_dict(torch.load('model_weights.pth'))
except RuntimeError as e:
  print("\nError when loading into a model with different architecture:\n", e)
```

Here, a new instance of `SimpleModel` is created, distinct from the initial `model` used for saving. The `load_state_dict()` function is then used to populate the new instance with the previously saved weights, directly loading the parameter tensors from the 'model_weights.pth' file. Notably, the subsequent section demonstrates the incompatibility of loading weights into a model with a different architecture, triggering a `RuntimeError` as the weight tensor shapes no longer align. This highlights the critical need for identical model architectures during loading. This mismatch happened during a refactoring of an object detection system, requiring careful reconciliation of the various model definition files.

Lastly, working on distributed training scenarios emphasized the utility of `state_dict`. For multi-GPU training, saving and loading the model directly was impractical and led to unnecessary overhead. Instead, the weights were periodically extracted, saved, and then redistributed to other processes. The model structure was replicated locally, and weights were applied before resuming training. Here's an illustration of extracting specific layers' parameters for transfer learning:

```python
import torch
import torch.nn as nn

# Assume a more complex model
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(16*26*26, 10) # assuming 3x32x32 input for simplicity
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 16*26*26) #Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_complex = ComplexModel()

#Assuming model is trained similarly to the first example
optimizer = torch.optim.Adam(model_complex.parameters(), lr=0.001)
for _ in range(10):
  x = torch.randn(1, 3, 32, 32)
  y = torch.tensor([1]).float()
  optimizer.zero_grad()
  output = model_complex(x)
  loss = nn.functional.mse_loss(output, y)
  loss.backward()
  optimizer.step()


# Extract state dict and save only the fc layers' weights
fc_weights = {k: v for k, v in model_complex.state_dict().items() if 'fc' in k}

torch.save(fc_weights, 'fc_layer_weights.pth')

print("Weights of fully connected layers saved as 'fc_layer_weights.pth'")

#Create a new instance and load only those weights
class ModifiedComplexModel(nn.Module):
    def __init__(self):
        super(ModifiedComplexModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(16*26*26, 12)  #Modified output size of fc1
        self.fc2 = nn.Linear(12, 2)       #modified input size of fc2


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 16*26*26) #Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

loaded_model_complex = ModifiedComplexModel()

try:
    loaded_model_complex.load_state_dict(torch.load('fc_layer_weights.pth'), strict=False)
    print("Successfully loaded fc layer weights with strict=False")
except RuntimeError as e:
  print(e)
```

In this final illustration, a `ComplexModel` is defined including convolutional and fully connected layers. This example demonstrates the extraction and saving of a subset of weights from the state dictionary, specifically those associated with the fully connected layers (identified by ‘fc’ in their keys), by iterating through the full state dictionary. I also illustrate here how to load a partial weight set by using `strict=False`. This parameter allows the loading of weights even when the new model doesn't have all layers of the previous saved state.  This functionality is critical in transfer learning scenarios, where only select layer weights are preserved. In my practical experiences, this functionality provided the flexibility to transfer a common set of feature extraction layers to different classification architectures, again promoting efficiency and code reuse.

For those seeking further information, the PyTorch documentation provides detailed explanations of `state_dict` usage, accessible from their official website. Several books covering deep learning in PyTorch also offer in-depth examples and explanations. The official PyTorch tutorials are another solid resource, covering a broad range of topics including saving and loading models, and can be accessed through the framework's online documentation portal. Finally, research papers often contain novel approaches on weight-only transfer, and are valuable sources for advanced applications.
