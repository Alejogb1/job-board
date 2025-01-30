---
title: "How can I access 'in_features' and 'fc' attributes of a sequential object?"
date: "2025-01-30"
id: "how-can-i-access-infeatures-and-fc-attributes"
---
The inability to directly access `in_features` and `fc` attributes within a generic `torch.nn.Sequential` object stems from its nature as a container, not a specific layer possessing those attributes. These attributes are typically associated with specific layer types, such as `torch.nn.Linear` or convolutional layers, rather than the sequential wrapper itself. This often leads to confusion, especially when users are dealing with dynamically constructed models where layers are added sequentially. I've personally encountered this hurdle multiple times while debugging custom architectures and have developed a systematic approach to extract this information.

The `torch.nn.Sequential` class itself is designed to hold an ordered sequence of layers. It's essentially a glorified list that applies each contained module to the input, one after the other. Consequently, properties like `in_features` and `fc` are not directly stored on the `Sequential` object but are specific to the individual layers residing *within* it. Therefore, accessing these requires traversing the sequence and identifying the layers that possess these attributes, usually a linear (fully connected) layer, or inspecting the layer that is immediately before the output layer of the model if `fc` refers to the final classification layer.

The general strategy for extracting this information involves iterating through the modules in the sequential object. During iteration, we must check the type of each module. If a module is of the desired type, we can then access its attributes. When working with linear layers, particularly in models aimed at classification, `in_features` represents the size of the input vector to the linear layer. This allows for calculating the number of parameters in the layer and can be crucial for debugging or modifications. Likewise, `fc` is not a standard attribute but often refers to the final fully connected layer in a classification model that directly computes scores for all classes. In such cases, we must identify this specific `torch.nn.Linear` layer and potentially modify it, if required.

Let’s consider three scenarios that illustrate this process and offer solutions:

**Example 1: Accessing `in_features` of the First Linear Layer**

In this first case, let’s assume we wish to get the `in_features` attribute from the *first* linear layer in a sequential network:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

first_linear_layer = None

for module in model:
    if isinstance(module, nn.Linear):
        first_linear_layer = module
        break

if first_linear_layer:
    in_features = first_linear_layer.in_features
    print(f"The in_features of the first linear layer: {in_features}")
else:
    print("No linear layer found in the sequential model.")

```
This example iterates through the layers of the `model`. Upon encountering a `nn.Linear` layer, it assigns that layer to the `first_linear_layer` variable and exits the loop. After the loop, we inspect if `first_linear_layer` was found. If so, we can retrieve and print the `in_features` attribute from that linear layer. This approach isolates and accesses the desired attribute safely, handling cases where a linear layer isn't present. The loop breaks after the first linear layer, to keep with the prompt requirement.

**Example 2: Identifying and Modifying the `fc` Layer (Last Linear Layer)**

Often, the final classification layer is a linear layer which can be considered the `fc` layer. Let's demonstrate how to identify this and potentially modify its output features:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 100),
    nn.ReLU(),
    nn.Linear(100, 5)
)

last_linear_layer = None

for module in reversed(model):
    if isinstance(module, nn.Linear):
        last_linear_layer = module
        break

if last_linear_layer:
    num_classes_old = last_linear_layer.out_features
    print(f"Original number of output features: {num_classes_old}")

    num_classes_new = 2 # For a binary classification problem
    last_linear_layer = nn.Linear(last_linear_layer.in_features, num_classes_new)
    
    #Replace the original layer with the new one:
    
    for i, module in enumerate(model):
        if isinstance(module, nn.Linear) and model[i] == model[-1]:
            model[i] = last_linear_layer
            break


    print(f"Modified number of output features: {last_linear_layer.out_features}")
    
    #Now check if last layer has the correct parameters.
    for module in reversed(model):
      if isinstance(module, nn.Linear):
        print(f"Confirmed number of output features after replacement: {module.out_features}")
        break

else:
    print("No linear layer found in the sequential model.")

```

In this scenario, we iterate backward using `reversed()` to efficiently find the last linear layer.  The code also demonstrates how to identify the last linear layer, modify its output dimension, create a new layer with the modified output feature, and replace the old layer with the new layer in place, within the sequential object. Note that directly modifying `out_features` attribute is not possible, so we need to create a new layer with the desired output size. The output shows the old and new number of output features to make it clear how this change takes effect.

**Example 3: Extracting `in_features` from Multiple Linear Layers**

This third example demonstrates how to extract and report in_features from multiple linear layers within the sequence.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.BatchNorm1d(30),
    nn.Linear(30, 5)
)

linear_layers = []

for module in model:
    if isinstance(module, nn.Linear):
        linear_layers.append(module)

if linear_layers:
    for i, layer in enumerate(linear_layers):
        print(f"Linear layer {i+1}: in_features = {layer.in_features}")
else:
    print("No linear layer found in the sequential model.")
```

This example is straightforward. It iterates through the modules, appending all linear layers to the `linear_layers` list. It then iterates over this list, printing the `in_features` of each identified layer, demonstrating how to access this attribute for all such layers within the network. The index in the output is incremented by one to provide a human-friendly indexing.

In summary, accessing attributes like `in_features` or identifying a specific layer that is acting as `fc` in a `torch.nn.Sequential` model requires an understanding of the layered structure. The `Sequential` class is just a container, therefore, we must iterate over the individual components to extract what we need. By using type checking and strategic looping, as demonstrated in my examples, we can efficiently access the required properties or layers, even if they are buried within the network.

For further understanding of PyTorch modules, the official PyTorch documentation provides extensive details on `torch.nn` and all the different layers available, such as linear, convolution, and recurrent layers. Additionally, numerous tutorials on building custom models and iterating through layer structures can provide valuable experience. The "Deep Learning with PyTorch" book and tutorials on the pytorch.org website, are especially beneficial. I have benefited from these resources while building complex models. Remember that understanding the module type is paramount to accessing its specific attributes.
