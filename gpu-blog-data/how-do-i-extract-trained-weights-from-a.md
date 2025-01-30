---
title: "How do I extract trained weights from a Skorch neural network?"
date: "2025-01-30"
id: "how-do-i-extract-trained-weights-from-a"
---
Extracting trained weights from a Skorch neural network, while not directly exposed as a single attribute, is a straightforward process involving accessing the underlying PyTorch model and its state dictionary. I've often needed to do this, especially when transferring learned representations to different architectures or for fine-grained model analysis. The key lies in understanding that Skorch acts as a wrapper around PyTorch, providing a convenient API for training while preserving the fundamental PyTorch model structure. The weights aren't Skorch's, they belong to the underlying PyTorch `nn.Module` instance.

The core mechanism for accessing the weights involves accessing the `module_` attribute of a fitted `NeuralNet` instance, and then using PyTorch's `state_dict()` method. The `module_` attribute holds the instantiated PyTorch model that was used for training, and the `state_dict()` method provides a dictionary containing the model's trainable parameters (weights and biases). This dictionary is organized with keys representing the module and layer names, and corresponding values that are PyTorch tensors holding the actual weight and bias values.

Let’s illustrate this with three practical scenarios. The first example involves a simple multi-layer perceptron (MLP), showcasing the extraction process for a basic fully connected network. I've encountered this situation frequently while experimenting with different neural net topologies.

```python
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import numpy as np

# 1. Define a simple PyTorch model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Create a Skorch NeuralNetClassifier
model = NeuralNetClassifier(
    module=MLP,
    module__input_size=10,
    module__hidden_size=20,
    module__output_size=2,
    max_epochs=5,
    lr=0.01,
)

# 3. Generate some dummy training data
X = np.random.rand(100, 10).astype(np.float32)
y = np.random.randint(0, 2, 100).astype(np.int64)

# 4. Train the model
model.fit(X, y)

# 5. Extract trained weights
trained_model = model.module_
weights = trained_model.state_dict()

# 6. Print keys and the shape of one weight tensor
print(f"Keys in the state dictionary: {weights.keys()}")
print(f"Shape of 'fc1.weight': {weights['fc1.weight'].shape}")
```
In this example, after training, we access the underlying PyTorch model through `model.module_` and obtain the state dictionary using `.state_dict()`. The dictionary keys reveal the naming convention used for the layers, such as `fc1.weight` for the weights of the first fully connected layer. The shape of the tensor confirms that it holds the expected matrix of the weight parameters for the layer. When dealing with more complex networks, this organization is essential to navigate the structure of the trained parameters.

My second case involves a convolutional neural network (CNN) commonly used for image classification tasks. I've had to extract weights from these models often for transfer learning or feature extraction, where the learned filters are crucial.

```python
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import numpy as np

# 1. Define a simple CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32*7*7, num_classes) # Assumes input of 28x28, post pooling will become 7x7

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# 2. Create a Skorch NeuralNetClassifier
model = NeuralNetClassifier(
    module=CNN,
    module__num_classes=2,
    max_epochs=5,
    lr=0.01,
)

# 3. Generate dummy image-like training data
X = np.random.rand(100, 3, 28, 28).astype(np.float32)
y = np.random.randint(0, 2, 100).astype(np.int64)

# 4. Train the model
model.fit(X, y)

# 5. Extract trained weights
trained_model = model.module_
weights = trained_model.state_dict()

# 6. Print the shapes of some of the weights
print(f"Shape of 'conv1.weight': {weights['conv1.weight'].shape}")
print(f"Shape of 'fc.weight': {weights['fc.weight'].shape}")
```

This demonstrates the extraction of convolutional filter weights from `conv1` and the linear layer weights from `fc`. The structure of `conv1.weight` shows the expected (out_channels, in_channels, kernel_height, kernel_width) format, which in this case is (16, 3, 3, 3) for a kernel with 16 output channels, 3 input channels and size 3x3. Understanding this format is essential when applying these weights in transfer learning contexts, where feature map dimensions need to be carefully aligned.

Finally, let's explore a case where we need to access the parameters of a custom-defined, more complex network module, incorporating batch normalization. This has been particularly useful in my experience while adapting pre-trained models.

```python
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import numpy as np

# 1. Define a custom complex network
class ComplexNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ComplexNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
# 2. Create a Skorch NeuralNetClassifier
model = NeuralNetClassifier(
    module=ComplexNet,
    module__input_size=10,
    module__hidden_size=20,
    module__output_size=2,
    max_epochs=5,
    lr=0.01,
)

# 3. Generate dummy training data
X = np.random.rand(100, 10).astype(np.float32)
y = np.random.randint(0, 2, 100).astype(np.int64)
# 4. Train the model
model.fit(X, y)

# 5. Extract trained weights
trained_model = model.module_
weights = trained_model.state_dict()

# 6. Print some weights
print(f"Keys in the state dictionary: {weights.keys()}")
print(f"Shape of 'fc1.weight': {weights['fc1.weight'].shape}")
print(f"Shape of 'batchnorm1.weight': {weights['batchnorm1.weight'].shape}")
print(f"Shape of 'batchnorm1.bias': {weights['batchnorm1.bias'].shape}")
```

This example not only demonstrates weight extraction from linear layers, but also shows the parameters for the batch normalization layers: scale (`weight`) and bias (`bias`). This is critical, especially when replicating inference or implementing complex transfer learning procedures that depend on the specific batch normalization statistics learned during training. The output confirms that these parameters can be individually extracted.

When working with model weights, it's advisable to familiarize oneself with resources provided by PyTorch, particularly their documentation on `torch.nn` modules. Additionally, exploring general materials on deep learning will provide a deeper understanding of how these weights function within the neural network. Understanding these resources will help you not only extract the weights effectively but also manipulate them for your particular needs.
