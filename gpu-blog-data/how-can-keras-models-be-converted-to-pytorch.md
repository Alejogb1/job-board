---
title: "How can Keras models be converted to PyTorch?"
date: "2025-01-30"
id: "how-can-keras-models-be-converted-to-pytorch"
---
Converting a Keras model to its PyTorch equivalent involves more than simply rewriting layer definitions; it requires a careful understanding of the underlying framework differences and ensuring numerical consistency in parameter initialization and data handling. I've navigated this process numerous times in past projects, migrating legacy Keras models to a PyTorch-centric production environment, and I've learned several effective strategies.

The core challenge arises from Keras and PyTorch's distinct approaches to computational graphs and model construction. Keras, with its higher-level API, implicitly builds a graph during model definition, often relying on TensorFlow’s backend for execution. PyTorch, on the other hand, is more explicit, requiring users to define the forward pass within a class inheriting from `torch.nn.Module`. This difference necessitates careful re-implementation of the model’s architecture, including its layers, activation functions, and potential regularization components. Furthermore, weights initialized within a Keras model are represented as NumPy arrays, requiring conversion to PyTorch tensors before being loaded into the equivalent PyTorch model.

The general approach can be broken down into three key phases: model architecture recreation, weight transfer, and verification. In the first phase, one must meticulously analyze the Keras model's structure using methods like `model.summary()` in Keras and then translate this into a PyTorch `nn.Module` class. This involves translating Keras layer types (e.g., `Dense`, `Conv2D`, `LSTM`) to their PyTorch counterparts (e.g., `nn.Linear`, `nn.Conv2d`, `nn.LSTM`). Careful attention should be paid to the parameters within each layer such as kernel sizes, strides, padding options, and input/output dimensionalities. This step requires referencing the respective library documentation to ensure the equivalent PyTorch layers behave as intended.

Once the model structure is replicated in PyTorch, the next phase revolves around transferring the model's trained weights. Keras stores weights as NumPy arrays, accessed via `model.get_weights()`. PyTorch, conversely, utilizes tensors stored within the model's state dictionary, accessible through `model.state_dict()`. Thus, each weight array retrieved from the Keras model needs to be transformed into a PyTorch tensor using `torch.from_numpy()`. These tensors can then be loaded into the corresponding layers of the PyTorch model. It's essential that this assignment is done correctly, matching the Keras weight names with the appropriate PyTorch layer parameters as indicated in the model's respective layers.

Finally, the correctness of the transfer needs to be verified. This can be done by feeding a small batch of inputs through both the original Keras model and the converted PyTorch model. The output tensors should be extremely close; any significant deviations indicate errors in model recreation, weight mapping, or data preparation. If possible, utilize the same preprocessing steps used during Keras model training for input data preprocessing to make verification as clear as possible. Comparing against the Keras model when in inference mode is recommended.

Below are three simplified code examples illustrating this conversion process.

**Example 1: Basic Dense Layer Model**

This example demonstrates conversion of a simple Keras model with two dense layers to PyTorch.

```python
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

# Keras Model
keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Initialize the weights of Keras model to be non-random
for layer in keras_model.layers:
    weights = layer.get_weights()
    if weights:
       layer.set_weights([np.ones_like(w) for w in weights])

# PyTorch Model
class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

pytorch_model = PyTorchModel()

# Transfer Weights
keras_weights = keras_model.get_weights()
with torch.no_grad():
    pytorch_model.fc1.weight.copy_(torch.from_numpy(keras_weights[0].T).float())
    pytorch_model.fc1.bias.copy_(torch.from_numpy(keras_weights[1]).float())
    pytorch_model.fc2.weight.copy_(torch.from_numpy(keras_weights[2].T).float())
    pytorch_model.fc2.bias.copy_(torch.from_numpy(keras_weights[3]).float())

# Verification
test_input = np.random.rand(1, 10).astype(np.float32)
keras_output = keras_model.predict(test_input)
pytorch_output = pytorch_model(torch.from_numpy(test_input).float()).detach().numpy()

print("Keras Output:\n", keras_output)
print("PyTorch Output:\n", pytorch_output)
print("Outputs close:", np.allclose(keras_output, pytorch_output))
```

This example shows the direct mapping of Keras `Dense` layers to PyTorch `nn.Linear` layers. The weight transfer process involves transposing weight matrices (`.T`) because of different storage conventions, and converting the biases as-is. We can also see that the input and output of the model has been defined such that we can pass in a numpy array to Keras and a tensor to PyTorch to compare the two models output. Also, note that both Keras and PyTorch have a `.float()` function, that can be used to cast to float 32, if not specified, can lead to unexpected behaviour during conversion.

**Example 2: Convolutional Layer Model**

Here's an example with convolutional layers, demonstrating the handling of channels, input dimensions, and weight format.

```python
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

# Keras Model
keras_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
# Initialize the weights of Keras model to be non-random
for layer in keras_model.layers:
    weights = layer.get_weights()
    if weights:
        layer.set_weights([np.ones_like(w) for w in weights])

# PyTorch Model
class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # 'same' padding = padding=1
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 28 * 28, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.softmax(self.fc(x))
        return x

pytorch_model = PyTorchModel()

# Transfer Weights
keras_weights = keras_model.get_weights()
with torch.no_grad():
   pytorch_model.conv1.weight.copy_(torch.from_numpy(keras_weights[0].transpose(3, 2, 0, 1)).float())
   pytorch_model.conv1.bias.copy_(torch.from_numpy(keras_weights[1]).float())
   pytorch_model.fc.weight.copy_(torch.from_numpy(keras_weights[2].T).float())
   pytorch_model.fc.bias.copy_(torch.from_numpy(keras_weights[3]).float())

# Verification
test_input = np.random.rand(1, 28, 28, 3).astype(np.float32)
keras_output = keras_model.predict(test_input)
pytorch_output = pytorch_model(torch.from_numpy(test_input.transpose(0, 3, 1, 2)).float()).detach().numpy()

print("Keras Output:\n", keras_output)
print("PyTorch Output:\n", pytorch_output)
print("Outputs close:", np.allclose(keras_output, pytorch_output))
```
The convolution layer weight transfer requires a more complex transpose. The Keras format is `(height, width, input_channels, output_channels)` while PyTorch expects `(output_channels, input_channels, height, width)`. Therefore, `keras_weights[0].transpose(3, 2, 0, 1)` correctly reshapes the kernel weight matrix. In addition, note that PyTorch convolution layers expect NCHW inputs, whereas Keras works with NHWC, therefore the input has to be permuted accordingly for this verification to work.

**Example 3: Recurrent Layer Model**
This example involves an LSTM layer, demonstrating the different ordering of weights in the two frameworks.

```python
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np


# Keras Model
keras_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(10, 20), return_sequences=False),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Initialize the weights of Keras model to be non-random
for layer in keras_model.layers:
    weights = layer.get_weights()
    if weights:
        layer.set_weights([np.ones_like(w) for w in weights])

# PyTorch Model
class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(20, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x) # h_n is the output, _ is all the sequences which we don't need
        x = self.softmax(self.fc(h_n.squeeze(0))) # h_n is of shape (num_layers, batch, num_directions * hidden_size) squeeze to make it (batch, hidden_size)
        return x

pytorch_model = PyTorchModel()

# Transfer Weights
keras_weights = keras_model.get_weights()
with torch.no_grad():
    lstm_weights_keras = keras_weights[0:4] # Keras groups all LSTM weights
    weight_ih, weight_hh, bias_ih, bias_hh = lstm_weights_keras
    pytorch_model.lstm.weight_ih_l0.copy_(torch.from_numpy(np.concatenate([weight_ih.T], axis=0)).float())
    pytorch_model.lstm.weight_hh_l0.copy_(torch.from_numpy(np.concatenate([weight_hh.T], axis=0)).float())
    pytorch_model.lstm.bias_ih_l0.copy_(torch.from_numpy(np.concatenate([bias_ih], axis=0)).float())
    pytorch_model.lstm.bias_hh_l0.copy_(torch.from_numpy(np.concatenate([bias_hh], axis=0)).float())
    pytorch_model.fc.weight.copy_(torch.from_numpy(keras_weights[4].T).float())
    pytorch_model.fc.bias.copy_(torch.from_numpy(keras_weights[5]).float())


# Verification
test_input = np.random.rand(1, 10, 20).astype(np.float32)
keras_output = keras_model.predict(test_input)
pytorch_output = pytorch_model(torch.from_numpy(test_input).float()).detach().numpy()


print("Keras Output:\n", keras_output)
print("PyTorch Output:\n", pytorch_output)
print("Outputs close:", np.allclose(keras_output, pytorch_output))
```

This illustrates the complexity involved in recurrent layers. Keras combines the input and recurrent weights for an LSTM into single matrices. PyTorch stores these separately within the `lstm` layer’s parameters. Therefore, the Keras weight matrices are concatenated and mapped to corresponding PyTorch parameters. Also note that the PyTorch LSTM requires `batch_first=True` to adhere to the expected input shape of keras.

For further reference, the official PyTorch documentation provides detailed explanations of all modules and functions (`torch.nn`, `torch.Tensor`). Also, the official TensorFlow documentation offers insight into the structure of Keras models and their corresponding weight representations. Consulting research papers on deep learning, especially those involving implementation details, can also provide helpful details.
