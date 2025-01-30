---
title: "How does a neural network retain learned patterns?"
date: "2025-01-30"
id: "how-does-a-neural-network-retain-learned-patterns"
---
The core mechanism by which a neural network retains learned patterns lies in the adjustment and storage of numerical values called weights, associated with the connections between its artificial neurons. These weights, established during training through iterative processes like backpropagation, directly influence the strength of activation signals passed from one neuron to the next. The configuration of these weight values, therefore, embodies the network’s learned representation of the training data, allowing it to generalize to new, unseen inputs exhibiting similar patterns.

Specifically, this process is not about storing the training data itself. Instead, it’s about encoding complex relationships and features found within the data through a distributed representation. This representation is spread across the network’s many layers and connections, such that no single weight value holds the complete information about a specific pattern. Instead, it's the intricate interplay and configuration of all weights that collectively represent these patterns. This is also why neural networks can display a high degree of fault tolerance, meaning that minor changes to individual weights don’t always disrupt learned functionality.

To understand this, consider a simplified feedforward neural network, consisting of input layers, hidden layers, and output layers. Each connection between neurons in adjacent layers has an associated weight. Initially, these weights are often assigned random small values. During training, input data is fed forward through the network. At each layer, the weighted sum of the previous layer's outputs (plus a bias term) is calculated, and this sum is then passed through an activation function (e.g., sigmoid, ReLU) to introduce non-linearity, crucial for learning complex relationships. The output is then compared to the true or desired output, yielding a loss or error. Backpropagation is employed to calculate the gradient of this loss with respect to each weight in the network. This gradient indicates how each weight affects the final loss, providing the information necessary to adjust the weights and decrease the overall error. A learning rate hyperparameter determines the magnitude of these adjustments; a small learning rate results in finer weight updates.

The weight adjustment is mathematically expressed in various forms of gradient descent, a common version of which can be shown as:

`weight_new = weight_old - learning_rate * gradient_of_loss_wrt_weight`.

This process iteratively updates the weights across all connections in the network with each training instance, gradually shaping the network to minimize prediction error and thus learn to recognize patterns inherent to the training data. After training concludes, the learned weights remain in place, enabling the network to accurately infer outputs for new input data within the same problem domain.

Consider a numerical classification example using Python and `numpy`:

```python
import numpy as np

# Simple network with 2 inputs, 2 hidden nodes, 1 output
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights randomly
input_nodes = 2
hidden_nodes = 2
output_nodes = 1

weights_ih = 2 * np.random.random((input_nodes, hidden_nodes)) - 1
weights_ho = 2 * np.random.random((hidden_nodes, output_nodes)) - 1

# Training data and labels (for demonstration)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

learning_rate = 0.1
epochs = 10000

for i in range(epochs):
  # Forward pass
  hidden_layer_input = np.dot(X, weights_ih)
  hidden_layer_output = sigmoid(hidden_layer_input)

  output_layer_input = np.dot(hidden_layer_output, weights_ho)
  output_layer_output = sigmoid(output_layer_input)

  # Backpropagation
  output_layer_error = y - output_layer_output
  output_layer_delta = output_layer_error * sigmoid_derivative(output_layer_output)

  hidden_layer_error = output_layer_delta.dot(weights_ho.T)
  hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

  # Update weights
  weights_ho += hidden_layer_output.T.dot(output_layer_delta) * learning_rate
  weights_ih += X.T.dot(hidden_layer_delta) * learning_rate

# After training, the weights hold the learned representation.
print("Weights between input and hidden layer:\n", weights_ih)
print("Weights between hidden and output layer:\n", weights_ho)
```

This example demonstrates a basic two-layer neural network performing binary classification. The weights (`weights_ih`, `weights_ho`) are initialized randomly and then iteratively updated via backpropagation. The training data, in this case, performs an XOR operation, and the final weights represent the network's learned representation of this relationship. The weights are adjusted so that the network correctly classifies the training data. Upon completion of training, the weights are the key mechanism the network uses to remember its learned relationship between inputs and outputs.

Now, let's consider a more complex example demonstrating weight adjustments in a simplified convolution layer, common in image processing. This layer uses a kernel, which is another set of weights, and that slides over the input, performing a weighted summation at each location.

```python
import numpy as np

# Simplified convolution with single input channel, single kernel
def convolution(input_matrix, kernel, stride=1):
  input_height, input_width = input_matrix.shape
  kernel_height, kernel_width = kernel.shape

  output_height = (input_height - kernel_height) // stride + 1
  output_width = (input_width - kernel_width) // stride + 1

  output_matrix = np.zeros((output_height, output_width))

  for i in range(0, input_height - kernel_height + 1, stride):
      for j in range(0, input_width - kernel_width + 1, stride):
          output_matrix[i // stride, j // stride] = np.sum(input_matrix[i:i + kernel_height, j:j + kernel_width] * kernel)
  return output_matrix

# Example 2D data, Kernel, initial random weights for the kernel
input_image = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])

kernel_size = 3
kernel = 2 * np.random.random((kernel_size, kernel_size)) - 1 #random kernel

# Simulate a training loop, adjusting a specific weight in the kernel
learning_rate = 0.1
epochs = 2

for _ in range(epochs):
  # Forward pass (simplified)
  output_feature_map = convolution(input_image, kernel)

  # Simulate an error (gradient w.r.t a specific kernel weight)
  gradient_error = 1 #Example gradient

  # Update a weight in kernel (example: at index [0,0])
  kernel[0,0] -= learning_rate * gradient_error #simulated weight update

# After training (simulated), kernel weight stores info for edge detection/feature extraction
print("Learned kernel:\n", kernel)
```
In this example, a random kernel is used to convolve over an input image. During training, only one specific weight is adjusted for demonstration purposes, although in a real situation backpropagation would compute gradients for all kernel weights. This highlights how the kernel's weight values are updated to learn to detect features from the input data.

Finally, let's showcase how these weights, represented as tensors in higher dimensions, are stored. This example shows how a tensor for the weights and biases can be structured for multiple layers of a simple network.

```python
import numpy as np

# Defining layers dimensions
input_size = 784  # Example: MNIST image size
hidden1_size = 256
hidden2_size = 128
output_size = 10  # 10 digits 0 to 9

# Layer weight and bias initialization (using tensors)
weights = {}
biases = {}

weights['w1'] = np.random.randn(input_size, hidden1_size) * 0.01 #random weights (input to hidden1)
weights['w2'] = np.random.randn(hidden1_size, hidden2_size) * 0.01 #random weights (hidden1 to hidden2)
weights['w3'] = np.random.randn(hidden2_size, output_size) * 0.01  #random weights (hidden2 to output)


biases['b1'] = np.zeros(hidden1_size) # Initial biases
biases['b2'] = np.zeros(hidden2_size)
biases['b3'] = np.zeros(output_size)

# Example of stored structure
print("Structure of weights:\n", weights.keys())
for k, v in weights.items():
    print(f"{k}: Shape:{v.shape}, Example weights:\n{v[0:2,0:2]}") #shape of weights and first 2x2 weights

print("\nStructure of biases:\n", biases.keys())
for k, v in biases.items():
    print(f"{k}: Shape:{v.shape}, Example biases:\n{v[0:2]}") # shape of biases and first two biases
```

This example illustrates how weights (and biases) are stored as numpy arrays. In a real scenario, these matrices would store values after being adjusted by backpropagation during the learning process. The shape of these tensors dictates how the information flows between the network layers.

For further learning, I would recommend exploring resources covering linear algebra, calculus, and statistics for the mathematical foundations of backpropagation and gradient descent. Texts on the architecture of different neural network types (e.g., convolutional networks, recurrent networks) and deep learning frameworks (like TensorFlow or PyTorch) are crucial for hands-on experience. Additionally, research papers on the specific problem domains you are interested in can provide insights into advanced techniques.
