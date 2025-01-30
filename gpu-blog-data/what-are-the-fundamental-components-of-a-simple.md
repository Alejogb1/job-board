---
title: "What are the fundamental components of a simple neural network?"
date: "2025-01-30"
id: "what-are-the-fundamental-components-of-a-simple"
---
A neural network, at its core, fundamentally attempts to approximate a function that maps inputs to outputs, achieved through a layered architecture of interconnected nodes, commonly known as neurons. This approximation capability relies on three primary components: input, hidden, and output layers, within which weights, biases, and activation functions facilitate learning. These elements work in concert to progressively refine the network's representation of the underlying function during the training process.

Let's break these components down in detail. The input layer, the initial point of data entry, receives raw features. Each node in this layer represents one of these features. Crucially, this layer does not perform any computation beyond simply holding the input value. Imagine you’re building a network to predict housing prices; features such as square footage, the number of bedrooms, and the distance from the city center would each be represented by an individual node in the input layer. The size of the input layer directly corresponds to the dimensionality of the feature space.

Next, hidden layers are where the majority of the computation takes place. Each node within a hidden layer takes the outputs from the preceding layer, applies a weighted sum to these inputs, adds a bias term, and then feeds the result through an activation function. This activation function introduces non-linearity, a critical aspect enabling the network to model complex relationships. Without non-linearity, any neural network, regardless of depth, could be reduced to a single linear transformation, severely limiting its expressiveness. Common activation functions include sigmoid, ReLU (Rectified Linear Unit), and tanh (hyperbolic tangent). The number of hidden layers, as well as the number of nodes within each layer, are hyperparameters and must be empirically determined; they are tuned to provide the best performance for the specific task. Deeper networks, featuring multiple hidden layers, are able to learn increasingly complex abstractions but are also more susceptible to overfitting.

Finally, the output layer generates the final prediction of the model. Similar to hidden layers, each output node calculates a weighted sum of the outputs from the preceding layer, adds a bias, and applies an activation function. However, the activation function here is chosen to align with the problem being addressed. For instance, for binary classification, a sigmoid function is frequently used to output a probability between 0 and 1; for multi-class classification, a softmax function is used to output a probability distribution over all classes; and for regression, a linear activation is typically used or none at all. The number of output nodes corresponds to the dimensionality of the desired output. Returning to the housing price prediction example, there would be a single node in the output layer representing the predicted price, while a handwritten digit classifier would have 10 nodes, each corresponding to a digit between 0 and 9.

Now, let’s consider the core operations within a hidden or output neuron: the weighted sum, bias, and activation. The weighted sum is calculated by multiplying each input value by a corresponding weight and adding these products together. These weights represent the strength of the connection between neurons; they are parameters that are adjusted during the learning process. The bias is a constant term that is added to this sum; it shifts the output of the neuron allowing for improved learning. It functions essentially as an adjustable intercept. The result of the weighted sum and bias is then passed through the activation function.

To illustrate these concepts practically, consider these simplified code examples using Python and NumPy.

**Example 1: Forward Pass through a Single Neuron**

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def forward_pass(inputs, weights, bias):
  weighted_sum = np.dot(inputs, weights) + bias
  output = sigmoid(weighted_sum)
  return output

# Example usage:
inputs = np.array([0.5, 0.8, -0.1]) # Example features
weights = np.array([0.2, -0.3, 0.5]) # Corresponding weights
bias = 0.1 # Bias value
output = forward_pass(inputs, weights, bias)
print(f"Output: {output}")
```

This example demonstrates the core calculation within a single neuron. We define a simple sigmoid activation function and a function representing a neuron's forward pass, which computes the weighted sum, adds the bias, and then applies the activation function. This basic structure is replicated across all neurons within a network.

**Example 2: Forward Pass Through a Simple Feedforward Network**

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def forward_pass(inputs, weights_1, bias_1, weights_2, bias_2):
    # First layer computation
    hidden_layer_output = sigmoid(np.dot(inputs, weights_1) + bias_1)
    # Second layer computation
    final_output = sigmoid(np.dot(hidden_layer_output, weights_2) + bias_2)
    return final_output

# Example usage:
inputs = np.array([0.5, 0.8, -0.1])
weights_1 = np.array([[0.2, -0.3, 0.5],
                     [0.1, 0.4, -0.2]]) # Weights of input to the first hidden layer
bias_1 = np.array([0.1, -0.2])          # Bias of the first hidden layer
weights_2 = np.array([[0.3], [0.6]])     # Weights of hidden to the output layer
bias_2 = np.array([0.05])             # Bias of the output layer

output = forward_pass(inputs, weights_1, bias_1, weights_2, bias_2)
print(f"Output: {output}")
```

This example extends the previous example by introducing a hidden layer. It demonstrates how the output from the input layer is passed through the hidden layer and finally to the output layer. The weights and biases are now represented as matrices, allowing for multiple neurons in each layer. It highlights the core structure of a multilayer feedforward neural network.

**Example 3: Defining a Simple Model Class**

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_1 = np.random.rand(input_size, hidden_size) # Random Initialization
        self.bias_1 = np.random.rand(hidden_size)
        self.weights_2 = np.random.rand(hidden_size, output_size)
        self.bias_2 = np.random.rand(output_size)

    def forward_pass(self, inputs):
        hidden_layer_output = sigmoid(np.dot(inputs, self.weights_1) + self.bias_1)
        final_output = sigmoid(np.dot(hidden_layer_output, self.weights_2) + self.bias_2)
        return final_output

# Example usage:
input_size = 3  # Number of input features
hidden_size = 2 # Number of hidden neurons
output_size = 1 # Number of output nodes
model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
inputs = np.array([0.5, 0.8, -0.1])
output = model.forward_pass(inputs)
print(f"Output: {output}")
```
This final example illustrates encapsulating the network's functionality within a Python class. This not only helps to organize the code more effectively, but also allows for easier management of the network’s parameters (weights and biases). Notably, this class also highlights the importance of weight initialization. In practice, random initialization is rarely the final method. There are other methods based on heuristic principles that aid the learning process and prevent issues with gradient propagation.

For deeper study of these topics, I'd recommend seeking out literature in the areas of machine learning and deep learning. Works such as *Pattern Recognition and Machine Learning* by Christopher Bishop provide a detailed theoretical foundation. For more practical and accessible guides, look at resources such as the *Deep Learning with Python* book by François Chollet. Furthermore, online courses offered by various universities offer both theoretical and hands-on approaches, including those from Andrew Ng, which are widely regarded as well-constructed educational offerings. Finally, extensive documentation for popular deep learning frameworks such as TensorFlow and PyTorch are crucial to becoming a proficient practitioner. This combination of theoretical understanding and practical coding experience is essential to building robust and effective neural networks.
