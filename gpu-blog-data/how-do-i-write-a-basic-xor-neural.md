---
title: "How do I write a basic XOR neural network program?"
date: "2025-01-30"
id: "how-do-i-write-a-basic-xor-neural"
---
The core functionality of an XOR neural network hinges on its ability to learn a non-linearly separable function, a task a single-layer perceptron cannot accomplish.  This limitation stems from the inherent linear nature of a single-layer perceptron's decision boundary.  Over the years, working on various pattern recognition projects, I've found that understanding this fundamental limitation is key to designing effective multi-layer perceptrons, including those used to solve the XOR problem.  A successful solution necessitates the introduction of at least one hidden layer, enabling the network to learn complex, non-linear relationships between inputs and outputs.

My approach involves a feedforward neural network with a single hidden layer containing two neurons. This architecture provides sufficient capacity to learn the XOR function. The network utilizes sigmoid activation functions in the hidden and output layers, providing a smooth, differentiable output crucial for efficient backpropagation training.

**1.  Explanation of the XOR Neural Network Architecture and Training:**

The network consists of two input neurons (x1 and x2), two hidden neurons (h1 and h2), and one output neuron (y).  Each connection between neurons possesses an associated weight (wij, where i denotes the source neuron and j denotes the destination neuron).  The hidden layer neurons receive weighted sums of the inputs, pass the result through a sigmoid activation function, and transmit the output to the output neuron. This process is repeated for the output neuron, resulting in a final prediction.

The sigmoid activation function is defined as:  σ(z) = 1 / (1 + exp(-z)), where z is the weighted sum of inputs to a neuron.  This function ensures the output of each neuron is within the range [0, 1].

Training involves adjusting the weights to minimize the difference between the network's predictions and the actual XOR outputs.  This is achieved using the backpropagation algorithm. Backpropagation calculates the gradient of the error function with respect to each weight, and the weights are then updated using gradient descent.  The error function often used is the mean squared error (MSE), defined as: MSE = (1/N) Σ(yi - ŷi)^2, where N is the number of training samples, yi is the actual output, and ŷi is the network's prediction.

The learning rate (η) determines the step size during weight updates.  A smaller learning rate leads to slower convergence but can result in a more accurate solution, whereas a larger learning rate can lead to faster convergence but might overshoot the optimal weights.  The choice of learning rate often involves experimentation to find a suitable value.


**2. Code Examples with Commentary:**

The following code examples utilize Python with the NumPy library for numerical computation.  These examples are simplified for clarity and may not incorporate all the optimizations found in dedicated machine learning libraries.

**Example 1:  NumPy-based Implementation**

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights randomly
weights_ih = np.random.rand(2, 2)  # Input to hidden
weights_ho = np.random.rand(2, 1)  # Hidden to output
bias_h = np.random.rand(1,2)
bias_o = np.random.rand(1,1)

# XOR training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_ih) + bias_h
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_ho) + bias_o
    output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)
    error_hidden_layer = d_output.dot(weights_ho.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_ho += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_ih += X.T.dot(d_hidden_layer) * learning_rate
    bias_o += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_h += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("Final Output:\n", output)
```

This example demonstrates a basic implementation of the backpropagation algorithm. The weights and biases are updated iteratively to minimize the error.

**Example 2:  Simplified Visualization (Conceptual)**

This example omits the actual training and focuses on demonstrating the forward pass calculation:

```python
import numpy as np

# Sample weights and inputs (for demonstration only)
weights_ih = np.array([[0.5, 0.8], [0.2, 0.3]])
weights_ho = np.array([[0.7], [-0.9]])
inputs = np.array([1, 0]) # Example input [x1, x2]

# Forward pass
hidden_layer_input = np.dot(inputs, weights_ih)
hidden_layer_output = np.array([sigmoid(x) for x in hidden_layer_input])
output_layer_input = np.dot(hidden_layer_output, weights_ho)
output = sigmoid(output_layer_input)

print("Output:", output)
```

This simplified example aids in understanding the flow of information through the network.

**Example 3:  Illustrative Data Structure**

This example shows a structured way of representing the network components:

```python
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        sum = np.dot(inputs, self.weights) + self.bias
        return sigmoid(sum)

# ... (rest of the network structure using Neuron objects)
```

This approach encapsulates the neuron's functionality, leading to more organized and potentially scalable code.  The weights and biases would need to be initialized and updated as in the previous examples.

**3. Resource Recommendations:**

I recommend consulting textbooks on neural networks and deep learning.  Furthermore, studying the source code of established machine learning libraries can offer valuable insights into efficient implementation strategies and advanced optimization techniques.  Finally,  carefully reviewing research papers on neural network architectures is highly beneficial for understanding the design choices and performance of various network configurations.  A strong mathematical foundation in linear algebra and calculus is also essential for comprehending the underlying principles of the backpropagation algorithm and neural network training.
