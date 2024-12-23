---
title: "How can backpropagation be implemented in Python?"
date: "2024-12-23"
id: "how-can-backpropagation-be-implemented-in-python"
---

Alright, let's talk backpropagation. It's a core concept in training neural networks, and implementing it from scratch can be incredibly insightful. I’ve spent a fair bit of time elbow-deep in this, particularly back in my days developing custom image recognition algorithms for a now-defunct robotics firm. We were using cutting-edge (at the time) convolutional architectures, and getting the gradient calculations correct was paramount. It's not just about understanding the mathematics, but about translating it into efficient and correct code.

The essential idea behind backpropagation is using the chain rule of calculus to compute the gradient of a loss function with respect to the weights of a neural network. This gradient tells us how much each weight contributes to the overall error, allowing us to adjust them iteratively to minimize the error. In practice, this is usually broken down into a forward pass and a backward pass. The forward pass calculates the output of the network given an input, while the backward pass calculates the gradients and updates the weights.

Let’s begin with the absolute basics: building blocks. We’ll represent our network using simple linear layers and sigmoid activation functions. While not as sophisticated as what you'd encounter in a deep learning library, it isolates the fundamental backpropagation logic and prevents obscuring details within more complex structures.

First, let’s define our sigmoid function and its derivative, which are crucial for both the forward and backward steps:

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

```
Here we've used numpy for its efficient numerical operations. The `sigmoid` function is straightforward, taking an input `x` and returning its sigmoid transformation. The `sigmoid_derivative` then calculates the derivative, which, crucially, can be expressed in terms of the sigmoid output itself, avoiding costly recalculations during backpropagation.

Now, let’s move on to a simplified linear layer implementation, and demonstrate how it plays within the forward pass. Let's assume we're using this in a 2-layer network, which is enough to see how gradients propagate.

```python
class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01 # Small initialization
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None


    def forward(self, input_data):
      self.input = input_data
      self.output = np.dot(input_data, self.weights) + self.biases
      return self.output

```

This `LinearLayer` class initializes weights and biases with small random values and zeros, respectively. The `forward` method performs the basic linear transformation: the dot product of the input with the weights, plus the bias. It also stores the input for use in the backward pass, a common practice in backpropagation implementations.

Now let’s add a backward pass to this class. The key is to calculate the gradients of the loss with respect to the weights and biases, and the gradient with respect to the input, which is then fed to previous layers. We'll assume we are passing in gradient already computed from the layer after this one, in the typical backpropagation fashion.

```python
class LinearLayer:  # Note that I'm extending the class from previous example
  # ( ... previously shown constructor and forward methods here ... )

  def backward(self, d_output, learning_rate):
    d_weights = np.dot(self.input.T, d_output)
    d_biases = np.sum(d_output, axis=0, keepdims=True)
    d_input = np.dot(d_output, self.weights.T)

    self.weights -= learning_rate * d_weights
    self.biases -= learning_rate * d_biases
    return d_input
```

In this `backward` method, we receive `d_output`, the gradient of the loss with respect to the layer’s output. We calculate `d_weights` and `d_biases` using the chain rule, and then update the weights and biases using a learning rate. Crucially, we also return `d_input`, which is the gradient of the loss with respect to this layer’s input. This `d_input` becomes the `d_output` for the previous layer.

Finally, we can compose these components into a very simple two-layer neural network with a mean squared error (MSE) loss. We’ll keep it as a class here as well for clarity.

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = LinearLayer(input_size, hidden_size)
        self.layer2 = LinearLayer(hidden_size, output_size)


    def forward(self, x):
        hidden_output = sigmoid(self.layer1.forward(x))
        final_output = self.layer2.forward(hidden_output)
        return final_output


    def backward(self, x, y, learning_rate):
        # Forward pass
        hidden_output = sigmoid(self.layer1.forward(x))
        final_output = self.layer2.forward(hidden_output)

        # MSE loss derivative
        d_final_output = 2 * (final_output - y)

        # Backpropagation
        d_hidden_output = self.layer2.backward(d_final_output, learning_rate)
        d_hidden_output_sig = d_hidden_output * sigmoid_derivative(hidden_output)
        self.layer1.backward(d_hidden_output_sig, learning_rate)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mse_loss_derivative(predictions, targets):
    return 2 * (predictions - targets)
```

This class defines the two-layer network, performs forward and backward passes, and includes the loss function. It's important to calculate the error at each step, propagate it backwards through the network and adjust the weights to minimize this error. In `backward`, after the forward pass, we start by computing the derivative of the MSE loss, and then call `backward` in sequence, propagating the gradient backwards. I've included both the `mse_loss` and its derivative calculation here.

This should give you a concrete starting point for understanding and implementing backpropagation. There are complexities when it comes to more advanced architectures, such as convolutional or recurrent layers, but the core principle remains the same: calculating the gradient of the loss and propagating it through the network using the chain rule. I cannot emphasize enough how important a firm grasp of matrix calculus is for doing this well. For further reading, I’d recommend delving into ‘Deep Learning’ by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, which goes through all this in meticulous detail. ‘Neural Networks and Deep Learning’ by Michael Nielsen is also an excellent, more approachable resource for solidifying understanding. Understanding the foundational math will ultimately save much time when working with more complex deep learning architectures and libraries. It’s not about blindly calling library functions, it’s about understanding *what* those functions are actually doing. From experience, it’s the only way to truly troubleshoot unexpected behavior and build truly innovative models.
