---
title: "How do supervised neural networks learn?"
date: "2024-12-23"
id: "how-do-supervised-neural-networks-learn"
---

, let’s tackle this one. It's a fundamental question, and I've seen enough variations of it over the years to understand where the confusion often lies. Supervised learning, in the context of neural networks, is at its core about minimizing error. Specifically, it's about adjusting the network's internal parameters — weights and biases — to map input data to desired output data as accurately as possible. The process is iterative and involves something akin to guided trial and error, but with a sophisticated mathematical framework underpinning everything.

I remember working on a project back in the mid-2010s where we were building a system to classify handwritten digits. We had a huge dataset, which was great, but also a testament to the importance of data preprocessing before diving into the modeling phase. That's a separate discussion, however. The neural network, a relatively simple feedforward model at the time, started from a place of near randomness in its predictions. Essentially, it was just guessing. The magic happened as we fed it training data.

So, how does it work in detail? Let’s break down the key concepts. First, you have the network architecture itself. This defines the layers of neurons and their connectivity. Each connection between neurons has an associated weight, and each neuron also has a bias. These weights and biases are the parameters that the learning algorithm manipulates. When an input data point is fed into the network, it propagates forward through the layers, each neuron performing a weighted sum of its inputs, adding a bias, and then passing the result through an activation function, like a sigmoid or ReLU. This ultimately produces an output, which is initially almost certainly incorrect.

This initial incorrect output is the genesis of the learning process. We compare the network's prediction to the actual target label (the "ground truth") using a loss function, often something like mean squared error or categorical cross-entropy. The loss function quantifies how wrong the network is. The goal of training is to minimize this loss. It’s the value we’re trying to drive down toward zero, indicating that the network's predictions are increasingly accurate.

The core of learning, the crucial step, is backward propagation or backpropagation. This algorithm uses the chain rule of calculus to compute the gradient of the loss function with respect to every weight and bias in the network. Put simply, it calculates how much each parameter contributes to the error. The gradient indicates the direction of steepest ascent, so we move the parameters in the *opposite* direction to minimize the loss. This is achieved using an optimization algorithm like stochastic gradient descent (sgd) or its more advanced variants like adam.

The learning rate, a hyperparameter, determines how big of a step we take down the loss landscape. Too large of a learning rate can cause the optimization to overshoot the minimum. Too small, and learning is agonizingly slow. It's a bit of an art to find the sweet spot for optimal training. This process of forward propagation, error calculation, backpropagation, and parameter update is iterated over many epochs – that is, many passes through the training dataset – until the network achieves acceptable accuracy on a separate validation set. We monitor this validation performance closely to avoid overfitting, which is where the network memorizes the training data rather than generalizing to new, unseen data.

To solidify this, let me provide some simplified, illustrative examples using python and numpy. These aren't production-level implementations but will clarify the fundamental principles.

**Example 1: Simple Linear Regression with One Neuron**

```python
import numpy as np

def linear_forward(x, w, b):
  return np.dot(x, w) + b

def mse_loss(y_pred, y_true):
  return np.mean((y_pred - y_true)**2)

def mse_gradient(y_pred, y_true, x):
  dw = 2 * np.dot(x.T, (y_pred - y_true)) / len(y_true)
  db = 2 * np.mean(y_pred - y_true)
  return dw, db

# Dummy data for simplicity
x = np.array([[1], [2], [3], [4]])  # Input features
y = np.array([2, 4, 5, 4.5])       # Target values

# Initialize parameters
w = np.array([[0.5]])
b = 0.0

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    y_pred = linear_forward(x, w, b)
    loss = mse_loss(y_pred, y)
    dw, db = mse_gradient(y_pred, y, x)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 200 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

print(f"Learned weight: {w}, Learned bias: {b}")
```
This first example illustrates the learning process with just one neuron and a linear relationship. It’s a simplification but shows the core concepts. We initialize a weight and bias, then iteratively calculate the loss (mean squared error) and the gradients and update these parameters with a basic gradient descent. The loss decreases during training until the model converges to a linear fit of the data.

**Example 2: Simple Neural Network with One Hidden Layer**

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

def forward_pass(x, w1, b1, w2, b2):
  z1 = np.dot(x, w1) + b1
  a1 = sigmoid(z1)
  z2 = np.dot(a1, w2) + b2
  return z2, a1

def mse_loss(y_pred, y_true):
  return np.mean((y_pred - y_true)**2)

def backward_pass(x, y, y_pred, a1, w2):
    m = len(y)
    dz2 = 2 * (y_pred - y) / m
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    dz1 = np.dot(dz2, w2.T) * sigmoid_derivative(np.dot(x, w1) + b1)
    dw1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)
    return dw1, db1, dw2, db2

# Dummy data
x = np.array([[0,0], [0,1], [1,0], [1,1]]) # Input
y = np.array([[0], [1], [1], [0]]) # Output (XOR like data)

# Initialize weights and biases
np.random.seed(0)
w1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
w2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 2000

for epoch in range(epochs):
    z2, a1 = forward_pass(x, w1, b1, w2, b2)
    loss = mse_loss(z2, y)
    dw1, db1, dw2, db2 = backward_pass(x, y, z2, a1, w2)

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    if epoch % 400 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")
```

This example introduces a hidden layer with sigmoid activations. The forward pass involves calculating activations through each layer, and the backward pass uses the derivative of the sigmoid function to calculate gradients. The process remains the same, but now we are backpropagating gradients across multiple layers.

**Example 3: Simplified Classification with One Output Neuron**

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_gradient(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)

def forward_pass(x, w, b):
  z = np.dot(x, w) + b
  a = sigmoid(z)
  return a

def backward_pass(x, y, y_pred, w):
  m = len(y)
  dz = binary_cross_entropy_gradient(y_pred, y) * sigmoid_derivative(np.dot(x,w) +b)
  dw = np.dot(x.T, dz) / m
  db = np.sum(dz, axis=0) / m
  return dw, db

# Dummy data
x = np.array([[1, 2], [2, 1], [4, 5], [5, 4]])
y = np.array([[0], [0], [1], [1]])

# Initialize weights and biases
np.random.seed(0)
w = np.random.randn(2, 1)
b = np.zeros((1,1))

learning_rate = 0.1
epochs = 15000

for epoch in range(epochs):
    y_pred = forward_pass(x, w, b)
    loss = binary_cross_entropy(y_pred, y)
    dw, db = backward_pass(x, y, y_pred, w)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 1500 == 0:
         print(f"Epoch: {epoch}, Loss: {loss:.4f}")

```

Here, I’ve used binary cross-entropy instead of mean squared error, appropriate when you want a single output to classify something as binary (for example, like deciding between two categories). This example has a sigmoid output which gives probabilities, and its loss function is the binary cross entropy, rather than mean squared error, which changes the gradients as well.

For a more comprehensive understanding, I’d highly recommend exploring "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This book is considered the definitive text on the topic. Additionally, papers detailing the original backpropagation algorithm, and papers on specific optimizers like Adam, are highly beneficial. I’d also recommend looking into "Pattern Recognition and Machine Learning" by Christopher Bishop for a solid statistical perspective on machine learning, including neural networks.

To summarize, supervised neural networks learn by iteratively adjusting their parameters through forward and backward passes, guided by a loss function and using gradient-based optimization. It is a fundamentally iterative process, and its effectiveness stems from this careful balancing of parameter updates and data. It's not magic; it's well-defined math executed meticulously.
