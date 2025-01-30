---
title: "Why is MSE greater than zero if input equals output and weights are initialized to one?"
date: "2025-01-30"
id: "why-is-mse-greater-than-zero-if-input"
---
Mean Squared Error (MSE), a ubiquitous loss function in machine learning, quantifies the average squared difference between predicted and actual values. Even when a model's input perfectly matches its desired output and all initial weights are set to one, MSE will almost always be greater than zero because of the bias term and, critically, how neural networks process data at the node level with weights and non-linear activation functions. The initial state of the network, though seemingly primed for minimal error, is in reality far from optimal due to the cumulative effect of even small deviations introduced during forward propagation.

To understand why, we need to break down how a basic feedforward neural network operates, even in the most simplistic case. Consider a single neuron in a layer; it receives inputs from the preceding layer, multiplies them by their corresponding weights, sums the results, adds a bias, and then passes this sum through an activation function. This entire process—weighted summation, bias addition, and non-linear activation—occurs at each node in each layer. If we set all initial weights to one and the bias to one (or any non-zero value), then even when the input and desired output are identical, these operations almost certainly won’t result in an output that exactly matches the input. The bias will almost always skew the output unless a compensatory combination of input, weights, and activation function happen to offset it exactly, which is statistically improbable with random input data. It’s extremely unlikely that the subsequent activation function will produce precisely the input value even if the weighted sum coincidentally matched it.

Let's consider a basic example: A single neuron with one input, one weight, one bias and a non-linear activation function such as ReLU. Our input (x) and desired output (y) are both one. The initial weight (w) is set to one, and the initial bias (b) is set to one.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

x = 1
y_true = 1
w = 1
b = 1

z = (x * w) + b
y_pred = relu(z)

error = mse(y_true, y_pred)

print(f"Input: {x}")
print(f"Output (Desired): {y_true}")
print(f"Pre-Activation Value (z): {z}")
print(f"Predicted Output: {y_pred}")
print(f"Mean Squared Error: {error}")
```

In this case, pre-activation value `z` will be 2. After applying the ReLU activation function, the predicted value will be 2, rather than the desired 1 and thus we get an MSE of 1. While a simple single-neuron example, it highlights how, even with identical input and desired output, initialization and activation functions introduce error. This occurs even before any backpropagation learning.

Now, let’s examine a slightly more complex example: a two-layer neural network where both input and desired output are represented by a vector. We'll assume two neurons in both input and hidden layer and a single neuron in the output layer. We will continue to use ReLU activations. Input and output will each be represented by `[1,1]`.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

# Input and desired output
x = np.array([1, 1])
y_true = np.array([1,1])

# Initialize weights and biases to one
w1 = np.ones((2, 2))
b1 = np.ones((2))
w2 = np.ones((2,1))
b2 = np.ones((1))


# Hidden layer calculations
z1 = np.dot(x, w1) + b1
a1 = relu(z1)

# Output layer calculations
z2 = np.dot(a1, w2) + b2
y_pred = relu(z2)

error = mse(y_true, y_pred)

print(f"Input: {x}")
print(f"Output (Desired): {y_true}")
print(f"Pre-Activation Layer 1 (z1): {z1}")
print(f"Activation Layer 1 (a1): {a1}")
print(f"Pre-Activation Layer 2 (z2): {z2}")
print(f"Predicted Output: {y_pred}")
print(f"Mean Squared Error: {error}")
```
In the preceding example, we perform matrix multiplications and bias additions, which transform our input such that by the time it reaches the output layer, the result (predicted output) will almost certainly differ from the desired output due to the effects of non-linear activations on each node and weighted inputs. Although we start with an input identical to the desired output and weights and bias of one, the non-linearities cause the final prediction to not align with the desired output. This results in a non-zero MSE.

The problem isn't that we've chosen bad initialization values, but that these non-linear calculations occur in the network’s forward pass. Each node makes modifications which in turn make the desired state unobtainable without iterations of back-propagation. Even setting all the weights to zero can sometimes result in a nonzero MSE if the biases are not also zero due to the activation function.

To demonstrate this with random input data, even with inputs equal to outputs, we will see that the initial MSE is not zero. This shows that even if the input and desired output are identical there will still be initial error due to the non-linear nature of neural networks during forward propagation.

```python
import numpy as np

def relu(x):
  return np.maximum(0, x)

def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

# Random input data where input equals output
input_size = 100
x = np.random.rand(input_size)
y_true = x

# Initialize weights and biases to one
w1 = np.ones((input_size, input_size))
b1 = np.ones((input_size))
w2 = np.ones((input_size, 1))
b2 = np.ones((1))


# Hidden layer calculations
z1 = np.dot(x, w1) + b1
a1 = relu(z1)

# Output layer calculations
z2 = np.dot(a1, w2) + b2
y_pred = relu(z2).flatten()

error = mse(y_true, y_pred)

print(f"Input (First 5 Values): {x[:5]}")
print(f"Output (Desired - First 5 Values): {y_true[:5]}")
print(f"Predicted Output (First 5 Values): {y_pred[:5]}")
print(f"Mean Squared Error: {error}")
```

In this last example, even with random data where input and output are identical, the resulting mean squared error after the initial forward propagation is far from zero due to the reasons described above.

In essence, the initial non-zero MSE even with input equal to output, and weights initialized to one, is inherent to the nature of neural network design and operation. These initial errors are not an indication that the network is broken, but rather represent a starting point from which it can optimize its parameters to minimize loss through backpropagation. The initialization process, as demonstrated, will almost certainly not result in a prediction that aligns perfectly with the target output. These initial imperfections are the very reason we need a learning process based on error gradients to guide the parameters to a more suitable configuration.

For further study into neural network operation and foundational math concepts, I recommend exploring textbooks focusing on Deep Learning fundamentals and numerical optimization techniques. Additionally, courses offered by leading universities on machine learning provide in-depth explanations of the material. Examining research papers that describe network architectures and common pitfalls will also be beneficial.
