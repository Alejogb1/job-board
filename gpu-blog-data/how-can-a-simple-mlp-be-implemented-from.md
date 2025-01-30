---
title: "How can a simple MLP be implemented from scratch using TensorFlow?"
date: "2025-01-30"
id: "how-can-a-simple-mlp-be-implemented-from"
---
Implementing a Multilayer Perceptron (MLP) from scratch using TensorFlow offers a valuable understanding of the underlying mechanics of neural networks.  My experience building custom image recognition systems for medical diagnostics highlighted the importance of this low-level understanding for debugging and optimization.  A key fact often overlooked is the explicit management of weight initialization, activation functions, and backpropagation, all of which directly impact performance and prevent reliance on higher-level abstractions.


**1. Clear Explanation:**

The core components of a simple MLP are: the input layer, one or more hidden layers, and an output layer.  Each layer consists of neurons, which perform weighted sums of their inputs and apply an activation function.  The weights are adjusted during training through backpropagation, an algorithm that calculates the gradient of the loss function with respect to the weights.  This gradient informs the update rule, typically using stochastic gradient descent (SGD) or its variants like Adam.  Forward propagation computes the network's output for a given input, while backpropagation calculates the error and updates the weights to minimize this error.  TensorFlow provides the computational framework for efficient matrix operations necessary for these processes.  Crucially, we bypass TensorFlow's high-level APIs like `tf.keras.Sequential` to explicitly define each layer's operations.


**2. Code Examples with Commentary:**


**Example 1: A simple MLP with one hidden layer for binary classification**

This example demonstrates a basic MLP with a single hidden layer, suitable for a binary classification problem. We'll use sigmoid activation in the hidden and output layers.  Note the explicit definition of weight matrices and bias vectors, and the manual implementation of forward and backward passes.

```python
import tensorflow as tf
import numpy as np

# Define hyperparameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 1000

# Initialize weights and biases
W1 = tf.Variable(tf.random.normal([input_size, hidden_size]))
b1 = tf.Variable(tf.zeros([hidden_size]))
W2 = tf.Variable(tf.random.normal([hidden_size, output_size]))
b2 = tf.Variable(tf.zeros([output_size]))

# Define activation functions
def sigmoid(x):
  return 1 / (1 + tf.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

# Generate sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training loop
for epoch in range(epochs):
  # Forward propagation
  z1 = tf.matmul(X, W1) + b1
  a1 = sigmoid(z1)
  z2 = tf.matmul(a1, W2) + b2
  a2 = sigmoid(z2)

  # Loss calculation (binary cross-entropy)
  loss = -tf.reduce_mean(y * tf.math.log(a2) + (1 - y) * tf.math.log(1 - a2))

  # Backpropagation
  da2 = a2 - y
  dz2 = da2 * sigmoid_derivative(a2)
  dW2 = tf.matmul(tf.transpose(a1), dz2)
  db2 = tf.reduce_sum(dz2, axis=0)
  da1 = tf.matmul(dz2, tf.transpose(W2))
  dz1 = da1 * sigmoid_derivative(a1)
  dW1 = tf.matmul(tf.transpose(X), dz1)
  db1 = tf.reduce_sum(dz1, axis=0)

  # Update weights and biases using gradient descent
  W1.assign_sub(learning_rate * dW1)
  b1.assign_sub(learning_rate * db1)
  W2.assign_sub(learning_rate * dW2)
  b2.assign_sub(learning_rate * db2)

  if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss: {loss.numpy()}")

print("Final weights and biases:", W1.numpy(), b1.numpy(), W2.numpy(), b2.numpy())
```


**Example 2: Incorporating ReLU activation and multiple hidden layers**

This example extends the previous one by using the Rectified Linear Unit (ReLU) activation function and adding a second hidden layer.  ReLU's non-linearity improves performance, especially in deeper networks.

```python
import tensorflow as tf
import numpy as np

# ... (Hyperparameters similar to Example 1, adjust hidden layer sizes) ...

# Initialize weights and biases (add for second hidden layer)
W1 = tf.Variable(tf.random.normal([input_size, hidden_size_1]))
b1 = tf.Variable(tf.zeros([hidden_size_1]))
W2 = tf.Variable(tf.random.normal([hidden_size_1, hidden_size_2]))
b2 = tf.Variable(tf.zeros([hidden_size_2]))
W3 = tf.Variable(tf.random.normal([hidden_size_2, output_size]))
b3 = tf.Variable(tf.zeros([output_size]))

# Define ReLU activation
def relu(x):
  return tf.maximum(0.0, x)

def relu_derivative(x):
  return tf.cast(tf.greater(x, 0.0), dtype=tf.float32) # Efficient derivative of ReLU

# ... (Data generation similar to Example 1) ...

# Training loop (modified for two hidden layers and ReLU)
for epoch in range(epochs):
  # Forward propagation
  z1 = tf.matmul(X, W1) + b1
  a1 = relu(z1)
  z2 = tf.matmul(a1, W2) + b2
  a2 = relu(z2)
  z3 = tf.matmul(a2, W3) + b3
  a3 = sigmoid(z3) #Output layer still uses sigmoid for binary classification.

  # Loss calculation (binary cross-entropy)
  loss = -tf.reduce_mean(y * tf.math.log(a3) + (1 - y) * tf.math.log(1 - a3))

  # Backpropagation (modified for two hidden layers and ReLU)
  # ... (Chain rule applied to calculate gradients for all weights and biases) ...

  # Update weights and biases using gradient descent
  # ... (Weight updates similar to Example 1) ...

  if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss: {loss.numpy()}")
```


**Example 3:  Multi-class classification using softmax**

This example demonstrates a modification for multi-class classification. We replace the sigmoid activation in the output layer with softmax and use categorical cross-entropy as the loss function.

```python
import tensorflow as tf
import numpy as np

# ... (Hyperparameters, adjust output size to number of classes) ...

# Initialize weights and biases
# ... (Similar to Example 1 or 2) ...

# Define softmax activation
def softmax(x):
  return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=1, keepdims=True)

# Generate sample data for multi-class (e.g., one-hot encoded)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]) # Example 3 classes

# Training loop (modified for softmax and categorical cross-entropy)
for epoch in range(epochs):
  # Forward propagation (modified output layer)
  # ... (Similar to Example 1 or 2, but a3 = softmax(z3)) ...

  # Loss calculation (categorical cross-entropy)
  loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(a3), axis=1))

  # Backpropagation (modified for softmax and categorical cross-entropy)
  # ... (Chain rule applied to calculate gradients, adjustments required for softmax) ...

  # Update weights and biases using gradient descent
  # ... (Weight updates similar to Example 1) ...

  if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss: {loss.numpy()}")
```


**3. Resource Recommendations:**

For deeper understanding, I recommend studying  "Deep Learning" by Goodfellow, Bengio, and Courville;  a linear algebra textbook focusing on matrix operations; and a calculus textbook covering multivariate calculus and gradient descent.  A strong grasp of these fundamentals is crucial for implementing and debugging custom neural network architectures.  Furthermore, exploring the TensorFlow documentation specifically on low-level tensor manipulation and automatic differentiation will prove invaluable.
