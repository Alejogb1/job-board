---
title: "How to build a neural network from scratch?"
date: "2024-12-14"
id: "how-to-build-a-neural-network-from-scratch"
---

alright, so you're looking to build a neural network from the ground up. that's a pretty solid challenge, and i'm totally on board with that kind of project. i’ve spent more time than i care to recall debugging some gnarly custom nn implementations, and let me tell you, it's a journey, but a very rewarding one.

first off, when i say 'from scratch', i assume we're not talking about using high-level libraries like tensorflow or pytorch. those are great for prototyping and production but bypass much of the core mechanics. we're going down to matrix operations, gradient calculation and the whole nine yards, right?.

i remember once, back in my early days of neural network tinkering, i thought i could whip up a multi-layer perceptron in an afternoon. boy, was i wrong. i spent two weeks battling with my own poorly implemented backpropagation algorithm. the error gradients were exploding and it turned out my initializations were terrible. i had naively used random numbers from 0 to 1 as initial weights and, obviously, everything went haywire. that was a painful lesson but i learned it.

anyway, here's the lowdown on how i tackle this kind of thing these days, focusing on the fundamentals:

the core ingredients:

*   **data preparation:** this is the unsung hero of ml. you can have the fanciest network architecture, but if your data is garbage or poorly formatted, you'll get garbage results. preprocessing like normalization or standardization are fundamental. also, if you are dealing with images, reshaping is key.
*   **linear algebra knowledge:** you need to be comfortable with matrices and vectors. matrix multiplication, transposition, and understanding their relationship to neural network layers is essential. if these are foggy areas, i recommend spending some time studying the mathematics behind it. a good start is "linear algebra and its applications" by gilbert strang, it covers all the bases.
*   **activation functions:** these introduce non-linearity into your network, which is crucial for learning complex patterns. sigmoid, relu, tanh and their variants are your standard tools here. you must know how their derivatives look like because they are needed in backpropagation.
*   **forward propagation:** this is where the input data travels through the network, layer by layer, getting transformed by the weights and biases. this is simple enough, basically matrix multiplication and application of the activation function, but it’s fundamental.
*   **loss function:** this measures how well your network's predictions match the actual values. it gives you a scalar value that tells how bad or good the prediction is. mean squared error and cross-entropy are popular choices. you must choose this well depending on your task, classification or regression.
*   **backpropagation:** this is the heavy lifting. it calculates gradients of the loss with respect to the network's weights, which then help you update those weights in the right direction. this often gets tricky in multilayer networks and you will find yourself debugging the backpropagation for hours.
*   **optimization algorithm:** stochastic gradient descent (sgd) and its variants like adam or rmsprop are often used to update the weights based on the computed gradients. if you decide to go for sgd you need to tune the learning rate well. this parameter changes everything.

let's break down a simple example: a single-layer perceptron (essentially a logistic regression)

here’s how it could look in python using numpy:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward_prop(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    return sigmoid(z)

def loss_func(y_predicted, y_actual):
   return np.mean((y_predicted - y_actual)**2)

def loss_derivative(y_predicted, y_actual):
    return 2 * (y_predicted - y_actual) / len(y_actual)

def back_prop(inputs, weights, bias, y_actual, learning_rate):
    y_predicted = forward_prop(inputs, weights, bias)
    dl_dy = loss_derivative(y_predicted, y_actual)
    dy_dz = sigmoid_derivative(np.dot(inputs, weights) + bias)
    dl_dz = dl_dy * dy_dz
    dl_dw = np.dot(inputs.T, dl_dz)
    dl_db = np.sum(dl_dz)

    weights = weights - learning_rate * dl_dw
    bias = bias - learning_rate * dl_db

    return weights, bias

# dummy data, just for testing
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# init parameters
weights = np.random.rand(2, 1)
bias = np.random.rand(1)
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
  weights, bias = back_prop(inputs, weights, bias, targets, learning_rate)
  if epoch % 100 == 0:
     y_pred = forward_prop(inputs, weights, bias)
     loss = loss_func(y_pred, targets)
     print(f"epoch {epoch} loss is: {loss}")
print(f"prediction after training {forward_prop(inputs, weights, bias)}")
```

this example is for a simple binary classification task. it shows you the basic forward prop with an activation function (sigmoid), the definition of a cost function (mse) and the backprop and parameter update. it will not give you the greatest accuracy but shows all the steps clearly. for this single layer problem there is no need to calculate gradients in respect to the inputs, but that would be necessary for a multi-layer perceptron.

now, building a multilayer perceptron (mlp)

things get a tad more intricate, but the underlying concepts are the same. you now have multiple layers of weights and biases. here's a snippet to extend the previous example to a two-layer network.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def forward_prop(inputs, weights1, bias1, weights2, bias2):
    z1 = np.dot(inputs, weights1) + bias1
    a1 = relu(z1) # first activation function
    z2 = np.dot(a1, weights2) + bias2
    a2 = sigmoid(z2) # second activation function
    return a2, a1

def loss_func(y_predicted, y_actual):
    return np.mean((y_predicted - y_actual)**2)

def loss_derivative(y_predicted, y_actual):
    return 2 * (y_predicted - y_actual) / len(y_actual)

def back_prop(inputs, weights1, bias1, weights2, bias2, y_actual, learning_rate):
    y_predicted, a1 = forward_prop(inputs, weights1, bias1, weights2, bias2)

    dl_dy = loss_derivative(y_predicted, y_actual)
    dy_dz2 = sigmoid_derivative(np.dot(a1, weights2) + bias2)
    dl_dz2 = dl_dy * dy_dz2

    dl_dw2 = np.dot(a1.T, dl_dz2)
    dl_db2 = np.sum(dl_dz2, axis=0, keepdims=True)

    da1_dz1 = relu_derivative(np.dot(inputs, weights1) + bias1)
    dl_da1 = np.dot(dl_dz2, weights2.T)
    dl_dz1 = dl_da1 * da1_dz1

    dl_dw1 = np.dot(inputs.T, dl_dz1)
    dl_db1 = np.sum(dl_dz1, axis=0, keepdims=True)

    weights1 = weights1 - learning_rate * dl_dw1
    bias1 = bias1 - learning_rate * dl_db1
    weights2 = weights2 - learning_rate * dl_dw2
    bias2 = bias2 - learning_rate * dl_db2

    return weights1, bias1, weights2, bias2

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

#init parameters
n_hidden = 2
weights1 = np.random.randn(2, n_hidden)
bias1 = np.random.randn(1, n_hidden)
weights2 = np.random.randn(n_hidden, 1)
bias2 = np.random.randn(1, 1)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    weights1, bias1, weights2, bias2 = back_prop(inputs, weights1, bias1, weights2, bias2, targets, learning_rate)
    if epoch % 1000 == 0:
      y_pred, _ = forward_prop(inputs, weights1, bias1, weights2, bias2)
      loss = loss_func(y_pred, targets)
      print(f"epoch {epoch} loss is: {loss}")
print(f"prediction after training {forward_prop(inputs, weights1, bias1, weights2, bias2)}")
```

this code introduces an extra hidden layer with relu activation, which is very important because it introduces non-linearity and lets the network learn more complex patterns. the backpropagation becomes a little more complex because we need to calculate gradients for each weight and bias of both layers. also note that the initialization of weights is now a standard normal distribution instead of uniform 0 to 1 which improves the overall training.

now, let's address something important which is about scaling up to larger datasets.

for larger datasets, you’ll almost certainly want to use mini-batch gradient descent to avoid needing to calculate gradients over the entire training set in each epoch, it's computationally heavy. here’s how you can change the previous code to implement that:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def forward_prop(inputs, weights1, bias1, weights2, bias2):
    z1 = np.dot(inputs, weights1) + bias1
    a1 = relu(z1)
    z2 = np.dot(a1, weights2) + bias2
    a2 = sigmoid(z2)
    return a2, a1

def loss_func(y_predicted, y_actual):
    return np.mean((y_predicted - y_actual)**2)

def loss_derivative(y_predicted, y_actual):
    return 2 * (y_predicted - y_actual) / len(y_actual)

def back_prop(inputs, weights1, bias1, weights2, bias2, y_actual, learning_rate):
    y_predicted, a1 = forward_prop(inputs, weights1, bias1, weights2, bias2)

    dl_dy = loss_derivative(y_predicted, y_actual)
    dy_dz2 = sigmoid_derivative(np.dot(a1, weights2) + bias2)
    dl_dz2 = dl_dy * dy_dz2
    dl_dw2 = np.dot(a1.T, dl_dz2)
    dl_db2 = np.sum(dl_dz2, axis=0, keepdims=True)
    da1_dz1 = relu_derivative(np.dot(inputs, weights1) + bias1)
    dl_da1 = np.dot(dl_dz2, weights2.T)
    dl_dz1 = dl_da1 * da1_dz1
    dl_dw1 = np.dot(inputs.T, dl_dz1)
    dl_db1 = np.sum(dl_dz1, axis=0, keepdims=True)

    weights1 = weights1 - learning_rate * dl_dw1
    bias1 = bias1 - learning_rate * dl_db1
    weights2 = weights2 - learning_rate * dl_dw2
    bias2 = bias2 - learning_rate * dl_db2

    return weights1, bias1, weights2, bias2

# dummy data, just for testing
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

n_hidden = 2
weights1 = np.random.randn(2, n_hidden)
bias1 = np.random.randn(1, n_hidden)
weights2 = np.random.randn(n_hidden, 1)
bias2 = np.random.randn(1, 1)

learning_rate = 0.1
epochs = 10000
batch_size = 2

def create_batches(inputs, targets, batch_size):
    num_samples = inputs.shape[0]
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]
        batches.append((batch_inputs, batch_targets))
    return batches

for epoch in range(epochs):
    batches = create_batches(inputs, targets, batch_size)
    for batch_inputs, batch_targets in batches:
        weights1, bias1, weights2, bias2 = back_prop(batch_inputs, weights1, bias1, weights2, bias2, batch_targets, learning_rate)
    if epoch % 1000 == 0:
        y_pred, _ = forward_prop(inputs, weights1, bias1, weights2, bias2)
        loss = loss_func(y_pred, targets)
        print(f"epoch {epoch} loss is: {loss}")

print(f"prediction after training {forward_prop(inputs, weights1, bias1, weights2, bias2)}")

```
in this code, instead of processing all the data at once, we process it in mini-batches. this greatly improves training speed for large datasets, since the gradient is calculated in smaller subsets of the training data and the parameters are updated more often. it also helps to reduce overfitting and helps the training find better local minimum in the cost function.

these snippets give you a starting point for building your own neural network from the ground up. building a neural network involves more than just matrix multiplications and derivatives. you’ll also want to consider stuff like initialization strategies, batch normalization and regularization. that's probably a conversation for another time, or you can explore more in "deep learning" by ian goodfellow, yoshua bengio, and aaron courville, which is a very comprehensive text. building a network is like constructing a house brick by brick, if you understand every part well, you will be able to build a very solid and interesting neural network. and hey, remember, don’t initialize your weights with random numbers from zero to one. i learned that one the hard way. or did i?.
