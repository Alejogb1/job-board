---
title: "Why am I getting a Neural Network From Scratch - Forward propagation error?"
date: "2024-12-14"
id: "why-am-i-getting-a-neural-network-from-scratch---forward-propagation-error"
---

alright, so you're hitting a snag with forward propagation in your neural network from scratch. i've been there, many times. it’s a classic debugging situation, and it usually boils down to a few common culprits. let’s walk through the things i typically check when i see this kind of error, drawing on my own history with these types of headaches.

first off, when you say "forward propagation error", i'm assuming we’re talking about the output of your network not being what you expect, or more often, it's spitting out NaNs (not a number) or infinities, or simply incorrect values, causing problems down the line when you start trying to calculate the loss or gradient. it's almost never the theoretical stuff, usually the devil is in the details of how you are moving the data.

my first go-to check is always data shapes. seriously, i cannot emphasize this enough. back in the early days when i was first getting into this, i spent an entire evening fighting a similar problem, only to realize that i had accidentally transposed my weights matrix. what looked like perfectly valid code on the surface was actually feeding my network data with completely incorrect dimensions. so, let's be precise about this:

*   **input shape:** make sure the input you’re feeding into your network has the correct number of columns or features. if your input samples are rows, and you have, say, 10 features each sample should be a row vector with length 10.

*   **weight shapes:** each layer's weights should be shaped to properly transform the output from the previous layer to the input of the next. if the previous layer outputs something with dimension n, and the current layer has m neurons, then this layer’s weight matrix should have the shape (m, n). a common error is to do (n, m) which will cause a shape error in the forward propagation multiplication.

*   **bias shapes:** biases are additive components, they should match the output dimension of the layer they are associated with. they are typically vectors, each element of which is added to the corresponding element of the layer’s output.

for example, let's say you have an input *x*, that’s a row vector of shape (1, 5) and a layer with 3 neurons, you would normally expect a matrix multiplication of *x* (1,5) with *w*, weight matrix of shape (3, 5), plus biases, a vector with shape (3, 1) resulting in *z* which will have a shape of (1, 3). then this becomes the input of the next layer. in code this may look like:

```python
import numpy as np

def linear_forward(x, w, b):
    """
    Performs the linear forward pass (matrix multiplication and bias addition).

    Args:
        x (numpy.ndarray): Input data (shape: (1, n)).
        w (numpy.ndarray): Weight matrix (shape: (m, n)).
        b (numpy.ndarray): Bias vector (shape: (m, 1)).

    Returns:
        z (numpy.ndarray): Output of the linear pass (shape: (1, m)).
    """
    z = np.dot(x, w.T) + b.T
    return z

# Example usage:
x = np.array([[1, 2, 3, 4, 5]])
w = np.random.randn(3, 5)
b = np.random.randn(3, 1)

z = linear_forward(x, w, b)
print(f"Input shape: {x.shape}")
print(f"Weight shape: {w.shape}")
print(f"Bias shape: {b.shape}")
print(f"Output shape: {z.shape}")
print(f"Output: \n{z}")
```

i specifically included a `.T` there in `np.dot(x, w.T)` because we’ve seen people do dot product without thinking carefully about the shapes and this can create problems. this code snippet includes checks so you can examine your input, weights, biases, and output.

next, i’d look at the activation functions and their derivatives. this is another area that can quietly ruin your day. have you checked the bounds of your activation function? for example, if you're using a sigmoid function and not clamping the value before, it's easy to get values that are so big or so small, due to the matrix multiplies, that when you calculate the activation they can turn into nans (not a number) or infinity, specially, in the later layers with bigger and bigger weights.

let's look at a typical sigmoid implementation:

```python
import numpy as np

def sigmoid(z):
  """
    Applies the sigmoid activation function.

    Args:
        z (numpy.ndarray): Input data.

    Returns:
        a (numpy.ndarray): Output after applying sigmoid.
    """
  a = 1 / (1 + np.exp(-z))
  return a

def sigmoid_derivative(a):
  """
    Computes the derivative of the sigmoid function.

    Args:
        a (numpy.ndarray): Output after applying sigmoid.

    Returns:
        dz (numpy.ndarray): Derivative of the sigmoid function.
    """
  dz = a * (1 - a)
  return dz


# Example usage:
z = np.array([-10, -1, 0, 1, 10])
a = sigmoid(z)
print(f"Sigmoid output: {a}")
dz = sigmoid_derivative(a)
print(f"Sigmoid derivative: {dz}")
```
this code snippet shows a typical sigmoid implementation and its derivative. make sure your activation and derivative match, because a wrong derivative function means no correct gradient calculations during backpropagation later on.

another critical piece is your initialization. how are you initializing weights and biases? if you’re using all zeros or random small numbers from a very narrow range, your network might struggle to learn. i remember trying to get something working with small random initializations and it simply could not converge, i then used Xavier initialization (also known as glorot initialization) and it worked right away. a very common approach that works quite well is to use a random initialization of small numbers based on a gaussian distribution scaled using the size of the layer. and if you have a very deep net, then you will want to look at even more advanced initializations.

here's an example of random weight initialization:
```python
import numpy as np
def initialize_weights_xavier(input_size, output_size):
    """
    Initializes weights using Xavier (Glorot) initialization.

    Args:
        input_size (int): Number of inputs to the layer.
        output_size (int): Number of neurons in the layer.

    Returns:
        w (numpy.ndarray): Initialized weight matrix.
        b (numpy.ndarray): Initialized bias vector (zeros).
    """
    limit = np.sqrt(6 / (input_size + output_size))
    w = np.random.uniform(-limit, limit, size=(output_size, input_size))
    b = np.zeros((output_size, 1))
    return w, b

# Example usage:
input_size = 5
output_size = 3
w, b = initialize_weights_xavier(input_size, output_size)
print(f"Initialized weight shape: {w.shape}")
print(f"Initialized bias shape: {b.shape}")
```
this piece shows the Xavier initializer, and you can also experiment with other initialization methods which are well documented.

i've seen people sometimes forget that forward propagation requires you to iterate through all layers of your network doing the transformations step by step. forgetting to apply the forward function in some layers, or applying in the wrong order, will definitely not get you to where you want. remember to go step by step, doing the transformations between layers using the output of the previous layer as the input for the next one. and if you have a loop for this operation, make sure that you are indexing into the correct variables and the variables are properly initialized. the same with your matrix multiplications, make sure they are done in the correct order. i’ve made this mistake so many times. it’s always worth checking.

finally, a last tip based on a situation i experienced. if you are dealing with regression problems, make sure your output layer activation function is suitable for that kind of problem. if you have outputs outside the range of the activation function of the last layer then your network won't be able to output them and will result in an error.

in terms of resources, i'd recommend looking into "deep learning" by goodfellow, bengio, and courville. this is the main resource for deep learning. and also "neural networks and deep learning" by michael nielsen. these are not only a reference but also show the fundamentals. there’s a wealth of detailed explanations that help you understand not just how, but also *why* these things are done this way.

so, to recap, focus on these key areas when debugging your forward propagation: data shapes, correct implementation of your activation functions and their derivatives, proper weight initialization, and correctly calling the forward step of all layers. if you meticulously check these, you’ll likely find the cause of your errors. sometimes it’s something completely dumb and trivial. once i spent an entire afternoon debugging, only to find i was initializing my biases wrong... let's not talk about that. i hope these tips help. let me know how it goes.
