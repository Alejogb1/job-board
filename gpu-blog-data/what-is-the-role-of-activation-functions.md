---
title: "What is the role of activation functions?"
date: "2025-01-30"
id: "what-is-the-role-of-activation-functions"
---
Activation functions are the linchpin of neural network non-linearity, enabling these architectures to approximate complex functions. Without them, a neural network would essentially function as a linear regression model, severely limiting its capacity for learning intricate patterns. I've observed this directly during my time optimizing convolutional networks for image recognition, where switching from a non-linear activation to a linear pass resulted in a drastic drop in performance and an inability to discern between different classes of images.

The fundamental role of an activation function is to introduce a non-linear transformation to the weighted sum of inputs that a neuron receives. Let's consider a single neuron within a neural network. It receives multiple input values, each multiplied by an associated weight. These weighted inputs are summed, and then a bias term is added. This sum, often referred to as the "pre-activation" value, is then passed through the activation function. This non-linear transformation ensures that the network's output is not simply a linear combination of its inputs, which would drastically limit its ability to model complex relationships in the data. The capacity to learn such non-linear patterns is crucial for tackling real-world problems. Specifically, imagine trying to classify images of cats and dogs. The raw pixel values do not possess any inherent linearity separating these two classes; it is only through the stacked non-linear transformations afforded by layers of neurons and activation functions that the network can extract features that allow for accurate classification.

The choice of activation function significantly impacts the network’s performance, learning speed, and tendency to suffer from issues like vanishing or exploding gradients. Different functions have different characteristics that make them more suitable for particular tasks. For instance, I have seen how using ReLU and its variants generally results in faster training than sigmoid or tanh functions in deep convolutional networks, as the latter suffer more from vanishing gradients. When training recurrent neural networks for sequence processing, using tanh might sometimes be preferable despite the vanishing gradient issue due to its better centering around zero, preventing biases towards certain ranges of numbers.

To understand the role of activation functions, it is useful to look at some common examples and their properties:

**1. Sigmoid Activation:**

The sigmoid function, defined as σ(x) = 1 / (1 + e<sup>-x</sup>), squashes its input to a range between 0 and 1. This property makes it suitable for scenarios like output layers of a binary classification model where probabilities are required. However, its use in hidden layers is less common in deeper networks due to the vanishing gradient problem. When the input magnitude is large in either the positive or negative direction, the gradient approaches zero. During backpropagation, the gradients are multiplied together across layers, and if these gradients are close to zero, they can diminish exponentially as they propagate backward, effectively stalling or severely slowing the learning process in the earlier layers. I encountered this personally when experimenting with a deep feedforward network employing sigmoid layers and noticed how the earlier layers’ weights barely changed after multiple iterations.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Example demonstrating the output range of sigmoid
print(f"Sigmoid(-10): {sigmoid(-10):.4f}")
print(f"Sigmoid(0): {sigmoid(0):.4f}")
print(f"Sigmoid(10): {sigmoid(10):.4f}")
```

**2. ReLU (Rectified Linear Unit) Activation:**

The ReLU function, defined as f(x) = max(0, x), is one of the most commonly used activation functions in deep learning. It is simple to compute and has been shown to work very well for many different tasks. ReLU mitigates the vanishing gradient problem as its derivative is 1 for positive inputs, avoiding the small gradient problem common in sigmoids and tanh. However, one downside of ReLU is the "dying ReLU" problem where, if the neuron input is consistently negative, its gradient will always be zero, effectively rendering the neuron inactive.  This is something that I have dealt with directly and often required careful initialization or the use of techniques to avoid this issue.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
  return np.maximum(0, x)

x = np.linspace(-10, 10, 100)
y = relu(x)

plt.plot(x, y)
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Example demonstrating the behavior of ReLU
print(f"ReLU(-10): {relu(-10)}")
print(f"ReLU(0): {relu(0)}")
print(f"ReLU(10): {relu(10)}")
```

**3. Leaky ReLU Activation:**

Leaky ReLU addresses the dying ReLU problem by allowing a small, non-zero gradient when the input is negative. It is defined as f(x) = max(αx, x), where α is a small constant, usually between 0.01 and 0.3.  This minor modification prevents the neuron from becoming entirely inactive.  In practical application, I found that replacing ReLU with Leaky ReLU can significantly increase training stability and accuracy when dealing with sparse datasets or certain network architectures. It is an easy substitution that is often worth experimenting with, even when ReLU is working well initially.

```python
import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

x = np.linspace(-10, 10, 100)
y = leaky_relu(x)

plt.plot(x, y)
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Example demonstrating the behavior of leaky ReLU
print(f"Leaky ReLU(-10): {leaky_relu(-10)}")
print(f"Leaky ReLU(0): {leaky_relu(0)}")
print(f"Leaky ReLU(10): {leaky_relu(10)}")
```

These examples only scratch the surface of the wide variety of activation functions available. Others like tanh, ELU, and variations of ReLU such as PReLU and Swish also have different characteristics suitable for different tasks.

In conclusion, the activation function is not merely an ancillary component of the neural network; it is the core that allows learning non-linear, high-dimensional relationships within data. Selection should be based upon a consideration of the training dynamics, desired output properties, and the specific characteristics of the problem being addressed. To gain further understanding, I would recommend exploring resources that cover various neural network architectures and their implementation details. Textbooks on deep learning, for example, often devote entire sections to the rationale for different activation functions and their effects on model performance. Technical research papers, especially those proposing or evaluating new activation functions, can offer insights into their mathematical foundations and the empirical results that justify their use. Additionally, online documentation for popular deep learning libraries such as TensorFlow and PyTorch provides practical guidance on activation functions and their effective usage.
