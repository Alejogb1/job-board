---
title: "How is activation applied in a neural network?"
date: "2024-12-23"
id: "how-is-activation-applied-in-a-neural-network"
---

Right then, let’s tackle activation functions in neural networks. It's a subject I've spent considerable time with, from my early days building image classifiers with limited hardware to more recent complex transformer-based architectures. So, let's break it down.

At its core, a neural network is designed to learn complex patterns from data. It achieves this by adjusting the weights and biases of its interconnected nodes, or neurons, across multiple layers. Crucially, between these layers, we don't just pass the raw weighted sums of inputs forward. Instead, we use activation functions. These functions introduce non-linearity into the network. Without them, a multi-layered neural network would essentially collapse into a single linear transformation, severely limiting its ability to learn intricate relationships in the data.

Essentially, each neuron computes a weighted sum of its inputs, adds a bias, and then that result passes through the activation function. This activation function decides whether or not the neuron should ‘fire’ based on that input. It determines the output of that neuron, which then serves as input to the next layer. Crucially, the choice of activation function influences how efficiently and effectively the network learns.

The activation function must be differentiable. This is non-negotiable. We need to calculate gradients during backpropagation, which is the mechanism through which the network learns to optimize its parameters. If the activation isn't differentiable, our gradients vanish, and learning grinds to a halt. Now, let's consider a few common activation functions and how they are applied.

First, the sigmoid function:
This was one of the earliest activations used. It squeezes its input between 0 and 1, effectively providing a probability-like output. Mathematically, it's represented as *σ(x) = 1 / (1 + e<sup>-x</sup>)*. While conceptually simple and useful in certain contexts (like binary classification), it suffers from the vanishing gradient problem, especially when inputs are very large or very small. In my early project with handwriting recognition, I often found my training getting stuck due to the gradient becoming practically zero.

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Example: output of a single neuron before activation
input_sum = 2.5  # weighted sum + bias

output = sigmoid(input_sum)
print(f"Sigmoid Output: {output}")  # Output: around 0.924
```

Then, we have the rectified linear unit, or ReLU. This is a very common choice today due to its computational efficiency and mitigation of the vanishing gradient problem (to some extent, at least compared to sigmoid). The function is simply *f(x) = max(0, x)*. This means for any negative input, the output is zero. For positive inputs, the output is the input value itself. This simple piecewise nature contributes significantly to computational speed. I recall switching from sigmoid to ReLU in a project involving speech recognition and observing significant speedup in training, as well as better results. However, ReLU can suffer from the 'dying ReLU' problem where neurons stop learning when all their inputs are negative.

```python
def relu(x):
  return max(0, x)

input_sum = -1.2 # weighted sum + bias

output = relu(input_sum)
print(f"ReLU Output: {output}")  # Output: 0
input_sum = 3.8 # weighted sum + bias

output = relu(input_sum)
print(f"ReLU Output: {output}") # Output: 3.8
```

Finally, let’s consider the hyperbolic tangent or tanh function, similar to sigmoid, but instead of squishing between 0 and 1, it squishes between -1 and 1. Mathematically, it's *tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)*. Tanh was often the go-to activation function after sigmoid because of its zero-centering property, which could help the gradients propagate slightly better during backpropagation in certain scenarios. I've successfully used it in several sequential model experiments in NLP tasks. But, similar to sigmoid, it also faces vanishing gradient problems as its output gets towards the extremes.

```python
import numpy as np

def tanh(x):
  return np.tanh(x)

input_sum = 1.5 # weighted sum + bias

output = tanh(input_sum)
print(f"Tanh Output: {output}") # Output: Around 0.905
```

Each of these activation functions introduces non-linearity, which is why we can build deep neural networks that learn complex patterns. However, they’re not interchangeable. The choice of activation often hinges on the specifics of your problem and the architecture you are employing. For example, ReLU and its variations are the dominant choice in computer vision tasks due to their efficiency, whereas for natural language processing, choices like tanh or even more advanced activations can be considered.

Now, in practice, it’s not just a matter of blindly picking an activation function. It requires experimentation and a solid understanding of the properties of each function. For example, if you see that your network is experiencing vanishing gradients, you might want to consider ReLU (or leaky ReLU), or perhaps a different initialization technique. Similarly, if you require outputs between -1 and 1, tanh could be a better choice than sigmoid. Many frameworks such as PyTorch and TensorFlow have implementations of a multitude of activation functions that can easily be integrated in your model.

To further enhance your understanding, I’d highly recommend delving into some authoritative resources. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville is a seminal work, covering not just activations but also a wide range of neural network concepts. Additionally, papers such as "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by Kaiming He et al. provide critical insights into the practical application and variations of ReLU. Finally, consider looking into resources such as "Understanding the difficulty of training deep feedforward neural networks," by Xavier Glorot and Yoshua Bengio which deals with the initialization issue which is also important in the context of activation functions, it's a key resource for anyone working on network optimization.

In conclusion, activation functions are not a mere detail. They are a core component of how neural networks learn and, more often than not, the choice of which activation function to use can have a big impact on how well your network trains and performs. While the theoretical underpinnings are important, practical experience in implementing and testing these functions across various use cases is, in my opinion, the only way to truly grasp their nuances and know which one is most suitable for the task at hand. This understanding grows with each practical project and by always striving to keep up with current advancements.
