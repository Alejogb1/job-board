---
title: "How does backpropagation calculate local gradients?"
date: "2024-12-16"
id: "how-does-backpropagation-calculate-local-gradients"
---

Let's tackle this. It's something I spent a considerable amount of time on back in my days working on a custom deep learning framework for satellite image analysis – a project that really forced me to get under the hood of these algorithms.

Backpropagation, at its core, is essentially a clever application of the chain rule of calculus, optimized for neural networks. The goal is to determine how much each weight and bias in the network contributes to the overall error, so we can adjust them and improve performance. Crucially, the process isn't about calculating some magical, global gradient; rather, we compute *local* gradients, step by step, and propagate them backward. That’s where the name comes from. It’s not a single calculation – it's a cascade.

To be precise, backpropagation calculates gradients in a layered manner, starting from the output layer and moving backward to the input layer. Each neuron in a given layer performs a computation, and this computation involves inputs (either from the previous layer or the raw input data), weights, biases, and an activation function. The local gradient calculation focuses *solely* on the components of this specific neuron's computation, and how the *output* of that computation changes with respect to changes in its inputs, weights and bias.

Consider, for a single neuron, the equation: *output = activation(sum(weight_i * input_i) + bias)*. For the sake of simplicity, let's use sigmoid as our activation function here, as it clearly demonstrates the gradient calculation. A sigmoid function is defined as *sigmoid(x) = 1 / (1 + exp(-x))*, and its derivative is *sigmoid(x) * (1 - sigmoid(x))*. I remember spending weeks staring at those equations trying to optimise the computation.

The local gradients, during backpropagation, are then:

1.  **Gradient with respect to the bias:** This is simply the derivative of the output with respect to the bias. Since the bias is directly added to the weighted sum, the local gradient with respect to the bias *before* the activation function is just 1. However, the full local gradient will include the derivative of the activation function itself. Let's call this derivative *sigmoid_deriv*. Then, *local_gradient_bias = sigmoid_deriv*. This gradient tells us how a small change in the bias affects the neuron's output.

2.  **Gradient with respect to a weight:** This is the derivative of the output with respect to the specific weight. Similar to the bias gradient, the gradient before the activation function is simply the corresponding input *input_i*. Again, we need to incorporate the derivative of the activation function. So, *local_gradient_weight_i = input_i * sigmoid_deriv*. This shows the influence of this specific weight on the final output.

3.  **Gradient with respect to the input:** This is where it gets interesting. Here we calculate how changing the input to the neuron affects its output. The gradient before the activation function here is the corresponding weight *weight_i*. Then we have: *local_gradient_input_i = weight_i * sigmoid_deriv*. This gradient is then passed backward to the previous layer to calculate its gradients, cascading the gradients towards the input layer. This, is the essence of backward propagation. This step, along with the weights, is what makes up the gradient signal to the preceding layers.

Now, let's demonstrate these calculations with some basic python using numpy. This first example will show the direct computation:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def neuron_output(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return sigmoid(weighted_sum)

def calculate_local_gradients(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    output = sigmoid(weighted_sum)
    sigmoid_deriv = sigmoid_derivative(weighted_sum)

    local_gradient_bias = sigmoid_deriv
    local_gradient_weights = inputs * sigmoid_deriv
    local_gradient_inputs = weights * sigmoid_deriv

    return local_gradient_bias, local_gradient_weights, local_gradient_inputs

# Example
inputs = np.array([0.5, 0.2])
weights = np.array([0.8, -0.3])
bias = 0.1

bias_grad, weight_grads, input_grads = calculate_local_gradients(inputs, weights, bias)

print(f"Local Gradient with respect to Bias: {bias_grad}")
print(f"Local Gradient with respect to Weights: {weight_grads}")
print(f"Local Gradient with respect to Inputs: {input_grads}")

```

In this first snippet, we are directly computing the local gradients for a given neuron. The `calculate_local_gradients` function is a direct translation of the previously defined gradient equations. This is a core implementation for any system doing backpropagation.

However, in practice, it's often more efficient to use a vectorized version, particularly when you have multiple neurons in a layer or multiple samples in a batch. For example:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def vectorized_neuron_output(inputs, weights, biases):
    weighted_sums = np.dot(inputs, weights.T) + biases
    return sigmoid(weighted_sums)

def vectorized_local_gradients(inputs, weights, biases):
    weighted_sums = np.dot(inputs, weights.T) + biases
    outputs = sigmoid(weighted_sums)
    sigmoid_derivs = sigmoid_derivative(weighted_sums)

    local_gradient_bias = sigmoid_derivs
    local_gradient_weights = np.dot(sigmoid_derivs.reshape(-1, 1), inputs) # Note the reshaping for correct broadcasting
    local_gradient_inputs = np.dot(sigmoid_derivs, weights)

    return local_gradient_bias, local_gradient_weights, local_gradient_inputs

# Vectorized Example
inputs = np.array([[0.5, 0.2], [0.7, 0.1]]) #batch of 2 examples each with 2 input features
weights = np.array([[0.8, -0.3], [0.6, 0.4]]) # 2 neurons each with 2 input weights
biases = np.array([0.1, -0.2]) #bias for each neuron

bias_grads, weight_grads, input_grads = vectorized_local_gradients(inputs, weights, biases)

print(f"Vectorized Local Gradient with respect to Bias: {bias_grads}")
print(f"Vectorized Local Gradient with respect to Weights: \n{weight_grads}")
print(f"Vectorized Local Gradient with respect to Inputs: \n{input_grads}")

```

This vectorized example processes multiple examples simultaneously and is a significantly more practical approach. Notice how the calculation for weight and input gradients uses matrix multiplication. The reshapes for the biases and gradients are important here to align matrix dimensions for efficient calculation. This, vectorized method, is much faster and scalable.

Finally, in practice frameworks like PyTorch or TensorFlow handle this automatically using graph-based computation. To illustrate how these concepts exist in a framework, we can do it using a very simple PyTorch implementation:

```python
import torch

def simple_pytorch_example():
    # Enable gradient tracking
    inputs = torch.tensor([0.5, 0.2], requires_grad=True)
    weights = torch.tensor([0.8, -0.3], requires_grad=True)
    bias = torch.tensor(0.1, requires_grad=True)

    # Forward pass
    weighted_sum = torch.dot(inputs, weights) + bias
    output = torch.sigmoid(weighted_sum)

    # Backward pass
    output.backward()

    # Access gradients
    print(f"PyTorch Gradient with respect to Bias: {bias.grad}")
    print(f"PyTorch Gradient with respect to Weights: {weights.grad}")
    print(f"PyTorch Gradient with respect to Inputs: {inputs.grad}")


simple_pytorch_example()
```

Here, pytorch's automatic differentiation handles the gradient calculation. Notice how I use `requires_grad=True`. This tells PyTorch that we need gradients for these tensors. The backward method automatically calculates the local gradients and stores them in `variable.grad`. This shows the power of modern deep learning frameworks which abstract away low level details.

For further study, I highly recommend delving into "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Also, "Neural Networks and Deep Learning" by Michael Nielsen is excellent, particularly for gaining a fundamental grasp. A great research paper to look into would be “Backpropagation Applied to Handwritten Zip Code Recognition” by Yann LeCun, et al.

In summary, backpropagation calculates *local* gradients, and then uses those local gradients in a layered manner to determine how changing each of the weights and biases in the network will affect the overall error. The core computation revolves around calculating these derivatives of the activation and weighted sums by applying the chain rule. From my own experience, a solid understanding of these calculations is fundamental when working at a low-level or trying to understand the inner mechanics of deep learning systems. While frameworks handle most of the complexity, this understanding is still critical.
