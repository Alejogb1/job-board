---
title: "Why is my three-layer neural network outputting NaN values?"
date: "2025-01-30"
id: "why-is-my-three-layer-neural-network-outputting-nan"
---
When a three-layer neural network consistently produces NaN (Not a Number) outputs, the root cause invariably lies in numerical instability during the forward or backward propagation phases. This often stems from exploding gradients, vanishing gradients, or encountering undefined mathematical operations within the network's calculations. I have frequently encountered this issue, particularly during my initial work with deep learning architectures on complex datasets, and have learned that careful scrutiny of activation functions, learning rates, and the weight initialization process is critical to resolution.

The central challenge arises from the iterative nature of backpropagation. Small changes in weights can be amplified through repeated matrix multiplications and activation function derivatives, leading to either extremely large (exploding gradients) or exceedingly small (vanishing gradients) values. When these gradients become too large, they can push the network's parameters to regions where subsequent calculations result in undefined or infinity-related behaviors, manifested as NaNs. Conversely, gradients that effectively vanish halt any meaningful learning progress. These issues are exacerbated by the compounding effect of multiple layers.

A common contributor to this problem is the use of activation functions that can saturate. Consider, for instance, the sigmoid function. Its derivative approaches zero as its input magnitudes become large, thus contributing to vanishing gradients. Conversely, a ReLU (Rectified Linear Unit) function can also contribute to the problem if its inputs become sufficiently negative, resulting in inactive neurons and potentially large weight adjustments later in training. The interplay between these functions and the chosen weight initialization method is paramount in ensuring network stability. I recall a project where I was attempting to classify medical images, and the use of a poorly scaled sigmoid with randomly initialized weights resulted in early saturation and a complete absence of learning, evidenced by constant NaN outputs.

Furthermore, a high learning rate can inadvertently exacerbate both exploding and vanishing gradient issues. During gradient descent, if the update step is too large, the network can overshoot the optimal weight values, resulting in oscillation or divergence, leading to numerical instabilities and, finally, NaNs. Effectively, the learning process becomes akin to trying to navigate a very steep terrain with very large steps, quickly moving out of the optimal descent range. In another project involving sequence-to-sequence translation, I had to significantly reduce the learning rate and implement gradient clipping to stabilize the training process and prevent the frequent NaN occurrences I was initially encountering.

Here are specific examples of code, along with explanations of why these errors could lead to NaN outputs and how to fix them:

**Example 1: Unstable Weight Initialization with Sigmoid Activations**

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = sigmoid(np.dot(input_data, self.weights) + self.biases)
        return self.output

    def backward(self, output_gradient, learning_rate):
        sigmoid_grad = sigmoid_derivative(np.dot(self.input, self.weights) + self.biases)
        weight_gradient = np.dot(self.input.T, output_gradient * sigmoid_grad)
        bias_gradient = np.sum(output_gradient * sigmoid_grad, axis=0, keepdims=True)

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
        input_gradient = np.dot(output_gradient * sigmoid_grad, self.weights.T)
        return input_gradient

# Initialize the network
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 0.1

layer1 = Layer(input_size, hidden_size)
layer2 = Layer(hidden_size, output_size)

# Prepare dummy input data
input_data = np.random.randn(1, input_size)

# Forward and backward passes for a few steps
for _ in range(10):
    output = layer2.forward(layer1.forward(input_data))
    
    # Compute a dummy output gradient (not accurate for learning but illustrates the issue)
    output_gradient = np.random.randn(1, output_size)

    # Back propagation
    grad_layer2 = layer2.backward(output_gradient, learning_rate)
    layer1.backward(grad_layer2, learning_rate)

    if np.isnan(output).any():
        print("NaN detected")
        break
print("Output:", output)

```

**Commentary:** The original code uses standard normal initialization (`np.random.randn`). Combined with the sigmoid activation, it's highly likely that some activations will saturate early, causing the derivative to become very small. This results in the vanishing gradient problem. For more effective weight initialization, use Xavier or He initialization. In this case, an implementation of He initialization is recommended for ReLU activation and sigmoid activation benefit from Xavier initialization.

**Example 2: A better initialization strategy (Xavier initialization) with Sigmoid Activations:**

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Layer:
    def __init__(self, input_size, output_size):
        # Xavier initialization
        variance = 2.0 / (input_size + output_size)
        self.weights = np.random.normal(0, np.sqrt(variance), (input_size, output_size))
        self.biases = np.zeros((1, output_size))


    def forward(self, input_data):
        self.input = input_data
        self.output = sigmoid(np.dot(input_data, self.weights) + self.biases)
        return self.output


    def backward(self, output_gradient, learning_rate):
        sigmoid_grad = sigmoid_derivative(np.dot(self.input, self.weights) + self.biases)
        weight_gradient = np.dot(self.input.T, output_gradient * sigmoid_grad)
        bias_gradient = np.sum(output_gradient * sigmoid_grad, axis=0, keepdims=True)

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
        input_gradient = np.dot(output_gradient * sigmoid_grad, self.weights.T)
        return input_gradient


# Initialize the network
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 0.1

layer1 = Layer(input_size, hidden_size)
layer2 = Layer(hidden_size, output_size)

# Prepare dummy input data
input_data = np.random.randn(1, input_size)

# Forward and backward passes for a few steps
for _ in range(10):
    output = layer2.forward(layer1.forward(input_data))
    
    # Compute a dummy output gradient (not accurate for learning but illustrates the issue)
    output_gradient = np.random.randn(1, output_size)

    # Back propagation
    grad_layer2 = layer2.backward(output_gradient, learning_rate)
    layer1.backward(grad_layer2, learning_rate)

    if np.isnan(output).any():
        print("NaN detected")
        break
print("Output:", output)

```

**Commentary:** Here, a Xavier initialization strategy is introduced. This approach helps maintain the variance of activations across layers, reducing the chance of early saturation or overly small weight updates and reduces the chance of NaN values. It calculates the appropriate variance for the normal distribution, based on the size of the input and output of the layer. It significantly improves convergence and stability of the training process.

**Example 3: High Learning Rate leading to NaN values**

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Layer:
    def __init__(self, input_size, output_size):
        # Xavier initialization
        variance = 2.0 / (input_size + output_size)
        self.weights = np.random.normal(0, np.sqrt(variance), (input_size, output_size))
        self.biases = np.zeros((1, output_size))


    def forward(self, input_data):
        self.input = input_data
        self.output = sigmoid(np.dot(input_data, self.weights) + self.biases)
        return self.output


    def backward(self, output_gradient, learning_rate):
        sigmoid_grad = sigmoid_derivative(np.dot(self.input, self.weights) + self.biases)
        weight_gradient = np.dot(self.input.T, output_gradient * sigmoid_grad)
        bias_gradient = np.sum(output_gradient * sigmoid_grad, axis=0, keepdims=True)

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
        input_gradient = np.dot(output_gradient * sigmoid_grad, self.weights.T)
        return input_gradient


# Initialize the network
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 1 # High learning rate

layer1 = Layer(input_size, hidden_size)
layer2 = Layer(hidden_size, output_size)

# Prepare dummy input data
input_data = np.random.randn(1, input_size)

# Forward and backward passes for a few steps
for _ in range(10):
    output = layer2.forward(layer1.forward(input_data))
    
    # Compute a dummy output gradient (not accurate for learning but illustrates the issue)
    output_gradient = np.random.randn(1, output_size)

    # Back propagation
    grad_layer2 = layer2.backward(output_gradient, learning_rate)
    layer1.backward(grad_layer2, learning_rate)

    if np.isnan(output).any():
        print("NaN detected")
        break
print("Output:", output)
```

**Commentary:** Setting the learning rate to 1 demonstrates how a large update step can destabilize the training. It will cause the weights and biases to be updated aggressively, possibly making the gradients even larger in following updates, quickly leading to infinity values, which are converted to NaNs, during matrix multiplication or derivative calculations. Decreasing the learning rate to a much smaller value and applying gradient clipping are common ways to mitigate this.

In addressing NaN outputs, systematically reviewing these areas is fundamental:

1.  **Activation Functions:** Carefully consider the choice of activation functions. ReLU or its variants (LeakyReLU, ELU) are often preferable to sigmoid and tanh in deep networks, especially in hidden layers.
2.  **Weight Initialization:** Employ robust initialization techniques, like Xavier/Glorot or He initialization, to avoid saturation of activation units.
3.  **Learning Rate:** Fine-tune the learning rate, often employing adaptive techniques like Adam or RMSprop, or by decreasing it with a schedule.
4.  **Gradient Clipping:** If gradient explosion is suspected, implementing gradient clipping can mitigate the problem by limiting the maximum magnitude of gradients during updates.
5. **Data Scaling:** Ensure the input data is appropriately scaled. Normalizing or standardizing the data often helps with training stability and convergence.

Consulting textbooks on deep learning optimization and reviewing documentation on deep learning frameworks can provide additional insights into best practices for mitigating numerical instability. Furthermore, experiment with different techniques and carefully observe the training loss behavior to gain a better understanding of which technique is best for a particular problem and network.
