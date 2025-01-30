---
title: "How does a single-hidden-layer neural network with ReLU activation function perform?"
date: "2025-01-30"
id: "how-does-a-single-hidden-layer-neural-network-with-relu"
---
The efficacy of a single-hidden-layer neural network using ReLU activation hinges heavily on its ability to model non-linear relationships in data, achieved through a combination of the linear transformation and the ReLU’s piecewise linear property. This architecture is a fundamental building block in understanding deeper neural networks, acting as a bridge between simple linear models and complex multi-layer systems. My experience developing various machine learning solutions has shown that, while powerful, its performance is highly sensitive to factors like the dimensionality of the input data, the hidden layer size, and the characteristics of the dataset itself.

The architecture essentially comprises three layers: an input layer, a single hidden layer, and an output layer. During forward propagation, the input vector *x* undergoes a linear transformation in the form *z = Wx + b*, where *W* represents the weight matrix and *b* the bias vector. These values are initially randomized. Crucially, this initial linear transformation alone would be insufficient for capturing complex patterns in real-world data. This is where the Rectified Linear Unit (ReLU) activation function comes in. The ReLU, defined as *ReLU(z) = max(0, z)*, introduces non-linearity by setting all negative values of *z* to zero and passing on positive values unchanged. This seemingly simple operation allows the network to approximate complex, non-linear decision boundaries.

The output of the hidden layer, denoted as *a = ReLU(z)*, is then fed into the output layer. In the case of regression, the output layer would perform a second linear transformation and likely no further activation. For a classification problem, the final layer could involve a softmax activation, providing probabilities for each class. The training process, typically using backpropagation, involves calculating gradients of a loss function with respect to the weights and biases, updating them to minimize the loss. The ReLU’s simple derivative, 1 for positive values and 0 for negative values, contributes to efficient gradient calculation, avoiding the vanishing gradient problem that can plague networks using sigmoid or tanh activations.

However, this simplicity also leads to a crucial consideration: "dead" neurons. If a neuron's weighted sum input (z) becomes negative during training, it will output 0 and its gradient will be 0. This prevents the neuron from learning, becoming permanently inactive. This vulnerability becomes more prominent if the weights are initialized improperly or the learning rate is excessively high, often requiring careful parameter tuning.

The model's performance is also intrinsically tied to the hidden layer’s size. A small hidden layer may be insufficient to capture the complexity inherent in the data, leading to underfitting, while an excessively large hidden layer can cause overfitting, making the model perform well on training data but fail to generalize to new unseen data. The determination of the optimal size is often empirically guided, relying on validation-based assessment.

To illustrate these concepts, consider the following code examples using Python with the `numpy` library:

**Example 1: Forward Propagation**

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def single_layer_forward(x, W1, b1, W2, b2):
  # x is input vector, W1,b1 are hidden layer weights/biases, W2,b2 are output layer weights/biases

    z = np.dot(W1, x) + b1  # Linear transformation for hidden layer
    a = relu(z) # ReLU activation
    y_hat = np.dot(W2, a) + b2 # Linear transformation for output layer
    return y_hat, a # Returns the output prediction and the hidden layer activations

# Example Usage
input_size = 3
hidden_size = 4
output_size = 1

np.random.seed(42) # for reproducibility
W1 = np.random.randn(hidden_size, input_size)
b1 = np.random.randn(hidden_size, 1)
W2 = np.random.randn(output_size, hidden_size)
b2 = np.random.randn(output_size, 1)
x_example = np.random.randn(input_size, 1) # Example input feature vector

prediction, hidden_activations = single_layer_forward(x_example, W1, b1, W2, b2)
print("Prediction: ", prediction)
print("Hidden Activations: ", hidden_activations)
```

This code demonstrates the core forward propagation calculation, showing the application of linear transformations and the ReLU activation. Here, the initial weights and biases are randomized. The `relu` function applies the piecewise linear non-linearity. The output of this function is used to make a prediction by passing it through a linear transformation in the output layer. The function returns both the output and the activations at the hidden layer. This is important, especially when calculating gradients.

**Example 2: ReLU derivative**

```python
def relu_derivative(z):
    # ReLU derivative is 1 for z > 0 and 0 for z <= 0
    dz = np.copy(z)
    dz[dz <= 0] = 0
    dz[dz > 0] = 1
    return dz

# Example Usage
z_example = np.array([-2, -1, 0, 1, 2])
derivative = relu_derivative(z_example)
print("Input Z:", z_example)
print("ReLU Derivative:", derivative)
```

This snippet demonstrates the ReLU derivative calculation, which is essential for backpropagation. The derivative is used to compute gradient updates for the hidden layer weights and biases, allowing the network to learn.

**Example 3: Impact of hidden layer size**

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # for final prediction

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x)) # for final prediction backprop

def single_layer_train(X, y, hidden_size, learning_rate=0.1, epochs=1000):
    input_size = X.shape[1]
    output_size = 1 # Binary Classification

    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size)
    b1 = np.random.randn(hidden_size, 1)
    W2 = np.random.randn(output_size, hidden_size)
    b2 = np.random.randn(output_size, 1)

    for epoch in range(epochs):
        # Forward propagation
        z1 = np.dot(W1, X.T) + b1
        a1 = relu(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(z2)

        # Backward propagation
        dz2 = a2 - y.reshape(1, -1)
        dW2 = np.dot(dz2, a1.T) / X.shape[0]
        db2 = np.sum(dz2, axis=1, keepdims=True) / X.shape[0]
        dz1 = np.dot(W2.T, dz2) * relu_derivative(z1)
        dW1 = np.dot(dz1, X) / X.shape[0]
        db1 = np.sum(dz1, axis=1, keepdims=True) / X.shape[0]

        # Update parameters
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    z1 = np.dot(W1, X.T) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    predictions = (a2 > 0.5).astype(int)
    return predictions.flatten()


# Generate Sample Data
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and test with different hidden layer sizes
hidden_sizes = [5, 20, 100]
for hidden_size in hidden_sizes:
    W1, b1, W2, b2 = single_layer_train(X_train, y_train, hidden_size)
    y_pred = predict(X_test, W1, b1, W2, b2)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Hidden size: {hidden_size}, Test Accuracy: {accuracy}")

```
This final example shows a basic training loop and how different hidden layer sizes affect classification performance. We create a synthetic dataset and split it into training and testing sets.  Three different hidden layer sizes are used to train a single-layer neural network.  The results demonstrate how accuracy can change as the size of the hidden layer changes. Too small of a size and the network may not learn enough (underfitting), too large, and the network may learn noise in the data (overfitting).

For further investigation, consider resources focusing on the fundamentals of neural network architectures and backpropagation. Texts covering linear algebra and calculus would also be helpful in understanding the mathematical foundations of these algorithms. Specifically, resources that cover optimization techniques for neural networks, focusing on the impact of learning rates and initialization schemes, would improve understanding of the nuances of training such models.
