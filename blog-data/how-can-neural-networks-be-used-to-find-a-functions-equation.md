---
title: "How can neural networks be used to find a function's equation?"
date: "2024-12-23"
id: "how-can-neural-networks-be-used-to-find-a-functions-equation"
---

Alright, let's talk function approximation with neural networks. It’s a topic I’ve spent quite a bit of time on, going back to my early days dealing with complex sensor data interpretation. Instead of trying to find a simple closed-form solution, we often found ourselves turning to the power of neural networks, especially when dealing with non-linear, high-dimensional data. The core idea is to leverage the neural network’s ability to act as a universal function approximator. It's not about *finding* the exact equation in the traditional symbolic sense; rather, it's about training a network to *mimic* the behavior of a function with a high degree of accuracy, essentially interpolating and extrapolating from the given data.

Essentially, what we're doing is using the network’s architecture to learn the mapping between inputs and outputs. Forget for a moment the idea of analytically determining `f(x) = 2x + 3`; instead, imagine you have a table of (x, y) pairs where `y` is approximately equal to `2x + 3`, perhaps with some noise. Our goal is to build a network that, when given an `x`, produces a `y` that’s close to what the actual function would generate.

The process begins with selecting the appropriate network architecture. For simpler, one-dimensional functions, a feedforward neural network with a few hidden layers will often suffice. The number of neurons in each layer, and the number of layers themselves, become hyperparameters we tune to achieve the best results. For more complex functions, you might need deeper networks or specialized layers, such as convolutional or recurrent layers, depending on the nature of the data. We often experimented with different activation functions for hidden layers, such as relu or tanh, observing how different choices impact training convergence. Remember, the choice of activation functions is crucial for non-linearity, which is needed to capture the non-linear relationships of most functions we try to approximate.

Then comes the data itself. You need a sufficient amount of data, covering a reasonable range of input values, to train the network effectively. Overfitting is a key concern here; if the dataset is too small, the network may memorize the training examples without generalizing well to unseen data. This leads to a model that might perform extremely well on training data but fails when it sees input it hasn’t encountered before. To counteract this, we used techniques like k-fold cross validation to ensure robustness of the trained network. You need to split the available data into three separate parts: the training set, the validation set (for tuning hyperparameters) and a holdout test set for evaluating the final model.

Finally, we optimize network parameters using a backpropagation algorithm, minimizing a loss function which quantifies the error between the network’s output and the desired output. Common loss functions for regression tasks, such as approximating function behavior, include mean squared error (mse) or mean absolute error (mae). The choice of optimizer (such as adam or stochastic gradient descent, or a variant thereof) and the learning rate are critical hyperparameters that will impact the convergence speed and the final model’s accuracy.

Let's look at some examples in python with tensorflow/keras:

**Example 1: Approximating a linear function**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Generate synthetic data for y = 2x + 3
np.random.seed(42)
x = np.linspace(-5, 5, 100)
y = 2 * x + 3 + np.random.normal(0, 0.5, 100) # add some noise
x = x.reshape(-1, 1)  # Reshape for neural network input

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(1,)) # Input of 1 dimension and output of 1 dimension
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, verbose=0)

# Evaluate the model on test data
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss}")

# Make prediction with the trained model:
test_input = np.array([[2.5]])
predicted_output = model.predict(test_input)
print(f"Prediction for x=2.5: {predicted_output[0][0]}")

```

In this example, we're training a very simple linear network to approximate a linear function. We use a dense layer with one neuron, which effectively learns the slope and the intercept. The mean squared error is a proper choice here given we are approximating the desired function.

**Example 2: Approximating a non-linear function**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Generate synthetic data for y = sin(x)
np.random.seed(42)
x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
y = np.sin(x) + np.random.normal(0, 0.1, 200)
x = x.reshape(-1, 1)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model with hidden layer to handle non-linearity
model = keras.Sequential([
    keras.layers.Dense(units=32, activation='relu', input_shape=(1,)),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=200, verbose=0)

# Evaluate the model on test data
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss}")

# Make prediction with the trained model
test_input = np.array([[np.pi/2]])
predicted_output = model.predict(test_input)
print(f"Prediction for x = pi/2: {predicted_output[0][0]}")
```

Here, we introduce non-linearity by adding two hidden layers with the relu activation function. This allows the network to approximate a more complex, non-linear sine function. The number of hidden neurons and the number of training epochs are important parameters to ensure optimal approximation.

**Example 3: Approximating a multivariate function**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Generate synthetic data for y = x1^2 + x2
np.random.seed(42)
num_samples = 200
x1 = np.random.uniform(-5, 5, num_samples)
x2 = np.random.uniform(-5, 5, num_samples)
x = np.column_stack((x1, x2))
y = x1**2 + x2 + np.random.normal(0, 0.5, num_samples)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model for multivariate function
model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape=(2,)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=200, verbose=0)

# Evaluate the model on test data
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss}")


# Make prediction with the trained model
test_input = np.array([[2, 3]])
predicted_output = model.predict(test_input)
print(f"Prediction for x1 = 2, x2 = 3: {predicted_output[0][0]}")
```

This last example demonstrates the flexibility to handle multivariate functions by increasing input dimensions. Here we’re using a function with two input variables.

In essence, this approach is about learning through examples, a data-driven perspective on function approximation. It does not produce symbolic equations as a mathematician might, but trained models that encapsulate the function’s input/output behavior. It is important to understand that we are not finding the function per se, rather we are approximating it.

If you want to deepen your knowledge, I suggest looking into "Deep Learning" by Goodfellow, Bengio, and Courville. It's an authoritative resource on deep learning concepts. For a more focused treatment of function approximation, papers on universal function approximators and neural network theory would be a great start. I also recommend exploring resources from the Stanford CS231n course, which offers detailed explanations with practical examples. These resources provide a solid theoretical grounding and the necessary practical understanding to use neural networks for function approximation. These sources have been instrumental in my own development and helped me tackle numerous function approximation challenges in my career.
