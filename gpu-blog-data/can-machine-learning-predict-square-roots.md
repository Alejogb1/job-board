---
title: "Can machine learning predict square roots?"
date: "2025-01-30"
id: "can-machine-learning-predict-square-roots"
---
The core challenge in using machine learning to predict square roots lies not in the complexity of the calculation itself, but rather in framing the problem in a way amenable to machine learning algorithms. A square root, denoted as √x, is fundamentally a deterministic mathematical function; given *x*, the result is uniquely defined. Machine learning models, particularly those based on neural networks or similar architectures, excel at learning approximations of complex, non-deterministic relationships from data. Therefore, training a model to predict square roots becomes an exercise in function approximation, rather than revealing any hidden structure or probabilistic nature inherent to the square root operation.

From my experiences in signal processing, where we often need to computationally handle non-linear operations efficiently, I've found that approaching this with supervised learning techniques proves most effective. Specifically, we generate a dataset of input values and their corresponding square roots, treat the inputs as features, and the square roots as target labels. A regression model then learns the mapping between the two.

Let’s illustrate with a simple example. Consider a linear regression model, a rudimentary choice, but one that will highlight a crucial point. We'd initially hypothesize a simple linear relationship: *y = mx + b*, where *y* is the predicted square root, *x* is the input number, *m* is the slope, and *b* is the y-intercept. This is a gross simplification because the square root function is non-linear. However, this will clarify the training process and its limitations.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate training data
x_train = np.linspace(0, 10, 100).reshape(-1, 1)  # Input values 0 to 10
y_train = np.sqrt(x_train)                   # Corresponding square roots

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Generate test data and predictions
x_test = np.linspace(0, 10, 50).reshape(-1, 1)
y_pred = model.predict(x_test)

# Plot results
plt.scatter(x_train, y_train, color='blue', label='Actual Square Root')
plt.plot(x_test, y_pred, color='red', label='Linear Regression Prediction')
plt.xlabel("Input (x)")
plt.ylabel("Square Root (√x)")
plt.title("Linear Regression Approximation of Square Root")
plt.legend()
plt.show()

print(f"Model parameters: Slope (m) = {model.coef_[0][0]:.4f}, Intercept (b) = {model.intercept_[0]:.4f}")
```

This first example demonstrates that a linear model performs poorly. The resulting plot shows a straight line attempting to approximate a curved function. The linear regression can find a best-fit straight line, but there are significant errors, especially at higher input values. The model is constrained by its linear nature and struggles to capture the non-linear relationship of the square root function. The calculated slope and intercept are simply the values that minimize the mean squared error for the linear approximation within the training data. This highlights that model selection is crucial.

For a more suitable approach, a polynomial regression model can better capture the curve inherent in the square root function. Let’s implement that using the same training and testing setup:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate training data (same as before)
x_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sqrt(x_train)

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)

# Train polynomial regression model
model = LinearRegression()
model.fit(x_train_poly, y_train)

# Generate test data
x_test = np.linspace(0, 10, 50).reshape(-1, 1)
x_test_poly = poly.transform(x_test)
y_pred = model.predict(x_test_poly)


# Plot results
plt.scatter(x_train, y_train, color='blue', label='Actual Square Root')
plt.plot(x_test, y_pred, color='red', label='Polynomial Regression Prediction')
plt.xlabel("Input (x)")
plt.ylabel("Square Root (√x)")
plt.title("Polynomial Regression Approximation of Square Root")
plt.legend()
plt.show()


print(f"Model coefficients: {model.coef_[0:]}")
print(f"Model intercept: {model.intercept_[0]}")
```

This second code example utilizes polynomial regression with a degree of 2. We transform our original data using scikit-learn's `PolynomialFeatures` module. This expands the feature space to include not only `x`, but also `x^2`.  This increased flexibility allows the model to approximate the curve of the square root function more closely.  The `model.coef_` output now gives us coefficients for `x^0`, `x^1`, and `x^2`.  The plot visually shows a much better fit than the straight line from the previous example, though there are still deviations, particularly at the very start and end of the range. The degree of the polynomial directly impacts the flexibility of the model. Higher degrees can capture more complex curves but can lead to overfitting, which is where the model fits the training data too closely and performs poorly on new, unseen data.

A more robust solution, particularly for complex non-linear relationships, involves neural networks. Let's build a basic neural network for this approximation.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate training data (same as before)
x_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sqrt(x_train)


# Build the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, verbose=0)

# Generate test data
x_test = np.linspace(0, 10, 50).reshape(-1, 1)
y_pred = model.predict(x_test)


# Plot the results
plt.scatter(x_train, y_train, color='blue', label='Actual Square Root')
plt.plot(x_test, y_pred, color='red', label='Neural Network Prediction')
plt.xlabel("Input (x)")
plt.ylabel("Square Root (√x)")
plt.title("Neural Network Approximation of Square Root")
plt.legend()
plt.show()
```

This third example employs a basic feedforward neural network with two hidden layers, utilizing the ReLU activation function. The model is trained using the Adam optimizer and minimizes mean squared error. The plot reveals that the neural network provides a highly accurate approximation of the square root function. Neural networks, with their ability to learn intricate patterns, are much better at capturing the non-linearity than simpler models like linear or polynomial regressions. This demonstrates the increased capacity of neural networks to learn complex relationships from data. The model learns the underlying function through iteratively adjusting the weights in its layers during training. While neural networks introduce more parameters and require more computational resources than simpler models, their capability to generalize complex functions is a crucial factor in their selection.

For further study, I recommend delving into fundamental texts covering regression models and neural network theory. Specifically, resources from machine learning pioneers like Hastie, Tibshirani, and Friedman on statistical learning offer thorough explanations of regression and related topics. For a deeper dive into neural networks, texts like Goodfellow, Bengio, and Courville provide comprehensive coverage. Additionally, practical tutorials using frameworks like scikit-learn and TensorFlow or PyTorch will solidify your understanding and ability to implement these concepts. These foundational resources provide both the theoretical and practical context necessary for a comprehensive grasp of the techniques for predicting the square root function (and others) using machine learning. The key is to recognize the deterministic nature of the operation and treat it as a function approximation problem.
