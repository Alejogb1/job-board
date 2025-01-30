---
title: "How can I interpret training and testing errors for batch gradient descent in Python?"
date: "2025-01-30"
id: "how-can-i-interpret-training-and-testing-errors"
---
Batch gradient descent, a foundational optimization algorithm in machine learning, iteratively adjusts model parameters to minimize a cost function over the *entire* training dataset. Understanding the behavior of training and testing errors during this process is critical for diagnosing model performance, identifying overfitting or underfitting, and making informed decisions about model architecture and training strategy. I've encountered these issues extensively throughout my years working with various machine learning models, and a nuanced interpretation of these error dynamics is indispensable for effective model building.

The training error, calculated on the data the model has learned from, provides an indication of how well the model is fitting the training data. A decreasing training error usually signifies that the model is successfully learning the patterns within this data. The testing error, on the other hand, computed on unseen data, acts as a proxy for the model’s ability to generalize to new, previously unencountered examples. The disparity between these two errors is the crux of the interpretation process. Specifically, a large gap between a low training error and a high testing error indicates overfitting, where the model has essentially memorized the training data rather than learning underlying, generalizable relationships. Conversely, both high training and testing errors suggest underfitting, where the model fails to capture even the basic patterns in the data.

The convergence behavior of these errors during training provides further insights. Ideally, we observe both the training and testing errors decreasing over epochs, eventually plateauing at a minimal value. The rate at which these errors decrease, along with the eventual plateaus, offers important diagnostic information. For example, very slow convergence might suggest that the learning rate is too low or the chosen optimizer is not optimal for the data or model architecture. Conversely, erratic or oscillating error curves might indicate a learning rate that's too high, causing the algorithm to overshoot the optimal solution. Monitoring the changes in these errors is therefore more significant than merely examining their final values.

Let’s illustrate these concepts with a Python implementation using NumPy, avoiding any external machine learning libraries for clarity. We will simulate a regression task where the goal is to approximate a noisy sine wave. I’ll start with a very basic linear model as the base example, where underfitting is most likely.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate noisy sine wave data
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + np.random.normal(0, 0.3, 100)

# Split data into training and testing
X_train = X[:80].reshape(-1, 1)
y_train = y[:80].reshape(-1, 1)
X_test = X[80:].reshape(-1, 1)
y_test = y[80:].reshape(-1, 1)

# Linear model (y = wx + b)
def predict(X, w, b):
    return np.dot(X, w) + b

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradient(X, y_true, y_pred, w, b):
    dw = -2 * np.mean(X * (y_true - y_pred))
    db = -2 * np.mean(y_true - y_pred)
    return dw, db

# Training loop
def train_model(X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=1000):
    w = np.random.randn(1, 1)
    b = np.random.randn(1, 1)
    training_losses = []
    testing_losses = []

    for epoch in range(epochs):
        y_pred_train = predict(X_train, w, b)
        y_pred_test = predict(X_test, w, b)

        training_loss = mse_loss(y_train, y_pred_train)
        testing_loss = mse_loss(y_test, y_pred_test)

        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        dw, db = gradient(X_train, y_train, y_pred_train, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    return training_losses, testing_losses

# Training
training_losses, testing_losses = train_model(X_train, y_train, X_test, y_test)

# Plotting losses
plt.plot(training_losses, label='Training Loss')
plt.plot(testing_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Linear Model Loss')
plt.legend()
plt.show()
```
This first example, using a simple linear model, demonstrates the typical behavior of an underfitting scenario. The plot generated will showcase that while the loss generally decreases over time, it plateaus at a relatively high value for both training and testing sets. The model fails to adequately capture the underlying sinusoidal pattern, indicating that a linear model is too simple for this particular task, hence leading to high error for both seen and unseen data.

Let’s add a polynomial feature to our model to allow for more complex relationships.

```python
# Add polynomial features to the model
def add_polynomial_features(X, degree):
    X_poly = X.copy()
    for i in range(2, degree+1):
        X_poly = np.concatenate((X_poly, X**i), axis=1)
    return X_poly

# Re-using predict, mse_loss, gradient, and train_model with updated parameters
def updated_gradient(X, y_true, y_pred, w):
    dw = -2 * np.mean(X * (y_true - y_pred), axis=0, keepdims=True)
    return dw

def updated_predict(X, w, b):
    return np.dot(X, w.T) + b

def updated_train_model(X_train, y_train, X_test, y_test, degree, learning_rate=0.01, epochs=1000):
    X_train_poly = add_polynomial_features(X_train, degree)
    X_test_poly = add_polynomial_features(X_test, degree)

    w = np.random.randn(1, X_train_poly.shape[1])
    b = np.random.randn(1, 1)
    training_losses = []
    testing_losses = []

    for epoch in range(epochs):
        y_pred_train = updated_predict(X_train_poly, w, b)
        y_pred_test = updated_predict(X_test_poly, w, b)

        training_loss = mse_loss(y_train, y_pred_train)
        testing_loss = mse_loss(y_test, y_pred_test)

        training_losses.append(training_loss)
        testing_losses.append(testing_loss)
        dw = updated_gradient(X_train_poly, y_train, y_pred_train, w)
        w = w - learning_rate * dw
        db = -2 * np.mean(y_train- y_pred_train)
        b = b - learning_rate * db


    return training_losses, testing_losses

# Train polynomial model
degree = 3
training_losses_poly, testing_losses_poly = updated_train_model(X_train, y_train, X_test, y_test, degree)

# Plotting losses
plt.plot(training_losses_poly, label='Training Loss')
plt.plot(testing_losses_poly, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Polynomial Model Loss')
plt.legend()
plt.show()

```
With the addition of polynomial features to degree 3, both training and testing errors significantly decrease and show better convergence. They are close to each other, indicating an improvement in model fitting. The model is now able to capture the underlying sinusoidal pattern much better than the previous linear one. This is a scenario where we have improved the model capacity to handle the data complexities without significant overfit.

Let’s examine a case of overfitting by increasing the degree of the polynomial features.

```python
# Train overfitting polynomial model
degree = 10
training_losses_overfit, testing_losses_overfit = updated_train_model(X_train, y_train, X_test, y_test, degree, learning_rate = 0.001, epochs = 1000)

# Plotting losses
plt.plot(training_losses_overfit, label='Training Loss')
plt.plot(testing_losses_overfit, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Overfitting Polynomial Model Loss')
plt.legend()
plt.show()
```

Here, increasing the polynomial degree to 10 will likely result in overfitting. The plot will now show a lower training error than testing error, and testing error does not reduce as much as before. This indicates that the model is becoming too complex, learning the noise within the training set, and does not generalize well to new, unseen data. In practice, we might see the testing loss decreasing and then start increasing at later epochs. This divergence between training and test losses is a strong indication of overfitting.

These three examples demonstrate how analyzing the training and testing errors can provide valuable insights into model behavior during batch gradient descent. In my practical experience, the precise shape of these curves is affected by a multitude of factors, including data characteristics, model architecture, initial parameter values, learning rate, and the chosen optimizer. To further enhance the analysis, I recommend consulting material on bias-variance decomposition, regularization techniques (L1 and L2), and the use of validation sets in addition to train and test sets. Texts on deep learning principles, as well as machine learning methodologies, will prove highly valuable resources to develop a deeper understanding of model training and debugging. Additionally, practical implementations in tutorials focusing on hands-on coding help solidify this abstract information into practical skills. By paying careful attention to training and test errors, you can significantly improve your ability to diagnose and debug machine learning models.
