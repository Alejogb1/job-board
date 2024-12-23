---
title: "How do I interpret training and testing errors when using batch gradient descent in Python?"
date: "2024-12-23"
id: "how-do-i-interpret-training-and-testing-errors-when-using-batch-gradient-descent-in-python"
---

, let's get into it. I've seen this pattern more than a few times in my years, and interpreting training and testing errors, especially with batch gradient descent, is a crucial skill for any practitioner of machine learning. It's not just about getting the numbers, but understanding the *why* behind them. A good grasp here can be the difference between a model that generalizes well and one that spectacularly fails on unseen data. I remember one project where we were building a predictive model for sensor data, and the initial training results were… well, concerning, but we cracked it by carefully examining these error trends.

So, when we talk about training error and testing error with batch gradient descent, we're basically looking at two different perspectives on how well our model is learning. Training error, which we often calculate on the training set, tells us how well our model is fitting the data it has already seen. Testing error, calculated on a held-out test set, gives us a view of how well our model is likely to perform on completely new, unseen data. The ideal situation is to have both errors decreasing concurrently, plateauing at a low value. This implies the model has learned the underlying patterns without overfitting.

However, the real world isn't so straightforward. We'll often observe scenarios where one error metric behaves differently than the other, which often indicates a problem with our model or training process.

Let's consider some common patterns and what they typically signify.

First, if your training error is significantly low, almost zero, but your testing error is high, you’ve got a clear case of overfitting. This means that your model has essentially memorized the training data but fails to generalize to new examples. It's like a student who has memorized the answers to practice questions but cannot solve similar problems. Batch gradient descent, by its nature, tends towards convergence, and with excessive capacity, it will fit the training data incredibly closely, potentially capturing noise and idiosyncrasies that aren't representative of the underlying phenomenon. To mitigate this, you might want to consider adding regularization techniques such as L1 or L2 regularization to the model cost function. You might also try early stopping or using simpler models. The capacity of the model is also vital here - a model with too many parameters will almost always overfit.

On the other hand, if both training and testing errors are high and not improving, this indicates underfitting. The model is too simplistic to capture the patterns in the data. This scenario suggests that you should consider using a more complex model or engineering richer features from your dataset. It could also mean your optimization process isn't converging, so things like adjusting the learning rate or increasing the number of epochs could help. Another potential reason could be poorly preprocessed input data.

It is worth noting that batch size also impacts the training trajectory. The batch size is a hyperparameter that determines how many samples the model sees at each gradient update. Smaller batches tend to introduce more noise in the gradient estimation, leading to more stochasticity in the training process. While this often leads to fluctuations in the training error, the advantage is that it can help the model avoid sharp minima, potentially leading to a better final result when generalising. In contrast, larger batches lead to smoother training, but they can also converge to local minima of suboptimal quality. The choice of batch size is an empirical choice that usually requires some experiments.

To illustrate these points, let's jump into some Python code using NumPy, which I find convenient for demonstrating these core concepts. I am deliberately keeping it simple, to focus on interpretation over more advanced implementations.

First, let's explore a case of *overfitting*. I'll simulate a dataset with some non-linear structure, and then try to fit it with an excessively complex model and examine the impact:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
# Generate noisy data
X = np.linspace(0, 1, 100)
y = 2*X**3 + np.random.normal(0, 0.2, 100)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

# Overly complex model: high-degree polynomial
degree = 10
W = np.random.rand(degree + 1, 1)

# Feature Engineering: Creating polynomial features
def create_polynomial_features(x, degree):
  X_poly = np.ones_like(x)
  for i in range(1, degree + 1):
    X_poly = np.hstack((X_poly, x**i))
  return X_poly

X_train_poly = create_polynomial_features(X_train, degree)
X_test_poly = create_polynomial_features(X_test, degree)

def batch_gradient_descent(X, y, W, learning_rate, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        predictions = X @ W
        errors = predictions - y.reshape(-1,1)
        gradient = (X.T @ errors) / m
        W = W - learning_rate * gradient
        cost = (1/(2*m)) * np.sum(errors**2)
        costs.append(cost)
    return W, costs

learning_rate = 0.001
iterations = 1000
W, costs = batch_gradient_descent(X_train_poly, y_train, W, learning_rate, iterations)

# Make Predictions
y_pred_train = X_train_poly @ W
y_pred_test = X_test_poly @ W


# Calculate error metrics
train_error = np.mean((y_pred_train.flatten() - y_train)**2)
test_error  = np.mean((y_pred_test.flatten() - y_test)**2)


print(f"Training Error: {train_error:.4f}")
print(f"Testing Error: {test_error:.4f}")

#Plot Learning Curve and model
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(costs)
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Cost')

plt.subplot(1,2,2)
plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, color='r', label='Testing Data')
x_values = np.linspace(0,1,100)
x_values_poly = create_polynomial_features(x_values[:, np.newaxis], degree)
y_model_fit = x_values_poly @ W
plt.plot(x_values, y_model_fit, color='g', label='Model fit')
plt.title('Data and Model Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()

```

In this case, you would likely see a low training error but a considerably higher test error, demonstrating overfitting. Note that I have used a polynomial of degree 10 to simulate an overly complex model.

Next, let's showcase an example of *underfitting*. I will use a model that is too simplistic, fitting the same dataset from before.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
# Generate noisy data
X = np.linspace(0, 1, 100)
y = 2*X**3 + np.random.normal(0, 0.2, 100)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

# Simplistic model: linear model
degree = 1
W = np.random.rand(degree + 1, 1)

# Feature Engineering: Creating polynomial features
def create_polynomial_features(x, degree):
  X_poly = np.ones_like(x)
  for i in range(1, degree + 1):
    X_poly = np.hstack((X_poly, x**i))
  return X_poly

X_train_poly = create_polynomial_features(X_train, degree)
X_test_poly = create_polynomial_features(X_test, degree)

def batch_gradient_descent(X, y, W, learning_rate, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        predictions = X @ W
        errors = predictions - y.reshape(-1,1)
        gradient = (X.T @ errors) / m
        W = W - learning_rate * gradient
        cost = (1/(2*m)) * np.sum(errors**2)
        costs.append(cost)
    return W, costs

learning_rate = 0.01
iterations = 1000
W, costs = batch_gradient_descent(X_train_poly, y_train, W, learning_rate, iterations)

# Make Predictions
y_pred_train = X_train_poly @ W
y_pred_test = X_test_poly @ W


# Calculate error metrics
train_error = np.mean((y_pred_train.flatten() - y_train)**2)
test_error  = np.mean((y_pred_test.flatten() - y_test)**2)


print(f"Training Error: {train_error:.4f}")
print(f"Testing Error: {test_error:.4f}")

#Plot Learning Curve and model
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(costs)
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Cost')

plt.subplot(1,2,2)
plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, color='r', label='Testing Data')
x_values = np.linspace(0,1,100)
x_values_poly = create_polynomial_features(x_values[:, np.newaxis], degree)
y_model_fit = x_values_poly @ W
plt.plot(x_values, y_model_fit, color='g', label='Model fit')
plt.title('Data and Model Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()
```

Here, both training and testing errors will remain relatively high, showing that the model is not capturing the underlying data pattern, and again, you might want to inspect the plots of the learning curve and the model fit.

Finally, I’ll show a situation where the learning is close to optimal, using a polynomial model of degree 3. The result is that the test and training errors will be reduced substantially.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
# Generate noisy data
X = np.linspace(0, 1, 100)
y = 2*X**3 + np.random.normal(0, 0.2, 100)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

# Polynomial model: degree 3
degree = 3
W = np.random.rand(degree + 1, 1)

# Feature Engineering: Creating polynomial features
def create_polynomial_features(x, degree):
  X_poly = np.ones_like(x)
  for i in range(1, degree + 1):
    X_poly = np.hstack((X_poly, x**i))
  return X_poly

X_train_poly = create_polynomial_features(X_train, degree)
X_test_poly = create_polynomial_features(X_test, degree)

def batch_gradient_descent(X, y, W, learning_rate, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        predictions = X @ W
        errors = predictions - y.reshape(-1,1)
        gradient = (X.T @ errors) / m
        W = W - learning_rate * gradient
        cost = (1/(2*m)) * np.sum(errors**2)
        costs.append(cost)
    return W, costs

learning_rate = 0.01
iterations = 1000
W, costs = batch_gradient_descent(X_train_poly, y_train, W, learning_rate, iterations)

# Make Predictions
y_pred_train = X_train_poly @ W
y_pred_test = X_test_poly @ W


# Calculate error metrics
train_error = np.mean((y_pred_train.flatten() - y_train)**2)
test_error  = np.mean((y_pred_test.flatten() - y_test)**2)


print(f"Training Error: {train_error:.4f}")
print(f"Testing Error: {test_error:.4f}")

#Plot Learning Curve and model
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(costs)
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Cost')

plt.subplot(1,2,2)
plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, color='r', label='Testing Data')
x_values = np.linspace(0,1,100)
x_values_poly = create_polynomial_features(x_values[:, np.newaxis], degree)
y_model_fit = x_values_poly @ W
plt.plot(x_values, y_model_fit, color='g', label='Model fit')
plt.title('Data and Model Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()
```

Beyond these simple examples, for a more thorough mathematical understanding, I’d strongly recommend delving into ‘Understanding Machine Learning: From Theory to Algorithms’ by Shai Shalev-Shwartz and Shai Ben-David. It provides a solid theoretical foundation for these concepts. Similarly, 'Pattern Recognition and Machine Learning' by Christopher M. Bishop is another authoritative resource with a more probabilistic approach, valuable for grasping the nuances of model generalization and error analysis. Also, don’t dismiss online lectures – the lectures by Andrew Ng on Coursera are an excellent resource, especially for building intuition.

In summary, interpreting training and testing errors is more than just looking at numbers; it requires understanding the underlying trends, and knowing how the error is evolving. These error trends are an important diagnostic that can help you improve the model and fine tune the learning process. These three examples, hopefully, give you a clear practical look at some of the most frequent scenarios you'll encounter. And always, keep iterating, keep testing, and keep learning.
