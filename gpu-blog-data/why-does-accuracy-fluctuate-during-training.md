---
title: "Why does accuracy fluctuate during training?"
date: "2025-01-30"
id: "why-does-accuracy-fluctuate-during-training"
---
Model accuracy during training is rarely a consistently upward trajectory; instead, it exhibits periods of improvement intermixed with plateaus and even temporary declines. This fluctuation is a direct consequence of the interplay between the optimization algorithm, the inherent complexity of the loss landscape, and the characteristics of the training data itself. Understanding these contributing factors is crucial for building robust and generalizable models.

Initially, let’s consider the optimization process. Gradient descent, or its variants, is used to minimize a loss function that quantifies the model’s error on the training data. This loss function defines a high-dimensional landscape, often non-convex, with local minima, saddle points, and flat regions. The optimization process involves iteratively adjusting model parameters by moving in the direction of the negative gradient. The magnitude of these adjustments is dictated by the learning rate. High learning rates can cause the optimization to overshoot local minima or bounce around the landscape, resulting in accuracy swings. Conversely, excessively low learning rates can result in slow convergence and getting trapped in suboptimal solutions. Adaptive learning rate algorithms, such as Adam or RMSprop, attempt to mitigate this by dynamically adjusting the learning rate for each parameter. However, even with these algorithms, accuracy can still fluctuate, especially in the initial stages of training.

The structure of the training data also contributes to accuracy fluctuations. During each training epoch, data is typically presented in batches. If the batch size is small, the gradient calculated is based on a limited subset of the training data, making it a noisy estimate of the true gradient across the entire dataset. This noise can lead to significant deviations in the parameter updates and result in accuracy swings. The variance of the batch samples can also introduce fluctuations. A batch with many challenging or atypical examples can lead to increased error, while a relatively easy batch can give the illusion of improvement. In extreme cases, the model might even temporarily overfit to a specific batch, causing a decrease in accuracy on other parts of the dataset. Using larger batch sizes typically results in more stable training because they average out the noise from individual samples. However, they can also lead to slower training time. Finding a good compromise between batch size and stable training is essential.

Furthermore, the representational capacity of the model plays a key role. If the model has too few parameters to capture the complexity of the underlying patterns in the training data, it can underfit. In this situation, accuracy will plateau at a relatively low level. Conversely, if the model is excessively complex, it may overfit the training data. This is characterized by initially low training error and a high test error. In the overfitting case, the model may capture noise present in training data, resulting in poor generalization and fluctuations in accuracy when presented with unseen data. Regularization techniques, such as dropout or L1/L2 regularization, mitigate overfitting by limiting model capacity. However, selecting optimal regularization hyperparameters can also impact training stability. These techniques often introduce an element of stochasticity, which can introduce short-term fluctuations while generally improving long-term generalization.

Below are three code examples illustrating various points discussed above. These examples are deliberately simplistic and intended for didactic purposes rather than practical usage.

**Example 1: Demonstrating the effect of learning rate.**

```python
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return x**2 - 4*x + 3

def gradient(x):
    return 2*x - 4

def gradient_descent(learning_rate, iterations):
    x = 0
    history = []
    for _ in range(iterations):
      x = x - learning_rate * gradient(x)
      history.append(objective_function(x))
    return history


learning_rates = [0.01, 0.1, 0.2, 0.5]
plt.figure(figsize=(10,6))
for lr in learning_rates:
  hist = gradient_descent(lr, 100)
  plt.plot(hist, label=f"Learning Rate: {lr}")

plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.title("Impact of Learning Rate on Optimization")
plt.legend()
plt.grid(True)
plt.show()
```

This example utilizes a simple quadratic function and gradient descent to demonstrate the effect of different learning rates. A low learning rate (0.01) results in slow but consistent progress towards the minimum. A moderate learning rate (0.1) shows relatively stable convergence. Larger rates (0.2 and 0.5) display oscillatory behavior, with larger jumps around the minimum. The learning rate of 0.5 even initially increases the value before coming back down, mimicking unstable learning dynamics. This illustrates how selecting an appropriate learning rate is crucial for stability.

**Example 2: Illustrating the impact of batch size in stochastic gradient descent.**

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_points, noise_scale=0.1):
    x = np.random.rand(num_points) * 10
    y = 2 * x + 1 + np.random.normal(0, noise_scale, num_points)
    return x.reshape(-1, 1), y

def linear_model(x, w, b):
  return w * x + b

def mse(y_true, y_pred):
  return np.mean((y_true - y_pred) ** 2)

def sgd_batch(X, y, batch_size, learning_rate, iterations):
    w = np.random.randn()
    b = np.random.randn()
    losses = []
    for _ in range(iterations):
        indices = np.random.choice(len(X), size=batch_size, replace=False)
        batch_x = X[indices]
        batch_y = y[indices]
        y_pred = linear_model(batch_x, w, b)
        error = y_pred - batch_y
        w = w - learning_rate * np.mean(error * batch_x)
        b = b - learning_rate * np.mean(error)
        losses.append(mse(y, linear_model(X, w, b)))
    return losses

num_points = 100
X, y = generate_data(num_points)

batch_sizes = [1, 5, 10, 20, 50, 100]
iterations = 100

plt.figure(figsize=(10,6))
for batch_size in batch_sizes:
  losses = sgd_batch(X, y, batch_size, 0.01, iterations)
  plt.plot(losses, label=f"Batch Size: {batch_size}")

plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Impact of Batch Size on Training Stability")
plt.legend()
plt.grid(True)
plt.show()

```

This code generates synthetic data and uses stochastic gradient descent to train a linear model. We vary the batch size and observe the training loss. Small batch sizes (1, 5, 10) exhibit a more fluctuating loss during training because each update is more significantly influenced by the specific examples in that batch. As the batch size increases (20, 50, 100), the loss stabilizes with lower fluctuations due to more accurate gradient estimation. A batch size of 100 corresponds to standard gradient descent using the entire training dataset in each iteration, providing a smooth learning curve. This illustrates the impact batch size has on the noisiness of gradient estimates.

**Example 3: An example using a complex model and showing overfitting.**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generate_data(num_points, noise_scale=10):
    x = np.random.rand(num_points) * 10
    y = x**3 + 2*x**2 - 5*x + 10 + np.random.normal(0, noise_scale, num_points)
    return x.reshape(-1, 1), y

def train_and_evaluate_model(X_train, y_train, X_test, y_test, degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    return train_error, test_error

num_points = 200
X, y = generate_data(num_points)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

degrees = [1, 3, 10]
train_errors = []
test_errors = []

for degree in degrees:
  train_error, test_error = train_and_evaluate_model(X_train, y_train, X_test, y_test, degree)
  train_errors.append(train_error)
  test_errors.append(test_error)

plt.figure(figsize=(8, 6))
bar_width = 0.25
index = np.arange(len(degrees))
plt.bar(index, train_errors, bar_width, label='Train MSE')
plt.bar(index + bar_width, test_errors, bar_width, label='Test MSE')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Test Error for Different Model Complexity")
plt.xticks(index + bar_width / 2, [f"Degree {d}" for d in degrees])
plt.legend()
plt.grid(True)
plt.show()
```

This example demonstrates the impact of model complexity on training and test error, showcasing overfitting. We fit polynomial regression models of different degrees (1, 3, and 10) to a cubic relationship plus noise. A linear model (degree 1) underfits the data, resulting in large errors on both the training and test sets. A cubic model (degree 3) fits the underlying relationship accurately. A high degree polynomial (degree 10) overfits the training data, resulting in low training error but high test error. This highlights how too much model capacity leads to overfitting and poor generalization, manifested as unstable accuracy on unseen data.

For further exploration and a deeper understanding of these topics, I recommend examining the following resources:
*   "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, for a comprehensive theoretical foundation of deep learning.
*   "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron, for practical implementation techniques and considerations.
*   Courses such as those offered by Andrew Ng, or fast.ai which provide a more applied perspective on building robust models.

These resources can offer further understanding in the areas of optimization theory, regularization techniques, and practical application of these concepts within the realm of machine learning.
