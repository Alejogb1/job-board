---
title: "How can stochastic gradient descent be applied to custom functions?"
date: "2024-12-23"
id: "how-can-stochastic-gradient-descent-be-applied-to-custom-functions"
---

,  I remember a particularly challenging project a few years back, developing a custom anomaly detection system for sensor data. We had this weird, non-standard loss function that wasn't playing nice with existing libraries, and that's where the deep dive into applying stochastic gradient descent (SGD) to custom functions really began. It's certainly more involved than just throwing data at a pre-built model, but absolutely achievable with a clear understanding of the underlying mechanisms.

Essentially, the power of SGD lies in its ability to iteratively adjust model parameters (weights, biases, etc.) to minimize a given cost or loss function. The function itself, in most deep learning scenarios, quantifies the error of our model’s predictions compared to the actual ground truth. When you move outside the realm of standard loss functions available in frameworks like tensorflow or pytorch, you need to be very explicit with how that process unfolds.

The core idea is this: we calculate the gradient (a fancy term for the direction of steepest increase) of our custom loss function with respect to the model’s parameters. We then take a step in the *opposite* direction of this gradient – because we want to minimize the loss, not maximize it. That’s the "descent" part. The "stochastic" part comes from the fact that we calculate this gradient not over the *entire* dataset at once, but using small, randomly chosen subsets of the data (batches). This is much more efficient computationally, especially with large datasets.

Now, how do you apply this to a *custom* function? There are two key areas that need careful attention: defining the custom loss function itself and then ensuring that you can compute its gradient efficiently, either through symbolic or numeric differentiation. The former is dependent entirely on your particular problem, but the latter is what I want to focus on for implementation and explanation.

**Example 1: A simple custom MSE-like function**

Let's start with something relatively straightforward. Suppose we have a custom loss that adds a penalty based on the magnitude of the model’s prediction, to prevent over-confident outputs. Our loss might be something like this (in Python-like pseudocode because math notations can be tricky in text):

`custom_loss(y_true, y_pred, lambda_val) = mean((y_true - y_pred)^2) + lambda_val * mean(abs(y_pred))`

Where `y_true` are the ground truth values, `y_pred` are the model's predictions, and `lambda_val` controls the strength of the penalty.

Here is the python code using numpy:

```python
import numpy as np

def custom_loss(y_true, y_pred, lambda_val):
    mse = np.mean((y_true - y_pred)**2)
    penalty = lambda_val * np.mean(np.abs(y_pred))
    return mse + penalty

def custom_loss_gradient(y_true, y_pred, lambda_val):
    diff = 2 * (y_pred - y_true)
    penalty_grad = lambda_val * np.sign(y_pred)
    return np.mean(diff + penalty_grad, axis=0) # gradient must be averaged across batch

# Example Usage (with some dummy data for a linear model)

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_true = np.array([3, 5, 7, 9])
W = np.array([0.1, 0.1]) # initial weights
b = 0.1 # initial bias
learning_rate = 0.01
lambda_val = 0.01

for epoch in range(1000):
    y_pred = np.dot(X, W) + b
    loss = custom_loss(y_true, y_pred, lambda_val)
    gradient = custom_loss_gradient(y_true, y_pred, lambda_val)

    W = W - learning_rate * np.dot(X.T , gradient) # weight update
    b = b - learning_rate * np.mean(gradient) # bias update


    if epoch % 100 == 0:
      print(f"Epoch: {epoch}, Loss: {loss}")

print(f"Final W: {W}, Final bias: {b}")
```

In this case, we are calculating the gradient analytically. The `custom_loss_gradient` function calculates the derivatives of the loss function, which is crucial for updating parameters in the right direction.

**Example 2: A Custom Loss with a Clipping Function**

Let’s explore a more complex custom loss function. Suppose our application needs a loss function where errors beyond a certain threshold are penalized less heavily. This could be useful in scenarios where outliers are common. We might design a loss something like this:

`custom_loss_clipped(y_true, y_pred, clip_threshold) = mean(clipped_squared_error(y_true - y_pred, clip_threshold))`

Where `clipped_squared_error` behaves like a squared error up to the `clip_threshold`, and then becomes linear beyond it. A common implementation might look like this:

```python
import numpy as np

def clipped_squared_error(error, clip_threshold):
    abs_error = np.abs(error)
    clipped_error = np.where(abs_error <= clip_threshold, error**2, 2 * clip_threshold * abs_error - clip_threshold**2)
    return clipped_error

def clipped_squared_error_gradient(error, clip_threshold):
    abs_error = np.abs(error)
    gradient = np.where(abs_error <= clip_threshold, 2 * error, 2 * clip_threshold * np.sign(error))
    return gradient


def custom_loss_clipped(y_true, y_pred, clip_threshold):
    error = y_true - y_pred
    return np.mean(clipped_squared_error(error, clip_threshold))


def custom_loss_clipped_gradient(y_true, y_pred, clip_threshold):
  error = y_true - y_pred
  return -np.mean(clipped_squared_error_gradient(error, clip_threshold), axis=0) # gradient needs negative sign

# Example Usage (using the same linear model)

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_true = np.array([3, 5, 7, 9])
W = np.array([0.1, 0.1])
b = 0.1
learning_rate = 0.01
clip_threshold = 1.0

for epoch in range(1000):
  y_pred = np.dot(X, W) + b
  loss = custom_loss_clipped(y_true, y_pred, clip_threshold)
  gradient = custom_loss_clipped_gradient(y_true, y_pred, clip_threshold)

  W = W - learning_rate * np.dot(X.T, gradient)
  b = b - learning_rate * np.mean(gradient)

  if epoch % 100 == 0:
    print(f"Epoch: {epoch}, Loss: {loss}")

print(f"Final W: {W}, Final bias: {b}")
```

Again, we compute the gradient of the modified error function, and apply that during the update of the model parameters. We define both the error function itself and its gradient.

**Example 3: Leveraging Automatic Differentiation**

For more complex loss functions, calculating the derivatives by hand can become a real headache. Fortunately, most deep learning frameworks support automatic differentiation. We'll use tensorflow to show an example. The idea is that we don't explicitly write out the gradient computation, but rely on the framework to handle the differentiation automatically.

```python
import tensorflow as tf

def custom_loss_tf(y_true, y_pred, lambda_val):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    penalty = lambda_val * tf.reduce_mean(tf.abs(y_pred))
    return mse + penalty

X = tf.constant([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
y_true = tf.constant([3., 5., 7., 9.])
W = tf.Variable([0.1, 0.1], dtype=tf.float32)
b = tf.Variable(0.1, dtype=tf.float32)
learning_rate = 0.01
lambda_val = 0.01

optimizer = tf.optimizers.SGD(learning_rate)

def train_step():
  with tf.GradientTape() as tape:
    y_pred = tf.matmul(X, tf.reshape(W, (2,1))) + b
    loss = custom_loss_tf(y_true, y_pred, lambda_val)
    gradients = tape.gradient(loss, [W,b])
    optimizer.apply_gradients(zip(gradients, [W,b]))
    return loss

for epoch in range(1000):
  loss = train_step()
  if epoch % 100 == 0:
    print(f"Epoch: {epoch}, Loss: {loss}")

print(f"Final W: {W.numpy()}, Final bias: {b.numpy()}")
```

In this snippet we see how we no longer need to explicitly calculate the gradient. With tensorflow and `GradientTape`, we define the loss function, and it takes care of the differentiation for us. This significantly reduces complexity and makes handling really complicated custom functions much easier.

**Key Considerations and Recommendations:**

*   **Gradient Checking:** Always, always double-check your gradient implementation, especially when doing it manually. Numerical differentiation using small perturbations can be a good check. Compare it against analytic gradients you derived.
*   **Numerical Stability:** Be mindful of numerical issues with your loss functions. Avoid operations that can lead to infinities or NaN values, especially when using exponentiations. For example, it might be worthwhile to use log loss instead of directly calculating error probabilities.
*   **Batching:** For large datasets, always train using mini-batches. This both provides noise that helps the optimization but also allows for a massive speed up due to parallelization.
*   **Regularization:** If you see your custom model overfitting, be prepared to add regularizers. These can be incorporated directly into the custom loss function.
*   **Learning Rates:** Tuning your learning rate will likely be necessary. You will probably see better results using adaptive optimizers (e.g. Adam, Adagrad) provided by your framework.
*   **Further Reading**: I recommend the following books/papers if you want to deepen your understanding on the topic:
    *   "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
    *   "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright
    *   "Understanding the difficulty of training deep feedforward neural networks" by Xavier Glorot and Yoshua Bengio (ICML 2010)
    *   Any good online course/MOOC in Neural Networks that teaches you the mathematics of backpropagation and differentiation.

Working with custom loss functions isn’t always straightforward, but it provides an incredible level of flexibility when tackling particular problems. The general process involves: define your custom loss function, calculate (either analytically or through autograd) its gradient, and then use that gradient to update the model's parameters in an iterative fashion. The examples provided are a good starting point and show the practical aspects of working on the problem. The key takeaway is that if you understand how gradients work you can apply your own functions. Remember that rigorous testing and validation are key to success.
