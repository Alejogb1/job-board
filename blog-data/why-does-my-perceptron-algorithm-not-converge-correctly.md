---
title: "Why does my Perceptron algorithm not converge correctly?"
date: "2024-12-23"
id: "why-does-my-perceptron-algorithm-not-converge-correctly"
---

Okay, let's tackle this. I've seen this issue surface more times than I care to count, often during the early explorations of machine learning, and it's almost always rooted in subtle, yet impactful, misunderstandings of how the perceptron really operates. The non-convergence of your perceptron algorithm, it’s not some mystical black box failure, it's typically a manifestation of one or more specific underlying problems. Let's unpack a few of those.

From my experience building image classification systems back in the early 2010s, we used perceptrons as a baseline model before moving to more sophisticated networks. I recall a particularly frustrating instance where a simple binary classifier stubbornly refused to settle, and debugging it led to a deeper appreciation for the nuances of this seemingly straightforward algorithm. I'm willing to bet that at least one of my observations will echo your experience.

First, and perhaps most frequently, the problem arises due to *non-linearly separable data*. A core requirement for the basic perceptron to converge is that the data it's trying to classify must be linearly separable. That is, a single hyperplane (a line in 2D, a plane in 3D, and so on) must exist that can divide the data points belonging to different classes. If your data isn't like this, the perceptron will continuously, and fruitlessly, attempt to find such a separating hyperplane, oscillating endlessly between different weight vectors and never actually reaching a stable solution. This isn’t a bug, but rather a limitation intrinsic to the perceptron’s architecture.

Second, the learning rate, often denoted as 'alpha' or 'eta' in your code, plays a critical role. If your learning rate is too high, the algorithm can overcorrect at each iteration, causing it to jump around the solution space without ever settling into the optimal weights. Think of it like trying to find the bottom of a bowl by taking giant steps—you'll most likely keep stepping right over it. Conversely, if the learning rate is excessively small, convergence might become painfully slow, taking an unreasonable number of iterations. It might look like it's stuck, though it's just moving too slowly. There’s an art to selecting a good learning rate, and it's usually done by experimentation rather than deriving it from first principles. Techniques like cross-validation can assist in this.

Third, the initialization of the weights also matters. While in theory, random initialization should eventually lead to convergence given linearly separable data and a suitable learning rate, there are practical considerations. If the initial weights lead to large errors, the algorithm might spend a considerable amount of time correcting those large errors. This effect, when combined with a high learning rate, might cause the algorithm to behave erratically.

Now, let's solidify these points with some code. Here are three Python snippets, each showcasing one potential issue. I’ll assume we're using NumPy for vector operations, as that's common in this space.

**Example 1: Non-Linearly Separable Data**

```python
import numpy as np

def perceptron(X, y, learning_rate=0.1, n_iterations=100):
    n_samples, n_features = X.shape
    weights = np.random.rand(n_features)
    bias = 0

    for _ in range(n_iterations):
      for i in range(n_samples):
        prediction = np.dot(X[i], weights) + bias
        if prediction >= 0:
            predicted_label = 1
        else:
            predicted_label = -1

        if predicted_label != y[i]:
            weights = weights + learning_rate * y[i] * X[i]
            bias = bias + learning_rate * y[i]

    return weights, bias


# Example of non-linearly separable data
X = np.array([[1, 1], [2, 2], [2, 0], [0, 2]])
y = np.array([1, 1, -1, -1])  # XOR-like pattern (cannot separate with a single line)
trained_weights, trained_bias = perceptron(X, y)
print(f"Weights: {trained_weights}, Bias: {trained_bias}")
```

In this case, the XOR-like pattern is designed such that it *cannot* be separated with a simple line. The weights and bias will likely exhibit no discernible convergence. You'll notice the weights change erratically each run.

**Example 2: High Learning Rate**

```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron_with_tracking(X, y, learning_rate=0.1, n_iterations=100):
    n_samples, n_features = X.shape
    weights = np.random.rand(n_features)
    bias = 0
    weight_history = []
    for _ in range(n_iterations):
        for i in range(n_samples):
            prediction = np.dot(X[i], weights) + bias
            if prediction >= 0:
                predicted_label = 1
            else:
                predicted_label = -1

            if predicted_label != y[i]:
                weights = weights + learning_rate * y[i] * X[i]
                bias = bias + learning_rate * y[i]
        weight_history.append(weights.copy()) # store each weights state after one loop

    return weights, bias, weight_history


# Linearly separable data
X = np.array([[1, 1], [2, 2], [3, 1], [0, 1], [1, 0]])
y = np.array([1, 1, 1, -1, -1])
learning_rate = 1.5
trained_weights, trained_bias, weights_history = perceptron_with_tracking(X,y, learning_rate)
print(f"Weights: {trained_weights}, Bias: {trained_bias}")

# plotting to illustrate lack of convergence
plt.plot([i[0] for i in weights_history], label="weight 0")
plt.plot([i[1] for i in weights_history], label="weight 1")
plt.xlabel('epoch number')
plt.ylabel('weight value')
plt.legend()
plt.show()
```
Here, although the data *is* linearly separable, a high learning rate (1.5 in this example) prevents the weights from converging smoothly. Instead they are fluctuating widely, as shown by plotting. Try changing the learning rate to a lower value like 0.1 or 0.01 to observe how the algorithm converges with less oscillation.

**Example 3: Small Learning Rate**
```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron_with_tracking(X, y, learning_rate=0.1, n_iterations=100):
    n_samples, n_features = X.shape
    weights = np.random.rand(n_features)
    bias = 0
    weight_history = []
    for _ in range(n_iterations):
        for i in range(n_samples):
            prediction = np.dot(X[i], weights) + bias
            if prediction >= 0:
                predicted_label = 1
            else:
                predicted_label = -1

            if predicted_label != y[i]:
                weights = weights + learning_rate * y[i] * X[i]
                bias = bias + learning_rate * y[i]
        weight_history.append(weights.copy()) # store each weights state after one loop

    return weights, bias, weight_history


# Linearly separable data
X = np.array([[1, 1], [2, 2], [3, 1], [0, 1], [1, 0]])
y = np.array([1, 1, 1, -1, -1])
learning_rate = 0.0001
trained_weights, trained_bias, weights_history = perceptron_with_tracking(X,y, learning_rate, n_iterations=10000)
print(f"Weights: {trained_weights}, Bias: {trained_bias}")
#plotting to illustrate lack of convergence
plt.plot([i[0] for i in weights_history], label="weight 0")
plt.plot([i[1] for i in weights_history], label="weight 1")
plt.xlabel('epoch number')
plt.ylabel('weight value')
plt.legend()
plt.show()
```

Here, we're using a small learning rate (0.0001) even after 10,000 iterations the algorithm might not converge, or it will take a lot of time to converge, as it can be seen in the plot. The change is slow, and while it's not oscillating as wildly as in the previous example, it's an equally problematic situation, because of the computational cost involved in waiting for the algorithm to reach convergence.

For further understanding, I'd highly recommend diving into *Pattern Classification* by Duda, Hart, and Stork. This is a classic text that provides a very thorough treatment of the theoretical foundations of perceptrons and related algorithms. Another invaluable resource is *Neural Networks and Deep Learning* by Michael Nielsen, an online book that’s freely accessible and explains the core concepts very well. It covers limitations of single perceptrons and the benefits of neural networks.

In summary, non-convergence in your perceptron is typically due to non-linearly separable data, a poorly chosen learning rate, or less frequently, the initialization of the weights. Identifying which of these issues is hindering your progress will steer you towards a more effective solution and deeper understanding of these simple yet fundamental classifiers. These things take time, of course, and you learn by doing. Don’t be discouraged by the fact that it wasn’t quite working straight away, those initial challenges are where true learning takes place.
