---
title: "Can a model avoid overfitting to a single data point?"
date: "2025-01-30"
id: "can-a-model-avoid-overfitting-to-a-single"
---
The inherent risk of overfitting, particularly to singular data points, stems from a model’s capacity to memorize the training set rather than generalize underlying patterns. I’ve encountered this specific challenge numerous times in my work, particularly when dealing with imbalanced datasets or sparse feature spaces, leading to models that perform exceptionally well on the training data, but poorly on unseen examples. The problem isn't that the model "sees" only one data point, but that its training process allows it to assign disproportionate weight to the characteristics of this lone data point, even if they're noise rather than signal.

The core issue is the model's capacity. Complex models, having many parameters, are more prone to overfitting because they possess the flexibility to fit virtually any training data, including the idiosyncratic features of a single data instance. Consider a very high-degree polynomial fit to a single datapoint; the curve perfectly goes through it, but will hardly generalize to other values. A model trained solely on a single data point essentially learns a delta function, a sharp spike in response at that precise value and zero everywhere else, if the model permits it. The model, rather than learning a relationship between features and the target variable, simply memorizes the training input and output.

This is not a complete disaster. Certain specialized techniques mitigate this, although these still need multiple datapoints, which is also part of the argument, and won’t be shown in the code snippets. A good example is regularisation, which includes techniques such as adding a penalty for large weights in linear regression, making the model less prone to follow the noise of single data points. However, a single data point would still present problems, even if the problem is only partially mitigated. Other techniques involve early stopping, which essentially monitors a held-out validation set to determine if it is getting worse or has stopped improving. Finally, more advanced models, such as decision trees (and similar models), can be adjusted to require a minimum of data points in order to create splits, preventing them from overfitting to single points. These techniques help mitigate overfitting to a degree but fail when confronted with a model being trained solely on a single data point.

The problem lies in two primary areas: the data scarcity and the model flexibility. If a complex model is given only a single training instance, there is little to constrain its flexibility, pushing it to overfit. To address this, one could consider techniques that effectively augment the available data or methods that reduce the model’s complexity, but neither resolves the situation of using *only* one datapoint.

Let's illustrate with some code examples. These are not fully working snippets, but demonstrations of the concepts.

**Example 1: Linear Regression**

Consider a linear regression model trained using gradient descent. In a typical scenario, the error function measures the model’s performance across *multiple* data points. Gradient descent updates the model’s parameters to reduce this aggregate error. However, if you use only a single data point, say (x=2, y=5), the update will directly focus on making the model fit this point, with no consideration of generalizability to other points.

```python
import numpy as np

def compute_error(w, b, x, y):
  y_hat = w * x + b
  return (y_hat - y)**2 / 2

def gradient_descent(w, b, x, y, learning_rate, iterations):
  for i in range(iterations):
     y_hat = w*x +b
     dw = (y_hat-y) * x #derivative of the squared error function with respect to the weight
     db = y_hat-y #derivative of the squared error function with respect to the bias
     w -= learning_rate*dw
     b -= learning_rate*db
  return w, b

# Single datapoint
x = 2
y = 5

#Initialize the weight and the bias
w = 0
b = 0
learning_rate = 0.01
iterations = 1000

# Gradient descent algorithm
w, b = gradient_descent(w,b,x,y,learning_rate,iterations)
print(f"Final Weight: {w}, Final Bias: {b}")
# After running, the model will output a set of parameters where the linear equation goes through the single datapoint (2,5)
# In this case, if the user were to try and predict the result of 1, it would fail.
```
This code demonstrates that gradient descent applied to a single data point will result in a model where, if x = 2, then the result is 5, no matter what the dataset is. This code shows how the model learns *exactly* the single data point, without generalising.

**Example 2: A Neural Network**

Here, the single data point will be used to train a neural network, further showing the memorization of the datapoint.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward_pass(inputs, weights, bias):
    output = np.dot(inputs, weights) + bias
    return sigmoid(output)

def compute_error(y_hat, y):
  return (y_hat - y)**2 / 2

def backward_pass(inputs, weights, bias, y_hat, y):
  dcost_dyhat = y_hat - y
  dyhat_dz = sigmoid_derivative(np.dot(inputs,weights)+bias)
  dz_dw = inputs
  dz_db = 1

  dw = dcost_dyhat * dyhat_dz * dz_dw
  db = dcost_dyhat * dyhat_dz * dz_db

  return dw, db

def gradient_descent(inputs, weights, bias, y, learning_rate, iterations):
  for i in range(iterations):
    y_hat = forward_pass(inputs, weights, bias)
    dw, db = backward_pass(inputs, weights, bias, y_hat, y)
    weights -= learning_rate * dw
    bias -= learning_rate * db
  return weights, bias


# Single datapoint
inputs = np.array([1,2])
y = 1
#Initialize weights and bias
weights = np.array([0.5, -0.5])
bias = 0.1
learning_rate = 0.1
iterations = 10000

# Gradient descent algorithm
weights, bias = gradient_descent(inputs, weights, bias, y, learning_rate, iterations)

y_hat = forward_pass(inputs, weights, bias)

print(f"Predicted Output: {y_hat}") #close to 1

# Similarly to the last snippet, we can see that the model has simply memorised the single data point, and will generate the output 1
# whenever a 1,2 is input.
```

The above two snippets demonstrate that even with gradient descent in a neural network, the model will still overfit on a single datapoint and fail to generalize.

**Example 3: Decision Trees**

Decision trees, whilst less susceptible to overfitting than neural networks, also overfit when trained on a single data point.

```python
class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value  # If leaf, output value; otherwise, split attribute
        self.left = left
        self.right = right
        
def build_tree(x, y):
    """Build a decision tree with a single split"""
    #This tree will just create a single branch.
    return Node(value=x[0], left=Node(value=y[0]), right=Node(value=y[0]))

def predict(tree, x):
  if tree is None:
    return None
  if tree.left is not None: # not a leaf
    return predict(tree.left,x) # the only possible prediction will be the left branch.
  else: #leaf
    return tree.value

x = np.array([2])
y = np.array([5])

tree = build_tree(x, y)

prediction = predict(tree, x)
print(f"Prediction for {x}: {prediction}")
# In this example, a decision tree will, with one data point, only ever give that same datapoint.
```
This decision tree will, given a single datapoint, generate a tree that, if a prediction is to be made, will only ever return the value of that singular datapoint.

**Conclusion**

As these examples demonstrate, a model trained *solely* on a single data point will not generalize. The model will merely memorize the input and corresponding output. No amount of clever tuning, regularization, or validation can overcome the fundamental issue of insufficient information. Techniques such as dropout or early stopping, which are often used to mitigate overfitting, rely on having a validation set that is different from the training set, and as such cannot solve the issue.

For further exploration of these topics, I recommend examining textbooks on statistical learning, specifically those covering bias-variance trade-off, regularisation techniques, and model evaluation. Additionally, research papers discussing the limitations of machine learning in low-data scenarios and synthetic data generation would provide a more thorough understanding. Study resources and tutorials on specific algorithms such as linear regression, decision trees, and neural networks will also prove valuable.
