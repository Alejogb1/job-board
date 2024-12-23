---
title: "How do I create an SVM scatter plot in matplotlib?"
date: "2024-12-23"
id: "how-do-i-create-an-svm-scatter-plot-in-matplotlib"
---

,  I remember vividly one particularly frustrating project back in 2016; I was knee-deep in analyzing sensor data from a rather temperamental robotic arm. Part of the challenge was visualizing the classification boundaries defined by a support vector machine (svm). Matplotlib, being my trusty visualization tool, was the logical choice, but rendering an informative svm scatter plot isn’t always straightforward. It involves understanding how SVMs operate and then translating that into something visually comprehensible. I found out quickly that plotting the decision boundary and support vectors in conjunction with the data points is crucial for proper analysis.

Creating an effective SVM scatter plot primarily involves these steps: First, we need to train an svm model using your chosen data. Second, we generate a grid of points across the feature space. This grid will allow us to plot the decision boundary. Third, we predict the class labels for each point in the grid, using the trained svm model. Fourth, we plot these points colored by their predicted class and finally, we overplot the actual data points and support vectors for context. It’s the combination of these layers that generates the comprehensive and informative visualization we are aiming for.

Before diving into the code, it's worth noting the key components of the plot. The background color plot represents the decision regions, visually separating the areas where the svm predicts different classes. The actual data points are overlaid to show how well the model fits the data. Finally, the support vectors are the key data points that define the decision boundary; these are crucial to understanding the model’s functionality.

Here's how we can achieve this in practice. We’ll use scikit-learn for the svm and matplotlib for plotting. I've seen several attempts over the years, and here are three illustrative examples:

**Snippet 1: Basic 2D SVM Scatter Plot**

This first snippet is a fundamental illustration and assumes you are working with two features. This is the most common scenario for initial understanding.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Generate some sample data
X, y = make_blobs(n_samples=100, centers=2, random_state=6)

# Train the SVM
clf = svm.SVC(kernel='linear', C=1)  # Linear kernel for simplicity
clf.fit(X, y)

# Create a grid of points
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class labels for the grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

# Plot the support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', marker='o')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.show()
```

This snippet starts with the generation of sample blobs using scikit-learn. We then instantiate and fit an `SVC` with a linear kernel. Critically, we generate a meshgrid covering the region of our data. Predictions are then made on this meshgrid, which is reshaped and plotted using `plt.contourf`. This creates the colored regions denoting the svm’s classification areas. The data points are overlaid using a scatter plot, and finally the support vectors are highlighted using distinct circular markers.

**Snippet 2: SVM with RBF Kernel and Customization**

The next example shifts from the simple linear kernel to a more commonly used radial basis function (rbf) kernel, and it also adds some visual customization to further enhance its clarity.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons

# Generate moon-shaped data
X, y = make_moons(n_samples=150, noise=0.1, random_state=42)

# Train the SVM with an RBF kernel
clf = svm.SVC(kernel='rbf', gamma=1, C=10)
clf.fit(X, y)

# Create a finer grid for a smoother boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict and reshape
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Customize plot colors
cmap_light = plt.cm.get_cmap('RdBu', 2)
cmap_bold = plt.cm.get_cmap('RdBu', 2)

# Plot decision boundary with custom colors
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Plot data points with custom colors
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=40)


# Highlight support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='k', marker='o', linewidth=2)


plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with RBF Kernel')
plt.show()
```

Here we’re using `make_moons` to create a more challenging classification dataset that requires a non-linear decision boundary. We use the rbf kernel, and notice how the `gamma` and `c` parameters significantly alter the boundary’s shape.  The finer grid is created by decreasing the step size in `np.arange` creating smoother boundaries, and I've introduced a custom color map for the plot for clarity. This illustrates how adjusting parameters can impact visual interpretation, which is crucial.

**Snippet 3: Adding a Legend**

This final snippet incorporates a legend to clarify the meaning of each element in the plot, which is often beneficial when presenting these visualizations, especially when you’re dealing with a larger group of audience.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

# Create dataset for classification
X, y = make_classification(n_samples=120, n_features=2, n_informative=2, n_redundant=0, random_state=10)

# Train the SVM
clf = svm.SVC(kernel='poly', degree=3, C=2)
clf.fit(X, y)

# Create grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot Decision Regions
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

# Plot data points, with custom labels and legend
scatter_data = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k', label='Data Points')

# Highlight support vectors, with custom labels and legend
scatter_support = plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with Legend')
plt.legend(handles=[scatter_data, scatter_support])
plt.show()
```

Here we use a polynomial kernel with a degree of three, demonstrating another type of boundary. Notice that we’ve added a legend using the `label` parameter in the scatter function and using `plt.legend`. This makes the plot more easily understandable. This is a crucial part of creating informative figures that are easy for other to interpret.

For further reading and deeper understanding, I’d recommend "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman, for a foundational understanding of SVMs. Also, "Pattern Recognition and Machine Learning" by Bishop is invaluable for a comprehensive treatment of the topic. For implementation specifics with scikit-learn, definitely refer to the official documentation; it contains examples and details that would answer any specific questions you might have while you’re working on your own implementations.

The process of generating a clear and informative SVM scatter plot is iterative. Start with the basic structure, understand the data you're working with, and progressively refine your visualization. I hope this breakdown helps you in your endeavors.
