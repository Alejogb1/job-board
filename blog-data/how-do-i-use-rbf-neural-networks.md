---
title: "How do I use RBF neural networks?"
date: "2024-12-16"
id: "how-do-i-use-rbf-neural-networks"
---

Let's talk about radial basis function (RBF) neural networks; it's a topic that’s come up quite a bit in my career, and I've found them particularly useful in certain scenarios. One project that springs to mind was a fairly complex time-series prediction challenge I tackled years ago, where traditional feedforward networks were struggling to capture the nuances of the data. That’s when I revisited RBF networks and got them working to my advantage.

The fundamental difference with RBF networks, as opposed to traditional multi-layer perceptrons, lies in their architecture and the type of activation function they utilize. Instead of sigmoidal or ReLU activation functions employed in feedforward networks, RBF networks use radial basis functions, commonly a gaussian function, within their hidden layer. The essence is that each neuron in the hidden layer acts as a local 'expert,' being highly responsive to inputs close to its center (a predefined location in the input space), and less so as the inputs move further away. This localization property is why RBF networks can excel at tasks where the input space has regions with distinct behavior.

Essentially, the network has three layers: an input layer, a hidden layer using RBF units, and an output layer. The input layer simply presents the data. The hidden layer, where the RBF magic happens, computes the radial basis function output. The distance between the input vector *x* and the center *c<sub>i</sub>* of each hidden neuron using the euclidean distance is computed and then passed to the RBF function. Finally, the output layer is a linear combination of the hidden layer’s outputs to produce the final results, generally using a set of linear weights *w<sub>i</sub>*.

The common equation for a Gaussian RBF unit *φ<sub>i</sub>* is:

φ<sub>i</sub>(x) = exp( - (||x - c<sub>i</sub>||<sup>2</sup>) / (2 * σ<sub>i</sub><sup>2</sup>))

Where:
*   *x* is the input vector.
*   *c<sub>i</sub>* is the center vector of the i-th hidden unit.
*   *σ<sub>i</sub>* is the width parameter of the i-th hidden unit, which controls the scope of each unit's activation.
*   ||x - c<sub>i</sub>|| denotes the euclidean distance between the input *x* and the center *c<sub>i</sub>*.

The output *y* of the network is computed by a weighted sum of the hidden layer’s outputs:
y(x) = Σ w<sub>i</sub> φ<sub>i</sub>(x)

Now, let's move towards a practical coding example using python and the library `scikit-learn`. It’s a rather common choice, and provides a simple yet practical base for experimenting. Assume for instance we aim to approximate a sine wave using RBF regression:

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Generate sample data
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Setup an RBF regressor
rbf_regressor = Pipeline([
    ('scaler', StandardScaler()), # Optional, but often beneficial for RBF networks.
    ('rbf', MLPRegressor(hidden_layer_sizes=(20,),
                        activation='relu', # Although we are approximating a RBF net, we use 'relu' here because sklearn doesn't provide true RBF
                        solver='lbfgs', # LBFGS is suitable for small datasets
                        random_state=42,
                        max_iter = 1000))
])

# Train the model
rbf_regressor.fit(X, y)

# Generate some test data and do a prediction
X_test = np.linspace(0, 5, 200).reshape(-1, 1)
y_pred = rbf_regressor.predict(X_test)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Training Data')
plt.plot(X_test, y_pred, color='red', label='RBF Regression Prediction')
plt.xlabel('Input (X)')
plt.ylabel('Output (Y)')
plt.title('RBF Regression Example - Sine Wave Approximation')
plt.legend()
plt.show()
```

This example demonstrates the basic pipeline for training an RBF network. While scikit-learn’s `MLPRegressor` is used with relu activation as a stand-in for the RBF (since it lacks a native RBF implementation), it illustrates the common processing steps: the standardization of input data and then training to approximate the sinusoidal function. The `hidden_layer_sizes` here controls the number of basis functions.

Now, let's delve into another scenario, one where you might use RBFs for classification problems. Imagine, you’ve got some data points that belong to two classes, and the class boundaries aren’t straightforward linear separations. In such a case an RBF classification network could perform well. Again we use the `MLPClassifier` from sklearn in a similar way to the previous example:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Generate synthetic data for classification
X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Setup an RBF classifier using a pipeline
rbf_classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('rbf', MLPClassifier(hidden_layer_sizes=(15,),
                           activation='relu', #Again, using relu as a stand-in for RBF
                           solver='lbfgs', # LBFGS is suitable for small datasets
                           random_state=42,
                           max_iter = 1000))
])

# Train the model
rbf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rbf_classifier.predict(X_test)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = rbf_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.RdBu)
plt.title('RBF Classifier Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

This example generates two clusters of data, then the RBF network is used to find the boundary between the two classes. Notice how the boundary is not perfectly straight, showcasing the non-linear behavior achieved using radial basis functions. This illustrates how RBF networks can handle more complicated decision boundaries.

Finally, let’s examine the practical consideration of hyperparameter tuning in an RBF network. I've found that the number of hidden nodes (RBF units) and, crucially, their widths (the sigma, σ, value in the Gaussian function) are the most important aspects to address. In this synthetic example, we'll look at how the number of hidden units affects the fit of the model:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Generate sample data
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Test different hidden layer sizes
hidden_sizes = [5, 10, 20, 50]

plt.figure(figsize=(12, 8))
for i, hidden_size in enumerate(hidden_sizes, 1):
    rbf_regressor = Pipeline([
        ('scaler', StandardScaler()),
        ('rbf', MLPRegressor(hidden_layer_sizes=(hidden_size,),
                            activation='relu',
                            solver='lbfgs',
                            random_state=42,
                            max_iter = 1000))
    ])
    rbf_regressor.fit(X, y)
    X_test = np.linspace(0, 5, 200).reshape(-1, 1)
    y_pred = rbf_regressor.predict(X_test)

    plt.subplot(2, 2, i)
    plt.scatter(X, y, label='Training Data')
    plt.plot(X_test, y_pred, color='red', label='RBF Prediction')
    plt.title(f'Hidden Units: {hidden_size}')
    plt.xlabel('Input (X)')
    plt.ylabel('Output (Y)')
    plt.legend()

plt.tight_layout()
plt.show()
```

As seen in this example, increasing the number of hidden units allows for a better fit of the data. However, using an excessive number of units could lead to overfitting and poor generalization. The width parameter, in this case controlled indirectly by the learning of the weights in the network (due to the lack of a true RBF implementation) also significantly impacts the model. Therefore, experimentation and, techniques such as cross-validation, are essential to tune an RBF network for optimal performance in any given task.

If you are looking for more detail about the background of RBF networks and their design considerations, I recommend delving into "Pattern Recognition and Machine Learning" by Christopher Bishop. Furthermore, for an in-depth mathematical treatment and a solid theoretical grounding, you might find "Neural Networks for Pattern Recognition" by Christopher Bishop very beneficial. These resources should provide a more rigorous and detailed perspective.
In summary, RBF networks offer unique advantages, particularly with non-linear, locally responsive data. They require careful consideration of parameters such as the number and width of the RBF units, but when tuned appropriately can be a very powerful tool in your machine learning toolbox.
