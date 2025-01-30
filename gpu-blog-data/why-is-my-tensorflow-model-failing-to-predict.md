---
title: "Why is my TensorFlow model failing to predict a quadratic function?"
date: "2025-01-30"
id: "why-is-my-tensorflow-model-failing-to-predict"
---
The inability of a TensorFlow model to accurately predict a quadratic function often stems from insufficient model complexity, inadequate training data, or inappropriate hyperparameter settings.  My experience debugging similar issues across numerous projects, particularly in time-series forecasting and polynomial regression tasks, points to these root causes.  Let's systematically examine these possibilities and illustrate them with concrete code examples.

**1. Insufficient Model Complexity:**  A simple linear model, regardless of its training, cannot accurately capture the curvature inherent in a quadratic function.  You need a model architecture capable of representing non-linear relationships.  A single-layer perceptron, for instance, would be insufficient; a multi-layer perceptron (MLP) with at least one hidden layer should be employed. Even better, tailored approaches like polynomial regression could be considered for this specific problem, although they can be prone to overfitting.

**2. Inadequate Training Data:** Insufficient data, particularly if it doesn't adequately span the range of the quadratic function's input, prevents the model from learning the underlying relationship.  The data needs to be sufficiently diverse to encompass the function's curvature; otherwise, the model will learn an approximation that might be accurate within the limited input range of the training data but performs poorly on extrapolation. The absence of data points in crucial areas (e.g., the vertex region of a parabola) severely restricts the model's ability to generalise.


**3. Inappropriate Hyperparameter Settings:**  The choices of optimizer, learning rate, batch size, and number of epochs significantly impact model performance. An overly aggressive learning rate can cause oscillations and prevent convergence, while an insufficient number of epochs might lead to underfitting.  Similarly, a too small batch size increases variance in the gradient estimates, hindering effective training.


Let's illustrate these points with TensorFlow/Keras examples.  I’ll assume you're familiar with basic TensorFlow/Keras concepts.  If not, resources on the Keras documentation and introductory TensorFlow tutorials are highly recommended.

**Code Example 1: Insufficient Model Complexity (Linear Model)**

```python
import numpy as np
import tensorflow as tf

# Generate quadratic data
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2*X**2 + 3*X + 1 + np.random.normal(0, 0.5, size=(100, 1))

# Linear model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mse')
model.fit(X, y, epochs=1000)

# Prediction and evaluation (poor performance expected)
y_pred = model.predict(X)
# Evaluate using metrics like Mean Squared Error (MSE)
```

This example demonstrates the limitation of a simple linear model.  It's fundamentally incapable of capturing the quadratic relationship, resulting in high prediction errors regardless of training.

**Code Example 2: Adequate Model Complexity (MLP)**

```python
import numpy as np
import tensorflow as tf

# Generate quadratic data (same as Example 1)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2*X**2 + 3*X + 1 + np.random.normal(0, 0.5, size=(100, 1))

# Multi-layer perceptron (MLP) model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500)

# Prediction and evaluation (improved performance expected)
y_pred = model.predict(X)
# Evaluate using MSE
```

This example uses an MLP with a hidden layer employing a ReLU activation function. This non-linear activation allows the model to learn the quadratic relationship. The Adam optimizer, generally more robust than SGD, is employed.


**Code Example 3: Addressing Inadequate Data (Data Augmentation)**

```python
import numpy as np
import tensorflow as tf

# Generate quadratic data (original data)
X_original = np.linspace(-5, 5, 100).reshape(-1, 1)
y_original = 2*X_original**2 + 3*X_original + 1 + np.random.normal(0, 0.5, size=(100, 1))


#Augment Data - Add more samples around the vertex
X_additional = np.linspace(-1, 1, 50).reshape(-1, 1) #Adding more data around vertex
y_additional = 2*X_additional**2 + 3*X_additional + 1 + np.random.normal(0, 0.2, size=(50,1)) #Lower noise

X = np.concatenate((X_original, X_additional))
y = np.concatenate((y_original, y_additional))

# MLP Model (same as Example 2)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500)

#Prediction and evaluation
y_pred = model.predict(X)
# Evaluate using MSE
```

This example demonstrates how augmenting the dataset with additional data points, focusing on the vertex region of the parabola (where data might be sparse), can improve model performance.  Careful consideration of the noise level in the augmentation process is crucial to avoid introducing artefacts that could negatively impact generalisation.


In summary, successfully training a TensorFlow model to predict a quadratic function requires careful consideration of the model's architecture, the quality and quantity of training data, and the selection of appropriate hyperparameters.  Addressing these aspects systematically, as illustrated in the examples above, should significantly improve predictive accuracy.  Thorough investigation into the data distribution and careful experimentation with different architectures and hyperparameter settings are essential for optimal model performance.  Remember to always validate your model on a separate test set to assess its generalization ability.
