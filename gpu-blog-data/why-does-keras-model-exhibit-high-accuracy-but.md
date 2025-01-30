---
title: "Why does Keras model exhibit high accuracy but poor prediction?"
date: "2025-01-30"
id: "why-does-keras-model-exhibit-high-accuracy-but"
---
The discrepancy between high training accuracy and poor prediction performance in a Keras model frequently stems from a mismatch between the training data distribution and the distribution of unseen data used for prediction.  This isn't simply a matter of insufficient data; it's a problem of data representativeness and the model's capacity to generalize effectively.  Over the course of my decade working on large-scale machine learning projects, I've encountered this issue numerous times, leading to the realization that robust model evaluation goes far beyond a simple training-validation split.

**1. Clear Explanation of the Problem**

High training accuracy indicates the model has learned the training data well – perhaps too well.  This often manifests as overfitting, where the model memorizes the training set's nuances instead of learning the underlying patterns that generalize to new, unseen data. Several factors contribute to this:

* **Data Imbalance:**  A skewed class distribution in the training data can lead to a model that performs exceptionally well on the majority class but poorly on the minority class.  High overall accuracy might mask this poor performance on a crucial subset.  This is particularly relevant in applications like fraud detection or medical diagnosis.

* **Data Leakage:**  Features in the training data might inadvertently contain information that shouldn't be present during prediction.  This could be due to accidental inclusion of target variables or correlated information.  The model learns these spurious correlations, yielding high training accuracy but failing to generalize when this leakage is absent in the prediction data.

* **Insufficient Regularization:**  Without sufficient regularization techniques (e.g., L1 or L2 regularization, dropout, early stopping), the model's complexity can increase excessively, enabling it to overfit the training data.  The high capacity allows the model to capture noise and irrelevant details, harming its generalization capabilities.

* **Suboptimal Hyperparameter Tuning:**  Inappropriate hyperparameter settings, such as learning rate, number of layers, or number of neurons, can hinder generalization.  An overly complex model with insufficient regularization is particularly prone to overfitting.

* **Non-representative Training Data:**  The training data might not accurately reflect the distribution of the data the model will eventually encounter during prediction.  This might stem from sampling bias, outdated data, or a fundamental mismatch between the data generation process during training and deployment.


**2. Code Examples with Commentary**

Here are three illustrative scenarios demonstrating potential causes and remedies:


**Example 1: Data Imbalance and Class Weighting**

```python
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Generate imbalanced data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, weights=[0.8, 0.2], random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

# Build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model with class weights
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weights)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
```

This example uses `class_weight` to address data imbalance.  Without class weighting, the model might achieve high overall accuracy by focusing primarily on the majority class.  The `compute_sample_weight` function automatically adjusts weights to balance the classes.

**Example 2: Data Leakage and Feature Selection**

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Simulate data leakage - feature X1 is strongly correlated with the target (y), but shouldn't be
X = np.random.rand(1000, 5)
y = X[:, 0] + np.random.normal(0, 0.1, 1000)
X[:,1] = y + np.random.normal(0,0.2,1000) #Introducing leakage

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

#Feature selection to address leakage (simplified illustration)
X_train_selected = X_train[:, 2:] #Remove correlated features
X_test_selected = X_test[:, 2:]

model_selected = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])

model_selected.compile(optimizer='adam', loss='mse')
model_selected.fit(X_train_selected, y_train, epochs=100, verbose=0)

loss_selected, accuracy_selected = model_selected.evaluate(X_test_selected, y_test)
print(f'Test loss with selected features: {loss_selected}')

```

This example demonstrates how a feature strongly correlated with the target (a form of leakage) can artificially inflate training accuracy.  Careful feature engineering and selection, possibly using techniques like feature importance from tree-based models, can mitigate this issue.


**Example 3: Overfitting and Regularization**

```python
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Overfit model (no regularization)
model_overfit = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1)
])

model_overfit.compile(optimizer='adam', loss='mse')
model_overfit.fit(X_train, y_train, epochs=100, verbose=0)
loss_overfit = model_overfit.evaluate(X_test, y_test, verbose=0)

#Regularized model with dropout
model_regularized = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
  tf.keras.layers.Dropout(0.5), #Dropout for regularization
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1)
])

model_regularized.compile(optimizer='adam', loss='mse')
model_regularized.fit(X_train, y_train, epochs=100, verbose=0)
loss_regularized = model_regularized.evaluate(X_test, y_test, verbose=0)

print(f"Overfit model loss: {loss_overfit}")
print(f"Regularized model loss: {loss_regularized}")
```

This highlights the impact of regularization (here, dropout) in preventing overfitting.  The overfit model, with many neurons and no regularization, likely achieves high training accuracy but poor generalization.  The regularized model, using dropout to randomly deactivate neurons during training,  demonstrates improved generalization performance.


**3. Resource Recommendations**

For a deeper understanding of overfitting and regularization techniques, I recommend studying comprehensive machine learning textbooks focusing on model selection and evaluation.  Furthermore, delve into research papers on ensemble methods, which can often mitigate generalization issues by combining predictions from multiple models.  Exploring techniques for data preprocessing and feature engineering is also crucial.  Finally, gaining familiarity with different types of cross-validation methods will enhance your model evaluation strategy.
