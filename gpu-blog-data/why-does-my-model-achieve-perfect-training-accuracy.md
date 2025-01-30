---
title: "Why does my model achieve perfect training accuracy but produce incorrect predictions?"
date: "2025-01-30"
id: "why-does-my-model-achieve-perfect-training-accuracy"
---
The phenomenon of a machine learning model achieving 100% training accuracy yet failing to generalize to unseen data, resulting in poor prediction accuracy, points to a critical issue: overfitting.  This isn't simply a matter of insufficient data; it's a structural problem within the model's learning process, where the model memorizes the training set instead of learning the underlying patterns.  My experience with this, spanning several years of developing and deploying predictive models for financial risk assessment, highlighted the subtle ways overfitting can manifest.


Overfitting occurs when a model's complexity exceeds the information capacity of the training data.  Essentially, the model learns the noise within the data, rather than the signal – the actual relationships between features and target variables.  This becomes particularly problematic with high-dimensional data or models with excessive parameters, allowing them to fit even the random fluctuations in the training set.  The consequence is excellent performance on familiar data, but disastrous performance on novel data the model hasn't encountered.

The manifestation of this problem can be nuanced.  One frequent symptom is a significant discrepancy between training and validation (or test) accuracy.  While training accuracy sits at 100%, validation accuracy remains stubbornly low, revealing the model's inability to generalize.  Another indicator is the model's sensitivity to small variations in the training data; slight changes can dramatically alter its predictions.  This instability further underscores the model’s overreliance on specific features or data points within the training set.

Addressing overfitting requires a multi-faceted approach, targeting both the model architecture and the training process.  Below, I’ll present three common strategies, illustrated with code examples.  Remember, the optimal solution depends heavily on the specific dataset and model employed.

**1. Regularization Techniques:**

Regularization adds a penalty to the model's loss function, discouraging excessively large weights.  This constraint prevents the model from becoming overly complex and fitting the noise.  L1 and L2 regularization are common methods.  L1 (LASSO) adds the absolute value of the weights to the loss function, promoting sparsity (many weights becoming zero), whereas L2 (Ridge) adds the square of the weights, leading to smaller but non-zero weights.

```python
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your own)
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# L2 Regularization (Ridge)
ridge = Ridge(alpha=1.0) # alpha controls the strength of regularization
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
y_pred_ridge = np.round(y_pred_ridge).astype(int) #convert probabilities to binary classifications
accuracy_ridge = accuracy_score(y_test, y_pred_ridge)
print(f"Ridge Regression Accuracy: {accuracy_ridge}")

#L1 Regularization (LASSO)
lasso = Lasso(alpha=1.0) # alpha controls regularization strength
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
y_pred_lasso = np.round(y_pred_lasso).astype(int)
accuracy_lasso = accuracy_score(y_test, y_pred_lasso)
print(f"LASSO Regression Accuracy: {accuracy_lasso}")
```

This code demonstrates the application of L1 and L2 regularization within a simple linear model.  The `alpha` parameter controls the regularization strength; higher values increase the penalty.  Experimentation is crucial to find the optimal value. Note that for classification problems, a sigmoid or softmax function would be necessary to map the model's output to probabilities, then threshold these probabilities to generate binary or multi-class predictions.  This example uses rounding for simplicity.


**2. Early Stopping:**

Early stopping monitors the model's performance on a validation set during training.  The training process is halted when the validation accuracy starts to decrease, preventing further overfitting.  This requires careful monitoring of both training and validation metrics.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple sequential model
model = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # For binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Use Keras callbacks for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy after early stopping: {accuracy}")
```

This Keras example incorporates early stopping using a callback.  The `patience` parameter defines how many epochs the model can continue training with deteriorating validation loss before stopping. `restore_best_weights` ensures the model loads the weights from the epoch with the best validation performance.


**3. Data Augmentation:**

Data augmentation artificially increases the size of the training dataset by creating modified versions of existing data points.  This reduces the model's reliance on specific instances and improves its ability to generalize.  For image data, common augmentations include rotations, flips, and crops. For tabular data, techniques like adding noise or creating synthetic samples might be used.  However, the success of data augmentation is highly context-specific.

```python
# Example using scikit-learn for synthetic sample generation (Illustrative, often requires domain-specific methods)
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE #For example, handling imbalanced datasets

X, y = make_classification(n_samples=100, n_features=10, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#Further model training using X_resampled and y_resampled
# ...
```

This code snippet demonstrates the use of SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance, a scenario that can sometimes contribute to overfitting. SMOTE generates synthetic samples for the minority class, helping balance the dataset and potentially mitigating overfitting issues resulting from imbalanced data.  This is only one example; other methods like bootstrapping or data transformation techniques can be tailored to specific datasets.

In summary, achieving perfect training accuracy while exhibiting poor prediction accuracy is a strong indication of overfitting.  Addressing this necessitates carefully considering model complexity, regularization techniques, early stopping criteria, and strategies for data augmentation.  Remember to thoroughly evaluate model performance using appropriate metrics on independent validation and test sets.  Consult resources on model selection, regularization methods, and hyperparameter tuning to refine your approach.  The effectiveness of these techniques is highly dependent on your specific dataset and the chosen model architecture, necessitating iterative experimentation and rigorous validation.
