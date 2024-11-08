---
title: "Tweaking Polynomial Regression: How to Use Regularization for Better Fit?"
date: '2024-11-08'
id: 'tweaking-polynomial-regression-how-to-use-regularization-for-better-fit'
---

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming you have your features (X) and target (y) data loaded

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a range of lambda values to experiment with
lambda_values = np.logspace(-3, 3, 10)

# Initialize lists to store the results
train_errors = []
val_errors = []

# Train models with different lambda values and record errors
for l in lambda_values:
    model = Ridge(alpha=l)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_error = mean_squared_error(y_train, y_train_pred, squared=False)
    train_errors.append(train_error)

    y_val_pred = model.predict(X_val)
    val_error = mean_squared_error(y_val, y_val_pred, squared=False)
    val_errors.append(val_error)

# Plot the training and validation errors vs. log(lambda)
import matplotlib.pyplot as plt
plt.plot(np.log(lambda_values), train_errors, label="Training Error")
plt.plot(np.log(lambda_values), val_errors, label="Validation Error")
plt.xlabel("log(lambda)")
plt.ylabel("Root Mean Squared Error")
plt.legend()
plt.show()

# Select the lambda value that gives the best performance on the validation set
best_lambda = lambda_values[np.argmin(val_errors)]

# Retrain the model with the optimal lambda value
model = Ridge(alpha=best_lambda)
model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_test_pred = model.predict(X_test)
test_error = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"Test RMSE: {test_error}")
```
