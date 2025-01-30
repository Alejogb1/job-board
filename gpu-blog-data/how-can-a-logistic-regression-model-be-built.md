---
title: "How can a logistic regression model be built and trained with two input features?"
date: "2025-01-30"
id: "how-can-a-logistic-regression-model-be-built"
---
Logistic regression, despite its simplicity, remains a powerful tool for binary classification.  My experience building and deploying models for fraud detection highlighted the effectiveness of logistic regression, even with limited feature sets.  Crucially, understanding the underlying mathematical assumptions and the impact of feature scaling is paramount for achieving optimal performance.  Let's examine the construction and training of a logistic regression model employing two input features.


1. **Clear Explanation:**

The core of logistic regression lies in modeling the probability of a binary outcome (typically represented as 0 or 1) as a function of predictor variables.  With two input features, x₁ and x₂, we model the probability  P(y=1 | x₁, x₂) using a sigmoid function:

P(y=1 | x₁, x₂) = 1 / (1 + exp(-(β₀ + β₁x₁ + β₂x₂)))

Where:

* y represents the binary outcome (0 or 1).
* x₁ and x₂ are the two input features.
* β₀ is the intercept.
* β₁, β₂ are the coefficients representing the effect of x₁ and x₂, respectively.

The sigmoid function maps any real number to a value between 0 and 1, providing a probability estimate. The coefficients (β₀, β₁, β₂) are estimated during the training process, aiming to maximize the likelihood of observing the training data.  This is typically achieved using maximum likelihood estimation (MLE) or, more commonly in practice, gradient descent-based optimization methods.  The process involves iteratively adjusting the coefficients to minimize a cost function, often the log-loss function.  Regularization techniques, such as L1 or L2 regularization, can be incorporated to prevent overfitting, particularly relevant when dealing with limited datasets or highly correlated features.  Feature scaling, standardizing or normalizing the input features, is a crucial preprocessing step to improve the convergence speed and stability of the optimization algorithm.  Failure to scale features can lead to slower convergence and suboptimal model performance.


2. **Code Examples with Commentary:**

These examples illustrate building and training a logistic regression model with two input features using Python's `scikit-learn` library.  I've chosen this library because of its widespread adoption and clear API.


**Example 1: Basic Logistic Regression**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample data (replace with your actual data)
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [1, 1], [2, 1], [3, 2], [4, 2]])
y = np.array([0, 1, 0, 1, 0, 0, 1, 1])

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

This example demonstrates a basic implementation.  Note the use of `StandardScaler` for feature scaling—a critical step I've learned to prioritize through repeated model development cycles.  The `random_state` ensures reproducibility.  The output includes model coefficients and the intercept, providing insights into the feature importance.


**Example 2: Logistic Regression with L2 Regularization**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample data (same as before)
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [1, 1], [2, 1], [3, 2], [4, 2]])
y = np.array([0, 1, 0, 1, 0, 0, 1, 1])

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with L2 regularization
model = LogisticRegression(penalty='l2', C=1.0) # C controls regularization strength
model.fit(X_train, y_train)

# Predictions and coefficients
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

This example incorporates L2 regularization (`penalty='l2'`).  The `C` parameter controls the regularization strength; a smaller `C` indicates stronger regularization.  This helps to prevent overfitting, especially important with small datasets or highly correlated features, a lesson learned from past projects involving customer churn prediction.


**Example 3:  Handling Imbalanced Data**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Imbalanced sample data
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [1, 1], [2, 1], [3, 2], [4, 2], [1,3], [2,4], [3,5]])
y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0])

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

This example tackles a common issue: imbalanced datasets.  SMOTE (Synthetic Minority Over-sampling Technique) is used to oversample the minority class, creating synthetic data points to balance the class distribution.  This is crucial for avoiding biased models, a pitfall I encountered during a project involving credit risk assessment.


3. **Resource Recommendations:**

For a deeper understanding of logistic regression, I recommend consulting textbooks on statistical modeling and machine learning.  Specifically, texts focusing on generalized linear models and their applications will provide a comprehensive foundation.  Furthermore, dedicated machine learning resources that cover model evaluation metrics, regularization techniques, and handling imbalanced datasets are invaluable.  Finally, exploring the documentation of relevant libraries like `scikit-learn` will prove beneficial in practical implementation.
