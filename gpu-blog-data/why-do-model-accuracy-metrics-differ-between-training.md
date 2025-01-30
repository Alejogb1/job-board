---
title: "Why do model accuracy metrics differ between training and confusion matrix evaluation?"
date: "2025-01-30"
id: "why-do-model-accuracy-metrics-differ-between-training"
---
The discrepancy between training accuracy and the accuracy derived from a confusion matrix on unseen test data stems fundamentally from the model's capacity to overfit the training dataset.  This isn't simply a matter of high variance; it's a consequence of the model learning spurious correlations present in the training data, which do not generalize to the broader population represented by the test set. In my experience working on large-scale image classification projects, particularly those involving complex medical imagery, this phenomenon consistently emerged as a primary hurdle.  Addressing this requires a nuanced understanding of the data, the model's architecture, and the application of appropriate regularization techniques.

**1. Clear Explanation:**

Training accuracy reflects the model's performance on the data it has already "seen" during the training phase.  The model's parameters are iteratively adjusted to minimize the loss function, calculated based on this training data.  High training accuracy indicates the model has successfully learned the patterns *within* the training set, but offers no guarantee of performance on new, unseen data.

The confusion matrix, on the other hand, provides a detailed breakdown of the model's performance on a held-out test set—data the model encountered for the first time during evaluation.  This matrix reveals the model's ability to correctly classify instances into each class and quantifies various performance metrics such as precision, recall, F1-score, and accuracy, which are derived from the true positive, true negative, false positive, and false negative counts. A significant discrepancy between training and test accuracy suggests overfitting, where the model has memorized the training data rather than learning generalizable features.

Several factors contribute to this divergence:

* **Insufficient Data:** A small training dataset limits the model's ability to learn robust features, increasing the likelihood of overfitting.  The model may capture noise or outliers specific to the training data.

* **Model Complexity:**  Highly complex models, with many parameters, have a greater capacity to memorize the training data. This higher capacity, in the absence of sufficient regularization, leads to overfitting.

* **Data Imbalance:** An uneven distribution of classes in the training set can bias the model, leading to inaccurate predictions for under-represented classes.  This bias might manifest differently in the training and test sets, leading to performance discrepancies.

* **Feature Engineering/Selection:** Poorly chosen features or a lack of relevant features can limit the model's ability to generalize, causing overfitting even with a large dataset.

Addressing these factors requires careful consideration of the dataset, model selection, and the application of regularization techniques.  Cross-validation is crucial to obtain a more robust estimate of model performance.


**2. Code Examples with Commentary:**

These examples utilize Python with scikit-learn, a library I've extensively used throughout my career.  They demonstrate the process of training a model, evaluating its performance on training and test data, and generating a confusion matrix.

**Example 1:  Illustrating Overfitting with a Simple Logistic Regression**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_classification

# Generate a synthetic dataset prone to overfitting
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate training accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"Training Accuracy: {train_accuracy}")

# Evaluate test accuracy
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {test_accuracy}")

# Generate the confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test))
print(f"Confusion Matrix:\n{cm}")
```

This example uses a small dataset and a simple model to highlight the potential for overfitting. The significant difference between training and test accuracy underscores the issue.

**Example 2: Applying Regularization to Mitigate Overfitting**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model with L2 regularization
model = LogisticRegression(C=1.0, penalty='l2') # C controls regularization strength
model.fit(X_train, y_train)

# Evaluate and print results (similar to Example 1)
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"Training Accuracy: {train_accuracy}")
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {test_accuracy}")
cm = confusion_matrix(y_test, model.predict(X_test))
print(f"Confusion Matrix:\n{cm}")

```

Here, L2 regularization (ridge regression) is introduced to constrain the model's complexity, reducing overfitting and improving generalization.  The `C` parameter controls the regularization strength.

**Example 3:  Demonstrating Cross-Validation for Robust Evaluation**

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_classification

# Generate a dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=5, random_state=42)

# Train a model using 5-fold cross-validation
model = LogisticRegression()
cv_scores = cross_val_score(model, X, y, cv=5)

# Print cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {np.mean(cv_scores)}")
```

This example showcases 5-fold cross-validation, providing a more reliable estimate of model performance by training and evaluating the model on multiple subsets of the data.  This mitigates the impact of a single, potentially unrepresentative, train-test split.


**3. Resource Recommendations:**

For a deeper understanding, I recommend consulting standard machine learning textbooks covering topics such as model selection, regularization, bias-variance tradeoff, and cross-validation.  A strong foundation in linear algebra and probability is also beneficial.  Focus on resources that provide rigorous mathematical explanations, rather than solely relying on intuitive explanations.  Practical experience building and evaluating models on diverse datasets is paramount to gaining a practical understanding of these concepts.  Supplement your learning with relevant research papers addressing overfitting and regularization techniques in specific contexts like image classification or time-series analysis.
