---
title: "Can F1 score be a valid target for cross-validation hyperparameter optimization?"
date: "2025-01-30"
id: "can-f1-score-be-a-valid-target-for"
---
The suitability of F1 score as a target metric for cross-validation hyperparameter optimization hinges on the specific characteristics of the problem and the desired trade-off between precision and recall. I’ve personally encountered situations where maximizing F1 score yielded significantly better model performance than optimizing for accuracy alone, particularly when dealing with imbalanced datasets. This experience solidified my understanding that a nuanced approach, rather than blindly maximizing any single metric, is crucial for effective model tuning.

Let’s break down why and when F1 score can be a valid choice, and its limitations. The core concept of F1 score is its calculation: the harmonic mean of precision and recall. Precision quantifies the proportion of predicted positives that were correct, while recall measures the proportion of actual positives that were correctly identified. Mathematically, F1 is given by:

F1 = 2 * (Precision * Recall) / (Precision + Recall)

This formulation is crucial because it penalizes models that excessively favor either precision or recall at the expense of the other. A model that predicts every instance as negative will achieve perfect precision (assuming no false positives) but terrible recall, thus a low F1. Conversely, a model predicting every instance as positive will achieve perfect recall but likely low precision. Maximizing F1 encourages a balance, which is often desirable when the cost of false positives and false negatives are relatively similar.

Now, when is F1 score a particularly compelling optimization target during cross-validation? The most common scenario is dealing with class imbalance, where one class vastly outnumbers the other. Accuracy, a seemingly straightforward metric, can be misleading in these cases. A model that simply predicts the majority class for every instance can achieve high accuracy, yet be useless in practice for detecting the minority class, which is often the class of interest. In fraud detection, for example, fraudulent transactions are far less common than legitimate ones, making accuracy an inadequate indicator of performance. F1 score, because of its sensitivity to both precision and recall, often provides a more realistic evaluation.

However, F1 isn't a universal solution. The importance of precision versus recall might vary depending on the specific application. In medical diagnoses, for instance, minimizing false negatives (high recall) might be paramount, even at the expense of some false positives (lower precision). In that instance, optimizing for F1 might not align with the core objective. Similarly, if the dataset is perfectly balanced, accuracy can be sufficient and F1's added complexity might not yield significantly better results, potentially increasing computation time unnecessarily in the hyperparameter optimization process. Furthermore, if a non-binary classification problem exists, using a weighted F1 approach may make more sense. I have personally encountered more value from focusing directly on relevant metrics, like recall or precision, within hyperparameter optimization when those specific trade-offs matter more.

To illustrate the practical application of F1 score, let's examine a few code examples, assuming a Python environment with libraries like `scikit-learn`.

**Example 1: Basic F1 Optimization in Cross-Validation**

This first example shows the use of F1 as the `scoring` parameter for `GridSearchCV`.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
import numpy as np

# Generate an imbalanced synthetic dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Define the hyperparameter grid
param_grid = {'C': np.logspace(-3, 3, 7)}

# Initialize the logistic regression model
logreg = LogisticRegression(solver='liblinear', random_state=42)

# Set up GridSearchCV using 'f1' as the scoring metric
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='f1')

# Run the grid search with cross-validation
grid_search.fit(X, y)

# Print the best hyperparameter and F1 score
print(f"Best Hyperparameter (C): {grid_search.best_params_['C']}")
print(f"Best F1 Score: {grid_search.best_score_}")
```

In this example, a simple `LogisticRegression` model’s ‘C’ hyperparameter is being optimized using a 5-fold cross-validation method. Importantly, the ‘scoring’ parameter is explicitly set to ‘f1’, indicating we want the `GridSearchCV` to optimize for this particular metric. The output will show the optimal hyperparameter value ‘C’ that maximized the average F1 score over the cross-validation folds.

**Example 2: Comparing F1 and Accuracy in a Logistic Regression**

Here, we demonstrate how the choice of optimization metric impacts the selected hyperparameter using an imbalanced dataset and evaluate both optimized parameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = {'C': np.logspace(-3, 3, 7)}

# Logistic Regression model initialization
logreg = LogisticRegression(solver='liblinear', random_state=42)

# GridSearchCV for F1 optimization
grid_search_f1 = GridSearchCV(logreg, param_grid, cv=5, scoring='f1')
grid_search_f1.fit(X_train, y_train)
best_C_f1 = grid_search_f1.best_params_['C']

# GridSearchCV for Accuracy optimization
grid_search_acc = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid_search_acc.fit(X_train, y_train)
best_C_acc = grid_search_acc.best_params_['C']

# Evaluate optimized models
logreg_f1 = LogisticRegression(solver='liblinear', C=best_C_f1, random_state=42)
logreg_f1.fit(X_train, y_train)
y_pred_f1 = logreg_f1.predict(X_test)

logreg_acc = LogisticRegression(solver='liblinear', C=best_C_acc, random_state=42)
logreg_acc.fit(X_train, y_train)
y_pred_acc = logreg_acc.predict(X_test)

# Print results
print(f"Best C optimized for F1: {best_C_f1}")
print(f"F1 Score (F1 Optimized Model): {f1_score(y_test, y_pred_f1)}")
print(f"Accuracy Score (F1 Optimized Model): {accuracy_score(y_test, y_pred_f1)}")
print("-" * 30)
print(f"Best C optimized for Accuracy: {best_C_acc}")
print(f"F1 Score (Accuracy Optimized Model): {f1_score(y_test, y_pred_acc)}")
print(f"Accuracy Score (Accuracy Optimized Model): {accuracy_score(y_test, y_pred_acc)}")
```

This example shows two separate `GridSearchCV` runs, one optimizing for ‘f1’ and the other for ‘accuracy’. Each optimized model is evaluated on the same test set, providing a comparison of which parameters achieve the best scores. In this case, ‘F1’ optimized model tends to perform better on F1 score than the accuracy-optimized model, indicating, in this particular case, F1 is the correct choice of optimization metric for the logistic regression model.

**Example 3: Custom F1 for a specific class in a multiclass problem**

This example illustrates extending the F1 score to specific classes within a multi-class scenario, where we want to focus on particular classes. This utilizes the `make_classification` function to generate 3 classes.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, make_scorer
import numpy as np

# Generate a multiclass synthetic dataset
X, y = make_classification(n_samples=1000, n_classes=3, weights=[0.2, 0.3, 0.5], random_state=42)

# Define custom F1 score function for a specific class
def custom_f1(y_true, y_pred, target_class=2):
    return f1_score(y_true, y_pred, labels=[target_class], average='macro')

# Create a scorer object from the custom function
custom_f1_scorer = make_scorer(custom_f1, greater_is_better=True)

# Define the hyperparameter grid
param_grid = {'C': np.logspace(-3, 3, 7)}

# Initialize the logistic regression model
logreg = LogisticRegression(solver='liblinear', random_state=42)

# Set up GridSearchCV using the custom F1 scorer
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring=custom_f1_scorer)

# Run the grid search with cross-validation
grid_search.fit(X, y)

# Print the best hyperparameter and F1 score
print(f"Best Hyperparameter (C): {grid_search.best_params_['C']}")
print(f"Best Custom F1 Score: {grid_search.best_score_}")
```

In this example, we define a custom `f1` function that accepts the target class as a parameter. This lets us focus on a particular class within our data, and `make_scorer` allows us to create a callable object that can be passed into the `GridSearchCV` function for use as the scoring metric. This custom method helps address specific use cases. The output displays the best ‘C’ value for the target class.

To conclude, F1 score is a potent tool for hyperparameter optimization when dealing with class imbalances, however, careful evaluation must be made on a case-by-case basis. It is not universally better than accuracy and should be considered within the context of specific needs. Consider exploring the scikit-learn documentation for metrics, such as `classification_report`, which can provide a more granular perspective. I also highly recommend investigating the impact of class weights on model performance alongside F1 score optimization. Understanding these nuances will invariably lead to more robust and effective models.
