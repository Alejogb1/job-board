---
title: "How to resolve a 'list object is not callable' error when calculating an ROC curve?"
date: "2025-01-30"
id: "how-to-resolve-a-list-object-is-not"
---
The error 'list object is not callable' during ROC curve calculation typically arises from attempting to invoke a Python list as if it were a function, a common mistake stemming from unintended name collisions or incorrect data handling in the preprocessing or scoring phases of a machine learning workflow. I’ve encountered this frequently, particularly when dealing with multi-class classification scenarios where probability scores need careful organization before being fed into tools like scikit-learn's `roc_curve`.

Here’s a breakdown of how this error surfaces and the approaches to resolving it.

The `roc_curve` function within `sklearn.metrics` expects two primary arguments: `y_true`, which represents the true binary labels, and `y_score`, which represents the predicted probability scores. These scores should either be the raw output of a decision function (in binary classification) or probabilities for the positive class (in multi-class scenarios using one-vs-rest or similar approaches). Critically, `y_score` is expected to be a one-dimensional array or an object that can be converted to one. When `y_score` accidentally becomes a list due to misassignment or transformation errors, the Python interpreter tries to call it like a function, hence the 'list object is not callable' message.

The most typical cause stems from how prediction probabilities are handled, specifically when dealing with models returning probabilities for multiple classes. In a binary case, a prediction method might output a single probability of the positive class. However, in multi-class scenarios, models will commonly return an array or a list of probabilities (one for each class), each representing the predicted probability that the given instance belongs to that specific class. Incorrectly using this multi-dimensional probability output directly as the `y_score` argument in `roc_curve` is the primary pitfall. Often, I’ve seen developers inadvertently assign such an array to a variable intended to store the prediction probabilities *after* properly handling the multi-class nature.

Consider the case where a classifier outputs prediction probabilities as a list of lists, like `[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]` where each inner list corresponds to probabilities for two classes. If a developer tries to directly pass this structure to `roc_curve` without appropriate selection of the probability related to a specific "positive" class, the error occurs. The list of lists is treated as `y_score`, which `roc_curve` attempts to call.

Here are three code examples demonstrating the problem and its solution, based on situations I have encountered:

**Example 1: Incorrect handling of multi-class output**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np

# Sample multi-class data
X = np.random.rand(100, 5)
y = np.random.randint(0, 3, 100)  # Three classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)  # Returns list of probabilites for all classes

try:
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Incorrect y_prob
    roc_auc = auc(fpr, tpr)
except Exception as e:
    print(f"Error: {e}")
```

In this case, `model.predict_proba(X_test)` returns an array where each row corresponds to a test instance, and the columns are the probabilities for each class (in this example, three classes). Passing this directly to `roc_curve` with multi-class target data results in the `y_score` being the probability matrix, which `roc_curve` attempts to treat as a function, hence the error.

**Example 2: Correct Implementation for One-vs-Rest with Binary Case as an Assumption**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np

# Sample multi-class data
X = np.random.rand(100, 5)
y = np.random.randint(0, 3, 100)  # Three classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)

# Apply proper transformation and use for class 1 vs all others
positive_class = 1
y_test_binary = (y_test == positive_class).astype(int) # Create Binary Data from One-Vs-Rest assumption
y_score = y_prob[:, positive_class] # Get the probabilities of class 1 from probability matrix


fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
roc_auc = auc(fpr, tpr)

print(f"AUC for class {positive_class} vs Rest : {roc_auc:.2f}")
```

Here, I corrected the previous issue by explicitly treating one class against the rest. I transformed `y_test` into a binary vector reflecting class 1 (positive) against all others (negative) and correctly selected the prediction probabilities for the specific positive class via array indexing (`y_prob[:, positive_class]`). This converts the multi-class problem into a binary case suitable for the single-ROC method, and is consistent with how a 'one vs rest' method is generally handled in the context of ROC. If this was intended to measure performance for all classes, a similar transformation and computation will need to be done in a looping fashion.

**Example 3: Incorrect List Assignment**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np

# Sample binary data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)  # Binary classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Assuming the user meant to extract class 1 probabilities
# but assigned a list instead
y_prob = model.predict_proba(X_test)

# Incorrect assignment
y_score = list(y_prob[:, 1]) # Accidentally casts into list when expecting an array

try:
    fpr, tpr, thresholds = roc_curve(y_test, y_score)  # Incorrect y_score
    roc_auc = auc(fpr, tpr)
except Exception as e:
    print(f"Error: {e}")
```

In this example, despite working with a binary case, I deliberately created the error by casting the correctly retrieved class 1 probabilities into a list through `list()`. Although the raw output from `model.predict_proba` would have been correct, the explicit casting resulted in the error. This demonstrates that even after doing a correct extraction of the probabilities, data type transformations need careful handling.

To summarize, the primary remedy is careful review of the `y_score` data structure before it gets passed into the `roc_curve` method. Ensure that the prediction probabilities are a one-dimensional array or object that can be converted to one, and not multi-dimensional. Handle multi-class probability outputs by selecting the scores corresponding to the positive class(es) or use a multi-class ROC approach if appropriate, and avoid any unnecessary conversions that alter the intended data structure.

For additional resources on this topic, consult the scikit-learn documentation for `roc_curve` and `predict_proba`, which explain the expected input formats. Books on the fundamentals of machine learning and classification metrics can provide further context. Furthermore, exploring coding exercises focused on multi-class classification will expose one to practical applications and help avoid this error.
