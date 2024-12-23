---
title: "How can I generate and display a valid confusion matrix for evaluation?"
date: "2024-12-23"
id: "how-can-i-generate-and-display-a-valid-confusion-matrix-for-evaluation"
---

Let's tackle this. I've certainly spent my fair share of time wrestling with the nuances of model evaluation, and confusion matrices are a cornerstone in that process. It's more than just visualizing numbers; it’s about understanding the types of errors your model is making, which, in turn, guides improvements. So, let’s break down how to generate and display a valid confusion matrix, covering the core concepts and then diving into some practical examples.

First and foremost, a confusion matrix, at its heart, is a tabular representation that visualizes the performance of a classification model. It compares the actual class labels against those predicted by the model. The rows typically represent actual values, while columns represent predicted values (or vice versa, depending on convention, but I usually stick to actuals as rows). The cells then contain counts representing the occurrences of various prediction outcomes.

The key elements within a confusion matrix for a binary classification task (e.g., positive and negative) are:

*   **True Positives (TP):** The model correctly predicted the positive class.
*   **True Negatives (TN):** The model correctly predicted the negative class.
*   **False Positives (FP):** The model incorrectly predicted the positive class (Type I error).
*   **False Negatives (FN):** The model incorrectly predicted the negative class (Type II error).

For multiclass problems, this extends to have more rows and columns, reflecting the number of classes. The diagonal then shows correctly classified examples and off-diagonal elements show which classes are being confused.

Generating such a matrix typically involves the following steps. Firstly, you need the true labels (ground truth) from your test set along with your model's predictions on that same test set. These form the foundation for our matrix. Once you have that, you can use libraries to generate the visual representation; I often find that using dedicated functions are better than doing it all by hand.

Now, let's look at three concrete examples using python and `scikit-learn`, because it is ubiquitous for these tasks:

**Example 1: Basic Binary Classification Matrix**

Let's say you are working on a spam email classifier. We have some ground truth, and our model makes predictions. Here's how I'd generate a basic confusion matrix using `scikit-learn`:

```python
from sklearn.metrics import confusion_matrix
import numpy as np

# Sample ground truth labels (0 for not spam, 1 for spam)
y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])

# Sample model predictions
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])


# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)

# Interpretation (based on above output):
#  [[4 1]
#  [1 4]]
# TP = 4 (bottom-right corner), TN = 4 (top-left), FP = 1 (top-right), FN = 1 (bottom-left).

# Displaying it nicely (optional, but beneficial)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

This simple script imports the necessary functions, creates some fake actuals and predicted outputs. It then produces the matrix and displays it, with a nice plot thanks to `seaborn` and `matplotlib`. This is usually more informative to look at than just raw numbers.

**Example 2: Multiclass Classification Matrix**

Now let's tackle a more complicated situation; a multiclass problem, for example, image recognition of different types of animals.

```python
from sklearn.metrics import confusion_matrix
import numpy as np

# Sample ground truth labels (0: Cat, 1: Dog, 2: Bird)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 2])

# Sample model predictions
y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 1, 2])

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)

# Interpretation of the output:
# [[3 0 0]
# [0 2 1]
# [0 1 2]]
# The diagonal represents correctly classified (e.g., 3 Cats classified as cat, 2 dogs classified as dog, etc.)
# Off-diagonals shows misclassifications (e.g., 1 dog was classified as bird and 1 bird as dog)

# Visualizing it for clarity
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Cat", "Dog", "Bird"], yticklabels=["Cat", "Dog", "Bird"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Multiclass)')
plt.show()
```

Here the structure is very similar, but we have expanded the number of categories, showing how the matrix easily scales up for multiclass problems, and demonstrating which classes are being confused with one another.

**Example 3: Normalization & More Detailed Metrics**

Sometimes looking at counts isn’t enough. We often need to look at proportions of classifications instead. We can normalize the matrix and obtain more metrics (recall, precision, f1-score). Here is an example demonstrating this:

```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Sample ground truth labels
y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0])

# Sample model predictions
y_pred = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1])


# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, normalize='true') # 'true' normalizes over rows, 'pred' normalizes over columns, 'all' over the total

print("Normalized Confusion Matrix (by actuals):\n", cm)


# Get a more complete report with multiple metrics
report = classification_report(y_true, y_pred)
print("\nClassification Report:\n", report)

# Visualize normalized matrix

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Reds", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.show()
```

This example normalizes the matrix based on the true labels (i.e. showing, out of the instances of class '0', what fraction were classified correctly, and same for class '1'), and also displays a more detailed classification report including recall, precision and f1 scores, which are usually helpful to provide more insights about your model, along with the confusion matrix itself.

Several technical books and papers address this area in depth. For instance, “Pattern Recognition and Machine Learning” by Christopher Bishop offers a rigorous foundation in classification and evaluation techniques. “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman also provides a comprehensive perspective. For a more practical approach, I'd recommend looking into the scikit-learn documentation and some of the relevant papers on evaluation measures. Researching papers specifically discussing the nuances of normalized matrices can also be quite helpful.

Generating and interpreting confusion matrices is a fundamental skill for anyone working in machine learning. By understanding their structure and purpose, you gain valuable insights into your model’s strengths and weaknesses, guiding you to more informed choices and targeted improvements. It’s a vital piece in the model evaluation process, and, frankly, I wouldn’t do machine learning without being able to use them effectively.
