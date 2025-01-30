---
title: "How do I create a confusion matrix?"
date: "2025-01-30"
id: "how-do-i-create-a-confusion-matrix"
---
A confusion matrix, at its core, provides a tabular visualization of the performance of a classification algorithm, detailing the counts of true positives, true negatives, false positives, and false negatives. Constructing one is fundamental for evaluating model efficacy, particularly when dealing with imbalanced datasets or scenarios where misclassifications have different costs. Over the years, I’ve found a robust understanding of this matrix essential for both initial model validation and subsequent refinement.

The core concept involves cross-referencing predicted classifications against actual ground truth labels. Each row in the matrix typically represents the actual (or true) class, while each column represents the predicted class. The intersection of a row and column then indicates how many instances of that actual class were classified as that specific predicted class. This allows for an immediate assessment of not only overall accuracy but also the specific types of errors a model is making.

Let's break down each component. A **true positive (TP)** signifies an instance where the model correctly predicted a positive class. Conversely, a **true negative (TN)** represents a correct prediction of a negative class.  A **false positive (FP)**, also known as a Type I error, occurs when a negative instance is incorrectly classified as positive. Finally, a **false negative (FN)**, or Type II error, occurs when a positive instance is misclassified as negative.  Understanding these four components forms the basis for any effective confusion matrix implementation.

I’ll now illustrate building a confusion matrix using Python with `scikit-learn` and then a manual approach to demonstrate the underlying mechanics.

**Example 1: Using `scikit-learn`**

The `sklearn.metrics` module provides a highly efficient function for generating a confusion matrix. I often leverage this module in initial model evaluations.

```python
from sklearn.metrics import confusion_matrix
import numpy as np

# Example of actual (y_true) and predicted (y_pred) labels
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])  # 1 represents 'positive', 0 'negative'
y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0])

# Construct the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Optionally, you can add labels for clarity
labels = ["Negative", "Positive"]
conf_matrix_labeled = confusion_matrix(y_true, y_pred, labels=labels)
print("\nLabeled Confusion Matrix:\n", conf_matrix_labeled)
```

**Commentary:**

This code snippet first imports the necessary modules. `numpy` is utilized to create sample arrays for true labels (`y_true`) and predicted labels (`y_pred`). Subsequently, `confusion_matrix` from `sklearn.metrics` computes the matrix. The output without labels displays a 2x2 array where the first row and column are associated with the label '0' and second row and column with '1'. When labeled, the output provides additional clarity for easier interpretation. It’s important that the labels argument in scikit learn match the integer values used in your classification. Using string-type labels in classification will lead to a type error. The result displays the counts of TP, TN, FP, and FN in a structured way. I typically follow this with functions calculating accuracy, precision, recall and F1 scores using this matrix as a foundational step in model assessment.

**Example 2: Building a Confusion Matrix from Scratch**

To solidify understanding, here’s a Python function to manually construct a matrix. This is rarely necessary in practice, but it clarifies the algorithm's logic.

```python
import numpy as np

def manual_confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(y_true)  # Determine all unique classes present
    num_classes = len(unique_classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)  # Initialize a matrix filled with zeros

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]

        # Determine the row and column index in the matrix based on the labels
        row_index = np.where(unique_classes == true_label)[0][0]
        col_index = np.where(unique_classes == pred_label)[0][0]

        conf_matrix[row_index, col_index] += 1  # Increment the corresponding cell
    return conf_matrix

# Example usage
y_true = np.array([1, 0, 2, 1, 0, 2, 1, 0, 1, 2])
y_pred = np.array([1, 1, 0, 0, 0, 2, 2, 0, 1, 2])
conf_matrix = manual_confusion_matrix(y_true, y_pred)
print("Manual Confusion Matrix:\n", conf_matrix)

# To verify against Scikit-Learn
from sklearn.metrics import confusion_matrix
conf_matrix_sk = confusion_matrix(y_true, y_pred)
print("\nScikit-learn Confusion Matrix:\n", conf_matrix_sk)
```

**Commentary:**

This `manual_confusion_matrix` function first identifies all unique class labels in the `y_true` array. It then initializes a matrix of zeros. The function iterates through each prediction-actual pair, determining the correct row and column index based on the unique class labels identified earlier. It increments the count at the corresponding matrix cell. This illustrates the core mechanism: each element of the matrix represents the frequency of each pairing between true and predicted class, providing an alternative, lower level understanding. The comparison against the scikit-learn version serves to demonstrate their functional equivalence. This approach helps to understand that the matrix represents a lookup table to count different classifications. In multi-class classification, this process extends to create N x N matrices where N is the number of classes.

**Example 3: Visualizing with Seaborn**

Visualizing the confusion matrix is crucial for easier interpretation, particularly for multi-class problems. Seaborn, a Python library built on Matplotlib, provides an excellent way to do this. While scikit learn can output a text based matrix, visualizing with seaborn is highly recommended for reporting and understanding model performance.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Same example data as before
y_true = np.array([1, 0, 2, 1, 0, 2, 1, 0, 1, 2])
y_pred = np.array([1, 1, 0, 0, 0, 2, 2, 0, 1, 2])

# Build the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class 0", "Class 1", "Class 2"],
            yticklabels=["Class 0", "Class 1", "Class 2"])
plt.title("Confusion Matrix Visualization")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.show()
```

**Commentary:**

This example first imports `matplotlib.pyplot` and `seaborn` for creating visualizations.  The `confusion_matrix` from `sklearn` is used to generate the raw matrix, followed by the construction of a heatmap using `sns.heatmap`. `annot=True` displays the numerical value on each cell, `fmt="d"` ensures integer formatting, and `cmap="Blues"` sets the color scheme. The `xticklabels` and `yticklabels` parameters set labels for clarity. This visual approach is valuable for presentations and quickly understanding trends like which classes are confused most frequently. In practice, I customize color maps to highlight imbalances or error patterns more effectively.

When working with a confusion matrix, it is important to use it to assess the model's performance. I typically calculate further metrics like:
*   **Accuracy**: The proportion of correct classifications (TP+TN)/(TP+TN+FP+FN). This metric however can be misleading in imbalanced datasets.
*   **Precision**: The proportion of true positives among all predicted positives TP/(TP+FP). I utilize this to evaluate model's performance when minimizing the occurrence of false positives is crucial.
*   **Recall**: The proportion of true positives among all actual positives TP/(TP+FN). This metric indicates how well the model can correctly identify all instances of the positive class.
*   **F1-Score**: Harmonic mean of precision and recall, representing a trade-off between precision and recall.

These metrics together with the confusion matrix create a foundation for a comprehensive evaluation.

For further exploration, I recommend investigating the following resources. While these sources will not have specific links, I have consistently found them valuable:

*   The documentation for `scikit-learn` provides extensive details on `confusion_matrix` and related metrics. The official website is always a first stop for me.
*   Texts on statistical modeling and pattern recognition often delve into the theoretical background of the confusion matrix and its utility in evaluating classification performance.
*   Various online data science education platforms offer courses covering model evaluation, frequently including detailed discussions and examples around confusion matrices. I advise checking multiple sources to get a balanced perspective.

By understanding the construction, interpretation, and visualization methods discussed here, anyone can effectively employ the confusion matrix as a foundational tool for building and refining classification models.
