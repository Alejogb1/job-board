---
title: "How do I create a confusion matrix?"
date: "2024-12-23"
id: "how-do-i-create-a-confusion-matrix"
---

, so, tackling the creation of a confusion matrix is something I've certainly spent a fair bit of time on across different projects. It’s a foundational tool in evaluating classification models, and getting it implemented correctly, and interpreting it meaningfully, is crucial. I remember one project, a predictive maintenance system for industrial machinery, where we initially struggled to differentiate between genuine failures and minor anomalies; a well-structured confusion matrix quickly became our best friend.

Let's break down what a confusion matrix *is* and how to implement it. In essence, it's a table that visualizes the performance of a classification model by showing the counts of:

*   **True Positives (TP):** Cases where the model correctly predicted the positive class.
*   **True Negatives (TN):** Cases where the model correctly predicted the negative class.
*   **False Positives (FP):** Cases where the model incorrectly predicted the positive class (also known as Type I error).
*   **False Negatives (FN):** Cases where the model incorrectly predicted the negative class (also known as Type II error).

The structure is typically presented as a 2x2 matrix for binary classification, but can be expanded to nxn for multi-class scenarios. The rows typically represent the actual (true) classes, and the columns represent the predicted classes (though some libraries or conventions may reverse this, always verify).

The core idea is to compare the actual class labels with the model's predicted class labels and tally up these categories. Manually calculating this for small datasets is feasible, but with even moderately sized data, it's best to automate the process.

Let's consider implementation using Python, since that’s a common choice in the field, and I've certainly found it to be versatile. I’ll show three approaches, each with varying levels of abstraction and demonstrating different scenarios.

**Example 1: Manual Calculation from Scratch**

This approach allows for absolute transparency and is valuable for understanding the inner workings, especially if you’re coming from a background without a lot of exposure to machine learning libraries. While not recommended for production due to its verbosity, it offers great learning potential.

```python
def calculate_confusion_matrix(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have the same length.")

    unique_classes = sorted(list(set(actual)))
    num_classes = len(unique_classes)

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for true_class, pred_class in zip(actual, predicted):
        row_index = unique_classes.index(true_class)
        col_index = unique_classes.index(pred_class)
        confusion_matrix[row_index][col_index] += 1

    return confusion_matrix, unique_classes

# Example usage
actual_labels = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
predicted_labels = ['cat', 'dog', 'dog', 'bird', 'cat', 'cat']

matrix, classes = calculate_confusion_matrix(actual_labels, predicted_labels)

print("Confusion Matrix:")
for row in matrix:
    print(row)
print("Classes:", classes)
```

In this snippet, we manually iterate through paired lists of actual and predicted values, creating a list of unique classes. We use the indexes of the true and predicted classes to increment the corresponding count in the confusion matrix. This builds the matrix from first principles.

**Example 2: Using `sklearn.metrics.confusion_matrix` for Binary Classification**

Scikit-learn provides a dedicated function for creating confusion matrices, greatly simplifying the process for many cases. This method is highly optimized and handles the counting process under the hood, allowing a focus on interpretation and further analysis.

```python
from sklearn.metrics import confusion_matrix

# Example binary classification
actual_labels = [0, 1, 0, 1, 0] # Assuming 0 for Negative, 1 for Positive
predicted_labels = [0, 1, 1, 0, 0]

matrix = confusion_matrix(actual_labels, predicted_labels)

print("Confusion Matrix:")
print(matrix)

# To label output, it can be done manually, assuming the class order in the provided lists are the same
# This is the case if using the original order of label list input to the sklearn method
classes = ["Negative", "Positive"]

print("        Predicted")
print("          Neg  Pos")
print("Actual Neg  " + str(matrix[0,0]) + " " + str(matrix[0,1]))
print("       Pos  " + str(matrix[1,0]) + " " + str(matrix[1,1]))
```

Here, `sklearn.metrics.confusion_matrix` takes two arguments: the actual labels and the predicted labels. It directly returns the matrix in the form of a numpy array. In this binary example we can easily identify the TP, TN, FP, and FN values based on their matrix positions. Remember that `sklearn` assumes the classes are ordered alphanumerically in the original input. To improve the output for this example, a custom print was added to label the output.

**Example 3: Handling Multi-Class Classification with `sklearn.metrics.confusion_matrix`**

The beauty of `sklearn` is that it handles multi-class scenarios effortlessly. This is valuable in real-world applications where you frequently have more than two classes to predict.

```python
from sklearn.metrics import confusion_matrix

# Example multi-class classification
actual_labels = ['apple', 'banana', 'cherry', 'apple', 'banana', 'cherry', 'cherry']
predicted_labels = ['apple', 'banana', 'banana', 'apple', 'cherry', 'cherry', 'cherry']

matrix = confusion_matrix(actual_labels, predicted_labels, labels = ['apple', 'banana', 'cherry'])

print("Confusion Matrix:")
print(matrix)

# To label output
classes = ['apple', 'banana', 'cherry']

print("         Predicted")
print("         apple  banana cherry")
print("Actual apple  " + str(matrix[0,0]) + "    " + str(matrix[0,1]) + "    " + str(matrix[0,2]))
print("       banana " + str(matrix[1,0]) + "    " + str(matrix[1,1]) + "    " + str(matrix[1,2]))
print("       cherry " + str(matrix[2,0]) + "    " + str(matrix[2,1]) + "    " + str(matrix[2,2]))
```

This snippet demonstrates that when provided with a list of unique classes, `sklearn.metrics.confusion_matrix` produces a matrix with dimensions equal to the number of unique classes. Again a manual print statement helps to visualize the output. The order of the labels passed to the `labels` parameter determines the rows and columns, which ensures the matrix makes sense.

**Interpreting the Confusion Matrix**

After creation, the confusion matrix allows us to calculate several performance metrics. Precision, recall, accuracy, and F1-score can all be derived from the numbers within the matrix. For example, precision is `TP / (TP + FP)`, and recall is `TP / (TP + FN)`. The choice of what metric to focus on is dependent on the problem. For example, in medical diagnostics, false negatives (FN) are often more critical to minimize than false positives (FP). So recall becomes a more focused metric.

**Recommendations for Further Study**

For a deeper understanding, I recommend exploring these resources:

*   **"The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman:** This book offers comprehensive coverage of statistical learning, including detailed explanations of model evaluation and performance metrics.
*   **"Pattern Recognition and Machine Learning" by Christopher Bishop:** This textbook provides a rigorous treatment of pattern recognition and machine learning algorithms, including classification and evaluation techniques.
*   **Scikit-learn documentation:** The official documentation for `sklearn.metrics.confusion_matrix` is the best reference for detailed usage and parameter explanations.

In my experience, starting with the manual implementation (Example 1) is incredibly helpful, but using pre-built functions like the ones found in `sklearn` is critical for scaling to larger projects. The insights gained from a well-formed confusion matrix, combined with an understanding of the strengths and weaknesses of our models, are essential for successful model development and deployment.
