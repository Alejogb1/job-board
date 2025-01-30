---
title: "How do I calculate the F1 score for a test set?"
date: "2025-01-30"
id: "how-do-i-calculate-the-f1-score-for"
---
The F1 score, a harmonic mean of precision and recall, provides a robust metric for evaluating a classifier, particularly in scenarios with imbalanced datasets where relying solely on accuracy can be misleading.  My experience developing anomaly detection systems for financial transactions highlighted the crucial role of the F1 score in assessing model performance, especially given the inherent class imbalance (legitimate transactions vastly outnumber fraudulent ones).  Calculating the F1 score for a test set requires a clear understanding of its constituent components: precision and recall.  This necessitates the correct identification of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).


**1. A Clear Explanation of F1 Score Calculation:**

The F1 score is defined as:

F1 = 2 * (Precision * Recall) / (Precision + Recall)

Where:

* **Precision:**  The proportion of correctly predicted positive instances out of all instances predicted as positive.  Formally, Precision = TP / (TP + FP).  A high precision indicates a low rate of false positives.

* **Recall (Sensitivity):** The proportion of correctly predicted positive instances out of all actual positive instances. Formally, Recall = TP / (TP + FN). A high recall indicates a low rate of false negatives.


Therefore, to compute the F1 score for a test set, one must first generate the confusion matrix from the model's predictions on the test data. The confusion matrix summarizes the counts of TP, TN, FP, and FN.  From these counts, precision and recall are calculated, and subsequently, the F1 score is derived using the formula above.  A perfect classifier achieves an F1 score of 1.0, while a completely inaccurate classifier would have an F1 score of 0.0.  Note that the F1 score is most informative when both precision and recall are considered simultaneously.


**2. Code Examples with Commentary:**

The following examples demonstrate F1 score calculation using Python.  I've used Scikit-learn and NumPy, libraries I've extensively employed in past projects involving large-scale classification tasks.


**Example 1: Using Scikit-learn's `f1_score` function:**

```python
import numpy as np
from sklearn.metrics import f1_score

# True labels for the test set
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])

# Predicted labels from the model
y_pred = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 0])

# Calculate the F1 score (default is binary F1)
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")

#For multiclass classification:
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred_multi = np.array([0, 1, 1, 0, 0, 2, 0, 1, 2, 1])
f1_macro = f1_score(y_true_multi, y_pred_multi, average='macro') #Macro average
f1_micro = f1_score(y_true_multi, y_pred_multi, average='micro') #Micro average
print(f"Macro F1 Score (Multiclass): {f1_macro}")
print(f"Micro F1 Score (Multiclass): {f1_micro}")

```

This example leverages Scikit-learn's built-in function, providing a concise and efficient method for calculating the F1 score.  The `average` parameter in `f1_score` allows for handling multi-class classification problems using macro and micro averaging. The macro average computes the unweighted mean of per-class F1 scores and  the micro average computes the F1 score from the global counts of true positives, false positives and false negatives.


**Example 2: Manual Calculation from a Confusion Matrix:**

```python
import numpy as np

# Confusion matrix
confusion_matrix = np.array([[5, 2], [1, 2]])

# Extract TP, TN, FP, FN
TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)

# Calculate F1 score
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
```

This example demonstrates the manual calculation, providing a deeper understanding of the underlying mechanics.  This approach is useful for situations where direct access to the confusion matrix is preferred or when using libraries that don't directly provide the F1 score.  It's crucial to ensure correct indexing when extracting TP, TN, FP, and FN from the confusion matrix.


**Example 3: Using a Pandas DataFrame:**

```python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Sample data
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 0])


#Create a confusion matrix using pandas
conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred),
                           index = ['Actual Negative', 'Actual Positive'],
                           columns = ['Predicted Negative', 'Predicted Positive'])
print(conf_matrix)

#Extract values from the dataframe for F1 calculation
TP = conf_matrix.loc['Actual Positive','Predicted Positive']
TN = conf_matrix.loc['Actual Negative','Predicted Negative']
FP = conf_matrix.loc['Actual Negative','Predicted Positive']
FN = conf_matrix.loc['Actual Positive','Predicted Negative']


precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
```

This approach uses Pandas DataFrames, offering a more organized and potentially more readable way to handle the data, particularly when dealing with larger datasets or more complex classification problems involving multiple classes. DataFrames also lend themselves well to further analysis and visualization of the results.


**3. Resource Recommendations:**

For further study, I recommend consulting standard machine learning textbooks focusing on classification metrics.  A strong grasp of probability and statistics is also essential for a thorough understanding of precision, recall, and the F1 score's implications.  Additionally, reviewing documentation for popular machine learning libraries like Scikit-learn will be beneficial.  Specific attention should be given to sections discussing model evaluation and the various metrics available.  Finally, examining research papers on applications of classification in relevant domains will offer practical insights and demonstrate the importance of choosing appropriate evaluation metrics based on the specific problem context.
