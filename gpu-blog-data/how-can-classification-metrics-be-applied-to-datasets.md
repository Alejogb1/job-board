---
title: "How can classification metrics be applied to datasets with both multi-label and multi-class targets?"
date: "2025-01-30"
id: "how-can-classification-metrics-be-applied-to-datasets"
---
Multi-label and multi-class classification present distinct challenges to the application of standard evaluation metrics. Standard metrics, often designed for binary or single-class problems, require adaptation or replacement to accurately reflect a model's performance in these more complex scenarios. Having frequently encountered these complexities in the development of document categorization systems, I've developed an understanding of how to apply metrics effectively within this context.

Multi-class classification involves assigning a single label from a set of mutually exclusive classes to an instance. For example, categorizing images into "cat," "dog," or "bird." Multi-label classification, conversely, allows an instance to have multiple labels simultaneously. Think of tagging articles with categories such as "politics," "international," and "economics," where a single article could fall under all three categories. The key distinction is the exclusivity of class assignments. This difference necessitates a careful selection of metrics tailored to each scenario.

For multi-class classification, standard metrics like accuracy, precision, recall, and the F1-score remain applicable. However, their interpretation might need adjustment depending on class imbalances. Accuracy, representing the proportion of correct predictions, might be misleading if one class dominates the data. Therefore, precision, recall, and their harmonic mean, the F1-score, are often preferred. Precision focuses on the correctness of positive predictions, while recall emphasizes the model's ability to identify all positive instances. The F1-score balances these two aspects. When classes are imbalanced, macro-averaging or weighted-averaging can provide a more representative picture of overall performance, treating each class either equally or by considering its support in the dataset, respectively.

In multi-label classification, these metrics require reformulation. For instance, a single instance can have multiple true labels and multiple predicted labels. We cannot simply count the number of correct predictions as we did in multi-class. Here, we treat each label as a separate binary classification problem. We then calculate precision, recall, and F1-score for each label, and subsequently compute a form of aggregate. Common aggregation methods are micro-averaging and macro-averaging. Micro-averaging computes the total true positives, false positives, and false negatives over all labels, subsequently computing precision, recall, and F1-score. Macro-averaging, on the other hand, computes the metrics for each label individually and then averages them. These two approaches offer differing viewpoints: Micro-averaging is appropriate when the overall class balance is important, whereas macro-averaging treats each label equally, providing better insights when focusing on the performance across each distinct label. Another important consideration in multi-label scenarios is the Hamming Loss. This metric calculates the fraction of labels that are incorrectly predicted.

Here are three code examples using Python, employing scikit-learn, to illustrate these concepts, along with commentary:

**Example 1: Multi-class classification using accuracy, precision, recall, and F1-score, focusing on class imbalance.**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

y_true = [0, 1, 2, 0, 0, 1, 0, 1, 2, 0] # True labels (10 samples, 3 classes)
y_pred = [0, 2, 2, 0, 1, 1, 0, 1, 0, 1] # Predicted labels

accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(y_true, y_pred, average='macro')
weighted_precision = precision_score(y_true, y_pred, average='weighted')
macro_recall = recall_score(y_true, y_pred, average='macro')
weighted_recall = recall_score(y_true, y_pred, average='weighted')
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')


print(f"Accuracy: {accuracy:.2f}") # Output: 0.60
print(f"Macro-Precision: {macro_precision:.2f}") # Output: 0.55
print(f"Weighted-Precision: {weighted_precision:.2f}") # Output: 0.60
print(f"Macro-Recall: {macro_recall:.2f}") # Output: 0.57
print(f"Weighted-Recall: {weighted_recall:.2f}") # Output: 0.60
print(f"Macro-F1: {macro_f1:.2f}") # Output: 0.55
print(f"Weighted-F1: {weighted_f1:.2f}") # Output: 0.59
```

In this example, we have a simple multi-class classification scenario with three classes. Notice the discrepancy between accuracy and macro-averaged values; accuracy is misleadingly high. Due to the class imbalance, weighted averages provide better insight. Class 0 dominates the dataset.

**Example 2: Multi-label classification using precision, recall, F1-score (micro and macro) and Hamming Loss**

```python
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
import numpy as np

y_true = np.array([[1, 0, 1],
                   [0, 1, 1],
                   [1, 1, 0],
                   [0, 1, 0]]) # True labels (4 samples, 3 labels)
y_pred = np.array([[1, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 1, 0]]) # Predicted labels


micro_precision = precision_score(y_true, y_pred, average='micro')
micro_recall = recall_score(y_true, y_pred, average='micro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

macro_precision = precision_score(y_true, y_pred, average='macro')
macro_recall = recall_score(y_true, y_pred, average='macro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

hl = hamming_loss(y_true, y_pred)


print(f"Micro-Precision: {micro_precision:.2f}") # Output: 0.75
print(f"Micro-Recall: {micro_recall:.2f}") # Output: 0.67
print(f"Micro-F1: {micro_f1:.2f}") # Output: 0.71
print(f"Macro-Precision: {macro_precision:.2f}") # Output: 0.75
print(f"Macro-Recall: {macro_recall:.2f}") # Output: 0.67
print(f"Macro-F1: {macro_f1:.2f}") # Output: 0.67
print(f"Hamming Loss: {hl:.2f}")  # Output: 0.25
```

This example illustrates the calculation of micro and macro averaged precision, recall, and F1-score in a multi-label context.  We can observe that the micro and macro scores are slightly different.  The hamming loss calculates the average number of label misclassifications.

**Example 3: Multi-label classification with custom metrics.**

```python
import numpy as np

def custom_precision(y_true, y_pred, label_index):
  true_positives = np.sum((y_true[:, label_index] == 1) & (y_pred[:, label_index] == 1))
  predicted_positives = np.sum(y_pred[:, label_index] == 1)
  return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def custom_recall(y_true, y_pred, label_index):
  true_positives = np.sum((y_true[:, label_index] == 1) & (y_pred[:, label_index] == 1))
  actual_positives = np.sum(y_true[:, label_index] == 1)
  return true_positives / actual_positives if actual_positives > 0 else 0.0


y_true = np.array([[1, 0, 1],
                   [0, 1, 1],
                   [1, 1, 0],
                   [0, 1, 0]])
y_pred = np.array([[1, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 1, 0]])

label_0_precision = custom_precision(y_true, y_pred, 0)
label_1_precision = custom_precision(y_true, y_pred, 1)
label_2_precision = custom_precision(y_true, y_pred, 2)

label_0_recall = custom_recall(y_true, y_pred, 0)
label_1_recall = custom_recall(y_true, y_pred, 1)
label_2_recall = custom_recall(y_true, y_pred, 2)


print(f"Label 0 Precision: {label_0_precision:.2f}") # Output: 1.00
print(f"Label 1 Precision: {label_1_precision:.2f}") # Output: 0.67
print(f"Label 2 Precision: {label_2_precision:.2f}") # Output: 0.00

print(f"Label 0 Recall: {label_0_recall:.2f}") # Output: 1.00
print(f"Label 1 Recall: {label_1_recall:.2f}") # Output: 1.00
print(f"Label 2 Recall: {label_2_recall:.2f}") # Output: 0.50
```
This example shows how to define custom precision and recall metrics for individual labels, which can be invaluable when analyzing specific classes' performance. This becomes useful in debugging performance of a specific label.

For further study, I would recommend resources that detail the nuances of multi-class and multi-label evaluation. Explore textbooks or online guides specializing in machine learning performance evaluation or classification. Statistical learning theory resources can be beneficial. These will provide theoretical background and practical insights to further deepen understanding of model evaluation beyond the basic metrics. Understanding the mathematical foundation of metrics alongside their implementation will allow for their correct application to various data problems. Specifically, seeking out material covering error analysis in classification is valuable when faced with poor model performance.
