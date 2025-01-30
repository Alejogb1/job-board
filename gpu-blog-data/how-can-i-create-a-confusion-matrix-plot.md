---
title: "How can I create a confusion matrix plot in PyTorch?"
date: "2025-01-30"
id: "how-can-i-create-a-confusion-matrix-plot"
---
PyTorch doesn't natively provide a function for generating confusion matrices.  The framework excels at tensor operations and model building, but visualization is typically handled by other libraries.  My experience working on several large-scale image classification projects has reinforced this understanding.  Generating a confusion matrix requires post-processing of prediction outputs and ground truth labels;  PyTorch provides the necessary tools to manipulate these data structures, but the plotting is best left to dedicated visualization libraries like Matplotlib or Seaborn.

**1. Clear Explanation:**

The confusion matrix visualizes the performance of a classification model by summarizing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.  Each cell (i, j) represents the number of instances where the actual class is 'i' and the predicted class is 'j'.  To create this matrix in conjunction with PyTorch, we first need to obtain the predicted class labels from our model's output and compare them to the ground truth labels.  Then, we utilize a suitable library (Matplotlib or Seaborn in the examples below) to generate the plot from this comparison.  The accuracy, precision, recall, and F1-score can all be easily calculated from the confusion matrix values.  This provides a comprehensive performance overview, beyond a simple accuracy metric.  Importantly, the choice of library will influence the aesthetic aspects of the plot; Matplotlib offers more granular control, while Seaborn provides a higher-level interface for visually appealing plots.  It's crucial to remember that the confusion matrix provides insights into the model's performance across all classes, revealing potential biases or areas requiring further improvement.


**2. Code Examples with Commentary:**

**Example 1: Using Matplotlib for a Basic Confusion Matrix**

This example demonstrates a straightforward approach, leveraging Matplotlib's `imshow` function for visualization.  I've frequently used this method in early stages of model evaluation for its simplicity and direct control over plot elements.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Sample predicted and true labels (replace with your model's output and ground truth)
predicted_labels = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0, 0, 2])
true_labels = torch.tensor([0, 1, 0, 0, 2, 1, 1, 0, 1, 2])

# Convert labels to NumPy arrays for Matplotlib compatibility
predicted_labels = predicted_labels.numpy()
true_labels = true_labels.numpy()

# Calculate the confusion matrix
num_classes = 3  # Adjust according to your number of classes
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
for i in range(len(predicted_labels)):
    confusion_matrix[true_labels[i], predicted_labels[i]] += 1

# Plot the confusion matrix
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, ['Class 0', 'Class 1', 'Class 2'])
plt.yticks(tick_marks, ['Class 0', 'Class 1', 'Class 2'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, confusion_matrix[i, j], ha='center', va='center')
plt.show()
```

This code first defines placeholder predicted and true labels. In a real application, you would replace these with the output of your PyTorch model's `predict` function (after applying `argmax` or a similar operation to get class labels) and your ground truth dataset labels respectively. The code then iterates through the predictions and updates the confusion matrix accordingly. Finally, it uses `imshow` to display the matrix as an image, adding labels, colorbar, and numerical values to each cell for better readability.  The `cmap` argument controls the colormap; 'Blues' is just one option; others are available within Matplotlib.


**Example 2: Utilizing Seaborn for Enhanced Visualization**

Seaborn offers a more concise and aesthetically pleasing approach. In my experience, Seaborn's higher-level functions make the plotting process more efficient, particularly for larger datasets or more complex visualizations.

```python
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ... (same predicted and true labels as Example 1) ...

# Create a Pandas DataFrame for Seaborn
data = {'True Label': true_labels, 'Predicted Label': predicted_labels}
df = pd.DataFrame(data)

# Create the confusion matrix using Seaborn's heatmap
confusion_matrix = pd.crosstab(df['True Label'], df['Predicted Label'], rownames=['True'], colnames=['Predicted'])
plt.figure(figsize=(8, 6)) # Adjust figure size as needed
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Seaborn)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```

This example leverages Pandas to create a DataFrame from the predicted and true labels.  Seaborn's `heatmap` function then directly generates the confusion matrix plot with annotations.  The `annot=True` and `fmt='d'` arguments ensure that cell values are displayed as integers.  The `cmap` parameter again sets the colormap.  This approach is often preferred for its readability and ease of use.


**Example 3:  Calculating Metrics from the Confusion Matrix (Matplotlib)**

This builds upon Example 1, explicitly calculating key performance metrics.  This is a crucial step in understanding model performance beyond a visual representation. I frequently incorporate these calculations into my evaluation scripts to automatically generate comprehensive reports.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# ... (same predicted and true labels as Example 1, and confusion matrix calculation) ...


# Calculate metrics
TP = np.diag(confusion_matrix)
FP = np.sum(confusion_matrix, axis=0) - TP
FN = np.sum(confusion_matrix, axis=1) - TP
TN = np.sum(confusion_matrix) - (TP + FP + FN)

accuracy = np.sum(TP) / np.sum(confusion_matrix)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

# ... (same plotting code as Example 1) ...
```

This extension calculates true positive (TP), false positive (FP), false negative (FN), and true negative (TN) counts directly from the confusion matrix.  It then derives accuracy, precision, recall, and F1-score.  These metrics provide a quantitative assessment of the model's performance, complementing the visual representation provided by the confusion matrix plot.  Handling potential `ZeroDivisionError` scenarios (e.g., when a class has no true positives) would require additional error handling which is omitted for brevity.


**3. Resource Recommendations:**

For further study, I would recommend consulting the official documentation for PyTorch, Matplotlib, Seaborn, and NumPy.  A comprehensive textbook on machine learning and its associated algorithms would also prove invaluable.  Exploring examples from established machine learning repositories would solidify your understanding of best practices.  Finally, examining published research papers that utilize confusion matrices in their evaluations can offer valuable insights into their application and interpretation in various contexts.
