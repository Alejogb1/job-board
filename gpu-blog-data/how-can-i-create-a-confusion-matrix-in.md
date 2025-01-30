---
title: "How can I create a confusion matrix in PyTorch?"
date: "2025-01-30"
id: "how-can-i-create-a-confusion-matrix-in"
---
The core challenge in generating a confusion matrix within the PyTorch framework lies not in PyTorch itself, but in the prerequisite step of obtaining accurate predictions and corresponding ground truth labels.  PyTorch excels at model training and inference, but the confusion matrix construction is fundamentally a post-processing task reliant on the output of your model and your dataset's labels.  My experience working on large-scale image classification projects, specifically within medical imaging, has highlighted this crucial distinction.  Many attempts to integrate confusion matrix generation directly within the training loop lead to inefficient and often incorrect results.  Let's proceed with a structured explanation and illustrative examples.

**1. Clear Explanation:**

The confusion matrix, a cornerstone of evaluating classification models, is a visual representation of the model's performance. It tabulates the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.  Its construction requires two key inputs:

* **Predictions:** These are the model's output classes for each data point in your test set.  Crucially, these must be in the same format as your ground truth labels.  For example, if your labels are one-hot encoded, your predictions should also be one-hot encoded. If they are integer class indices, your predictions must be integer class indices.
* **Ground Truth Labels:** These are the actual classes for each data point in your test set. This represents the correct classification for each instance.

Once you possess both predictions and ground truth labels, the confusion matrix can be constructed.  This is typically achieved using libraries like NumPy or Scikit-learn, operating independently of PyTorch's core functionalities.  The matrix's rows represent the predicted classes, and its columns represent the true classes.  The entry at row *i*, column *j* signifies the number of instances predicted as class *i* but actually belonging to class *j*.


**2. Code Examples with Commentary:**

**Example 1:  Basic Confusion Matrix using NumPy and Scikit-learn**

This example leverages NumPy for efficient array manipulation and Scikit-learn's `confusion_matrix` function for a straightforward implementation.  This is the approach I frequently employ for its simplicity and readability.

```python
import numpy as np
from sklearn.metrics import confusion_matrix

# Assume 'predictions' and 'labels' are NumPy arrays of predictions and ground truth labels respectively.
#  'predictions' and 'labels' should be 1D arrays of class indices.
predictions = np.array([0, 1, 1, 0, 2, 1, 0, 2, 2, 1])
labels = np.array([0, 1, 0, 0, 2, 1, 1, 2, 2, 0])

cm = confusion_matrix(labels, predictions)
print(cm)

#Further processing to calculate metrics:
accuracy = np.trace(cm) / np.sum(cm)
print(f"Accuracy: {accuracy}")
```

This code snippet directly calculates the confusion matrix and the accuracy.  Note the crucial step of ensuring `predictions` and `labels` are correctly formatted NumPy arrays.  Errors here frequently arise from mismatched data types or dimensions.


**Example 2: Handling One-Hot Encoded Outputs:**

In scenarios where your model outputs one-hot encoded vectors, the prediction needs to be converted to class indices first.  This was a common issue I faced when working with multi-label classification tasks.

```python
import numpy as np
from sklearn.metrics import confusion_matrix

#Assume 'predictions' is a NumPy array of one-hot encoded vectors, and 'labels' is an array of class indices
predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]])
labels = np.array([0, 1, 0, 0, 2, 1, 0, 2, 2, 1])

#Convert one-hot predictions to class indices
predicted_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(labels, predicted_classes)
print(cm)
```

This illustrates the necessary pre-processing step to convert one-hot encoded predictions into a format compatible with `confusion_matrix`.  The `np.argmax` function efficiently identifies the index of the maximum value in each row (representing the predicted class).


**Example 3:  Visualization with Matplotlib:**

While the numerical confusion matrix is informative, visual representation often enhances understanding.  Matplotlib is a powerful tool for this.

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

predictions = np.array([0, 1, 1, 0, 2, 1, 0, 2, 2, 1])
labels = np.array([0, 1, 0, 0, 2, 1, 1, 2, 2, 0])

cm = confusion_matrix(labels, predictions)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=np.arange(cm.shape[1]),
       yticklabels=np.arange(cm.shape[0]),
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
fig.tight_layout()
plt.show()

```

This example extends the previous ones by visualizing the confusion matrix using Matplotlib, enhancing interpretability.  The code includes annotations for clarity.


**3. Resource Recommendations:**

For a deeper understanding of confusion matrices and their interpretation, I recommend consulting standard machine learning textbooks.  Focus on sections dedicated to classification model evaluation.  Furthermore, the Scikit-learn documentation offers comprehensive explanations of its `confusion_matrix` function and related metrics.  Finally, review the NumPy documentation for efficient array manipulation techniques, crucial for effective matrix handling.  These resources provide a solid foundation for mastering this aspect of model evaluation.
