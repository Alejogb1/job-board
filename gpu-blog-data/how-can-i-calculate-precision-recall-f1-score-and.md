---
title: "How can I calculate precision, recall, F1-score, and confusion matrix for a multiclass classification problem using MobileNet?"
date: "2025-01-30"
id: "how-can-i-calculate-precision-recall-f1-score-and"
---
Successfully evaluating a multiclass classification model like one based on MobileNet necessitates a nuanced understanding of performance metrics beyond simple accuracy. Accuracy, while intuitive, can be misleading when class distributions are imbalanced. Instead, precision, recall, F1-score, and the confusion matrix provide a granular view of a model’s strengths and weaknesses across each class. My experience deploying such models in real-time image recognition systems highlights the importance of these metrics for debugging and refinement.

**Precision** addresses the proportion of correctly predicted positives out of all predicted positives. In the multiclass scenario, precision is calculated for each class individually. It essentially asks, "Of all the examples I predicted as belonging to this class, how many were actually in this class?". High precision implies a low rate of false positives. The formula for precision for class *i* is:

`Precision_i = True Positives_i / (True Positives_i + False Positives_i)`

**Recall**, or sensitivity, conversely measures the proportion of correctly predicted positives out of all actual positives. It asks, "Of all the examples that actually belong to this class, how many did I correctly predict?". A high recall indicates a low rate of false negatives. The formula for recall for class *i* is:

`Recall_i = True Positives_i / (True Positives_i + False Negatives_i)`

**F1-score** is the harmonic mean of precision and recall, providing a single metric that balances both. It is especially useful when precision and recall are competing concerns. A high F1-score demonstrates that the model is both precise and has good recall, signifying a strong performance for the class. The formula for the F1-score for class *i* is:

`F1-score_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)`

The **Confusion Matrix** provides a comprehensive breakdown of the model’s predictions, visualizing the true and predicted class labels. It is an N x N matrix, where N is the number of classes. The diagonal entries represent the correct classifications, and off-diagonal entries represent misclassifications. This matrix is a crucial tool for identifying specific areas of model confusion between classes. It is often represented visually, allowing for immediate pattern identification.

Let’s illustrate how these metrics can be practically calculated using Python, specifically leveraging libraries like NumPy and scikit-learn, common in my workflows when working with MobileNet’s output. Suppose we have a pre-trained MobileNet model that classifies images into three classes: “cat”, “dog”, and “bird”. We’ll assume the model produces probabilities for each class for each input, and we then convert these to predicted labels by taking the argmax.

**Example 1: Calculation of per-class metrics**

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Assume model predictions (e.g., argmax of MobileNet's output) and true labels
true_labels = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1])  # 0: cat, 1: dog, 2: bird
predicted_labels = np.array([0, 1, 2, 1, 0, 2, 0, 1, 1])

# Calculate precision, recall, and f1-score for each class
precision_per_class = precision_score(true_labels, predicted_labels, average=None)
recall_per_class = recall_score(true_labels, predicted_labels, average=None)
f1_per_class = f1_score(true_labels, predicted_labels, average=None)

print("Precision per class:", precision_per_class)
print("Recall per class:", recall_per_class)
print("F1-score per class:", f1_per_class)
```

In this example, `precision_score`, `recall_score`, and `f1_score` from scikit-learn are used with `average=None` to compute these metrics for each class separately. The output will be arrays corresponding to “cat,” “dog,” and “bird,” respectively. When working with more classes, these calculations naturally scale.

**Example 2: Calculation of overall metrics using weighted average**

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Assume model predictions and true labels as in Example 1
true_labels = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1])
predicted_labels = np.array([0, 1, 2, 1, 0, 2, 0, 1, 1])


# Calculate precision, recall, and f1-score with weighted average
precision_weighted = precision_score(true_labels, predicted_labels, average='weighted')
recall_weighted = recall_score(true_labels, predicted_labels, average='weighted')
f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')


print("Weighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
print("Weighted F1-score:", f1_weighted)
```

In this second example, I show how to use `average='weighted'` within scikit-learn to compute overall metrics. The weighted average accounts for class imbalances by weighting the per-class scores based on the number of instances for each class. This overall weighted average gives a more balanced perspective on the model’s performance across all classes than simple accuracy, especially in the case of unbalanced data.

**Example 3: Confusion Matrix Computation and Visualization**

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Assume model predictions and true labels as in Example 1
true_labels = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1])
predicted_labels = np.array([0, 1, 2, 1, 0, 2, 0, 1, 1])


# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["cat", "dog", "bird"],
            yticklabels=["cat", "dog", "bird"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

Here, the `confusion_matrix` function generates the matrix, and then Seaborn and Matplotlib are used for a visual representation, which can help identify specific confusions. For example, the heat map will clearly display if the model frequently misclassifies dogs as cats, or vice versa.  I often rely on this visual to intuitively grasp the areas where the model is weakest, guiding focused model improvement strategies. The annotated cells provide the absolute numbers of instances per combination of true and predicted labels.

In practice, I would leverage established libraries within deep learning frameworks such as TensorFlow or PyTorch to handle the actual inference from the MobileNet model. I've found that those libraries are very efficient for calculating the predicted probabilities from the model, leaving me to focus on the application of the evaluation metrics detailed above. These metrics and their corresponding analysis tools allow a much deeper understanding of how the model performs on specific classes, leading to targeted improvements like adding more data for less well-performing classes, or adjusting the model architecture to handle specific challenges.

For further learning, research on the concepts of statistical classification, performance evaluation metrics for machine learning models, and practical examples of building a multiclass classification pipeline with deep learning frameworks are crucial. Additionally, focusing on the details of imbalanced data handling and strategies to mitigate their impact on metric calculations is valuable. Exploration of more specialized metrics for multiclass classifications beyond the basics can provide even more detailed analysis. The ability to understand and calculate these metrics is not just a theoretical exercise; it is a critical skill for anyone working with classification problems using complex models like MobileNet.
