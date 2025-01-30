---
title: "How do I log a confusion matrix to Weights & Biases?"
date: "2025-01-30"
id: "how-do-i-log-a-confusion-matrix-to"
---
Logging confusion matrices to Weights & Biases (WandB) requires a nuanced understanding of how WandB handles custom data structures and the inherent characteristics of confusion matrices.  My experience in deploying and monitoring several deep learning models across diverse classification tasks underscores the importance of properly formatted data for effective visualization and analysis within the WandB platform.  The key is structuring the matrix in a way WandB's logging mechanisms can readily interpret and display.  It's not simply a matter of feeding it a NumPy array;  efficient logging demands careful consideration of data types and the intended visualization.

**1. Clear Explanation**

WandB primarily uses dictionaries and lists to ingest and display custom data.  A confusion matrix, fundamentally a 2D array representing predicted versus true class labels, needs to be transformed into a format WandB can easily parse and render as a visual representation. This usually involves converting the NumPy array into a dictionary where keys represent the predicted class labels, and values are dictionaries mapping each true class label to its corresponding count.  This hierarchical structure allows WandB to generate a clear and informative visualization, unlike directly logging the raw NumPy array which would result in a less intuitive display.  Furthermore,  metadata, such as the epoch number, needs to be included to facilitate tracking the matrix across training iterations.  Failing to integrate these metadata points severely limits the usefulness of the logged matrix within the context of the overall training process.

**2. Code Examples with Commentary**

The following examples demonstrate different ways to log confusion matrices to WandB, each tailored to specific needs and complexities.  I've personally employed all three methods during the course of my projects, selecting the appropriate method based on project needs and existing code structure.

**Example 1: Basic Confusion Matrix Logging**

This example provides a straightforward approach suitable for simpler scenarios.  It utilizes a helper function to convert the confusion matrix to a dictionary compatible with WandB.

```python
import wandb
import numpy as np

def format_confusion_matrix(cm, labels):
    """Converts a NumPy confusion matrix into a dictionary suitable for WandB logging."""
    formatted_cm = {}
    for i, predicted_label in enumerate(labels):
        formatted_cm[predicted_label] = {}
        for j, true_label in enumerate(labels):
            formatted_cm[predicted_label][true_label] = cm[i, j]
    return formatted_cm

# ... (Your model training code) ...

# Assuming 'cm' is your NumPy confusion matrix and 'labels' is a list of class labels
cm = np.array([[100, 10, 5], [5, 80, 15], [2, 8, 90]])
labels = ['Class A', 'Class B', 'Class C']

wandb.log({"confusion_matrix": format_confusion_matrix(cm, labels)})

wandb.finish()
```

This method prioritizes clarity and readability. The `format_confusion_matrix` function encapsulates the conversion logic, improving code maintainability. The direct logging of the formatted dictionary ensures easy integration with WandB's visualization capabilities.


**Example 2:  Logging with Metadata and Multiple Metrics**

This example expands on the basic approach by incorporating metadata such as the epoch number and logging additional metrics alongside the confusion matrix. This provides a richer context for evaluating model performance across multiple training epochs.

```python
import wandb
import numpy as np

wandb.init(project="confusion_matrix_example")

# ... (Your model training loop) ...

for epoch in range(10):
    # ... (Your model training and evaluation code) ...
    cm = np.array([[100, 10, 5], [5, 80, 15], [2, 8, 90]]) #Replace with your actual confusion matrix
    labels = ['Class A', 'Class B', 'Class C']
    accuracy = 0.85 # Replace with your actual accuracy

    formatted_cm = format_confusion_matrix(cm, labels)

    wandb.log({
        "epoch": epoch,
        "confusion_matrix": formatted_cm,
        "accuracy": accuracy
    })

wandb.finish()
```

This enhanced example demonstrates a more robust logging strategy, critical for comprehensive model analysis. The inclusion of the epoch number allows for tracking performance changes across the training process, directly within WandB's visualization tools.


**Example 3: Handling Imbalanced Datasets and Normalization**

This advanced example addresses the challenges posed by imbalanced datasets. It normalizes the confusion matrix to highlight the relative proportions of correctly and incorrectly classified instances, providing a more insightful perspective, particularly when dealing with class imbalances.

```python
import wandb
import numpy as np

def normalize_confusion_matrix(cm):
    """Normalizes a confusion matrix by row to show proportions."""
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm / row_sums

# ... (Your model training code) ...

cm = np.array([[100, 10, 5], [5, 80, 15], [2, 8, 90]])
labels = ['Class A', 'Class B', 'Class C']
normalized_cm = normalize_confusion_matrix(cm)
formatted_cm = format_confusion_matrix(normalized_cm, labels)

wandb.log({"normalized_confusion_matrix": formatted_cm})

wandb.finish()
```

This example shows a practical application of data preprocessing before logging, crucial for effective interpretation in datasets with unequal class distributions. Normalizing the confusion matrix offers a more balanced perspective on model performance, preventing misleading conclusions drawn from raw counts in imbalanced scenarios.


**3. Resource Recommendations**

The official Weights & Biases documentation provides comprehensive guidelines on logging custom data. Consulting the API reference for detailed information on supported data structures and formatting conventions is invaluable.  Furthermore, exploring example notebooks provided by WandB, focusing on those demonstrating custom metric logging, will offer practical insights.  Finally, understanding NumPy array manipulation is essential for preparing your confusion matrices for effective logging.  A strong grasp of Python dictionaries is equally important due to WandB's reliance on dictionary structures for visualization.
