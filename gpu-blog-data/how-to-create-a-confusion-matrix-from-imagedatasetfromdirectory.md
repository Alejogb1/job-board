---
title: "How to create a confusion matrix from `image_dataset_from_directory` in TensorFlow 2.x?"
date: "2025-01-30"
id: "how-to-create-a-confusion-matrix-from-imagedatasetfromdirectory"
---
Generating a confusion matrix directly from a `tf.keras.utils.image_dataset_from_directory` dataset requires careful consideration of the workflow.  The core challenge lies in effectively separating the prediction phase from the data loading and preprocessing steps inherent in the `image_dataset_from_directory` function.  My experience optimizing image classification models for large-scale medical imaging datasets highlighted this specific hurdle.  The naive approach of directly applying a model's `predict` function to the dataset generator often leads to performance bottlenecks and memory issues, particularly with high-resolution images or substantial dataset sizes.  Therefore, a more structured approach, involving explicit data iteration and batch processing, is necessary.


**1. A Structured Approach:**

The optimal method involves three distinct stages: data preparation, model prediction, and confusion matrix construction.  First, we prepare the dataset using `image_dataset_from_directory`, ensuring proper batching and preprocessing. Second, we iterate through the prepared dataset, making predictions for each batch. Finally, we aggregate these predictions to create the confusion matrix. This prevents overwhelming memory with the entire dataset and allows for efficient batch-wise processing.

**2. Code Examples:**

The following examples demonstrate this workflow using TensorFlow 2.x, assuming a pre-trained model `model` and a dataset created via `image_dataset_from_directory`.  Error handling and specific preprocessing steps (like image resizing or augmentation) are omitted for brevity but are crucial in a production environment based on my past experience.  Remember to adjust paths and parameters according to your specific dataset structure.

**Example 1: Basic Confusion Matrix Generation**

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# Assume 'model' is a pre-trained model and 'test_ds' is the dataset from image_dataset_from_directory
test_ds = tf.keras.utils.image_dataset_from_directory(
    '/path/to/test/images',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32
)

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

This example directly uses `sklearn.metrics.confusion_matrix` for simplicity. It iterates through the test dataset, predicts the class for each batch of images, and appends the true and predicted labels to respective lists.  Finally, it computes the confusion matrix using scikit-learn.  Note the use of `np.argmax` to convert one-hot encoded labels to class indices.


**Example 2: Handling Class Imbalance with Weighted Confusion Matrix**

In scenarios with class imbalance, a weighted confusion matrix provides a more nuanced evaluation.  This involves assigning weights based on class frequencies.

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter

# ... (Dataset loading and model prediction as in Example 1) ...

class_counts = Counter(y_true)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

weighted_cm = np.zeros_like(cm, dtype=float)
for i, true_label in enumerate(y_true):
    weighted_cm[true_label, y_pred[i]] += class_weights[true_label]

print(weighted_cm)
```

This example calculates class weights inversely proportional to their frequencies. The weighted confusion matrix then incorporates these weights during its computation, providing a more representative evaluation in imbalanced datasets.  This addresses a common issue I encountered while working with medical image datasets where certain pathologies were underrepresented.


**Example 3:  Visualization with Matplotlib**

Visualizing the confusion matrix enhances interpretability.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ... (Dataset loading and model prediction as in Example 1) ...

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names) # Assuming class_names is a list of class labels
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

This employs `seaborn` and `matplotlib` to generate a heatmap visualization of the confusion matrix. The `annot=True` argument displays the counts in each cell, enhancing readability.  Using a suitable colormap and clear labels greatly aids in understanding the model's performance across different classes. This step is essential for effective communication of results, a skill critical in my role.



**3. Resource Recommendations:**

For a deeper understanding of confusion matrices and their interpretation, I recommend exploring relevant chapters in standard machine learning textbooks.  Furthermore, the official TensorFlow documentation and tutorials on model evaluation are invaluable resources.  Familiarize yourself with different metrics derived from the confusion matrix, such as precision, recall, F1-score, and accuracy, to gain a comprehensive perspective on model performance.  Studying the documentation for `sklearn.metrics` will be beneficial for understanding the various functionalities offered by this library. Finally, actively engaging in online forums and communities focused on machine learning and deep learning, similar to this one, can provide further insight and support.
