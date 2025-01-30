---
title: "How can a confusion matrix be generated from a pre-trained TensorFlow model?"
date: "2025-01-30"
id: "how-can-a-confusion-matrix-be-generated-from"
---
TensorFlow's high-level API, particularly when employing Keras, often obscures the process of manually extracting predictions and ground truth labels required for creating a confusion matrix. My experience developing image classifiers for automated inspection systems revealed that while the model training pipeline often provides overall metrics like accuracy, precision, and recall, a fine-grained analysis via a confusion matrix is crucial for understanding the specific misclassification patterns within a model. These misclassifications frequently reveal subtle biases in the training dataset or weaknesses in the model architecture that might be missed by aggregated metrics alone. This detailed matrix visualization enables targeted model refinement and data augmentation strategies.

To generate a confusion matrix from a pre-trained TensorFlow model, I find that the fundamental process involves the following steps. Firstly, obtaining predictions on a designated test or validation dataset. Secondly, extracting the corresponding true labels associated with this dataset. Lastly, utilizing a suitable function, usually provided by libraries such as scikit-learn, to create the confusion matrix based on the predicted and true labels. This often necessitates bypassing higher-level Keras evaluation mechanisms which typically present summary statistics rather than individual prediction outputs.

The following Python code examples, leveraging TensorFlow and scikit-learn, demonstrate this process, along with necessary adjustments for different data formats.

**Example 1: Processing NumPy Array Data**

Assume we have a pre-trained model `model`, a test dataset `x_test` (a NumPy array of features), and corresponding labels `y_test` (a NumPy array of one-hot encoded labels).

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# Assume model, x_test, and y_test are already loaded/defined

# Load the saved model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Generate predictions
y_pred_probs = model.predict(x_test)

# Convert prediction probabilities to predicted class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert one-hot encoded true labels to class labels
y_true = np.argmax(y_test, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)
```

Here, `model.predict(x_test)` outputs probabilities for each class. We use `np.argmax` to select the class with the highest probability, transforming prediction probabilities into class labels. The same `np.argmax` is applied to `y_test` to convert from one-hot encoding into true class labels. Finally, scikit-learn's `confusion_matrix` function is utilized to construct the confusion matrix which displays the number of samples that were classified correctly and misclassified for each class. This resulting `conf_matrix` array directly quantifies misclassification patterns.

**Example 2: Processing TensorFlow Dataset API Data**

If the data is processed using the TensorFlow Dataset API, specific adjustments are required due to the lazy evaluation and batched structure. Suppose we have a `test_dataset`.

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# Assume model and test_dataset are already loaded/defined

# Load the saved model
model = tf.keras.models.load_model('path_to_your_model.h5')

y_true = []
y_pred = []

for images, labels in test_dataset:
    # Generate predictions in batches
    y_pred_probs_batch = model.predict(images)
    y_pred_batch = np.argmax(y_pred_probs_batch, axis=1)
    y_true_batch = np.argmax(labels.numpy(), axis=1)

    # Extend lists with the predictions and true labels from this batch
    y_true.extend(y_true_batch)
    y_pred.extend(y_pred_batch)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

```
In this example, I iterate through the batches in `test_dataset`. Inside the loop, I extract predictions on the current batch of images and the corresponding batch of true labels using similar logic as the previous example. The `labels` are in the form of a TensorFlow tensor so I convert it to numpy array using `.numpy()` before applying `np.argmax` to it. The predicted and true class labels from the batch are appended to accumulating lists `y_true` and `y_pred`. The process is carried out for all the batches of data. Once finished, a single confusion matrix is generated from the full lists of `y_true` and `y_pred`. Note that the `.numpy()` method is necessary to extract raw data from TensorFlow tensors.

**Example 3: Handling Categorical Data and Class Names**

If dealing with categorical data represented as string labels, I typically map them to numerical indices for use in the confusion matrix, and then generate a human-readable plot later, linking numerical indices back to original class labels using a mapping dictionary or a list of class names. Let's assume the `test_dataset` yields string labels.

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Assume model and test_dataset are already loaded/defined

# Load the saved model
model = tf.keras.models.load_model('path_to_your_model.h5')


class_names = ['class_a', 'class_b', 'class_c'] # Example class names

y_true = []
y_pred = []


for images, labels in test_dataset:

    y_pred_probs_batch = model.predict(images)
    y_pred_batch = np.argmax(y_pred_probs_batch, axis=1)
    y_true_batch = [class_names.index(label.decode()) for label in labels.numpy()]

    y_true.extend(y_true_batch)
    y_pred.extend(y_pred_batch)

conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```

This example iterates through the dataset as before, but now the true labels are string encoded. I decode each label to string using `.decode()` and obtain its corresponding numerical index using `class_names.index()`, using a predefined list of class names. The confusion matrix generation remains the same as before, but the visualization stage is added which plots the confusion matrix using `seaborn`'s heatmap functionality, along with the string class names for each axis and appropriate annotations. This step makes interpretation significantly easier.

The challenges encountered during the generation of confusion matrices from pre-trained models often stem from inconsistencies in data formats or a lack of clear understanding of the model’s output structure. Thorough inspection of the model’s output using print statements or debugging tools is always beneficial. The need to convert probabilities into class labels and decode potentially categorical labels are common sources of error. The three examples presented demonstrate the typical scenarios encountered in my projects.

For additional knowledge, I would recommend exploring the official scikit-learn documentation for the `confusion_matrix` function. Also the TensorFlow guide on working with `tf.data` is essential for understanding and processing input data through the TensorFlow API. Additionally, exploring visualization libraries like seaborn and matplotlib provides the tools for creating informative graphical representations of these matrices, aiding in the diagnostic process.
