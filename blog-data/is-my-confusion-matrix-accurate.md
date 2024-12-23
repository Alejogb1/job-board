---
title: "Is my confusion matrix accurate?"
date: "2024-12-23"
id: "is-my-confusion-matrix-accurate"
---

Let's unpack this confusion matrix quandary. I've seen more than my fair share of these, and “accuracy” isn’t quite the straightforward answer it sometimes seems to be at first glance. It’s a critical point, so let’s go beyond the initial calculation and really explore what a confusion matrix tells you, and more importantly, what it *doesn't* tell you, and how it can be misleading. It's a classic situation where the devil is often in the details, and that’s particularly true with evaluation metrics.

First off, a confusion matrix, at its core, is a table that visualizes the performance of a classification model. It lays out how well your model distinguishes between different classes by detailing the counts of true positives (tp), true negatives (tn), false positives (fp), and false negatives (fn). Think of it as the raw scorecard, providing a ground truth comparison against your predictions. The accuracy derived from this matrix is simply (tp + tn) / (tp + tn + fp + fn), which calculates the overall correct predictions out of all predictions made. This *seems* straightforward, right?

Not quite. I remember working on a fraud detection model a few years back, where the dataset was heavily imbalanced - less than 1% of transactions were fraudulent. We achieved a fantastic accuracy of 99.5%. The initial thought was, “great job everyone, time for tea!” But digging a little deeper into the confusion matrix, we found that our model was doing a superb job of predicting legitimate transactions, but failing horribly on identifying the fraudulent ones (high fp, very low tp, high fn). This highlights the problem: high accuracy doesn’t necessarily imply a model performs well in practice, especially when dealing with imbalanced data. That's where simply relying on a single accuracy score fails.

Let me break down three examples to make this concrete:

**Example 1: A Balanced Dataset**

Let's imagine we're classifying images of cats and dogs, with a near perfect 50/50 split.

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # 0: cat, 1: dog
y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1])

conf_mat = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("Confusion Matrix:\n", conf_mat)
print("Accuracy:", accuracy)

# Expected Output:
# Confusion Matrix:
#  [[4 1]
#  [1 4]]
# Accuracy: 0.8
```
In this example, we have 4 true positives (correctly predicting dogs), 4 true negatives (correctly predicting cats), 1 false positive (incorrectly predicting a cat as a dog) and 1 false negative (incorrectly predicting a dog as a cat). The overall accuracy of 0.8 (or 80%) reflects a reasonably good performance here. This example highlights a scenario where accuracy is a useful metric.

**Example 2: Imbalanced Data - Fraud Detection**

Now let’s return to the fraud detection example. Assume out of 100 transactions, only 5 were fraudulent. Let’s see what a confusion matrix might look like when the model performs poorly on the minority class (fraud):

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

y_true = np.array([0] * 95 + [1] * 5)  # 0: not fraud, 1: fraud
y_pred = np.array([0] * 94 + [1] + [0] * 4 + [1]) # The model failed to detect most fraud
conf_mat = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("Confusion Matrix:\n", conf_mat)
print("Accuracy:", accuracy)

# Expected Output:
# Confusion Matrix:
#  [[94 1]
#  [ 4 1]]
# Accuracy: 0.95
```

Here, the accuracy is 95% but the model has significant issues with identifying fraud. We had 94 true negatives, one false positive (incorrectly flagged as fraud) and one true positive (correctly classified fraud) while missing four actual fraudulent instances. The high accuracy masks the poor performance on the critical class (fraud). In such situations, we need to look beyond basic accuracy. We’d typically focus on measures like precision, recall, and the f1-score or the auc-roc curve.

**Example 3: Imbalanced Data - Disease Detection**

Let’s consider another imbalanced scenario: detecting a rare disease. Let's say 990 people are healthy, and 10 people have the disease:

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

y_true = np.array([0] * 990 + [1] * 10)  # 0: healthy, 1: diseased
y_pred = np.array([0] * 990 + [0] * 10) # The model completely failed to identify the disease
conf_mat = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("Confusion Matrix:\n", conf_mat)
print("Accuracy:", accuracy)

# Expected output
# Confusion Matrix:
#  [[990   0]
#  [ 10   0]]
# Accuracy: 0.99
```

In this case, our model predicted everyone to be healthy, and yet our accuracy is 99%. It’s a fantastic accuracy score but useless because it didn't identify any positive cases. This demonstrates the danger of blindly trusting accuracy in the presence of imbalanced classes. We would need measures like sensitivity (recall) and specificity to understand what is truly happening.

So, is your confusion matrix accurate? It’s a qualified yes. The numbers it contains are, presumably, accurate counts based on the predictions and ground truth. However, the real question isn’t about the numbers being *correct*, but rather whether the derived *accuracy* is a suitable metric for your problem. It is a good starting point to help build a baseline, but as you can see, it rarely tells the entire story. It's merely a building block. It's important to contextualize this metric, especially when dealing with imbalanced datasets.

To understand performance more comprehensively, I highly recommend delving deeper into metrics like precision, recall, f1-score and the roc-auc curve; these are discussed extensively in textbooks like "Pattern Recognition and Machine Learning" by Christopher M. Bishop, or in the scikit-learn documentation for metrics which is invaluable. Specifically for imbalanced datasets, have a look at the research paper 'Learning from Imbalanced Data' by He and Garcia. For those looking into metrics with consideration of costs I would recommend the book 'The Elements of Statistical Learning' by Hastie, Tibshirani and Friedman. These resources are essential to truly understand and evaluate a model's performance beyond accuracy. It's important to look at things from all angles and pick the right metric or metrics that speak to the specific needs of your problem, and that usually means going beyond simple accuracy.
