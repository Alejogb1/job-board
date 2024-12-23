---
title: "How accurate are model predictions, visualized by confusion matrix and plot?"
date: "2024-12-23"
id: "how-accurate-are-model-predictions-visualized-by-confusion-matrix-and-plot"
---

Let's unpack the accuracy of model predictions, focusing on the insights we can glean from confusion matrices and plots. It's something I've spent a fair bit of time navigating, especially during a previous stint working on a large-scale fraud detection system. We were dealing with incredibly imbalanced datasets, and the seemingly straightforward performance metrics could often mask underlying issues. The journey made me appreciate the nuances of these visualizations beyond the headline numbers.

First off, let’s talk about confusion matrices. These aren't just pretty grids; they're fundamental for understanding the performance of classification models. In their simplest form, they show the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). The columns usually represent the predicted classes, while the rows represent the actual classes. A diagonal heavy with large values signals good performance, sure, but the deviations from this ideal, the off-diagonal elements, are where the true insights lie. For instance, in our fraud system, the matrix wasn't symmetrical; we had far more false negatives (fraud cases missed) than false positives (legitimate transactions flagged as fraudulent). This meant our model was overly cautious and not aggressive enough in identifying fraud, leading to significant financial losses despite respectable overall accuracy.

We might, for example, observe a high accuracy in a binary classification problem, say 95%. But if the positive class (e.g., the fraudulent transactions) makes up only 1% of the total data, the model could be achieving this by simply classifying every instance as the negative class. The confusion matrix, however, would immediately reveal a large number of false negatives and consequently a very low recall. This discrepancy underscores that the overall accuracy metric, although helpful, can be entirely misleading. It's important to consider precision, recall, and the f1-score, each derived from confusion matrix components.

To understand this practically, consider the following python snippet using sklearn:

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Actual and predicted labels
y_true = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Generate a classification report
print("\nClassification Report:\n", classification_report(y_true, y_pred))
```

This simple snippet illustrates how to generate a confusion matrix and a classification report. The output reveals that while accuracy might appear reasonable given the limited data here, it is the classification report that gives insight into performance for each class.

Next, let’s delve into the visualisations beyond just the confusion matrix, such as precision-recall curves and receiver operating characteristic (ROC) curves. These plot a model's performance across different thresholds. The ROC curve plots the true positive rate (recall) against the false positive rate, effectively showing how well a classifier can distinguish between classes. In contrast, the precision-recall curve focuses on the trade-off between these metrics, which is often more informative when dealing with imbalanced datasets, like the fraud dataset that I worked with. The Area Under the Curve (AUC), calculated for both ROC and PR curves, gives us a scalar measure of the classifier's ability to rank positive cases higher than negative cases. For instance, an AUC of 1 signifies perfect prediction, and a value of 0.5 indicates a performance no better than random guessing.

Back to my fraud project, a high ROC-AUC didn't necessarily mean the model was good at finding fraud; the precision-recall curve often gave us the true picture, revealing areas where our model had very high recall but abysmal precision. This led to flagging a significant number of non-fraudulent transactions and causing unnecessary disruptions for our users.

We then explored using different thresholds, as the default 0.5 in many algorithms may not be optimal. This adjustment, visualized by the ROC and PR curves, allowed us to find a sweet spot for the specific business problem, which involved a trade-off between capturing most fraud and minimizing false alarms.

Let's move to coding an example which illustrates generating ROC and precision-recall curves:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression

# Example binary classification data
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)
model = LogisticRegression()
model.fit(X,y)
y_score = model.predict_proba(X)[:, 1] # Predict probabilities, for use in curve generation

fpr, tpr, _ = roc_curve(y, y_score) # Generate the data for ROC curves
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y, y_score) # Generate the data for PR curves
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
```
In this snippet, we generate the ROC and Precision-Recall Curves for a given binary classification. This allows us to assess how good our model is performing, and to compare models to one another.

Lastly, when assessing the accuracy, always remember to validate these metrics on a hold-out dataset and ensure that the dataset used reflects the actual distribution observed in production. Bias and variance issues often become apparent during this validation step. Cross-validation, which repeatedly splits data into training and validation sets, can often give a more robust estimate of the model's performance. It’s also worth considering the confidence intervals around these metrics, as metrics derived from small sample size can be highly variable and sometimes not dependable.

One practical technique I have used is visualizing prediction distributions via histograms. For example, let's assume that we are using a classifier that outputs a probability as a prediction. If the probabilities are highly concentrated at the top end of the spectrum or at the bottom, it can indicate issues with your training setup or algorithm selection. By plotting the prediction probabilities in a histogram, you can often see patterns that may indicate the model is performing poorly or has been trained sub-optimally.

Consider this example using python:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Example binary classification data
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)
model = LogisticRegression()
model.fit(X,y)
y_score = model.predict_proba(X)[:, 1] # Predict probabilities, for use in histogram

plt.figure(figsize=(8, 6))
plt.hist(y_score, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.show()
```

Here, we generate a simple histogram to visualise the distribution of predicted probabilities of a given logistic regression model, which provides visual information that could highlight problems.

In essence, evaluating model accuracy isn't just about obtaining a single number; it's about understanding the full picture. Confusion matrices and related plots are essential tools that reveal not only how well a model is performing, but also *where* it excels and *where* it falters, facilitating the process of making informed decisions to improve the system. For those diving deeper, I strongly recommend “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman and “Pattern Recognition and Machine Learning” by Bishop. These resources provide a strong theoretical foundation and can guide you beyond simple metrics towards understanding the underlying assumptions of these performance measures. They definitely helped me navigate similar problems in my career.
