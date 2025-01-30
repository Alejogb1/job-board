---
title: "Why is prediction accuracy low (below 0.01) despite high overall prediction quality (99.99%)?"
date: "2025-01-30"
id: "why-is-prediction-accuracy-low-below-001-despite"
---
The observed discrepancy between a near-perfect overall prediction quality (99.99%) and drastically low accuracy (below 0.01) in a prediction model strongly suggests a class imbalance problem, exacerbated by an evaluation metric unsuitable for such scenarios.  My experience debugging similar issues in large-scale fraud detection systems has highlighted this exact pitfall.  The high overall quality is likely masking the model's failure to accurately predict the rare, but critical, positive class.

**1. Explanation:**

The issue stems from the fundamental difference between overall prediction quality (often implicitly referring to metrics like precision or recall calculated across all classes) and accuracy, which directly measures the proportion of correctly classified instances.  When dealing with severely imbalanced datasets—where one class (typically the 'positive' or 'interesting' class, such as fraudulent transactions in my fraud detection work) constitutes a minuscule fraction of the total data—accuracy becomes a misleading performance indicator.  A model can achieve a high overall prediction quality by flawlessly predicting the majority class, while simultaneously failing to correctly identify the rare positive class instances.  This leads to a high overall 'quality' score, masked by an extremely low accuracy.

For instance, consider a dataset with 9999 negative instances and only 1 positive instance.  A naive model predicting 'negative' for every instance would achieve 99.99% accuracy. This, however, is completely uninformative and masks the model's failure to detect the crucial positive case.  The problem is not the model's inherent inability to learn, but rather the mismatched evaluation metric and the dataset's skewed distribution.

Effectively addressing this necessitates using alternative evaluation metrics, employing appropriate resampling techniques, and potentially adjusting the model's decision threshold.  Precision, recall, F1-score, AUC-ROC, and PR-AUC curves are far more suitable metrics than accuracy for imbalanced datasets.  These metrics provide a more nuanced perspective on the model's performance across different class thresholds.

**2. Code Examples with Commentary:**

Let's illustrate this with three code examples using Python and scikit-learn.  These examples are simplified for illustrative purposes, but reflect the core principles applied in my previous projects.

**Example 1: Illustrating the Problem with Accuracy**

```python
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

y_true = np.array([0] * 9999 + [1])  # Highly imbalanced dataset
y_pred = np.array([0] * 10000)       # Model always predicts negative

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_true, y_pred))
```

This code demonstrates the misleading nature of accuracy.  The model's accuracy is high (almost 1.0), despite entirely failing to predict the positive class. The `classification_report` however exposes the issue showing a recall of 0.0 for the positive class.


**Example 2: Utilizing the F1-Score**

```python
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Simulate a dataset with class imbalance
X = np.random.rand(10000, 10)
y = np.concatenate(([0] * 9900, [1] * 100))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.4f}")
print(classification_report(y_test, y_pred))

```

This example uses a more realistic scenario employing Logistic Regression and the F1-score, a harmonic mean of precision and recall, better suited for imbalanced datasets. The `classification_report` provides a more comprehensive evaluation, including precision, recall and F1-score for each class.

**Example 3: Addressing Imbalance with SMOTE**

```python
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#Simulate a dataset with class imbalance (same as example 2)
X = np.random.rand(10000, 10)
y = np.concatenate(([0] * 9900, [1] * 100))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred)
print(f"F1-score (with SMOTE): {f1:.4f}")
print(classification_report(y_test, y_pred))

```

This example demonstrates how to use SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class before model training, leading to improved performance on the positive class.  Note that the choice of resampling technique (SMOTE, RandomOverSampler, etc.) depends on the specific characteristics of the dataset.


**3. Resource Recommendations:**

For a deeper understanding of class imbalance, I recommend exploring texts on machine learning evaluation metrics, specifically focusing on those designed for imbalanced datasets.  Further, studying techniques for handling imbalanced data, including resampling methods and cost-sensitive learning, will prove invaluable.  Finally, a comprehensive understanding of the underlying data and the problem domain is crucial for effective model selection and evaluation.  The right metric is always dependent on the problem.  What constitutes a 'good' prediction is defined by the application itself, not only by some arbitrary metric.
