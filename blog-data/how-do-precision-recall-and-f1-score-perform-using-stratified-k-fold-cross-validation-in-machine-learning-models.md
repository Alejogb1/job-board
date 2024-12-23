---
title: "How do precision, recall, and F1 score perform using stratified k-fold cross-validation in machine learning models?"
date: "2024-12-23"
id: "how-do-precision-recall-and-f1-score-perform-using-stratified-k-fold-cross-validation-in-machine-learning-models"
---

, let's unpack this. I've seen firsthand how the interplay of precision, recall, and f1 score gets a little nuanced, especially when you throw stratified k-fold cross-validation into the mix. This isn't just theory for me; a few years back, I was working on a fraud detection system where the class imbalance was absolutely brutal, and how we handled validation made a massive difference. It's easy to get tricked by simple accuracy, and that's precisely why these metrics become critical, particularly when you're dealing with skewed data sets.

So, let's start with the basics. Precision answers the question: *of all the instances the model predicted as positive, how many were actually positive?* Mathematically, it's true positives (tp) divided by the sum of true positives and false positives (fp). Recall, on the other hand, asks: *of all the actual positive instances, how many did the model correctly identify?* That's true positives divided by the sum of true positives and false negatives (fn). And the f1 score? Think of it as the harmonic mean of precision and recall. It balances the two, giving you a single metric that’s often more informative than looking at them in isolation. It's calculated as 2 * (precision * recall) / (precision + recall).

Now, why do we care about these, and why bring in stratified k-fold cross-validation? Well, simply dividing your data into training and testing sets can be incredibly misleading, especially if your data isn't perfectly uniform across categories. Say you’re classifying spam emails. If your test set happens to have disproportionately fewer spam emails than your training set, your model may perform well on the test data without being actually good at spotting real-world spam. This is where k-fold cross-validation comes in. It splits your dataset into *k* folds, trains your model on *k-1* folds, and tests on the held-out fold. This repeats *k* times with each fold becoming the test set, and the performance results are averaged.

But here's the kicker – basic k-fold can still stumble if you have imbalanced data. If a fold has very few instances of the minority class (in our example, the "fraudulent transaction" class, or the spam email class), your model might never learn to identify those examples properly. That's where *stratified* k-fold shines. Stratified k-fold ensures that each fold maintains the same class distribution as the whole dataset. For instance, if the positive class represents 10% of the whole data, each fold will have roughly 10% positive class examples. This way, your validation scores become more stable and representative of real-world performance.

Let’s see this in action. Let’s use Python with scikit-learn to illustrate the point.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Generate a highly imbalanced synthetic dataset
X = np.random.rand(1000, 10)
y = np.concatenate([np.zeros(900), np.ones(100)])

# Perform Stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LogisticRegression(solver='liblinear') # Using Logistic Regression for simplicity
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

print(f"Average Precision: {np.mean(precision_scores):.3f}")
print(f"Average Recall: {np.mean(recall_scores):.3f}")
print(f"Average F1 Score: {np.mean(f1_scores):.3f}")
```

This snippet generates a synthetic dataset with a 90/10 class imbalance. It then applies stratified k-fold cross-validation with five folds. The `precision_score`, `recall_score`, and `f1_score` are calculated for each fold and averaged at the end. Notice the use of `StratifiedKFold`, which is crucial here.

Now, let’s tweak it and see how it might differ with simple (non-stratified) k-fold, just to highlight the importance of stratification:

```python
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Same imbalanced synthetic dataset
X = np.random.rand(1000, 10)
y = np.concatenate([np.zeros(900), np.ones(100)])

# Perform standard k-fold cross-validation (not stratified)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
precision_scores_nonstrat = []
recall_scores_nonstrat = []
f1_scores_nonstrat = []


for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LogisticRegression(solver='liblinear') # Using Logistic Regression for simplicity
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    precision_scores_nonstrat.append(precision)
    recall_scores_nonstrat.append(recall)
    f1_scores_nonstrat.append(f1)


print(f"Non-Stratified Avg Precision: {np.mean(precision_scores_nonstrat):.3f}")
print(f"Non-Stratified Avg Recall: {np.mean(recall_scores_nonstrat):.3f}")
print(f"Non-Stratified Avg F1 Score: {np.mean(f1_scores_nonstrat):.3f}")
```

Comparing the output of these two code segments will often show slightly better and less volatile performance metrics with the stratified version, especially the recall score, where we’re focused on catching as many of the "true" positives (minority class) as possible.

Finally, a scenario where these metrics become absolutely vital is in situations where you care more about either precision or recall. Let's say you’re building a system to predict if a patient has a serious illness. A high recall would be more valuable to avoid missing a sick patient, even if it means some healthy patients get false alarms, which would then prompt further tests. On the flip side, in a system where false positives are expensive or time-consuming, then a high precision will be more important. Here's an adjustment to the previous stratified example, showing how it would look if you were focusing only on the recall of the minority class during the model selection phase:

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Same imbalanced synthetic dataset
X = np.random.rand(1000, 10)
y = np.concatenate([np.zeros(900), np.ones(100)])

# Perform Stratified k-fold cross-validation, focus on maximizing recall
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_recall_scores = []

for train_index, test_index in skf.split(X, y):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  # Added a preprocessing step for real scenarios
  pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))
    ])

  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)

  recall = recall_score(y_test, y_pred)
  all_recall_scores.append(recall)


print(f"Average Recall: {np.mean(all_recall_scores):.3f}")

```
Here, we explicitly prioritize recall during model evaluation. Note that we are also now incorporating standard scaling and setting class weights to “balanced” in order to further help the model learn from the minority class data and prevent overfitting.

In short, precision, recall, and the f1 score are your go-to metrics for evaluating models, especially in imbalanced data scenarios, and stratified k-fold is absolutely critical for robust validation of your model's performance, providing more realistic and stable performance estimates. These aren't mere academic exercises but tools I've relied on repeatedly in real-world scenarios. If you want to delve deeper into the theoretical underpinnings, I’d recommend looking into "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman, as well as the scikit-learn documentation which provides very detailed usage guidelines and explanation on the functions that are involved here. They provide a good foundation for understanding the math and rationale behind these methods.
