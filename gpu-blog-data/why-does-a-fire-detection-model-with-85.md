---
title: "Why does a fire detection model with 85% accuracy fail to detect any fires?"
date: "2025-01-30"
id: "why-does-a-fire-detection-model-with-85"
---
The crux of the issue lies not solely in the model's accuracy metric, but rather in the inherent biases present within the training data and the mismatch between the training and deployment environments.  My experience developing large-scale anomaly detection systems for industrial applications has shown that a seemingly high accuracy rate can be utterly misleading when the underlying data distribution is skewed or insufficiently representative of real-world scenarios.  An 85% accuracy rate on a dataset heavily weighted towards non-fire events will inevitably produce a system that struggles – or even fails – to identify the minority class, even if the model's internal performance metrics appear satisfactory.

The first key consideration is the class imbalance in the training data.  Fire detection inherently involves a significantly larger number of non-fire events compared to actual fires.  If the training data disproportionately reflects this imbalance, the model, even with a sophisticated algorithm, may become overly optimized for classifying non-fire events, sacrificing its ability to accurately detect the less frequent positive cases (fires).  This leads to a high overall accuracy score – because it correctly classifies the vast majority of non-fires – while simultaneously exhibiting a low recall for fire events.  In essence, the model learns to say "no fire" very well, but struggles to say "fire" even when it should.

Secondly, the features used in training are crucial.  If the model relies on features that are not reliable indicators of fire in real-world deployment scenarios, its performance will suffer drastically.  For example, relying solely on temperature readings without accounting for environmental factors (e.g., ambient temperature fluctuations, proximity to industrial heat sources) might lead to numerous false positives in a controlled environment during training, but frequent false negatives in real-world deployments with diverse and unpredictable conditions.  This highlights the critical importance of feature engineering and careful selection based on a thorough understanding of the problem domain and likely confounding variables.

Thirdly, the deployment environment itself can differ significantly from the training environment, leading to a phenomenon known as concept drift.  Factors such as lighting conditions, camera angles, the presence of smoke obscurants, and even subtle changes in the background noise can significantly impact the model's ability to generalize well and accurately detect fires.  If the training data doesn't account for this variability, the model will likely fail to perform as expected in real-world applications.

Let's illustrate this with some code examples using Python and scikit-learn.  These examples demonstrate potential issues and mitigation strategies.


**Example 1: Class Imbalance and its impact**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Highly imbalanced dataset
X = np.random.rand(1000, 5)  # Features
y = np.concatenate([np.zeros(900), np.ones(100)]) # Labels (90% non-fire, 10% fire)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

This example highlights how even a simple model can achieve seemingly high accuracy with a highly imbalanced dataset, while simultaneously having very low recall for the minority class (fires).  The confusion matrix will clearly show a large number of false negatives.


**Example 2: Addressing Class Imbalance with Resampling**

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model_resampled = LogisticRegression()
model_resampled.fit(X_train_resampled, y_train_resampled)

y_pred_resampled = model_resampled.predict(X_test)

print("Accuracy (Resampled):", accuracy_score(y_test, y_pred_resampled))
print("Recall (Resampled):", recall_score(y_test, y_pred_resampled))
print("Precision (Resampled):", precision_score(y_test, y_pred_resampled))
print("Confusion Matrix (Resampled):\n", confusion_matrix(y_test, y_pred_resampled))
```

Here, SMOTE (Synthetic Minority Over-sampling Technique) is used to oversample the minority class in the training data, creating synthetic samples to address the imbalance.  This often leads to improved recall for the fire events.


**Example 3: Feature Engineering and Model Selection**

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Simulate data with relevant and irrelevant features
data = {'temp': np.random.normal(25, 5, 1000),
        'smoke_density': np.random.normal(0.1, 0.05, 1000),
        'irrelevant_feature': np.random.rand(1000),
        'fire': np.concatenate([np.zeros(900), np.ones(100)])}
df = pd.DataFrame(data)

X = df[['temp', 'smoke_density', 'irrelevant_feature']]
y = df['fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestClassifier(random_state=42) #A more robust model
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

print("Accuracy (RandomForest):", accuracy_score(y_test, y_pred_rf))
print("Recall (RandomForest):", recall_score(y_test, y_pred_rf))
print("Precision (RandomForest):", precision_score(y_test, y_pred_rf))
print("Feature Importances:", model_rf.feature_importances_)
```

This example demonstrates the importance of feature selection.  A RandomForestClassifier is used, known for its ability to handle higher dimensional data and identify important features.  Examining the `feature_importances_` attribute reveals the relative importance of each feature, guiding further feature engineering or selection.  The irrelevant feature might be dropped in a refined model.

In conclusion, a high accuracy score alone is insufficient to guarantee robust performance in a real-world fire detection system.  Addressing class imbalance through resampling techniques, meticulous feature engineering,  selection of appropriate models, and a rigorous validation process encompassing diverse deployment scenarios are crucial to ensuring reliable fire detection.  Furthermore, continuous monitoring and retraining of the model to account for concept drift are vital for maintaining its effectiveness over time.  Consulting specialized literature on imbalanced classification, anomaly detection, and model validation is highly recommended for further understanding and improvement.
