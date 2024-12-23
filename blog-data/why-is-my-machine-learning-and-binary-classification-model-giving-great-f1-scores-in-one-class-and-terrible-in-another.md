---
title: "Why is my Machine Learning and Binary Classification model giving great f1 scores in one class and terrible in another?"
date: "2024-12-23"
id: "why-is-my-machine-learning-and-binary-classification-model-giving-great-f1-scores-in-one-class-and-terrible-in-another"
---

Okay, let's unpack this. I’ve seen this pattern emerge more times than I care to count in my years working with machine learning models, and it's almost never a simple fix. The symptom—great f1-score for one class and abysmal for another in a binary classification task—usually points to a combination of issues, often interlinked. It’s not necessarily indicative of an *inherently bad* model, but rather a model that hasn't been sufficiently challenged or refined against the underlying data dynamics.

Let's break down the likely culprits. Firstly, and most commonly, we are dealing with *class imbalance*. In many real-world scenarios, one class dominates the dataset by a wide margin. Think of fraud detection, where fraudulent transactions are significantly outnumbered by legitimate ones, or medical diagnoses, where the incidence of a particular disease may be low. If, let's say, 90% of your training data belongs to class ‘A’ and only 10% belongs to class ‘B’, a model that predicts everything as ‘A’ would still achieve a 90% accuracy. But, accuracy alone is a terrible metric in such cases. F1 score, being the harmonic mean of precision and recall, addresses this better, but not perfectly. If you look into the f1-score calculation, you see it still benefits from correctly predicting the dominant class, hence it can appear great for it and terrible for minority classes. The model gets rewarded for learning well what it already has in abundance in the dataset.

Secondly, we might be dealing with *feature imbalance*. Even if class labels are reasonably balanced, the representation of features might not be equally informative for all classes. If crucial feature attributes are present more abundantly for one class than another, your model will naturally learn to classify that class more effectively. It effectively means the signal to noise ratio is unequal across your classes and this can be as critical as the class imbalance issue. For example, if you are classifying different types of documents, you might find that documents from class 'A' feature specific keywords or text structures which makes the learning task simpler for the model.

Thirdly, and sometimes overlooked, the *choice of model and its regularization*. Some models may inherently favor learning from dominant classes, or require specific hyperparameter tuning to tackle skewed datasets. For example, if you're using a regularized logistic regression model, the regularization might be hindering the model's capacity to learn from the minority class because those smaller classes have weaker signals. The penalty for misclassifying the dominant class might be disproportionately influencing the model parameters.

Fourth, a more subtle issue is insufficient or misleading *data preprocessing*. If you applied heavy feature scaling or some kind of dimension reduction without considering individual class characteristics, you may have inadvertently made the minority class even harder to classify. The scale of the features might be more informative for the majority class than for the minority class, or vice versa, and a blanket scaling may dilute useful signals.

Okay, let's solidify this with some examples. Imagine I was tasked with building a model to identify rare species of plants from images a couple of years ago, and it performed great on common plants but poorly on the ones I needed. This wasn’t a toy problem; it had conservation implications. Here's the breakdown using python, with some code snippets:

**Example 1: Addressing Class Imbalance with Class Weights in Scikit-Learn**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

# Simulate imbalanced data
np.random.seed(42)
X = np.random.rand(1000, 10)
y = np.concatenate([np.zeros(900), np.ones(100)]) # 90% zeros, 10% ones

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model without class weights
model_no_weights = LogisticRegression(solver='liblinear')
model_no_weights.fit(X_train, y_train)
y_pred_no_weights = model_no_weights.predict(X_test)
f1_no_weights = f1_score(y_test, y_pred_no_weights, average=None) # F1 for each class

# Model with class weights
model_weights = LogisticRegression(solver='liblinear', class_weight='balanced')
model_weights.fit(X_train, y_train)
y_pred_weights = model_weights.predict(X_test)
f1_weights = f1_score(y_test, y_pred_weights, average=None)

print(f"F1 score without class weights: {f1_no_weights}")
print(f"F1 score with class weights: {f1_weights}")
```
In this snippet, notice how the F1 score for the minority class (class '1') significantly improves when we use `class_weight='balanced'` in Logistic Regression. This automatically adjusts the weight for each class, penalizing misclassifications in the minority class more heavily.

**Example 2: Investigating Feature Imbalance and Data Transformation**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Simulate imbalanced feature distributions
np.random.seed(42)
data = {
    'feature1': np.concatenate([np.random.normal(0, 1, 900), np.random.normal(3, 1, 100)]),
    'feature2': np.concatenate([np.random.normal(2, 0.5, 900), np.random.normal(0, 0.5, 100)]),
    'target': np.concatenate([np.zeros(900), np.ones(100)])
}
df = pd.DataFrame(data)
X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model without scaling
model_no_scale = RandomForestClassifier(random_state=42)
model_no_scale.fit(X_train, y_train)
y_pred_no_scale = model_no_scale.predict(X_test)
f1_no_scale = f1_score(y_test, y_pred_no_scale, average=None)

#Model with scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_with_scale = RandomForestClassifier(random_state=42)
model_with_scale.fit(X_train_scaled, y_train)
y_pred_scale = model_with_scale.predict(X_test_scaled)
f1_scale = f1_score(y_test, y_pred_scale, average=None)


print(f"F1 score without scaling: {f1_no_scale}")
print(f"F1 score with scaling: {f1_scale}")
```
Here, the `feature1` is more distinguishing for class '1' and `feature2` is more distinguishing for class '0'. Sometimes feature distributions or value ranges vary across classes, and a StandardScaler often helps, as it brings all the features into a similar distribution, and can significantly impact model performance across classes. This snippet highlights that scaling might be helpful for better feature contribution in both classes, even when no clear feature imbalance is evident initially.

**Example 3: Exploring Different Model Architectures and Regularization**

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np

np.random.seed(42)
X = np.random.rand(1000, 10)
y = np.concatenate([np.zeros(900), np.ones(100)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model with standard parameters
model_no_regul = MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=300)
model_no_regul.fit(X_train_scaled, y_train)
y_pred_no_regul = model_no_regul.predict(X_test_scaled)
f1_no_regul = f1_score(y_test, y_pred_no_regul, average=None)

# Model with L2 regularization
model_regul = MLPClassifier(hidden_layer_sizes=(100,), random_state=42, alpha=0.1, max_iter=300)
model_regul.fit(X_train_scaled, y_train)
y_pred_regul = model_regul.predict(X_test_scaled)
f1_regul = f1_score(y_test, y_pred_regul, average=None)

print(f"F1 score without L2 regularization: {f1_no_regul}")
print(f"F1 score with L2 regularization: {f1_regul}")
```

Here we see that, even with class balancing and feature scaling, our model can still underperform on the minority class. The `MLPClassifier` is a relatively complex model, and regularization, like the L2 regularization done in the code snippet, can help to better generalize across all classes, even when there are small sample sizes. Sometimes it also helps to change the model architecture to improve the model performance across all classes.

In conclusion, improving F1 scores across both classes often requires a multi-faceted approach. There isn’t a single magic bullet. Instead of thinking of it as 'fixing' your model, it's usually more helpful to view it as iteratively exploring and addressing the specific issues present within your data and the chosen approach. I strongly recommend diving into "Pattern Recognition and Machine Learning" by Christopher Bishop and "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman for a deep understanding of the underlying theory, as well as research papers on class imbalance and their proposed solutions. Remember, machine learning is as much about understanding the data as it is about selecting models.
