---
title: "How to Plot about Machine Learning Variable Importance?"
date: "2024-12-15"
id: "how-to-plot-about-machine-learning-variable-importance"
---

alright, so you’re looking to visualize feature importance from a machine learning model, i get that. it's a common need when you're trying to understand *why* your model is making the decisions it is. i've been there, staring at a bunch of numbers and thinking, "there *has* to be a better way to see this".

over the years, i've wrestled (oh i tried, i have to say) with different approaches. back in my early days, i remember working on a fraud detection project. we had this gigantic dataset with like a hundred different features. the model was working reasonably well but nobody on my team understood what was driving its predictions. the feature importance output? just a giant text dump. we spent hours manually trying to sort and find patterns. i almost went cross-eyed. that was a lesson in the necessity of visualization tools, definitely a bad time.

essentially, we want to graphically represent which variables the model deems to be most influential in its decision-making process. this helps not just with understanding the model but also with feature selection, meaning removing the unnecessary features, which can really improve training time and model performance. there are several methods, of which i'm going to highlight my favorites:

first and foremost, if you have a model that provides attribute `feature_importances_` (like many tree based models in scikit-learn), a simple bar plot is usually the go-to. it's straightforward and effective for a basic overview.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance_basic(model, feature_names, top_n=10):
  """plots top n feature importances from a scikit-learn model."""
  importances = model.feature_importances_
  indices = np.argsort(importances)[::-1]
  top_indices = indices[:top_n]
  top_feature_names = [feature_names[i] for i in top_indices]
  top_importances = importances[top_indices]
  plt.figure(figsize=(10, 6))
  plt.barh(top_feature_names, top_importances)
  plt.xlabel("Importance")
  plt.title("Feature Importance")
  plt.gca().invert_yaxis()
  plt.show()

# example usage with a scikit-learn random forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]
model = RandomForestClassifier(random_state=42).fit(X, y)

plot_feature_importance_basic(model, feature_names, top_n=7)

```

this snippet provides a function that will pull feature importance from models that has it and gives a basic horizontal bar plot of the top n most important features. the `invert_yaxis()` call is just to make most important at the top of the plot, something i learned the hard way, after spending hours reading graphs upside down, haha. if you have a model that doesn’t directly have feature importances you will have to calculate them which leads to the second method, permutation importance.

permutation importance, it doesn't rely on any internal model attributes, instead it measures importance by observing how the model's performance changes when you randomly shuffle a feature. this works with almost any machine learning model. if shuffling a feature significantly reduces performance, that feature is considered important. that's something that can be applied everywhere. my first time i used it i saw immediately how it would make model interpretation much simpler. you need to do cross-validation though.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

def plot_permutation_importance(model, X, y, feature_names, scoring='accuracy', n_repeats=10, cv=5, top_n=10):
  """plots top n permutation importances for any scikit-learn model."""
  cv_method = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=42)
  result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=42)

  importances = result.importances_mean
  indices = np.argsort(importances)[::-1]
  top_indices = indices[:top_n]
  top_feature_names = [feature_names[i] for i in top_indices]
  top_importances = importances[top_indices]

  plt.figure(figsize=(10, 6))
  plt.barh(top_feature_names, top_importances)
  plt.xlabel("Importance")
  plt.title("Permutation Feature Importance")
  plt.gca().invert_yaxis()
  plt.show()

# Example usage with a scikit-learn logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(random_state=42, solver='liblinear').fit(X_train, y_train)

plot_permutation_importance(model, X_test, y_test, feature_names, top_n=7)
```

this second example provides a permutation importance calculation which does not rely on specific model’s internal structures, allowing for use on any model that can be used for scoring. i also added cross validation to be a bit more robust to randomness.

finally, when we’re dealing with tree-based models, like random forests and gradient boosting, partial dependence plots can be invaluable. these show how a feature influences the model's prediction while keeping the other variables constant. they really help in visualizing the relationship between features and predicted outcomes. i remember when working on a medical diagnostic project, i used this method to show how different lab test results impacted the probability of a diagnosis. the ability to not just show importance but also show how the features effect the result was invaluable to convey to doctors the behavior of the model.

```python
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

def plot_partial_dependence(model, X, feature_names, features, sample_size=100, n_cols=2):
    """plots the partial dependence of specified features."""

    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), constrained_layout=True)
    axs = axs.flatten()

    for i, feature in enumerate(features):
        PartialDependenceDisplay.from_estimator(model, X, [feature], ax=axs[i], feature_names=feature_names)
        axs[i].set_title(f"Partial Dependence of {feature_names[feature]}")
        axs[i].set_ylabel("Prediction")
    for j in range(i+1, len(axs)):
        axs[j].set_axis_off()
    plt.show()

# Example usage with scikit-learn gradient boosting model
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]
model = GradientBoostingClassifier(random_state=42).fit(X, y)
features_of_interest = [0, 3, 5, 8]
plot_partial_dependence(model, X, feature_names, features=features_of_interest)

```

this snippet will help you to visualize partial dependencies of the model, which can help on understanding how the features influence model’s output.

so, as you can see, visualizing feature importance isn’t just about making pretty pictures, it's about understanding the inner workings of your model. for a deeper dive into the concepts i would suggest reading "interpretable machine learning" by christoph molnar, it is a great resource with different methods for feature importance explanations. also the scikit-learn documentation itself for permutation importance is worth exploring, it is very comprehensive. if you do end up using tree based methods "the elements of statistical learning" by hastie, tibshirani and friedman, is also an amazing book.

it really pays off to master this, i swear, you would spend less time trying to debug models and also have better model building skills. it's a skill that gets used in virtually all data science and machine learning problems, and it can make all the difference when it comes time to communicate those results and that was my experience. good luck and if you have any other questions i'm here.
