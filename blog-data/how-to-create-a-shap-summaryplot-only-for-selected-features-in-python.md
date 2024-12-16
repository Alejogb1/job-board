---
title: "How to create a SHAP summary_plot only for selected features in Python?"
date: "2024-12-16"
id: "how-to-create-a-shap-summaryplot-only-for-selected-features-in-python"
---

Okay, let's tackle this. Having spent a fair chunk of my career knee-deep in model explainability, particularly with SHAP, I've encountered this specific need more times than I care to count. The default `shap.summary_plot` is powerful, but it can be overwhelming and sometimes, frankly, irrelevant when you’re focused on just a subset of your model's inputs. Filtering it down efficiently and correctly is a crucial skill for targeted analysis. Let me walk you through how I've typically approached this, with code examples and rationale.

The core challenge with `shap.summary_plot` is that it's designed to present a holistic view of all features by default. When you only want to focus on, say, the top 3 most impactful features or features related to a specific business domain, the standard approach can be cumbersome. The key here is to manipulate the shap values *before* you hand them over to the `summary_plot` function, not trying to filter within the plotting function itself. We accomplish this by selective slicing of the underlying shap values matrix.

Fundamentally, the `shap_values` obtained after calling `explainer.shap_values()` for a model, and subsequently fed to the `summary_plot`, is a numpy array representing feature contributions. When we have `n` instances and `m` features, the output will typically have shape `(n, m)`. If you are dealing with multi-class classification, this will be a list of such `(n, m)` arrays. Therefore, we are essentially subsetting the `m` dimension to select specific columns.

My first encounter with this limitation was while building a fraud detection system for a financial institution. We had hundreds of features, but the primary interest from the risk team was in a small set of transactional and user behavioral features. Generating the full SHAP summary plots for every model iteration was a time waste and, to be honest, it confused the stakeholders rather than providing insights. What we needed was a method to zero in on specific features and their impact.

Here are three distinct methods I found most efficient, each with its own use case and associated code snippet:

**Method 1: Feature Selection by Index**

This method is the most straightforward when you know the exact positional indices of the features you're interested in. Let's say, for example, that you wanted to analyze the 1st, 3rd and 5th features within your model’s feature set. We need to pull out those specific columns from the overall shap values array.

```python
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Simulate some data
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature indices to select
feature_indices = [0, 2, 4] # first, third, fifth feature

# Filter the shap values
if isinstance(shap_values, list): #for multi-class use cases
    filtered_shap_values = [sv[:, feature_indices] for sv in shap_values]
else:
    filtered_shap_values = shap_values[:, feature_indices]

# Filter the features names
filtered_feature_names = X_test.columns[feature_indices]


# Generate the summary plot
if isinstance(shap_values, list):
    for class_index, filtered_sv in enumerate(filtered_shap_values):
        shap.summary_plot(filtered_sv, features=X_test.iloc[:, feature_indices],
                          feature_names=filtered_feature_names,
                          show=False)
        plt.title(f"Class {class_index}")
        plt.show()

else:
    shap.summary_plot(filtered_shap_values, features=X_test.iloc[:, feature_indices],
                      feature_names=filtered_feature_names,
                      show=True)
```
In the code above, we are first simulating some sample data, then we are training a simple Random Forest classifier on the data. After obtaining shap values from the trained model using TreeExplainer, we proceed to select columns using index based slicing. Then we utilize the resulting slice along with the corresponding feature names to plot a shap summary plot, limited to just our selected features.

**Method 2: Feature Selection by Name**

This approach is better suited when you have the feature names rather than their index. Let's say, instead of the positional indices, we want to analyze features named "feature_1", "feature_3" and "feature_5". The code would be as follows:

```python
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# Simulate some data
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature names to select
feature_names = ["feature_1", "feature_3", "feature_5"]

# Retrieve indices by feature names.
feature_indices = [X_test.columns.get_loc(feature_name) for feature_name in feature_names]

# Filter the shap values
if isinstance(shap_values, list): #for multi-class use cases
    filtered_shap_values = [sv[:, feature_indices] for sv in shap_values]
else:
    filtered_shap_values = shap_values[:, feature_indices]


# Generate the summary plot
if isinstance(shap_values, list):
    for class_index, filtered_sv in enumerate(filtered_shap_values):
        shap.summary_plot(filtered_sv, features=X_test[feature_names],
                          feature_names=feature_names,
                           show=False)
        plt.title(f"Class {class_index}")
        plt.show()
else:
    shap.summary_plot(filtered_shap_values, features=X_test[feature_names],
                      feature_names=feature_names,
                      show=True)

```
Here, rather than hardcoding the column indices, we use pandas' `get_loc` method on the feature columns to get the corresponding numerical indices. Everything else regarding the slicing and plotting stays consistent with the previous method, making it highly flexible.

**Method 3: Using Feature Importance as a Filter**

In the real world, you’ll often want to focus on the top `k` most influential features, rather than pre-determined ones. This requires calculating some sort of feature importance metric first, then filtering based on it. For tree-based models, you can use the model's built-in feature importance, while for others, SHAP's average absolute contribution itself could be used.

```python
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Simulate some data
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Get feature importances
feature_importances = model.feature_importances_

# Select top 3 features
top_k = 3
top_feature_indices = np.argsort(feature_importances)[-top_k:]

# Filter the shap values
if isinstance(shap_values, list): #for multi-class use cases
    filtered_shap_values = [sv[:, top_feature_indices] for sv in shap_values]
else:
    filtered_shap_values = shap_values[:, top_feature_indices]


# Filter the features names
top_feature_names = X_test.columns[top_feature_indices]


# Generate the summary plot
if isinstance(shap_values, list):
    for class_index, filtered_sv in enumerate(filtered_shap_values):
        shap.summary_plot(filtered_sv, features=X_test.iloc[:, top_feature_indices],
                          feature_names=top_feature_names,
                           show=False)
        plt.title(f"Class {class_index}")
        plt.show()
else:
    shap.summary_plot(filtered_shap_values, features=X_test.iloc[:, top_feature_indices],
                      feature_names=top_feature_names,
                      show=True)

```

Here, we first obtain feature importances from our model, we are using model.feature\_importances in the random forest case here, but you could similarly use permutation based importances or the average SHAP importance as a metric. After obtaining the top feature indices, we proceed with the normal filtering and visualization process. This makes the plot highly dynamic based on what the model thinks is important.

These techniques, when applied correctly, can significantly streamline your model analysis, making the SHAP plots much more informative and manageable. For a deeper understanding of SHAP and its underlying mathematics, I recommend consulting "Interpretable Machine Learning" by Christoph Molnar—a thorough resource. For hands-on experience with SHAP itself, the official documentation, which you can find easily online, is indispensable. Additionally, for detailed coverage of tree-based models and their feature importance, consider "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman. Each of these sources provides in-depth knowledge necessary for understanding and applying SHAP and similar explainability methods effectively. Remember, focused analysis leads to clearer insights and better decisions.
