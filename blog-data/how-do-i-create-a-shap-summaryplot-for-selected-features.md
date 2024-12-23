---
title: "How do I create a SHAP summary_plot for selected features?"
date: "2024-12-16"
id: "how-do-i-create-a-shap-summaryplot-for-selected-features"
---

Alright,  I recall a particularly challenging project back in my days at the data science division of a fintech firm. We were building a complex credit risk model, and accurately communicating feature importance to stakeholders was paramount. Simply presenting aggregate importance scores wouldn't cut it; we needed to delve into how each feature influenced predictions across different value ranges. This is where the need for selective shap summary plots became crucial, beyond the all-encompassing view.

When working with shap (SHapley Additive exPlanations) values, the `summary_plot` function from the `shap` library provides an incredibly powerful visual tool for understanding model behavior. However, the standard output often displays the impact of all features, which can sometimes be overwhelming or, frankly, irrelevant when we are focusing on specific aspects of our model. Creating summary plots for *selected* features, therefore, allows for greater precision in analysis and communication. This isn't a capability directly provided as a named parameter, but we can achieve it via clever manipulation of the `shap_values` array before plotting.

The core concept hinges on creating a reduced `shap_values` matrix and a corresponding reduced feature set for the plot. The original `shap_values` array has dimensions corresponding to the number of samples and the number of features, so by selecting columns of the appropriate features, we get the desired subset. Let's consider this in detail.

First, generating the shap values is a fundamental step, of course. We need a model and data to compute this, which I'll represent with a basic random forest model using scikit-learn in the examples. The focus, of course, will be the subsequent selective plotting.

Here is the first example, using a simple regression task:

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate dummy data
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 5), columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
y = np.random.rand(100)

# Fit a basic random forest model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Select features
selected_features = ['feature_2', 'feature_4']
selected_indices = [X.columns.get_loc(feature) for feature in selected_features]

# Create reduced SHAP values matrix
reduced_shap_values = shap_values[:, selected_indices]

# Create reduced features dataframe
reduced_features_df = X[selected_features]


# Generate summary plot for selected features
shap.summary_plot(reduced_shap_values, reduced_features_df, show=False)

import matplotlib.pyplot as plt
plt.show()
```

In this example, we start by creating a dummy dataset and a basic random forest regressor model. We then calculate the SHAP values using a `TreeExplainer`. The critical part is the subsequent selection: we specify the desired features (`feature_2` and `feature_4`), determine their column indices in the original feature dataframe, and extract the corresponding SHAP values using numpy's slicing capabilities. We also create a new dataframe with the reduced features. Finally, we pass the reduced SHAP values matrix and the reduced feature dataframe to `shap.summary_plot`.

This allows us to visualize only the contributions of `feature_2` and `feature_4`, omitting the rest. You'll note I'm including `show=False` and using matplotlib's `plt.show()` separately. This avoids plot conflicts in some environments.

Let's move to a slightly more complex example, this time showcasing a classification scenario:

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate dummy data for classification
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 5), columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
y = np.random.randint(0, 2, 100) # Binary classification

# Fit a basic random forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For classification, explainer.shap_values returns a list, so we pick the SHAP values for the positive class (index 1)
if isinstance(shap_values, list):
  shap_values = shap_values[1]


# Select features
selected_features = ['feature_1', 'feature_3', 'feature_5']
selected_indices = [X.columns.get_loc(feature) for feature in selected_features]

# Create reduced SHAP values matrix
reduced_shap_values = shap_values[:, selected_indices]

# Create reduced features dataframe
reduced_features_df = X[selected_features]

# Generate summary plot for selected features
shap.summary_plot(reduced_shap_values, reduced_features_df, show=False)

import matplotlib.pyplot as plt
plt.show()

```

The core difference here lies in handling shap values for classifiers. `shap.TreeExplainer` with a classifier model will return a *list* of shap values - one set per class. We are usually interested in the values corresponding to the positive class, typically found at index `1`. We then proceed identically as in the regression case: selecting features and plotting. This example also demonstrates the ability to select multiple features as opposed to just two.

Finally, let's look at a third example where the original dataset could be a NumPy array and not a Pandas DataFrame:

```python
import shap
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Generate dummy data as numpy array
np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.rand(100)

feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

# Fit a basic gradient boosting regressor model
model = GradientBoostingRegressor(random_state=42)
model.fit(X, y)

# Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Select features
selected_features = ['feature_2', 'feature_3', 'feature_4']
selected_indices = [feature_names.index(feature) for feature in selected_features]

# Create reduced SHAP values matrix
reduced_shap_values = shap_values[:, selected_indices]


# Create dummy feature array for reduced features, since it will be a matrix rather than a DataFrame
reduced_features_array = X[:, selected_indices]
# You could create an actual dataframe here if needed, like this:
#reduced_features_df = pd.DataFrame(reduced_features_array, columns=selected_features)


# Generate summary plot for selected features
shap.summary_plot(reduced_shap_values, reduced_features_array, feature_names=selected_features, show=False)

import matplotlib.pyplot as plt
plt.show()
```
Here we are showcasing what to do when you have a numpy array rather than a dataframe. While the approach is identical in the sense that we slice the shap values, in this example, we just pass the reduced `X` array rather than a DataFrame with reduced feature names and instead pass `selected_features` as `feature_names`. This provides the same visual output, but we do so without needing to convert our array back to a dataframe.

For further reading, I highly recommend consulting the original SHAP paper by Lundberg and Lee ("A Unified Approach to Interpreting Model Predictions"). This paper, available on ArXiv, provides the theoretical underpinning of SHAP values. For a practical guide to implementations, the official SHAP documentation on GitHub is invaluable. Additionally, Chapter 10 of "Interpretable Machine Learning" by Christoph Molnar will offer more context and practical insights around interpreting SHAP plots, while Chapter 7 of "Python Machine Learning" by Sebastian Raschka will cover the specifics of model building and explanation using scikit-learn that was relevant for these examples. These are the primary sources I relied on in my daily work and continue to recommend.

The beauty of this method is in its adaptability and clarity. By understanding the underlying structure of `shap_values` and leveraging numpy's array manipulation capabilities, we gain granular control over what's visualized by our plots, allowing for targeted analysis and effective communication of complex model behaviors. This allows for more focused analysis when presenting model outcomes, which I've found is critical when working with stakeholders across varying technical backgrounds.
