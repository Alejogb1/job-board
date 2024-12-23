---
title: "How can I make a SHAP summary_plot only for selected features from a list in Python?"
date: "2024-12-23"
id: "how-can-i-make-a-shap-summaryplot-only-for-selected-features-from-a-list-in-python"
---

Alright,  It's a common scenario, and I've certainly found myself needing to visualize only a subset of features from SHAP (SHapley Additive exPlanations) output, especially when dealing with datasets with numerous variables. It's inefficient and often visually overwhelming to plot everything. So, focusing on the problem of creating a `summary_plot` for specific features, we need to carefully slice the SHAP values. I'll share my past experiences and how I approached it.

Early in my career, working on a churn prediction model, I encountered this directly. We had a model using over 100 features, and the default `shap.summary_plot()` was utterly unreadable. That’s when I had to develop this feature-selection technique with SHAP values. The core concept is to filter the SHAP values array itself before passing it to the plot function. Here's how I usually approach it.

First, let's understand the typical output from SHAP. After training your model and generating SHAP values (let's say, `shap_values`), it is typically a 2D array or a pandas DataFrame. The rows correspond to instances (rows from your original dataset), and columns correspond to your features. The values within this structure represent each feature’s contribution to a particular instance's prediction.

So to slice it, instead of giving everything to `summary_plot`, I use index or label-based filtering to select only the column(s) or feature(s) I'm interested in. Now, it’s not just about doing it. It also entails thinking about how to do this efficiently so you are not doing unnecessary computation.

Now, let's get to some code examples.

**Example 1: Slicing Using Feature Indices (Assuming SHAP Values in NumPy Array)**

Let's say your SHAP values are stored in a NumPy array, `shap_values_array`, and you have a list of feature indices you want to plot, `feature_indices_to_plot`.

```python
import shap
import numpy as np
import matplotlib.pyplot as plt

# Sample SHAP values (replace with your actual values)
np.random.seed(42)
shap_values_array = np.random.rand(100, 10)  # 100 instances, 10 features
feature_indices_to_plot = [1, 3, 5]  # Indices of features you want to plot
feature_names = [f"feature_{i}" for i in range(10)] # sample feature names

# 1. Slice the SHAP values array based on the selected indices
sliced_shap_values = shap_values_array[:, feature_indices_to_plot]

# 2. Optionally, slice the feature name to match the sliced shap values.
sliced_feature_names = [feature_names[i] for i in feature_indices_to_plot]


# 3. Generate and display the summary plot
plt.figure() # added for independent plot calls
shap.summary_plot(sliced_shap_values, features=None if not sliced_feature_names else np.array(sliced_feature_names),show=False)
plt.show()
```

In this example, we use NumPy's array slicing (`[:, feature_indices_to_plot]`) to grab only the relevant columns from `shap_values_array`. Note that if feature names are not provided, `summary_plot` assigns default index names. I also added code for slicing feature names, which makes it clear which features we are plotting. I also added the `plt.figure()` command to ensure plot calls do not overlap.

**Example 2: Slicing Using Feature Names (If SHAP Values are in Pandas DataFrame)**

If your SHAP values are conveniently stored in a Pandas DataFrame, filtering becomes more readable using label-based selection. Let’s imagine your `shap_values_df` is a dataframe with your SHAP values.

```python
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample SHAP values DataFrame (replace with your actual DataFrame)
np.random.seed(42)
data = np.random.rand(100, 10)
columns = [f"feature_{i}" for i in range(10)]
shap_values_df = pd.DataFrame(data, columns=columns)

feature_names_to_plot = ["feature_2", "feature_7", "feature_9"] # list of feature names

# 1. Slice the DataFrame based on column names
sliced_shap_values_df = shap_values_df[feature_names_to_plot]

# 2. Generate and display the summary plot
plt.figure()
shap.summary_plot(sliced_shap_values_df.values, features=sliced_shap_values_df.columns, show=False)
plt.show()
```

Here, `sliced_shap_values_df` will now contain only the SHAP values for the features listed in `feature_names_to_plot`. We used `.values` to convert the dataframe to a numpy array as that is what `summary_plot` expects. Additionally, the column names are passed as `features`.

**Example 3:  Handling SHAP Object and Feature Names from the Model**

Often, when you are doing model explanation, SHAP doesn’t only provide the SHAP values, it provides an object with these values. Also, feature names may be embedded in the model itself. This code example illustrates a typical workflow using the SHAP object. Here we will assume we are using the tree explainer and an XGBoost model.

```python
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data and model (replace with your actual model and data)
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
y = np.random.randint(0, 2, 100)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)


# Create explainer
explainer = shap.TreeExplainer(model)
# compute shap values
shap_values = explainer.shap_values(X)

feature_names_to_plot = ["feature_3", "feature_6", "feature_8"]

# 1. Find indices of feature names
feature_indices = [list(X.columns).index(name) for name in feature_names_to_plot]

# 2. Slice SHAP values and names
sliced_shap_values = shap_values[:, feature_indices]
sliced_feature_names = feature_names_to_plot

# 3. Generate and display the summary plot
plt.figure()
shap.summary_plot(sliced_shap_values, features= sliced_feature_names,show=False)
plt.show()
```

In this scenario, I first calculate the index location of the provided feature names and slice the shap values and feature names.

These three examples should give you a pretty solid foundation for extracting the specific features you need. I have found that carefully pre-processing the `shap_values` array or dataframe, and especially slicing both the data and the names for consistency, is the key to generating clean and informative summary plots.

Now, a word on learning resources. While online tutorials can be helpful, for a deeper understanding of SHAP and model interpretability, I recommend checking out the original SHAP paper by Lundberg and Lee, titled "A Unified Approach to Interpreting Model Predictions." It provides the mathematical basis for SHAP. Furthermore, "Interpretable Machine Learning" by Christoph Molnar is an excellent resource that delves into various interpretable machine learning techniques, including SHAP. These should give you a solid understanding that goes beyond just the how-to and explains the core principles behind SHAP analysis.

Remember, the ultimate goal is to extract meaningful insights from your machine learning model, and targeted visualizations using tools like SHAP are a crucial step in achieving that goal. And having the flexibility to focus on the critical features can make all the difference.
