---
title: "How can I scale a dataset with both numerical and categorical variables?"
date: "2024-12-23"
id: "how-can-i-scale-a-dataset-with-both-numerical-and-categorical-variables"
---

Alright, let's tackle this. Scaling datasets, especially those with a mix of numerical and categorical features, is a problem I’ve encountered more times than I can count over the years. It's a crucial step, often overlooked, that can drastically affect the performance of your machine learning models. Back in my early days working on predictive maintenance models for industrial machinery, we had datasets overflowing with sensor readings (numerical) and equipment types (categorical). Figuring out the best way to handle that was a baptism by fire.

The core issue isn't just about putting everything on the same scale, though that's part of it. It’s about ensuring your algorithms aren't unfairly biased toward features with larger ranges. Raw numerical data can vary wildly - temperature might range from -20 to 50, while pressure might be in the thousands. Similarly, categorical variables represented numerically after encoding can have skewed distributions.

For the numerical part, you've got a couple of primary approaches: normalization and standardization. Normalization (min-max scaling) squishes all values into a range, typically between 0 and 1. Standardization, on the other hand, centers data around a mean of 0 and scales it to unit variance (standard deviation of 1). I've found that standardization is generally more robust, particularly when dealing with outliers. Normalization can be quite sensitive to them. The choice between the two, however, should be informed by the model and the data itself.

For categorical variables, it gets a bit more complex. These are non-numeric features (like 'red,' 'blue,' 'green' or 'low', 'medium', 'high') that need to be converted into numerical representations for algorithms to process. The usual culprits here are one-hot encoding and ordinal encoding.

One-hot encoding creates a new binary column for each unique category. The presence of that category is indicated by a '1,' and its absence by '0'. This works exceptionally well when there's no inherent order among categories. Ordinal encoding assigns numerical values according to a predefined order. For example, 'low' might be 1, 'medium' 2, and 'high' 3. You'd use ordinal encoding when the order of categories is significant, like education levels.

Now, the catch: you don’t scale categorical variables after encoding. Scaling typically applies to the magnitude of numeric data, not the indicator values generated during one-hot or the numeric values allocated during ordinal encoding. Doing so would destroy the meaning of your categorical encodings and would often lead to inaccurate results.

Here's the nitty-gritty, with a few code examples using Python and scikit-learn to demonstrate:

**Example 1: Standardizing Numerical Features**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample numerical data (simulating sensor readings)
data = {'temperature': [20, 25, 18, 30, 22, 28, 15, 32],
        'pressure': [1200, 1350, 1100, 1400, 1250, 1300, 1000, 1450],
        'humidity': [60, 65, 58, 70, 63, 68, 55, 72]}
df = pd.DataFrame(data)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform numerical columns
numerical_cols = ['temperature', 'pressure', 'humidity']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(df)
```

This first snippet shows how to take a raw dataframe, identify the numerical columns, and apply `StandardScaler`. Notice how the original values are replaced with their standardized versions.

**Example 2: One-Hot Encoding Categorical Features**

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample categorical data (simulating equipment types)
data = {'equipment_type': ['pump', 'valve', 'pump', 'compressor', 'valve', 'pump', 'compressor', 'valve']}
df_cat = pd.DataFrame(data)


# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform categorical columns
encoded_features = encoder.fit_transform(df_cat[['equipment_type']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['equipment_type']))

df_cat = pd.concat([df_cat,encoded_df], axis=1)
df_cat = df_cat.drop(columns=['equipment_type'])
print(df_cat)
```

Here, `OneHotEncoder` is used to convert a categorical variable into multiple binary columns. The 'handle_unknown' parameter ensures that if during prediction, you encounter a category that was not present in training, you will get zeros for all categories.

**Example 3: Combining Numerical Scaling and Categorical Encoding**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sample mixed data
data = {'temperature': [20, 25, 18, 30, 22, 28, 15, 32],
        'pressure': [1200, 1350, 1100, 1400, 1250, 1300, 1000, 1450],
         'equipment_type': ['pump', 'valve', 'pump', 'compressor', 'valve', 'pump', 'compressor', 'valve']}
df_mixed = pd.DataFrame(data)

# Separate numerical and categorical columns
numerical_cols = ['temperature', 'pressure']
categorical_cols = ['equipment_type']

# Scale numerical features
scaler = StandardScaler()
df_mixed[numerical_cols] = scaler.fit_transform(df_mixed[numerical_cols])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df_mixed[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

df_mixed = pd.concat([df_mixed,encoded_df], axis=1)
df_mixed = df_mixed.drop(columns=categorical_cols)
print(df_mixed)
```

This example shows how the steps can be combined: separate numerical and categorical columns; scale the numerical ones, encode the categorical ones, and recombine. This is pretty standard in many data preparation pipelines.

For further learning, I'd recommend looking into *Feature Engineering for Machine Learning* by Alice Zheng and Amanda Casari; it's a practical book that thoroughly covers these methods. Also, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron provides an excellent overview of machine learning pipelines, where data preprocessing has an important role. I found papers by Sebastian Raschka on feature scaling and encoding extremely helpful in my initial learning phase. I would also specifically look at research papers exploring the effect of scaling and encoding on the specific learning model you are intending to use. For example, the effect of scaling is often higher for models using gradient descent methods, but less important for tree based methods. It is important to understand which approaches are more appropriate in any situation.

In practice, you'll want to carefully choose the scaling and encoding strategies according to your data characteristics and the algorithms you intend to use. There isn't one-size-fits-all approach, as I learned early on. It’s an iterative process, often involving cross-validation to find the best combination. Be systematic and document your choices. It can save you from future headaches.
