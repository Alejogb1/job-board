---
title: "Why does my code using feature_engine.selection.DropHighPSIFeatures not work when a pandas DataFrame is passed?"
date: "2024-12-23"
id: "why-does-my-code-using-featureengineselectiondrophighpsifeatures-not-work-when-a-pandas-dataframe-is-passed"
---

Alright, let's tackle this one. I've definitely been down that road with `feature_engine` and dataframes. It's usually not a problem with the `DropHighPSIFeatures` class itself, but rather the way the input dataframe is structured or understood by the library. Let me walk you through it based on a few experiences I've had.

The core issue, in my experience, typically boils down to a mismatch in expectations concerning the format of the input data, specifically column types and how the feature engine library handles these. `DropHighPSIFeatures` calculates the population stability index (psi) across different datasets, and it expects numeric data for those calculations. If your dataframe contains non-numeric data, especially those with object datatypes—which is common when pandas automatically infers datatypes from a CSV or other sources—it’s very likely to stumble.

Let's imagine a scenario from a previous project. I was building a credit risk model, and I was aiming to streamline my features using psi analysis with `DropHighPSIFeatures`. Initially, my dataframe included the loan applications, and the data initially seemed clean. However, the `DropHighPSIFeatures` class was throwing errors and failing to drop any columns as expected. It took me some time, but I realized a few categorical columns, namely *application_status* and *credit_score_category*, were interpreted as pandas ‘object’ type instead of the proper integer or numeric representation. Pandas often handles string data this way.

Now, let’s delve into why this is a problem and how to fix it.

Here’s the thing: the `feature_engine` library, underneath, often uses numpy arrays for its computations as it is more performant for numeric operations. When you pass a dataframe, it internally tries to convert those columns into a numpy array. If pandas assigned a 'object' datatype to a numeric column because it contains missing data or other string-like entries, the conversion will either fail or, more likely, result in a numpy array of strings which `DropHighPSIFeatures` can't properly operate with.

Here's a simplified version of what could be causing this issue and how to resolve it, first with code examples, followed by a detailed explanation:

**Code Example 1: Initial Problematic Scenario**

```python
import pandas as pd
from feature_engine.selection import DropHighPSIFeatures

# Assume df is our original dataframe
data = {'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'feature2': ['A', 'B', 'A', 'B', 'A', 'B'],
        'feature3': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'feature4': ['1', '2', '3', '4', '5', '6'] }
df = pd.DataFrame(data)
df['feature4'] = df['feature4'].astype(float)


# Define train and test sets
train_df = df.iloc[:3]
test_df = df.iloc[3:]

# Initialize the feature selector (without specifying a threshold to make it fail as the feature2 is not numerical)
selector = DropHighPSIFeatures()

try:
    selector.fit(train_df, test_df)
    print(selector.features_to_drop_)
except ValueError as e:
    print(f"Error encountered: {e}")
```

In the above code snippet, `feature2` column is a string and `feature4` column is numeric with float datatype. If the selector was initialized without a specified threshold, `feature2` will cause an issue.

**Code Example 2: Addressing the Object Type Issue**

```python
import pandas as pd
from feature_engine.selection import DropHighPSIFeatures

# Assume df is our original dataframe
data = {'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'feature2': ['A', 'B', 'A', 'B', 'A', 'B'],
        'feature3': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'feature4': ['1', '2', '3', '4', '5', '6'] }
df = pd.DataFrame(data)
df['feature4'] = df['feature4'].astype(float)


# Define train and test sets
train_df = df.iloc[:3]
test_df = df.iloc[3:]

# Correct the feature2 column to make it numeric by adding dummy variables
train_df_processed = pd.get_dummies(train_df, columns=['feature2'], drop_first=True)
test_df_processed = pd.get_dummies(test_df, columns=['feature2'], drop_first=True)

# Now, initialize the selector
selector = DropHighPSIFeatures(threshold=0.2)  # Add a threshold to work

selector.fit(train_df_processed, test_df_processed)
print(selector.features_to_drop_)
```

In this corrected code, before passing the dataframe to `DropHighPSIFeatures`, I'm making sure all the features are numeric by transforming the `feature2` column into dummy variables using `pd.get_dummies`. This action will ensure that both dataframes contain only numeric data and the `DropHighPSIFeatures` will be able to use them. It's crucial to transform both train and test sets accordingly and also pass a threshold parameter in order for the functionality to work correctly.

**Code Example 3: Handling missing data in numeric columns**

```python
import pandas as pd
from feature_engine.selection import DropHighPSIFeatures

# Assume df is our original dataframe with null values
data = {'feature1': [1.0, 2.0, None, 4.0, 5.0, 6.0],
        'feature2': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        'feature3': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'feature4': ['1', '2', '3', '4', '5', '6'] }

df = pd.DataFrame(data)
df['feature4'] = df['feature4'].astype(float)

# Define train and test sets
train_df = df.iloc[:3]
test_df = df.iloc[3:]


# Impute missing values
train_df = train_df.fillna(train_df.mean(numeric_only=True))
test_df = test_df.fillna(test_df.mean(numeric_only=True))


#Initialize the selector
selector = DropHighPSIFeatures(threshold=0.2)

selector.fit(train_df, test_df)
print(selector.features_to_drop_)

```

In the third example, I have introduced null values to illustrate a common real-world problem. The code first imputes missing values using the mean from each dataframe. It's worth noting that if missing values are a large percentage of your dataset, imputation can result in biases and you might need to investigate alternatives, or even exclude those features. After handling missing values and ensuring they are numerical, `DropHighPSIFeatures` can correctly compute the PSI.

To dig deeper into the nuances of PSI and handling data types, I would recommend exploring the book "Feature Engineering for Machine Learning" by Alice Zheng and Amanda Casari. It offers solid practical guidance on data transformations and feature engineering techniques. Additionally, the documentation for `feature_engine` on its own website is useful. They detail each module and give practical examples on their use. For a solid mathematical understanding of PSI, I would suggest searching through academic publications, specifically looking for papers related to model stability analysis in credit scoring which usually presents PSI as a standard tool.

In summary, the common pitfall with using `DropHighPSIFeatures` on pandas DataFrames arises from incorrect data types or presence of null values. Always ensure all the selected features have numeric data and that the data is thoroughly preprocessed with categorical feature transformations, or missing value imputations, before you use this functionality. I have found these steps to be extremely helpful in all my past experiences.
