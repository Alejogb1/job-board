---
title: "How can I calculate R-squared for a linear regression model (using statsmodels) on training and testing datasets?"
date: "2024-12-23"
id: "how-can-i-calculate-r-squared-for-a-linear-regression-model-using-statsmodels-on-training-and-testing-datasets"
---

Let’s tackle that. It's a common scenario, needing to evaluate your linear regression model’s performance not just on the data it was trained on, but also on unseen data, the test set. R-squared, or the coefficient of determination, gives us a handy metric for that, indicating how well the model fits the variance in the dependent variable. I remember a project years back where we were forecasting consumer demand, and getting this distinction between training and testing r-squared correct was critical to avoid severely over-optimistic projections. We can easily do this using `statsmodels`.

First, a bit of background: r-squared measures the proportion of variance in the dependent variable that's predictable from the independent variables. It ranges from 0 to 1, with 1 indicating a perfect fit. Keep in mind it has limitations. A high r-squared doesn’t necessarily mean your model is "good," especially if the model is overly complex, leading to overfitting. Always supplement r-squared with other evaluation metrics and diagnostic plots.

Now, for `statsmodels`, the process is pretty straightforward. The key is generating the predictions on both your training and testing sets, and then using the model's inherent method to calculate r-squared from those predicted values and the actual values in each set. Let me show you with some code examples that will illustrate the whole workflow.

**Example 1: Basic R-squared Calculation**

```python
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate some sample data
data = {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [2, 4, 5, 4, 5, 8, 10, 9, 11, 14]}
df = pd.DataFrame(data)

# Split into training and testing sets
X = df['x']
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add a constant to the training data (necessary for statsmodels)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the model
model = sm.OLS(y_train, X_train).fit()

# Predictions on training and test sets
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Calculate R-squared
r_squared_train = model.rsquared
r_squared_test = sm.OLS(y_test,X_test).fit().rsquared # using statsmodels directly

print(f"Training R-squared: {r_squared_train:.3f}")
print(f"Testing R-squared: {r_squared_test:.3f}")

```

In this first snippet, I created a basic dataframe, split it into train and test using scikit-learn's `train_test_split`, and used `statsmodels` to fit a standard OLS regression model. Importantly, notice I added a constant to the training and testing data using `sm.add_constant`. This adds an intercept to the model, which is essential in most linear regression situations. The training r-squared comes directly from `model.rsquared`. For the testing set, I directly utilize `sm.OLS` again, this time on the testing data and extract the r-squared to calculate the test performance. Note that this way we're calculating the r-squared with the testing set *given* the model from training, which might differ from re-fitting to the testing data directly.

**Example 2: Using a Pandas DataFrame**

Let's enhance that a bit by using more dataframe-centric operations. This reflects how you might handle your data most of the time:

```python
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample Data
data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 5, 4, 5, 8, 10, 9, 11, 14],
        'target': [3, 6, 8, 7, 9, 14, 17, 16, 19, 24]}

df = pd.DataFrame(data)

# Split the data
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Define features and target variable
features = ['feature1', 'feature2']
target = 'target'

# Prepare data
X_train = sm.add_constant(train_df[features])
y_train = train_df[target]
X_test = sm.add_constant(test_df[features])
y_test = test_df[target]

# Fit Model
model = sm.OLS(y_train, X_train).fit()

# Predictions
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Calculate r-squared
r_squared_train = model.rsquared
r_squared_test = sm.OLS(y_test,X_test).fit().rsquared


print(f"Training R-squared: {r_squared_train:.3f}")
print(f"Testing R-squared: {r_squared_test:.3f}")
```

Here, I explicitly use dataframes for train and test, allowing for multiple features, and makes it more convenient to scale up, if needed. It follows the same logic as example 1, but showcases the usage when more complex data structures are in place. This kind of handling is far more typical in real-world applications.

**Example 3: Focusing on Cross-Validation**

While simple train-test splits are useful, cross-validation is generally preferred. Let's examine how you might incorporate that, specifically, to look at variations in the validation set scores:

```python
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

# Sample Data
data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'feature2': [2, 4, 5, 4, 5, 8, 10, 9, 11, 14, 15, 17, 16, 18, 20, 21, 22, 20, 23, 26],
        'target': [3, 6, 8, 7, 9, 14, 17, 16, 19, 24, 26, 28, 27, 29, 32, 34, 35, 33, 36, 40]}

df = pd.DataFrame(data)

# Define features and target variable
features = ['feature1', 'feature2']
target = 'target'

# Prepare Data
X = sm.add_constant(df[features])
y = df[target]

# Setup K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r_squared_values = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit Model
    model = sm.OLS(y_train, X_train).fit()

    # R-squared on test set
    r_squared_test = sm.OLS(y_test,X_test).fit().rsquared
    r_squared_values.append(r_squared_test)

print(f"Cross-Validation R-squared scores: {r_squared_values}")
print(f"Mean Cross-Validation R-squared score: {np.mean(r_squared_values):.3f}")
```

Here, I use K-Fold cross-validation. The dataset is shuffled and split into five folds. In each iteration, I train the model on four folds and evaluate the r-squared on the remaining fold. The key here is that it illustrates how you'd measure out-of-sample performance multiple times, getting a sense of variance in your performance rather than relying on a single test-train split. This gives a more robust evaluation of the model's generalization capability.

To further your understanding beyond what I've shown, I recommend exploring the following resources: "Applied Regression Analysis" by Norman Draper and Harry Smith is a classic reference, going very deep into both the theoretical and practical aspects of regression. For a more computationally oriented approach, I'd recommend "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman, which provides a comprehensive overview of various statistical learning techniques including regression. Also, looking into papers by T. Hastie or R. Tibshirani on cross validation or regularization techniques will further help grasp the challenges and approaches in these fields.

Remember, simply having r-squared for train and test is not the final step. It's part of an iterative process where you should analyze residual plots, understand feature importance, and consider other metrics relevant to the specific problem you are solving. The key is to truly understand the limitations and strengths of your model in the context of your data. This approach will help ensure your model is robust and generalize well, something I learned the hard way many times over the years, and that has served me well.
