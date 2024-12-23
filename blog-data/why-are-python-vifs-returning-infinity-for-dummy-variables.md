---
title: "Why are Python VIFs returning infinity for dummy variables?"
date: "2024-12-23"
id: "why-are-python-vifs-returning-infinity-for-dummy-variables"
---

Alright, let's talk about infinite variance inflation factors (VIFs) when dealing with dummy variables in Python. This is a scenario I've encountered a few times, and it usually points to a specific, rather fundamental, issue with your model setup. It's not necessarily a bug in Python’s VIF calculation itself, but rather a symptom of multicollinearity—a problem you absolutely need to address in regression modeling. I recall one project, a predictive modeling task for a customer behavior dataset, where we faced this exact situation. It nearly derailed our entire timeline until we got to the root cause.

The core problem isn't that Python miscalculates VIFs; it's that your dummy variables are perfectly predictable from each other. This results in the calculation of the R-squared in the VIF formula becoming exactly equal to 1, leading to division by zero—hence the infinity.

To be more precise, VIF is calculated using the formula:

`VIF_i = 1 / (1 - R^2_i)`

Where `R^2_i` is the R-squared value obtained when regressing the i-th predictor variable against all other predictor variables in your model. When your dummy variables are set up in such a way that any of them is a linear combination of the others (or when one variable perfectly predicts the others), their `R^2` becomes 1, causing `1 - R^2` to be zero. Division by zero, of course, throws us into infinite VIF territory.

Typically, this problem manifests when you've created dummy variables from a categorical variable without dropping one category. Consider a scenario with a variable called 'color,' which has three possible values: 'red,' 'blue,' and 'green.' If you create dummy variables for each of these colors (let's say, `is_red`, `is_blue`, and `is_green`), the last one, `is_green`, becomes perfectly predictable using the other two. If the instance is neither red nor blue, it must necessarily be green. This sets up perfect linear dependency.

Let me provide you with some practical python examples to showcase this:

**Example 1: The problem of not dropping a dummy variable**

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create sample dataframe
data = {'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
        'value': [10, 20, 30, 12, 22, 32]}
df = pd.DataFrame(data)

# create dummy variables without dropping the first category
df_dummies = pd.get_dummies(df['color'])
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['color'], axis = 1)
# add intercept to the model
df['intercept'] = 1

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(df.shape[1])]

print(vif_data)
```

In this example, if you run this, you will see `inf` values in the VIF column for at least some of the dummy variables, which means that multicollinearity is a major problem.

The solution here is to always drop one of the dummy variables from the set. That becomes the 'base case'. Let’s show you an example of that in action:

**Example 2: Dropping the first category**

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create sample dataframe
data = {'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
        'value': [10, 20, 30, 12, 22, 32]}
df = pd.DataFrame(data)

# create dummy variables and drop first column (i.e. drop_first=True)
df_dummies = pd.get_dummies(df['color'], drop_first=True)
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['color'], axis = 1)

# add intercept to the model
df['intercept'] = 1

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(df.shape[1])]

print(vif_data)
```

Now the VIFs should be finite and indicate how much the variance of one regressor increases when compared to the others, or not, in this case. Specifically, each VIF indicates how much the variance of the coefficient is increased by this multicollinearity issue. Dropping the first variable is, of course, just one possibility; you can drop any of the dummy variables.

It's crucial to note that the 'intercept' variable won't have an infinite VIF in these particular examples due to its constant values and how we coded the dummies. This could be quite different depending on the type of variables used to create the model (continuous or categorical). This also assumes there are no other multicollinearity issues within the dataset.

Furthermore, even if you did drop a dummy variable, infinite VIF might still appear in some special cases: it could be caused by another column that is the perfect linear combination of the other columns, or a column that does not have any variations. Let me showcase a possible example:

**Example 3: Adding perfectly correlated column with no variation**

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create sample dataframe
data = {'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
        'value': [10, 20, 30, 12, 22, 32]}
df = pd.DataFrame(data)

# create dummy variables and drop first column (i.e. drop_first=True)
df_dummies = pd.get_dummies(df['color'], drop_first=True)
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['color'], axis = 1)

# add intercept to the model
df['intercept'] = 1

# Add a column that causes issues
df["problematic_column"] = [1,1,1,1,1,1]

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(df.shape[1])]

print(vif_data)
```

In this example, we will again see `inf` values in the VIF column for the `problematic_column` because all the values are identical. This is an example of multicollinearity which is not caused by dummy variables, but it results in the same issue. So, if you fix the issue of multicollinearity within your dummies and still have infinite values, this should be your next thing to check.

So, to recap, if you’re getting infinite VIFs for dummy variables, the primary cause is that you haven't dropped one category during the dummy variable creation process. However, it is important to remember that although this is the most common cause, multicollinearity can arise from a number of issues with the data being used to create the model.

For further understanding, I'd recommend the following resources:

1.  **"Applied Regression Analysis" by Norman Draper and Harry Smith**: This is a classic text that thoroughly covers regression concepts, including multicollinearity and VIFs. It explains the theory behind these calculations and provides a detailed statistical perspective.

2.  **"An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani**: While this book focuses on machine learning, it dedicates an important section to linear regression, which includes a clear explanation of multicollinearity and its practical implications.

3.  **"Regression Modeling Strategies" by Frank Harrell**: This book offers a practical perspective on building regression models and includes in-depth chapters about variable selection and dealing with multicollinearity.

These resources will help you deepen your knowledge not only of the practicalities of calculating VIFs, but also the statistical underpinnings of the methods used when working with regression models. They also showcase other forms of multicollinearity and how to mitigate them.
