---
title: "How to apply a stepwise regression on all CSV files in a folder?"
date: "2024-12-23"
id: "how-to-apply-a-stepwise-regression-on-all-csv-files-in-a-folder"
---

Let's tackle that. I recall a rather thorny project from a few years back where we needed to build a predictive model across numerous datasets, all conveniently formatted as csv files within a single directory. The challenge, as you might expect, involved not just running the stepwise regression itself, but automating the entire process for hundreds of files. It quickly became clear that manual processing was a non-starter. So, the key, in this scenario, lies in wrapping the regression process within a loop, which iterates through each csv file, applies the regression, and consolidates the results. Here's how we can approach this, broken down into manageable components, along with some specific python code examples.

First things first: we need a way to efficiently iterate through all csv files in the directory. Python's `os` and `glob` modules are invaluable here. `os` provides system-level functionality, including path manipulation, while `glob` allows us to use filename patterns to filter our files, which will be key to ensure we're only targeting csv files. After loading each file into a pandas dataframe, we'll apply a stepwise regression. For the regression itself, we'll use `statsmodels`, a package which provides a rich collection of statistical models, including stepwise regression. We'll also utilize `sklearn` for some data preprocessing steps such as the handling of categorical variables and feature scaling, which is vital before performing regression analysis.

Let’s solidify this with some example code snippets.

**Snippet 1: File Iteration and Data Loading**

This snippet shows how to locate csv files and read each one into a pandas dataframe.

```python
import os
import glob
import pandas as pd

def process_csv_files(folder_path):
    all_results = {} # To store regression results for each file
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            all_results[filename] = perform_stepwise_regression(df)
        except Exception as e:
           print(f"Error processing file {csv_file}: {e}")

    return all_results
```

This function `process_csv_files` first finds all the csv files within the provided folder path. Then, within a loop, each file is read into a pandas DataFrame. Error handling with `try...except` is included, a must for dealing with potentially malformed files. The file name is extracted and used as a key in the results dictionary. Note the usage of `os.path.join` and `os.path.basename`; these ensure the code is platform agnostic and robust.

Next we'll get into the `perform_stepwise_regression` function, which is where the statistical modeling happens. We need to consider how to handle different variable types. Categorical variables require encoding (e.g., one-hot encoding), and continuous variables may require scaling. The function below is where these operations are performed along with the core stepwise regression logic.

**Snippet 2: Stepwise Regression Implementation**

This is the core logic for performing stepwise regression.

```python
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def perform_stepwise_regression(df, target_variable='target', significance_level=0.05):
    # Handle categorical and numerical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if target_variable in categorical_cols:
        categorical_cols.remove(target_variable)
    if target_variable in numerical_cols:
        numerical_cols.remove(target_variable)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough'
    )

    #Handle missing target variable
    if target_variable not in df.columns or df[target_variable].isnull().all():
        print(f"Skipping regression: Target variable missing or all nulls in this dataset")
        return None

    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Splitting the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Data Transformation
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    #Ensure that statsmodels accepts our transformed data
    X_train_transformed = sm.add_constant(X_train_transformed)
    X_test_transformed = sm.add_constant(X_test_transformed)


    included = list(range(X_train_transformed.shape[1]))
    excluded = []
    while True:
        changed = False
        #Forward Step
        pval = pd.Series(index=included, dtype='float64')
        for feature_index in included:
            model = sm.OLS(y_train, X_train_transformed[:, included]).fit()
            pval.at[feature_index] = model.pvalues[feature_index]
        if pval.min() >= significance_level:
            break
        min_pval = pval.min()
        min_index = pval.idxmin()
        excluded.append(min_index)
        included.remove(min_index)
        changed = True
        #Backward step
        pval = pd.Series(index=excluded, dtype='float64')
        for feature_index in excluded:
            model = sm.OLS(y_train, X_train_transformed[:,included+[feature_index]]).fit()
            pval.at[feature_index] = model.pvalues[-1]
        if pval.max() < significance_level:
            break
        max_pval = pval.max()
        max_index = pval.idxmax()
        included.append(max_index)
        excluded.remove(max_index)
        changed = True
        if not changed:
            break

    final_model = sm.OLS(y_train, X_train_transformed[:, included]).fit()
    return {'significant_features': list(X_train.columns[included]), 'model_summary': final_model.summary().as_text()}
```

This function handles variable preprocessing via the column transformer, allowing one-hot encoding and scaling. It then splits the data into training and testing sets for model evaluation, implements a forward/backward stepwise feature selection algorithm based on p-values, and returns a summary including final model statistics, and the selected features. The significance level is controlled by the `significance_level` parameter. The key here is handling the transformed data with `sm.add_constant` to ensure that the statsmodels library can properly model the linear regression with a constant value.

The final function aggregates the information and is called to perform the operation on all of the files within the folder.

**Snippet 3: Main Execution**

This is how to run the entire process:

```python
if __name__ == '__main__':
   folder_path = "path_to_your_csv_files" # Replace with the actual folder path
   all_results = process_csv_files(folder_path)
   for filename, result in all_results.items():
       print(f"Results for {filename}:")
       if result:
          print(f" Significant Features: {result['significant_features']}")
          print(f" Model Summary: {result['model_summary']}")
       else:
          print("Regression failed or missing target data")
       print("---")
```

This snippet shows a main execution section that calls our processing function and prints out the results. Remember to replace `"path_to_your_csv_files"` with the actual path to your folder. Error handling has been included, which allows the process to skip over malformed files, which would otherwise cause the process to fail.

From practical experience, I can suggest that if you start working with more complex datasets or have constraints like feature interactions, you might want to move beyond the classical stepwise algorithms, which can be computationally expensive and might lead to unstable models. In such instances, consider exploring methods like regularization (L1 or L2) or more advanced model selection procedures.

For further learning, I would recommend delving into *'The Elements of Statistical Learning'* by Hastie, Tibshirani, and Friedman. This book provides a strong foundation in statistical learning techniques, including regression and feature selection. Also, for a deeper dive into statistical modeling using python, the documentation of *statsmodels* itself is an invaluable resource. Finally, for the specifics of pandas data manipulation, I'd advise keeping the *pandas documentation* readily available. By systematically applying these concepts, you can effectively and efficiently conduct stepwise regression on multiple datasets, which was certainly key for my previous project.
