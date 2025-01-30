---
title: "How can I perform a paired Wilcoxon test on corresponding rows of two dataframes?"
date: "2025-01-30"
id: "how-can-i-perform-a-paired-wilcoxon-test"
---
Performing a paired Wilcoxon signed-rank test on corresponding rows of two dataframes requires careful alignment and an understanding of how the test's assumptions are met within this structure. The core challenge lies in correctly pairing observations across the two dataframes before applying the statistical test, ensuring that each row represents a related pair of measurements rather than independent observations. I've encountered this frequently when comparing pre- and post-intervention measurements where each row corresponds to an individual participant.

The Wilcoxon signed-rank test is a non-parametric alternative to the paired t-test, used when the assumption of normality for the differences between paired observations is violated. In our case, each row in the two input dataframes represents a single paired observation. The test evaluates whether the median of these differences is significantly different from zero. Crucially, the test works on the *differences* between paired observations, not directly on the raw values. Before proceeding, it's essential to confirm that the dataframes have the same number of rows, and that the corresponding rows are indeed the intended pairs. A mismatch could lead to invalid conclusions. Additionally, ensure there are sufficient non-zero differences. If most paired values are identical, the Wilcoxon test might not be informative.

The general procedure involves the following steps: 1) Verify dataframe alignment and dimensions. 2) Calculate the differences between corresponding rows. 3) Apply the Wilcoxon signed-rank test to these calculated differences. Let's explore how this is implemented using Python and `pandas` along with the `scipy.stats` module for the statistical analysis.

**Code Example 1: Basic Implementation**

```python
import pandas as pd
from scipy import stats

def paired_wilcoxon(df1, df2):
    """
    Performs a paired Wilcoxon signed-rank test on corresponding rows of two dataframes.

    Args:
        df1: First pandas DataFrame.
        df2: Second pandas DataFrame.

    Returns:
        A tuple containing:
            - The W statistic of the test.
            - The p-value of the test.
        Returns None if an error occurs or the test can't be performed.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        print("Error: Input must be pandas DataFrames.")
        return None

    if df1.shape != df2.shape:
        print("Error: DataFrames must have the same shape.")
        return None

    if df1.empty or df2.empty:
      print("Error: DataFrames cannot be empty.")
      return None

    try:
        differences = df1.values - df2.values
        w_statistic, p_value = stats.wilcoxon(differences.flatten(), zero_method="pratt") # Flattening is crucial for scipy
        return w_statistic, p_value
    except Exception as e:
        print(f"Error during calculation: {e}")
        return None

#Example Usage
data1 = {'A': [10, 12, 15, 18, 20],
         'B': [25, 23, 28, 30, 27],
         'C': [5, 7, 9, 11, 13]}
data2 = {'A': [9, 11, 14, 17, 19],
         'B': [24, 22, 27, 29, 26],
         'C': [6, 8, 10, 12, 14]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

results = paired_wilcoxon(df1, df2)
if results:
    w_stat, p_val = results
    print(f"Wilcoxon Statistic: {w_stat}")
    print(f"P-value: {p_val}")

```

This first example establishes the core logic. The function `paired_wilcoxon` first checks the inputs for correct types and matching shapes. Then the crucial part – element-wise subtraction using `.values` ensures numerical arrays are handled effectively for subsequent statistical analysis. The `differences` matrix is flattened using `.flatten()` before being passed to `stats.wilcoxon`.  The `zero_method="pratt"` argument handles zero differences by discarding them, as recommended for the Wilcoxon test; this prevents potential errors caused by many zero values.  Error handling includes explicit checks for invalid inputs, like empty dataframes.

**Code Example 2: Handling Specific Columns**

Sometimes, you'll want to apply the test to a specific set of columns common between the dataframes. Here is an example demonstrating such scenario.

```python
import pandas as pd
from scipy import stats

def paired_wilcoxon_subset(df1, df2, columns):
    """
    Performs a paired Wilcoxon test on specified columns of two dataframes.

    Args:
        df1: First pandas DataFrame.
        df2: Second pandas DataFrame.
        columns: A list of column names to consider.

    Returns:
        A tuple containing:
            - The W statistic of the test.
            - The p-value of the test.
            - None if an error occurs or the test can't be performed.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        print("Error: Input must be pandas DataFrames.")
        return None

    if df1.shape != df2.shape:
      print("Error: DataFrames must have the same shape.")
      return None

    if not columns:
      print("Error: Columns list cannot be empty.")
      return None

    try:
        df1_subset = df1[columns]
        df2_subset = df2[columns]

        differences = df1_subset.values - df2_subset.values
        w_statistic, p_value = stats.wilcoxon(differences.flatten(), zero_method="pratt")
        return w_statistic, p_value
    except Exception as e:
        print(f"Error during calculation: {e}")
        return None

#Example Usage
data1 = {'A': [10, 12, 15, 18, 20],
         'B': [25, 23, 28, 30, 27],
         'C': [5, 7, 9, 11, 13],
         'D' : [3, 4, 5, 6, 7]}
data2 = {'A': [9, 11, 14, 17, 19],
         'B': [24, 22, 27, 29, 26],
         'C': [6, 8, 10, 12, 14],
         'D' : [4, 5, 6, 7, 8]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

selected_cols = ['A', 'B', 'C']

results = paired_wilcoxon_subset(df1, df2, selected_cols)
if results:
    w_stat, p_val = results
    print(f"Wilcoxon Statistic (subset): {w_stat}")
    print(f"P-value (subset): {p_val}")

```

This example adds flexibility by processing only a defined subset of columns. The `paired_wilcoxon_subset` function accepts a `columns` parameter that acts as a filter using DataFrame's indexing capabilities. This allows applying the test to the relevant columns for your specific analysis. The core logic for calculation is identical; we first extract the relevant columns, perform the subtraction, and apply the statistical test as before.

**Code Example 3:  Handling Missing Values**

Real-world data often contains missing values.  While the Wilcoxon test itself doesn’t directly handle missing values, we need to appropriately pre-process the data. The most straightforward approach is to drop rows with any missing value before performing the test, since pairwise comparison requires both values to be present for a given row.

```python
import pandas as pd
from scipy import stats

def paired_wilcoxon_missing(df1, df2):
    """
    Performs a paired Wilcoxon test on corresponding rows of two dataframes,
    handling missing data by removing rows with any NaN values.

    Args:
        df1: First pandas DataFrame.
        df2: Second pandas DataFrame.

    Returns:
        A tuple containing:
            - The W statistic of the test.
            - The p-value of the test.
            - None if an error occurs or the test can't be performed
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        print("Error: Input must be pandas DataFrames.")
        return None

    if df1.shape != df2.shape:
      print("Error: DataFrames must have the same shape.")
      return None

    try:
        # drop rows that have NaN in either dataframe, maintaining a valid pairing across dfs
        df1_cleaned = df1.dropna()
        df2_cleaned = df2.dropna()

        # now ensure they still have the same shape, in case dropping rows changed that
        if df1_cleaned.shape != df2_cleaned.shape:
            print("Error: DataFrames must have the same shape after removing missing data.")
            return None

        if df1_cleaned.empty or df2_cleaned.empty:
            print("Error: DataFrames cannot be empty after removing missing data.")
            return None

        differences = df1_cleaned.values - df2_cleaned.values
        w_statistic, p_value = stats.wilcoxon(differences.flatten(), zero_method="pratt")
        return w_statistic, p_value
    except Exception as e:
        print(f"Error during calculation: {e}")
        return None

# Example with missing values
data1 = {'A': [10, 12, None, 18, 20],
         'B': [25, 23, 28, 30, None],
         'C': [5, 7, 9, 11, 13]}
data2 = {'A': [9, 11, 14, 17, 19],
         'B': [24, 22, 27, 29, 26],
         'C': [6, None, 10, 12, 14]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

results = paired_wilcoxon_missing(df1, df2)

if results:
    w_stat, p_val = results
    print(f"Wilcoxon Statistic (missing data handled): {w_stat}")
    print(f"P-value (missing data handled): {p_val}")
```

In this variant, the function `paired_wilcoxon_missing` removes rows with missing values using the `.dropna()` method before proceeding. It is crucial to apply `dropna()` to *both* dataframes to ensure a matched pairing is preserved. I've incorporated additional checks after dropping NA's to verify that the shapes remain the same and the dataframes are not empty. It’s important to remember that dropping missing values is just one approach; imputation techniques can be considered as an alternative depending on the characteristics of the data and the potential impact of missing values.

For further exploration, I would recommend researching statistical analysis methods using a combination of textbooks, online courses, and statistical documentation sites. Look at resources covering nonparametric statistics, specifically the Wilcoxon signed-rank test. Consult documentation for the `pandas` and `scipy` libraries as well. Finally, consider reading material on data cleaning and preprocessing before statistical modeling.
