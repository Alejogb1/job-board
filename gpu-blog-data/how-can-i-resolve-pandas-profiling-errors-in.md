---
title: "How can I resolve pandas profiling errors in Jupyter Notebook?"
date: "2025-01-30"
id: "how-can-i-resolve-pandas-profiling-errors-in"
---
Specifically, I am encountering issues like 'TypeError: unhashable type: 'list'' and 'AttributeError: ‘NoneType’ object has no attribute ‘lower’' when generating reports using pandas profiling.

When profiling data with pandas, the `TypeError: unhashable type: 'list'` and `AttributeError: ‘NoneType’ object has no attribute ‘lower’` are frequently symptomatic of data cleaning deficiencies prior to report generation. These errors indicate that the profiling library, pandas-profiling in this case, is encountering data structures or values it cannot process within the input dataframe.  My experience has shown that while the core pandas operations might appear to function, pandas-profiling imposes stricter constraints on data types and content.  It is paramount to scrutinize and preprocess data thoroughly before attempting to generate a profiling report.

The `TypeError: unhashable type: 'list'` error emerges from the pandas-profiling library attempting to use list-valued cells within a dataframe as keys, typically during frequency or value counting. Hashable data types are immutable, such as integers, strings, and tuples, permitting them to be used as keys in dictionaries or set members.  Lists, being mutable, are not hashable, leading to this specific error when pandas-profiling encounters them. This issue often arises when you have loaded data without properly handling nested structures or when you've performed a transformation that resulted in list-like entries within dataframe columns.

The `AttributeError: ‘NoneType’ object has no attribute ‘lower’` signals that the profiling function encountered a `None` value where a string method, specifically `lower()` in this case, was expected.  Pandas profiling applies string operations for certain statistical calculations or visualization labels. This commonly appears in textual columns or when null values (represented as `None`) were not explicitly converted to strings or handled properly. The underlying issue is that a `NoneType` does not have a `lower()` method, which leads to the `AttributeError`.

To resolve these types of errors effectively, a methodical approach is needed, focusing on data inspection and specific pre-processing.  I find that targeting the most likely culprits by first identifying columns that might contain lists or null values provides the most expedient solution.

**Code Example 1: Handling List-Like Data**

This example demonstrates a scenario where list-like entries exist within a dataframe column and how to convert them to strings before profiling.

```python
import pandas as pd
from pandas_profiling import ProfileReport

# Example dataframe with a list in 'tags' column
data = {'id': [1, 2, 3],
        'name': ['Product A', 'Product B', 'Product C'],
        'tags': [['red', 'blue'], ['green'], ['yellow', 'purple', 'black']]}
df = pd.DataFrame(data)

# Identify the column with list-like entries (using df.applymap() is safer than .apply() when we want to apply transformation to every cell)
list_columns = df.applymap(type).apply(lambda col: any(t==list for t in col)).to_dict() # returns dict of column name : True/False

for col, isList in list_columns.items():
   if isList:
      # Convert list-like cells to comma-separated strings
      df[col] = df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

# Now generate report (this should not throw TypeError related to list)
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("list_handling_report.html")
```

In this example, I am first explicitly identifying list-type columns by checking the type of each element within the dataframe. Then, for the columns identified as containing list-type elements, I'm converting them to comma-separated strings. I achieve that using a combination of `applymap` and `apply`.  The conditional statement ensures that only list cells are processed.   This approach ensures that pandas-profiling only receives string data, circumventing the `TypeError`. The output is then written to an html file which allows easy inspection of the profiled data.

**Code Example 2: Handling Null Values**

This example illustrates how to address null values causing `AttributeError` errors.

```python
import pandas as pd
from pandas_profiling import ProfileReport

# Example dataframe with null values in 'description' column
data = {'id': [1, 2, 3, 4],
        'description': ['Short Description', None, 'Another Description', 'Final entry']}
df = pd.DataFrame(data)


# Identify columns that might have null values.
null_columns = df.isnull().any().to_dict()

# Replace null values with an empty string (or another placeholder as needed)
for col, hasNull in null_columns.items():
  if hasNull:
    df[col] = df[col].fillna('')

# Alternatively, we could convert to string
# for col, hasNull in null_columns.items():
#    if hasNull:
#        df[col] = df[col].astype(str)

#Generate the profiling report
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("null_handling_report.html")
```

Here, the `isnull().any()` method quickly identifies columns with missing values.  The crucial step is using `fillna('')` to substitute `None` values with an empty string.  Alternatively, one can use `astype(str)` to convert the column data type to a string to avoid the null issue. This ensures that string methods in pandas-profiling, like `lower()`, will not be applied to `None` values, preventing the `AttributeError`. Both strategies effectively remove the cause of the error. I opted for `fillna('')` as it is often a more robust approach to missing string data. The output is then written to an html file which allows easy inspection of the profiled data.

**Code Example 3: Comprehensive Data Preprocessing**

This example combines handling both list-like data and null values in one comprehensive function.

```python
import pandas as pd
from pandas_profiling import ProfileReport

# Example dataframe with combined issues
data = {'id': [1, 2, 3, 4],
        'tags': [['red', 'blue'], None, ['green'], 'yellow'],
        'description': ['Short Description', None, ['Another', 'Description'], 78]}
df = pd.DataFrame(data)


def preprocess_for_profiling(df):
    """Preprocesses dataframe for pandas-profiling, handling lists and nulls."""
    for col in df.columns:

        # Check for nulls first before converting to strings in case nan is present
        if df[col].isnull().any():
            df[col] = df[col].fillna('')


        # Convert list-like cells to comma separated strings.
        if df[col].apply(lambda x: isinstance(x, list)).any():
          df[col] = df[col].apply(lambda x: ', '.join(map(str,x)) if isinstance(x, list) else str(x))

        # Convert non-string column to string to handle potential errors during profiling
        if df[col].apply(lambda x: not isinstance(x, str)).any():
          df[col] = df[col].astype(str)
    return df


df = preprocess_for_profiling(df)

profile = ProfileReport(df, title="Combined Preprocessing Report")
profile.to_file("combined_preprocessing_report.html")

```
This function, `preprocess_for_profiling`, encapsulates the previously shown strategies for list-like entries and null values, but now, it also casts any non-string column as a string to handle other potential problems that profiling library might encounter. Iterating over the columns, I check for null values first and use `fillna('')`. Then, I identify list-like entries using `.apply` and convert them to strings. Finally, if the column still contains non-string cells, it converts the column to a string. This approach provides a reusable and comprehensive solution, handling the most common data issues during pandas profiling.  The resulting output is again an html file suitable for reviewing profiled data.

To further improve one’s capability with pandas and profiling, I would suggest exploring the official pandas documentation, especially the sections on data cleaning and string operations. Additionally, the pandas-profiling GitHub repository contains a wealth of information about issues and solutions.  Further investigation of articles and tutorials focused on data preprocessing and data cleaning using pandas will also provide helpful context and techniques.  Understanding the specific data types and their implications within pandas is crucial to avoiding these common profiling errors. The ability to systematically analyze and preprocess the data, as demonstrated, ensures more efficient and successful profiling reports.
