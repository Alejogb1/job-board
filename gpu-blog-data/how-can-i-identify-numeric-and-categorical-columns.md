---
title: "How can I identify numeric and categorical columns in a Pandas DataFrame using Pandas Profiling, focusing solely on dtype information?"
date: "2025-01-30"
id: "how-can-i-identify-numeric-and-categorical-columns"
---
The `pandas` library automatically assigns data types (`dtypes`) to DataFrame columns, and this inherent information is the basis for determining whether a column is numeric or categorical. I've encountered numerous data analysis scenarios where explicitly using `dtypes` for this purpose, rather than relying on more complex heuristics or value inspection, is the most direct and reliable approach, particularly when combined with a tool like `pandas-profiling`. Using the `dtype` attribute is also generally faster for large dataframes.

To illustrate, consider a typical workflow involving data exploration and preprocessing. The goal is often to quickly understand the characteristics of columns, dividing them into those suitable for mathematical operations (numeric) and those that represent discrete groups (categorical). This initial segmentation is crucial for later stages like feature engineering and model selection. We need a method that extracts this core type information without delving into the actual data values.

The fundamental way to identify columns of specific types using only `dtypes` lies in accessing the `.dtypes` attribute of the `DataFrame` object. This returns a `pandas.Series` with column names as the index and their corresponding data types as values. To find numeric columns, I'd typically filter this series to identify `dtypes` associated with numerical representations, such as `int64`, `float64`, and so on. Similarly, I would filter for object types that represent strings, which I usually treat as categorical. It’s crucial to note that `datetime` columns, while not technically numeric in the same way as integer or float columns, often require special handling similar to categoricals, as a straightforward numeric analysis is not usually desired.

Here's how this translates to practical code:

```python
import pandas as pd
from pandas_profiling import ProfileReport

# Assume 'df' is our DataFrame

def identify_columns_by_dtype(df):
    """
    Identifies numeric and categorical columns based on dtype alone.

    Args:
        df: pandas DataFrame.

    Returns:
        tuple: (list of numeric columns, list of categorical columns).
    """
    numeric_dtypes = ['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'uint64', 'uint32', 'uint16', 'uint8']
    categorical_dtypes = ['object', 'string', 'category']

    dtypes_series = df.dtypes
    numeric_cols = dtypes_series[dtypes_series.isin(numeric_dtypes)].index.tolist()
    categorical_cols = dtypes_series[dtypes_series.isin(categorical_dtypes)].index.tolist()
    
    return numeric_cols, categorical_cols

# Create a sample DataFrame for demonstration
data = {
    'order_id': [1, 2, 3, 4, 5],
    'customer_id': ['A101', 'A102', 'A101', 'A103', 'A102'],
    'order_amount': [25.50, 120.75, 50.00, 75.20, 150.00],
    'order_date': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-15', '2023-01-17', '2023-01-18']),
    'status': ['shipped', 'pending', 'shipped', 'processing', 'delivered'],
    'product_rating': [4, 5, 3, 5, 4],
    'category_id': pd.Categorical([10, 11, 10, 12, 11]),
}
df = pd.DataFrame(data)

numeric_cols, categorical_cols = identify_columns_by_dtype(df)
print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True,
                     variables={'descriptions': {
                                                    'order_id' : "The unique id of the order",
                                                    'customer_id': "Unique identifier for customer",
                                                    'order_amount': "Total value of the order",
                                                    'order_date' : "Date of order",
                                                    'status' : "Current status of the order",
                                                    'product_rating' : "Rating provided by the user",
                                                    'category_id': "Id of the category associated with the order"
                                                },
                                    'categorical': categorical_cols,
                                    'numeric': numeric_cols,
                                    'date': ['order_date']
                                  })
profile.to_file("pandas_profiling_report_dtypes.html")
```

In this first example, I defined a `identify_columns_by_dtype` function to encapsulate the logic. I've created lists of `numeric_dtypes` and `categorical_dtypes` to filter the output of `.dtypes`. This direct comparison with the known `dtypes` avoids any interpretation of the data itself. The sample DataFrame includes a range of typical column types, including an explicit `category` data type and a date for future reference. The print statements output the lists, verifying the logic. The `pandas_profiling` call then uses these explicit lists when generating the report. The `variables` argument is used to pass a dictionary to the profiler. The `description` keys provide information for each column. In the `categorical`, `numeric` and `date` sections, we use the lists computed to tell the profiler how to process each column. This ensures that the profiler uses the correct processing methodology for each column type, based solely on the dtype information.

Now, let's examine a slightly different scenario where we need to be a little more flexible: sometimes numeric data is imported with object types, and we want to force those to be numeric before doing profiling, or identify them separately.

```python
import pandas as pd
from pandas_profiling import ProfileReport

# Sample DataFrame with mixed type
data2 = {
    'item_id': ['1001', '1002', '1003', '1004', '1005'], # Numeric in reality but loaded as object
    'quantity': ['2', '5', '3', '1', '4'],             # Numeric in reality but loaded as object
    'price': [25.99, 42.50, 12.20, 10.00, 5.99],      # Already numeric
    'brand': ['Brand A', 'Brand B', 'Brand A', 'Brand C', 'Brand B'],
    'city' : ['New York', 'Los Angeles', 'Chicago', 'New York', 'Los Angeles'],
    'shipping_date': ['2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09'],
}
df2 = pd.DataFrame(data2)


def identify_numeric_object_columns(df):
    """
    Identifies columns that are object dtype but can be converted to numeric.

    Args:
        df: pandas DataFrame.

    Returns:
        list: List of columns names with object dtype that can be converted to numeric.
    """
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_object_cols = []
    for col in object_cols:
        try:
            df[col] = pd.to_numeric(df[col])
            numeric_object_cols.append(col)
        except ValueError:
            pass #If conversion failed, it's not a numeric column disguised as object

    return numeric_object_cols

numeric_object_cols = identify_numeric_object_columns(df2)
print("Numeric Object Columns:", numeric_object_cols)
numeric_cols_2, categorical_cols_2 = identify_columns_by_dtype(df2)
print("Numeric Columns:", numeric_cols_2)
print("Categorical Columns:", categorical_cols_2)


profile2 = ProfileReport(df2, title="Pandas Profiling Report With Forced Conversions", explorative=True,
                     variables={'descriptions': {
                                                    'item_id' : "The unique id of the item",
                                                    'quantity': "Number of items purchased",
                                                    'price': "Price of a single item",
                                                    'brand' : "Brand of the item",
                                                    'city' : "Shipping city",
                                                    'shipping_date' : "Shipping date",
                                                },
                                    'categorical': categorical_cols_2,
                                    'numeric': numeric_cols_2,
                                    'date': []
                                  })
profile2.to_file("pandas_profiling_report_forced_conversions.html")

```
In this example, I introduced a function `identify_numeric_object_columns` that tries converting object columns to numeric and stores the names of those columns that were successfully converted. After running this, we run the original `identify_columns_by_dtype` function. The resulting lists reflect the converted columns. This example demonstrates how to identify hidden numeric columns encoded as object dtypes and force conversions before running profiling. Also notice that the shipping date, which was previously a string, has not been processed as a date, so the profiler will not assume any date processing on this.

Finally, let's demonstrate how to treat datetime columns as categorical-like. Although they contain numeric information, their analysis is usually better performed using specialized libraries:
```python
import pandas as pd
from pandas_profiling import ProfileReport

# Sample DataFrame with mixed type
data3 = {
    'start_date': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-15', '2023-01-17', '2023-01-18']),
    'end_date': pd.to_datetime(['2023-01-20', '2023-01-22', '2023-01-21', '2023-01-23', '2023-01-24']),
    'score': [85, 92, 78, 88, 95],
    'comments': ['Good', 'Excellent', 'Average', 'Good', 'Outstanding'],
    'region' : ['North', 'South', 'East', 'North', 'West'],
}
df3 = pd.DataFrame(data3)


def identify_columns_for_profiling(df):
    """
    Identifies numeric and categorical columns including datetime as categorical.

    Args:
        df: pandas DataFrame.

    Returns:
        tuple: (list of numeric columns, list of categorical columns, list of date columns).
    """
    numeric_dtypes = ['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'uint64', 'uint32', 'uint16', 'uint8']
    categorical_dtypes = ['object', 'string', 'category']
    date_dtypes = ['datetime64[ns]']

    dtypes_series = df.dtypes
    numeric_cols = dtypes_series[dtypes_series.isin(numeric_dtypes)].index.tolist()
    categorical_cols = dtypes_series[dtypes_series.isin(categorical_dtypes)].index.tolist()
    date_cols = dtypes_series[dtypes_series.isin(date_dtypes)].index.tolist()
    return numeric_cols, categorical_cols, date_cols

numeric_cols_3, categorical_cols_3, date_cols_3 = identify_columns_for_profiling(df3)
print("Numeric Columns:", numeric_cols_3)
print("Categorical Columns:", categorical_cols_3)
print("Date Columns:", date_cols_3)


profile3 = ProfileReport(df3, title="Pandas Profiling Report With Datetime", explorative=True,
                     variables={'descriptions': {
                                                    'start_date' : "Start of the interval",
                                                    'end_date': "End of the interval",
                                                    'score': "Score of the entity",
                                                    'comments' : "User comments",
                                                    'region' : "Geographical region",
                                                },
                                    'categorical': categorical_cols_3,
                                    'numeric': numeric_cols_3,
                                    'date': date_cols_3
                                  })
profile3.to_file("pandas_profiling_report_datetime.html")
```

In this last example, the `identify_columns_for_profiling` function has been expanded to also identify and extract datetime columns, using an explicit `date_dtypes` variable. In this way, while not technically categorical, we are setting a specific type for the profiler, which will lead to the desired date-related processing.  The lists of numeric, categorical and date columns are printed, and then used by the profiler.

For further learning, the official pandas documentation is essential, particularly the section on data types and indexing. Additionally, exploring resources detailing best practices in data cleaning and preprocessing, especially those that emphasize handling numerical and categorical data, will prove very valuable. Finally, looking into the documentation of `pandas-profiling` will enhance your understanding of how the tool processes different dtypes based on information provided via the variables argument.
