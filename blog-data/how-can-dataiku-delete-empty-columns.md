---
title: "How can DATAIKU delete empty columns?"
date: "2024-12-23"
id: "how-can-dataiku-delete-empty-columns"
---

Alright, let's talk about dealing with those pesky empty columns in Dataiku. I've certainly spent my fair share of time battling those, especially during larger data migrations a few years back when we inherited a particularly... *robust* dataset. The sheer volume of empty columns—present but entirely devoid of meaningful data—was staggering. It was like archaeology, sifting through layers of potentially useful data only to find vast stretches of nothingness.

There isn't a single, magic-bullet 'delete empty columns' button in Dataiku, and frankly, that's a good thing. We need a bit more control to ensure we aren’t inadvertently nuking something valuable. Instead, we leverage Dataiku's flexible data manipulation tools, primarily using Python recipes for this sort of task. It offers the most granular approach to filtering these columns based on our specific needs.

The core challenge isn't just identifying empty columns (that’s relatively straightforward); it’s defining what *empty* actually means in the context of your data. Are we talking purely null values (`None` in Python/Pandas terms, or SQL `NULL`)? Or do we also consider columns that contain just empty strings, or perhaps whitespace-only strings? These are nuances that can have big implications.

So, in practice, I found the most reliable method is to construct a small Python recipe that handles these different scenarios elegantly. First, let's look at the basic case, where we're dealing with columns that are *strictly* full of null values. This leverages pandas, which is already available in a Dataiku python recipe environment.

```python
import pandas as pd
from dataiku import pandasutils as pdu

# Retrieve the input dataset as a Pandas dataframe
input_dataset = dataiku.Dataset("your_input_dataset_name")
df = input_dataset.get_dataframe()

# Identify columns with only null values
null_columns = df.columns[df.isnull().all()].tolist()

# Drop the identified columns
df_cleaned = df.drop(columns=null_columns)

# Write the cleaned dataframe to the output dataset
output_dataset = dataiku.Dataset("your_output_dataset_name")
output_dataset.write_dataframe(df_cleaned)
```

In this first example, `df.isnull().all()` produces a boolean series that is `True` if all values in a column are `null`, and `False` otherwise. We use this boolean mask to filter out only the columns that have null for every row. We then retrieve those column names with `.tolist()`. This list is used to drop the identified columns. This is the most conservative version – it will only remove columns that are *completely* empty of any data. This is a good place to start.

However, let's say we also want to handle columns containing only empty strings, or even strings containing whitespace. We need a slightly more nuanced approach. Here's a refined version to accommodate those cases:

```python
import pandas as pd
from dataiku import pandasutils as pdu

# Retrieve the input dataset as a Pandas dataframe
input_dataset = dataiku.Dataset("your_input_dataset_name")
df = input_dataset.get_dataframe()

def is_empty(series):
    """Checks if a pandas Series is "empty" considering nulls, empty strings and whitespace."""
    return series.isnull().all() or (series.astype(str).str.strip() == "").all()

# Identify columns that are "empty" based on our function
empty_columns = df.apply(is_empty).index[df.apply(is_empty)].tolist()

# Drop the identified columns
df_cleaned = df.drop(columns=empty_columns)

# Write the cleaned dataframe to the output dataset
output_dataset = dataiku.Dataset("your_output_dataset_name")
output_dataset.write_dataframe(df_cleaned)
```

Here, we've created a helper function, `is_empty`. This function first checks if all the values are null. If that fails, then it casts the column to strings, removes leading and trailing whitespace, and then tests if all the elements are empty strings.  The `apply()` method then applies this function to every column in our dataset, and we only select those columns where our `is_empty` function returns `True`. This handles the cases where you might have a column containing `""`, `" "`, or similar.

Now, this is usually sufficient, but sometimes, one may want to refine that logic even further. Perhaps, you might need to exclude specific columns from removal, or have a more complex criterion for what constitutes “empty”. Imagine a situation where, in addition to the prior case, you want to keep the `id` or `record_number` column, even if it's currently full of `null` values. In some cases these IDs are generated before we have values.

```python
import pandas as pd
from dataiku import pandasutils as pdu

# Retrieve the input dataset as a Pandas dataframe
input_dataset = dataiku.Dataset("your_input_dataset_name")
df = input_dataset.get_dataframe()

def is_empty(series):
    """Checks if a pandas Series is "empty" considering nulls, empty strings and whitespace."""
    return series.isnull().all() or (series.astype(str).str.strip() == "").all()

# Define columns to exclude from removal
exclude_columns = ['id', 'record_number']

# Identify columns that are "empty", excluding specified columns
empty_columns = df.apply(is_empty).index[df.apply(is_empty) & ~df.columns.isin(exclude_columns)].tolist()

# Drop the identified columns
df_cleaned = df.drop(columns=empty_columns)

# Write the cleaned dataframe to the output dataset
output_dataset = dataiku.Dataset("your_output_dataset_name")
output_dataset.write_dataframe(df_cleaned)
```

In this final example, we've incorporated the `exclude_columns` list and added that filtering inside the column selection expression `df.apply(is_empty) & ~df.columns.isin(exclude_columns)`. The `~` character means "not" allowing us to only select columns that are both empty and not on the exclusion list.

For more profound understanding of data manipulation techniques using pandas, consider exploring "Python for Data Analysis" by Wes McKinney, the creator of pandas. This book is a foundational resource. Additionally, for a deeper grasp of the nuances of working with missing data in data processing pipelines, the documentation for pandas' handling of null and `NA` values is essential. Understanding how the `isnull()`, `dropna()`, and `fillna()` functions operate will considerably bolster your ability to manage complex datasets. Also, "Data Wrangling with Python" by Jacqueline Nolis and Liudmila Sultanova can be useful for broader data cleaning practices.

Finally, remember to thoroughly test your cleaning recipe on a subset of your dataset before applying it to the entire dataset, just as a habit. Also consider adding logging and metrics for monitoring the removal. This can help catch subtle bugs that may manifest with larger data volumes. After this you should have an easy way to keep your data neat and tidy.
