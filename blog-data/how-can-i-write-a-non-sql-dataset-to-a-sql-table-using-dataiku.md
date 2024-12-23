---
title: "How can I write a non-SQL dataset to a SQL table using DataIku?"
date: "2024-12-23"
id: "how-can-i-write-a-non-sql-dataset-to-a-sql-table-using-dataiku"
---

Let's tackle this. I remember a particularly frustrating project a few years back. We had this system that churned out data in, let's call it, a semi-structured format—think nested json objects mixed with some flat csv-like data. The target was, of course, a well-structured postgresql database. Dataiku’s inherent capabilities couldn't directly consume it, so we had to get a bit inventive to get it into the SQL tables. It's a very common scenario, and Dataiku, while powerful, often needs a helping hand when faced with non-standard data structures moving into a rigid schema. The key here is understanding data transformations and leveraging Dataiku’s flexibility.

The core issue isn't about a direct "write non-sql to sql" action—Dataiku doesn’t offer magic that simply bypasses the structured nature of databases. Instead, we need to shape the non-sql data into a tabular format that mirrors our target SQL table schema before inserting it. Think of it as translation; you're converting from a language the database doesn’t speak into one it understands.

First, let's address the general process and then I'll show some example code snippets:

1.  **Data Ingestion and Parsing:** The initial step is reading your non-sql data into Dataiku. Depending on the format, you might use Dataiku's built-in connectors, custom Python recipes, or, in my case with semi-structured files, a combination of both. This involves parsing the data and extracting the key-value pairs or individual data points you need.
2.  **Data Transformation and Restructuring:** This is where the bulk of the work usually happens. You'll typically need to use Python scripting or other data manipulation tools within Dataiku to restructure your data. This might include flattening nested structures, renaming fields, data type conversions, or applying business logic to cleanse and derive data. The goal is to create a tabular structure (a pandas dataframe, for instance) that matches your SQL table's columns.
3.  **SQL Table Creation or Update:** If your SQL table doesn't already exist, you can create it through Dataiku's SQL connectors, or, if preferred, using SQL scripts outside of Dataiku and then connecting. If the table exists, ensure your transformed data structure aligns with the existing schema.
4.  **Data Loading (Write):** Finally, once you have a structured dataframe mirroring your target SQL table, you can leverage Dataiku's connectors to append the dataframe to the table. It can do bulk inserts, or, with more complexity, handle upserts and similar operations.

Now, let me illustrate with some examples. Note that these snippets assume you're working inside a Dataiku Python recipe.

**Example 1: Flattening a JSON structure and loading it into a table**

Suppose you have a json-like string field in your input dataset, and you need to flatten it into columns of your SQL table.

```python
import pandas as pd
import json
from dataiku.dataset import Dataset
from dataiku import pandasutils as pdu

# 'input_dataset' is the name of your Dataiku dataset containing the json-like field
input_dataset = Dataset("input_dataset")
df = input_dataset.get_dataframe()

def flatten_json(row):
    try:
        data = json.loads(row['json_field']) # Assuming your json field is named 'json_field'
        return pd.Series(data)
    except (json.JSONDecodeError, TypeError):
        return pd.Series()

# Apply the function to create new columns
df = df.join(df.apply(flatten_json, axis=1))

# Select only columns necessary for the SQL table.  Adapt the column names as needed.
columns_to_keep = ['id', 'name', 'value', 'timestamp']
df_sql_ready = df[columns_to_keep]

# 'sql_output_dataset' is your SQL output dataset configured in Dataiku.
output_dataset = Dataset("sql_output_dataset")
pdu.write_to_dataset(output_dataset, df_sql_ready) # or use output_dataset.write_dataframe(df_sql_ready)

```

In this example, `flatten_json` attempts to parse the JSON, turning it into a pandas series with each key as a column, and the values corresponding. We then join these series back to our main dataframe, choose the pertinent columns that match the SQL table schema, and write the resulting dataframe to our output SQL table via a Dataiku connection.

**Example 2: Pivoting data and inserting it**

Let's say you have data in a "long" format where categories are in one column, and values are in another, but the SQL table has a column for each category. This means pivoting will be necessary.

```python
import pandas as pd
from dataiku.dataset import Dataset
from dataiku import pandasutils as pdu

# 'input_dataset_long' is the name of your Dataiku dataset in long format
input_dataset_long = Dataset("input_dataset_long")
df_long = input_dataset_long.get_dataframe()

# pivot the data. The "id" and "timestamp" are identifiers, "category" holds categories and 'value' contains values
df_pivot = df_long.pivot(index=['id', 'timestamp'], columns='category', values='value').reset_index()

# Rename columns to match SQL table.
df_pivot = df_pivot.rename(columns={"category1": "col_category1", "category2": "col_category2"})

# 'sql_output_dataset_pivoted' is your SQL output dataset configured in Dataiku.
output_dataset_pivoted = Dataset("sql_output_dataset_pivoted")
pdu.write_to_dataset(output_dataset_pivoted, df_pivot)
```

Here, `pivot` is used to transform a long-formatted dataset into a wide-format one, so it matches the desired output table schema with each category as a dedicated column. Column renaming ensures data mapping is correct.

**Example 3: Handling non-standard CSV files**

Suppose your input csv files are not standard, perhaps missing headers or with inconsistent delimiters.

```python
import pandas as pd
from dataiku.dataset import Dataset
from dataiku import pandasutils as pdu
import io

# 'input_dataset_raw_csv' is your Dataiku dataset pointing to the csv files
input_dataset_raw_csv = Dataset("input_dataset_raw_csv")

# Read the files, skipping first few rows and using a different delimiter
def read_csv_and_apply(stream, path, file_name, context):
   csv_string = stream.read().decode()
   df = pd.read_csv(io.StringIO(csv_string), delimiter=";", skiprows=2, header=None)
   df.columns = ['id', 'col_1', 'col_2', 'timestamp'] # Set headers manually
   return df

df_all = input_dataset_raw_csv.read_with_schema(read_csv_and_apply)

# 'sql_output_dataset_csv' is your SQL output dataset configured in Dataiku.
output_dataset_csv = Dataset("sql_output_dataset_csv")
pdu.write_to_dataset(output_dataset_csv, df_all)

```
This example shows how you might read from multiple non-standard csv files. It includes a function `read_csv_and_apply` to manipulate the data using pandas before writing it to the dataset. This demonstrates the flexibility of applying custom transformations to the input data.

**Key points to consider**

*   **Data quality:** Pre-processing steps are crucial. Handle missing data, type mismatches, and inconsistencies to avoid problems down the line.
*   **Schema alignment:** Ensure your transformed dataframe's columns precisely match your SQL table's schema. Pay special attention to data types (integer vs. string, timestamp formats, etc.).
*   **Performance:** For very large datasets, optimizing transformations in Python and using batch inserts can improve the write performance.
*   **Error handling:** Implement proper error catching and logging during transformation and write processes for debugging. Dataiku's logging capabilities can be very helpful for this.

**Further Reading:**

For a deeper dive into data manipulation with pandas, the official pandas documentation is essential. Specifically, familiarize yourself with functions for reading data (like `read_csv`, `read_json`), data transformations (`pivot`, `melt`, `apply`), and writing. Also, exploring books like "Python for Data Analysis" by Wes McKinney is very helpful for gaining a solid understanding. For understanding database concepts and SQL, "SQL for Dummies" can be a surprisingly practical resource, covering many common SQL dialects. In terms of advanced data engineering concepts, "Designing Data-Intensive Applications" by Martin Kleppmann is great for a solid theoretical foundation, if you find the technical side fascinating.

In summary, the process of writing non-SQL data to a SQL table in Dataiku requires careful transformation. Dataiku gives you the flexibility to tackle most scenarios with code snippets like the examples. Remember the process is always about transforming data into the specific structure expected by the target table. With the proper transformations and code, it becomes a very manageable process. I hope this gives you a solid start.
