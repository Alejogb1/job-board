---
title: "How can pandas profiling .html output be converted to a readable tabular format in Python and PySpark?"
date: "2025-01-30"
id: "how-can-pandas-profiling-html-output-be-converted"
---
Pandas profiling, while exceptionally useful for exploratory data analysis, generates HTML reports that aren't directly compatible with tabular manipulation in Python or PySpark. I've often encountered the need to extract key statistics from these reports programmatically, especially when automating data quality checks and comparisons across datasets, instead of relying on manual interpretation of the generated HTML pages. The challenge lies in parsing the complex nested structure of the HTML to extract the relevant data points and reorganizing them for tabular presentation, and ultimately, conversion to PySpark.

First, the core issue: pandas-profiling creates an interactive HTML report encompassing multiple sections, including overall dataset statistics, individual column summaries, and interactions. The structure is not easily consumable as a simple key-value pair or a CSV/JSON format. We must treat this HTML as a document to be parsed.

The most effective approach I’ve found uses Python's `BeautifulSoup4` library for HTML parsing combined with judicious use of regular expressions and dictionary comprehension to extract information into a structured dictionary, which can then be transformed to tabular formats. This allows us to programmatically isolate sections, like the overview, column statistics, and variable properties. I generally begin by targeting specific HTML tags and classes employed by the `pandas-profiling` reports. A typical section within the report, like the summary statistics for a variable, contains nested tags such as `<div>`, `<span>`, and `<li>` with specific CSS classes that can be used as locators.

Here's a practical breakdown with code examples:

**Example 1: Extracting Overview Statistics (Python)**

```python
from bs4 import BeautifulSoup
import re

def extract_overview_stats(html_file_path):
    """Extracts overview statistics from pandas-profiling HTML report.

    Args:
        html_file_path (str): Path to the HTML report.

    Returns:
        dict: A dictionary containing overview statistics.
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    overview_div = soup.find('div', class_='overview')

    if not overview_div:
        return {}

    stats_dict = {}
    for li_item in overview_div.find_all('li'):
        span_tags = li_item.find_all('span')
        if len(span_tags) == 2:
          key = span_tags[0].get_text(strip=True)
          value = span_tags[1].get_text(strip=True)
          stats_dict[key] = value
        elif len(span_tags) > 2:  #handles cases with extra nested span, e.g., duplicate values
            key = span_tags[0].get_text(strip=True)
            value = span_tags[1].get_text(strip=True)
            if value.isnumeric():
               stats_dict[key] = value
            else:  #handle duplicate cases
               value = " ".join(span.get_text(strip=True) for span in span_tags[1:])
               stats_dict[key] = value
    return stats_dict

# Example usage:
# overview_data = extract_overview_stats("profiling_report.html")
# print(overview_data)
```

In this example, I open the HTML file, then use `BeautifulSoup` to locate the `<div>` with class `overview`. I iterate through the list items (`<li>`) within this section, extracting the text from the corresponding `<span>` tags. I perform a length check on `span_tags` to determine whether it's a key-value pair, or a potential key-value pair with a potential nested span due to duplicate values, then add that key value to the `stats_dict`, handling nested spans by joining the extracted text values together. The returned dictionary represents a flat key-value structure, making it suitable for loading into a Pandas DataFrame.

**Example 2: Extracting Column Statistics (Python)**

```python
import re
from bs4 import BeautifulSoup
import pandas as pd

def extract_column_stats(html_file_path):
    """Extracts column statistics from pandas-profiling HTML report.

    Args:
        html_file_path (str): Path to the HTML report.

    Returns:
        pandas.DataFrame: A Pandas DataFrame containing column statistics.
    """

    with open(html_file_path, 'r', encoding='utf-8') as f:
      html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')

    columns_data = []
    for section in soup.find_all('div', class_='variable'):
        col_name_element = section.find('h2')
        if col_name_element:
            col_name = col_name_element.get_text(strip=True)
        else:
            continue #if there's no header, skip

        col_stats = {'column': col_name} #initialize with name
        for li_item in section.find_all('li'):
            spans = li_item.find_all('span')
            if len(spans) == 2:
              key = spans[0].get_text(strip=True)
              value = spans[1].get_text(strip=True)
              col_stats[key] = value
            elif len(spans) > 2:  #handles cases with extra nested span, e.g., duplicate values
              key = spans[0].get_text(strip=True)
              value = " ".join(span.get_text(strip=True) for span in spans[1:])
              col_stats[key] = value
        columns_data.append(col_stats)

    return pd.DataFrame(columns_data)


# Example usage:
# column_data = extract_column_stats("profiling_report.html")
# print(column_data)

```

This function parses HTML to create a Pandas DataFrame representing column statistics, which is more suitable for further manipulation. It extracts the column name from the `<h2>` tag, and then iterates through the `<li>` tags within that section to extract key-value pairs related to that column's specific statistics. Like the first example, it uses a check on the number of spans to correctly handle cases where there are nested span tags due to duplicate values. The dictionaries for each column are then appended to a list, which is later used to initialize the Pandas DataFrame. This DataFrame can be analyzed programmatically and also be used for PySpark conversion.

**Example 3: Converting to PySpark DataFrame**

```python
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql import functions as F

def convert_to_pyspark(pandas_df, spark_session):
    """Converts a pandas DataFrame to a PySpark DataFrame.
    Args:
         pandas_df (pd.DataFrame): pandas DataFrame
         spark_session(SparkSession): pyspark spark session
    Returns:
         pyspark.sql.DataFrame: PySpark DataFrame.
    """
    if pandas_df.empty:
        return spark_session.createDataFrame([], schema=None)  # Return empty df with empty schema
    return spark_session.createDataFrame(pandas_df)

# Example usage:
# spark = SparkSession.builder.appName("ProfilingData").getOrCreate()
# column_stats_pandas = extract_column_stats("profiling_report.html")
# if not column_stats_pandas.empty:
#   column_stats_spark = convert_to_pyspark(column_stats_pandas, spark)
#   column_stats_spark.show()
# else:
#    print("No Data Retrieved from HTML Report.")
# spark.stop()
```

This snippet demonstrates how a previously extracted Pandas DataFrame can be converted to a PySpark DataFrame. First a Spark session must be established. I use a basic check to determine if a DataFrame is empty, and return an empty PySpark dataframe with a schema. The Pandas DataFrame is ingested directly via `spark.createDataFrame()`. The subsequent example shows how I call this function after extracting the column stats from the previous example, in the process converting it to a PySpark DataFrame, which can be used for Spark data analysis.

**Resource Recommendations:**

* **BeautifulSoup4 Documentation:** A comprehensive guide to parsing HTML and XML documents. Learning to navigate its API, specifically the find and find_all methods for element selection, is vital.
* **Pandas Documentation:** For tabular data manipulation after extraction from HTML reports, specifically methods for DataFrame creation, selection, and operations.
* **PySpark Documentation:** The definitive source for using PySpark DataFrames. The usage of the `createDataFrame` method, as well as methods to apply Spark SQL functions, are crucial here.
* **Python Standard Library `re` Module:** For using regular expressions, primarily when cleaning text data extracted from HTML tags, or for more complex tag selection patterns.

In my experience, focusing on specific CSS classes and HTML tag structures employed by the pandas-profiling report ensures reliable extraction. The keys of the returned dictionaries can serve as schema for the PySpark DataFrames. This approach allows for automated quality checks, enables comparisons between various profiling reports, and can be used as a basis for more complex ETL processes. Ultimately, it converts the interactive HTML reports into machine-readable formats suitable for programmatic analysis and large-scale data processing.
