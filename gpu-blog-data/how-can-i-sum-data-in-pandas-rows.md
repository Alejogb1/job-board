---
title: "How can I sum data in Pandas rows matching keywords using `str.contains`?"
date: "2025-01-30"
id: "how-can-i-sum-data-in-pandas-rows"
---
The efficiency of vectorized operations in Pandas allows for rapid data processing when summing row values based on keyword matches within string columns. Direct application of `str.contains` coupled with boolean indexing and summation methods provides a highly performant solution. Over years working with financial datasets, I've frequently utilized this approach to categorize and aggregate data based on textual descriptions, making this a cornerstone technique for my work.

The core mechanism involves the `str.contains` method, which, when applied to a Pandas Series containing strings, returns a boolean mask indicating where the specified pattern is found. This mask can then be used to index the DataFrame, selecting only the rows where the condition evaluates to True. Subsequently, numeric columns from these filtered rows can be summed to arrive at aggregated values. The flexibility lies in the pattern argument of `str.contains`, allowing for both simple string matches and more sophisticated regular expressions for complex keyword searches.

It's crucial to understand the implications of case sensitivity and potential handling of missing values (NaN) when using `str.contains`. By default, the method is case-sensitive. This can be controlled using the `case` argument. Further, if the column containing strings includes `NaN`, `str.contains` might return `NaN` for those rows depending on the chosen behaviour. Proper handling is crucial to ensure accurate results.

Here are a few examples illustrating various use cases, each accompanied by a discussion of their implementation and nuances.

**Example 1: Basic Keyword Search and Summation**

In this scenario, assume a DataFrame representing transaction data, with columns for 'Description' and 'Amount'. The task is to sum all transaction amounts where the description contains the keyword "refund".

```python
import pandas as pd

data = {'Description': ['Purchase of goods', 'Refund of order', 'Service fee', 'Refund of items', 'Payment received'],
        'Amount': [100, -50, 20, -30, 75]}
df = pd.DataFrame(data)

keyword = 'refund'
refund_amounts = df[df['Description'].str.contains(keyword, case=False)]['Amount'].sum()

print(f"Total refund amount: {refund_amounts}")

```

**Commentary:**

This example directly addresses the basic problem. `df['Description'].str.contains(keyword, case=False)` creates a boolean mask that evaluates to `True` where the 'Description' column contains the word "refund" (case-insensitive due to `case=False`). This mask is then used to filter the DataFrame, selecting only the rows that meet the condition. The `.sum()` method is then applied to the 'Amount' column of the filtered DataFrame, producing the total refund amount. It is crucial to specify `case=False` if you intend to match “Refund”, “refund”, and “REFUND” etc.

**Example 2: Handling Multiple Keywords with Regular Expressions**

Consider a similar scenario, but this time needing to sum amounts that contain either “insurance” or “premium” in the description. This requires using regular expressions within `str.contains`.

```python
import pandas as pd

data = {'Description': ['Insurance payment', 'Premium collection', 'Goods purchase', 'Insurance adjustment', 'Monthly fee', 'Premium refund'],
        'Amount': [150, 200, 80, -25, 30, -50]}
df = pd.DataFrame(data)

keywords = 'insurance|premium'
insurance_premium_amounts = df[df['Description'].str.contains(keywords, case=False, regex=True)]['Amount'].sum()

print(f"Total insurance/premium amount: {insurance_premium_amounts}")

```

**Commentary:**

Here, `keywords` is set to a string representing a regular expression: `"insurance|premium"`. The `|` character acts as an OR operator within the regular expression, meaning the `str.contains` will evaluate to `True` if the string in the ‘Description’ column matches either “insurance” or “premium”. The parameter `regex=True` activates regex functionality. Consequently, the boolean mask will flag rows containing either keyword and the `.sum()` method operates on the ‘Amount’ column. This technique can significantly broaden the flexibility of keyword searches. Again, `case=False` makes the search insensitive to casing.

**Example 3:  Handling NaN Values and Conditional Summation**

In situations where the description column may contain missing values (NaN), special handling might be needed. In some cases, the `NaN` entries may need to be excluded or handled as a specific case. Here's an example of excluding them by making sure only valid strings are considered in the match process.

```python
import pandas as pd
import numpy as np

data = {'Description': ['Maintenance costs', np.nan, 'Repair work', 'Maintenance service', 'Payment due', None],
        'Amount': [50, 10, 80, 40, -10, np.nan]}
df = pd.DataFrame(data)


keyword = 'maintenance'

maintenance_amounts = df[df['Description'].notna() & df['Description'].str.contains(keyword, case=False)]['Amount'].sum()

print(f"Total maintenance amount: {maintenance_amounts}")
```

**Commentary:**

This example demonstrates handling missing data effectively. First, `df['Description'].notna()` creates a boolean mask that returns `True` for rows where the 'Description' is not NaN or None. This mask is combined with the result of the `str.contains` using the logical AND operator `&`. The result is a mask that only returns `True` for the rows that: 1) have a valid 'Description' value and 2) have a ‘Description’ which contains the keyword. If the `NaN` values are not handled explicitly in this way then the program would generate an error.

Resource recommendations for delving deeper into these techniques include the official Pandas documentation, particularly sections on string methods, boolean indexing, and grouping/aggregation. Additionally, the Python standard library documentation regarding regular expressions provides valuable information for constructing more complex search patterns. Textbooks focusing on data analysis with Python, such as those covering Pandas and NumPy, are also crucial. For practical exercises, open-source projects that involve textual data analysis, available on platforms such as GitHub, are an excellent way to hone proficiency. Furthermore, experimentation with varying datasets and keywords is essential to solidify understanding and develop expertise. Proficiency in the combination of string manipulation, logical indexing, and aggregation techniques is key for effective data analysis using pandas.
