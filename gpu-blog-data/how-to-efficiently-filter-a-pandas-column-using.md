---
title: "How to efficiently filter a Pandas column using multiple `str.contains` conditions?"
date: "2025-01-30"
id: "how-to-efficiently-filter-a-pandas-column-using"
---
The performance of filtering a Pandas DataFrame column based on multiple string patterns using iterative `str.contains` can degrade significantly with larger datasets. Vectorized operations are critical for efficient data manipulation in Pandas, and several techniques can dramatically improve filter performance over naive approaches. My experience working with financial transaction data, where I frequently had to isolate records based on complex text descriptions, highlighted the limitations of iterative filtering and spurred the need for optimized techniques.

The primary issue with chained `df[df['column'].str.contains('pattern1')] [df['column'].str.contains('pattern2')]` operations is that each filter generates a new boolean mask and effectively copies a potentially large DataFrame. This intermediate copying contributes heavily to processing time and memory usage. Instead, we aim to create a single boolean mask representing the combined filtering criteria.

One robust method is to use the power of regular expressions within a single `str.contains` operation. This approach consolidates multiple search patterns into a unified regex, leveraging the internal optimization Pandas offers for regular expression matching. The "or" operator, represented by `|`, can efficiently combine multiple string patterns within a single regex.

**Code Example 1: Regex-based Filtering**

```python
import pandas as pd

# Sample DataFrame
data = {'description': ['Transaction A - Approved', 'Transaction B - Declined', 
                        'Transaction C - Review', 'Transaction D - Approved - High Risk',
                        'Transaction E - Fraudulent', 'Transaction F - Under Review']}
df = pd.DataFrame(data)

# Desired patterns
patterns = ['Approved', 'Review', 'Fraudulent']

# Create the regex pattern
regex_pattern = '|'.join(patterns)

# Filter using the combined regex
filtered_df = df[df['description'].str.contains(regex_pattern, na=False)]

print("Filtered DataFrame (Regex):\n", filtered_df)
```

In this first example, I first generate a sample Pandas DataFrame to mimic text data that one would find in transaction records. Next, a list named "patterns" is constructed that contains the various substrings to find in the data. I use the `join` method on a pipe character `'|'` to take that list of strings and create a single, combined regex string, which serves as the search pattern for `str.contains`. The `na=False` handles NaN values appropriately, preventing any errors in cases of missing text data. This entire operation is vectorized within pandas. This avoids creating temporary dataframes. The resulting filtered DataFrame contains only rows where the description matches one of the patterns.

While the regex approach is efficient and powerful, for very large lists of filter patterns or scenarios with exact string matches, using a method like `isin` with a set can be more performant than regex. The `isin` method checks if elements within a series are present in a list or set, allowing direct, rather than regex, pattern comparisons.

**Code Example 2: isin-based Filtering with Preprocessing**

```python
import pandas as pd

# Sample DataFrame (expanded)
data = {'description': ['Transaction A - Approved', 'Transaction B - Declined',
                        'Transaction C - Review', 'Transaction D - Approved - High Risk',
                        'Transaction E - Fraudulent', 'Transaction F - Under Review',
                        'Transaction G - Approved', 'Transaction H - Reviewer',
                        'Transaction I - Fraud', 'Transaction J - Reviewed']}
df = pd.DataFrame(data)

# Desired patterns
patterns = ['Approved', 'Review', 'Fraudulent', 'Reviewed']

# Convert column to lowercase and exact match
filtered_df = df[df['description'].str.lower().isin([pattern.lower() for pattern in patterns])]

print("Filtered DataFrame (isin):\n", filtered_df)
```

Here, an expanded DataFrame is created with more complex text that may not match the patterns directly due to case sensitivity or extra words. I then use `str.lower()` and list comprehension to handle case sensitivity by converting both the descriptions and the patterns to lowercase before comparison with the `isin()` function. I chose to do an exact match with `isin()` here, which allows for different behavior compared to `str.contains()`, namely a pattern must exactly equal a description string, in a case insensitive way. It is important to note that although `isin()` is usually faster for matching against exact substrings, pre-processing the dataframe with `.str.lower()` adds overhead, which is why regex is usually a better default. However, if you want to match full strings only, `isin()` might be preferrable.

Sometimes, you may need to use a combination of filtering techniques. For instance, if you need to match one set of strings exactly, and another based on regex, the most straightforward way to do so is to create a mask using each filtering technique, and then combine the masks using the `|` bitwise operator.

**Code Example 3: Combined Filtering**

```python
import pandas as pd

# Sample DataFrame
data = {'description': ['Transaction A - Approved', 'Transaction B - Declined',
                        'Transaction C - Review', 'Transaction D - Approved - High Risk',
                        'Transaction E - Fraudulent', 'Transaction F - Under Review',
                        'Transaction G - Approved', 'Transaction H - Reviewer',
                         'Transaction I - Fraud', 'Transaction J - Reviewed',
                         'Transaction K - Cancelled - Fraudulent']}
df = pd.DataFrame(data)

# Exact match patterns
exact_patterns = ['Approved', 'Reviewed']

# Regex patterns
regex_patterns = ['Fraud', 'Review']

# Create masks
exact_match_mask = df['description'].str.lower().isin([pattern.lower() for pattern in exact_patterns])
regex_mask = df['description'].str.contains('|'.join(regex_patterns), na=False, case=False)

# Combine masks using OR operator
combined_mask = exact_match_mask | regex_mask

# Apply combined mask
filtered_df = df[combined_mask]

print("Filtered DataFrame (Combined):\n", filtered_df)
```

In this third example, I demonstrate a more complex filtering scenario. I split the filtering criteria into two groups: those requiring exact matching and those using a regex match. I create two masks, named `exact_match_mask` and `regex_mask`, using the filtering methods demonstrated earlier. Then, I use the bitwise OR operator `|` to combine the two masks into one single mask. This results in a DataFrame filtered by any text that matches the exact match criteria, or the regex criteria, or both. This can be an extremely powerful method for efficiently and effectively filter data using complex logic.

In practice, I have found these approaches to be substantially more efficient than iteratively applying `str.contains`, particularly when processing datasets with millions of rows. Choosing the most appropriate method involves considering factors such as the complexity of the patterns, their relative quantity, and whether exact matches or pattern searches are needed. When working with a single list of strings for searching, the regex approach tends to be the fastest option. When dealing with exact string matches, `isin` can be more efficient. For a mixture, combined methods can be best.

For further learning, resources focusing on Pandas performance best practices and regular expressions are highly beneficial. Specifically, studying vectorized operations within Pandas documentation and tutorials can provide a comprehensive understanding of the optimization mechanisms available. Exploring literature focused on regular expressions can further refine your ability to construct efficient search patterns. Finally, studying boolean indexing methods within pandas can provide additional ways to efficiently filter data.
