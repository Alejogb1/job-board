---
title: "How to filter a Pandas DataFrame for rows containing specific words, excluding rows with words not on the list?"
date: "2024-12-23"
id: "how-to-filter-a-pandas-dataframe-for-rows-containing-specific-words-excluding-rows-with-words-not-on-the-list"
---

Alright, let’s tackle this. It's a challenge I remember facing often, especially back when I was working on large text datasets for sentiment analysis – getting granular control over data based on the presence (and absence) of specific terms is absolutely crucial. So, we're talking about filtering a Pandas DataFrame, focusing on rows that *contain* certain words from a predefined list, and specifically excluding rows that have words not on that same list. This is more nuanced than just a simple ‘contains’ search, and a straight loop would be too inefficient for anything but the smallest data sets.

The key here is understanding that we need to combine regular expressions for flexible string matching with Pandas' powerful boolean indexing. The direct approach, using `.str.contains()` repeatedly, while workable, can quickly become unwieldy and less performant as the number of words or the data size increases. I’ve definitely seen projects slow to a crawl because of poorly optimized string filtering, so let’s skip that and focus on clean, efficient methods.

Essentially, we will construct a regular expression pattern that dynamically accounts for the inclusion and exclusion criteria, then use that pattern to generate a boolean mask. This mask, in turn, will be used to slice our DataFrame and produce the filtered result.

First, let's define a general approach. I'll use a hypothetical scenario where I have a DataFrame with a 'text' column, and I want to filter for rows containing words from `included_words` but only *if* those rows do not also contain any words from `excluded_words`.

Here’s the first code example, breaking down how to structure this:

```python
import pandas as pd
import re

def filter_dataframe_by_words(df, text_column, included_words, excluded_words):
    """
    Filters a Pandas DataFrame to include rows containing words from included_words,
    excluding rows with words from excluded_words.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text.
        included_words (list): A list of words to include in the filter.
        excluded_words (list): A list of words to exclude from the filter.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    included_pattern = r'\b(' + '|'.join(included_words) + r')\b'
    excluded_pattern = r'\b(' + '|'.join(excluded_words) + r')\b'

    # Create masks
    inclusion_mask = df[text_column].str.contains(included_pattern, case=False, regex=True)
    exclusion_mask = ~df[text_column].str.contains(excluded_pattern, case=False, regex=True)

    combined_mask = inclusion_mask & exclusion_mask
    return df[combined_mask]


if __name__ == '__main__':
    data = {'text': [
        "This is an apple.",
        "This is a banana, and it is yellow.",
        "The quick brown fox jumps over the lazy dog.",
        "I like apple and orange.",
        "This is a bad apple.",
        "I ate a single banana.",
    ]}
    df = pd.DataFrame(data)
    included = ["apple", "banana"]
    excluded = ["bad", "orange"]
    filtered_df = filter_dataframe_by_words(df, 'text', included, excluded)
    print(filtered_df)

```

In this code, we construct the `included_pattern` and `excluded_pattern` using regular expression word boundaries (`\b`). The `\b` ensures that we are matching whole words, avoiding partial matches like "applesauce" when looking for "apple". We then create boolean masks based on whether the `text_column` contains at least one word from the `included_words` and *does not contain* any words from the `excluded_words` lists. Finally, we combine them with a logical `AND` to obtain the final `combined_mask`, which filters our DataFrame.

Now, it’s worth expanding on why regular expressions are vital for this task. If you try string comparisons without them, the process is quite inflexible and error-prone. You might inadvertently match sub-strings and it becomes impossible to handle complex criteria without manual iteration which, as I said, is not practical for anything but toy examples. Regular expressions give us the power of precise text matching with boundary conditions, case-insensitivity, and conditional searches.

Here’s another example, demonstrating case-insensitivity and slightly different data:

```python
import pandas as pd
import re

def filter_dataframe_case_insensitive(df, text_column, included_words, excluded_words):
    """
    Filters a DataFrame using case-insensitive word matching.

    Args:
         df (pd.DataFrame): The input DataFrame.
         text_column (str): The name of the column containing text.
         included_words (list): List of words to include.
         excluded_words (list): List of words to exclude.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    included_pattern = r'\b(' + '|'.join(included_words) + r')\b'
    excluded_pattern = r'\b(' + '|'.join(excluded_words) + r')\b'
    inclusion_mask = df[text_column].str.contains(included_pattern, case=False, regex=True)
    exclusion_mask = ~df[text_column].str.contains(excluded_pattern, case=False, regex=True)
    combined_mask = inclusion_mask & exclusion_mask
    return df[combined_mask]


if __name__ == '__main__':
    data = {'text': [
        "I love APPLES.",
        "Bananas are great but I also like a pear.",
        "This is a bad APPLE indeed",
        "An awesome banana dessert is ready.",
        "Oranges and apples are nutritious.",
        "The dog is running around the yard.",
    ]}
    df = pd.DataFrame(data)
    included = ["apple", "banana"]
    excluded = ["bad", "oranges", "pear"]
    filtered_df = filter_dataframe_case_insensitive(df, 'text', included, excluded)
    print(filtered_df)

```
Notice the `case=False` in the `str.contains` method. This is how you enforce case-insensitive searches. This snippet makes sure that “Apples” is treated the same as “apple,” adding flexibility.

Now, let’s consider a slightly more complex scenario. Say you want to filter for rows containing *any* of the included words, but *only* if *none* of the excluded words are present.

```python
import pandas as pd
import re

def filter_dataframe_any_included_no_excluded(df, text_column, included_words, excluded_words):
    """
    Filters rows where any included word is present, but no excluded words are.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text.
        included_words (list): List of words to include.
        excluded_words (list): List of words to exclude.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    included_pattern = r'\b(' + '|'.join(included_words) + r')\b'
    excluded_pattern = r'\b(' + '|'.join(excluded_words) + r')\b'

    inclusion_mask = df[text_column].str.contains(included_pattern, case=False, regex=True)
    exclusion_mask = ~df[text_column].str.contains(excluded_pattern, case=False, regex=True)

    combined_mask = inclusion_mask & exclusion_mask
    return df[combined_mask]


if __name__ == '__main__':
    data = {'text': [
        "A red car.",
        "A blue truck and a red car.",
        "The quick brown fox.",
        "A green car and a bicycle.",
         "The red dog and a yellow cat.",
    ]}
    df = pd.DataFrame(data)
    included = ["car", "truck"]
    excluded = ["fox", "dog"]
    filtered_df = filter_dataframe_any_included_no_excluded(df, 'text', included, excluded)
    print(filtered_df)

```

This version is structurally identical to the first and second examples. It demonstrates that the core logic of combining inclusion and exclusion via regular expressions is consistent, adapting only to the data and specific lists of words.

For more in-depth knowledge on the power of regular expressions, I'd highly recommend Jeffrey Friedl’s "Mastering Regular Expressions." It is a canonical resource that will serve you well. Additionally, for detailed understanding of pandas functionality related to string operations, the official pandas documentation is always your best bet, particularly the sections on `str` methods, as well as the text processing documentation in Python’s standard library, namely the `re` module.

In practice, I’ve found that these principles form the basis of practically all text-based data filtering tasks. The key takeaway is to use regular expressions effectively to construct precise patterns and combine boolean masks in Pandas to filter your DataFrames cleanly and efficiently.
