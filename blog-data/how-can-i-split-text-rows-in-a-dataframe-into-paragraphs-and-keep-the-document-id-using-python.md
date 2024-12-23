---
title: "How can I split text rows in a dataframe into paragraphs and keep the document ID using Python?"
date: "2024-12-23"
id: "how-can-i-split-text-rows-in-a-dataframe-into-paragraphs-and-keep-the-document-id-using-python"
---

,  I've seen this problem pop up in various data science projects over the years, particularly those dealing with unstructured text documents. We often get dataframes where each row contains potentially multiple paragraphs lumped together, and for deeper analysis, we need to break these down. Keeping track of the original document id is crucial, of course. I’ll walk you through how I’ve approached this using python, ensuring each paragraph is associated with its originating document.

The core idea revolves around identifying paragraph delimiters, which are often represented by one or more newline characters (`\n`). However, edge cases always seem to find a way in, so we need to be a bit more robust. For example, sometimes you'll find consecutive newline characters, extra whitespace, or even a mix of carriage returns and newlines (`\r\n`). I'll provide a solution that takes those into account, using `pandas` and `re` (regular expressions).

**Conceptual Approach**

First, we’ll iterate through each row of your dataframe. Within each row, the text needs to be split into paragraphs using our defined delimiter. A regular expression will help us capture and clean up various newline formats. We then reconstruct a new dataframe from the paragraphs and their associated document ids. This way, each paragraph will become its own row, while the `document_id` is preserved. Let's get into the details.

**Code Implementation & Explanation**

Here’s the first method I’ve found useful, using list comprehensions for a concise solution:

```python
import pandas as pd
import re

def split_dataframe_paragraphs_1(df, text_column, id_column):
    """Splits text rows into paragraphs and keeps the document ID using list comprehension.

    Args:
        df (pd.DataFrame): The input dataframe.
        text_column (str): The name of the column containing text.
        id_column (str): The name of the column containing document IDs.

    Returns:
        pd.DataFrame: A new dataframe with each paragraph in a separate row.
    """
    new_rows = []
    for index, row in df.iterrows():
        doc_id = row[id_column]
        text = row[text_column]

        if isinstance(text, str): # check if is a string to avoid errors
           paragraphs = [para.strip() for para in re.split(r'\s*\n+\s*', text) if para.strip()] # cleaning
           for paragraph in paragraphs:
                new_rows.append({'paragraph': paragraph, id_column: doc_id})
        else:
           new_rows.append({'paragraph':"", id_column: doc_id}) # Handling non-string
    return pd.DataFrame(new_rows)

# Example usage
data = {'document_id': [1, 2, 3],
        'text': ["This is the first paragraph.\n\nThis is the second.  \n",
                "Paragraph one\nParagraph two.\r\nParagraph three",
                 123]
       }
df = pd.DataFrame(data)
result_df = split_dataframe_paragraphs_1(df, 'text', 'document_id')
print(result_df)
```

In this implementation, the core logic lies within the list comprehension: `[para.strip() for para in re.split(r'\s*\n+\s*', text) if para.strip()]`. We use `re.split(r'\s*\n+\s*', text)` to split on one or more newlines that may or may not be surrounded by whitespace. Then we filter out empty strings after cleaning leading/trailing whitespace with `strip()` using `if para.strip()` to prevent adding empty paragraphs. If the text is not a string, we are adding an empty paragraph to the resulting dataframe. The result is a list of cleaned paragraphs, and we then build up the `new_rows` list with our new paragraph rows and document ids. Finally, we convert `new_rows` into a dataframe. This method is very efficient in terms of readability and execution.

Now, let’s explore a variation using `apply` and a helper function. This can sometimes be clearer when the logic becomes more complex:

```python
import pandas as pd
import re

def _split_text_to_paragraphs(text):
     """Helper function to split text into paragraphs.
    Args:
       text (str): The input text.
    Returns:
        list: A list of paragraphs.
    """
     if isinstance(text, str):
        return [para.strip() for para in re.split(r'\s*\n+\s*', text) if para.strip()]
     else:
        return [""]

def split_dataframe_paragraphs_2(df, text_column, id_column):
    """Splits text rows into paragraphs and keeps the document ID using apply.

    Args:
        df (pd.DataFrame): The input dataframe.
        text_column (str): The name of the column containing text.
        id_column (str): The name of the column containing document IDs.

    Returns:
        pd.DataFrame: A new dataframe with each paragraph in a separate row.
    """
    df['paragraphs'] = df[text_column].apply(_split_text_to_paragraphs) # applying the helper function
    exploded_df = df.explode('paragraphs') # exploding list of paragraphs to single row
    exploded_df.rename(columns={'paragraphs': 'paragraph'}, inplace=True) # renaming
    exploded_df = exploded_df[exploded_df['paragraph']!=""] # removing non-string entries
    return exploded_df.drop(columns=[text_column])


# Example usage
data = {'document_id': [1, 2, 3],
        'text': ["This is the first paragraph.\n\nThis is the second.  \n",
                "Paragraph one\nParagraph two.\r\nParagraph three",
                 123]
       }
df = pd.DataFrame(data)
result_df = split_dataframe_paragraphs_2(df, 'text', 'document_id')
print(result_df)
```

Here, we first create a helper function `_split_text_to_paragraphs` that isolates our paragraph splitting logic and returns a list of paragraphs. This makes our code easier to read and maintain. We then apply this function to the `text_column` using the `.apply()` method on the `pandas` series. This creates a new column named `paragraphs` containing a list of paragraphs. `df.explode('paragraphs')` then transforms each list of paragraphs into individual rows. `exploded_df.rename(columns={'paragraphs': 'paragraph'}, inplace=True)` allows us to rename the column for clarity. The `exploded_df.drop(columns=[text_column])` drops the original text. We also have removed entries which are not strings.

For a final variation, let's consider a solution with `itertools.chain` that could be more efficient for larger dataframes, even though the performance benefit might not be dramatic for smaller ones:

```python
import pandas as pd
import re
from itertools import chain


def split_dataframe_paragraphs_3(df, text_column, id_column):
    """Splits text rows into paragraphs and keeps the document ID using itertools.chain.

    Args:
        df (pd.DataFrame): The input dataframe.
        text_column (str): The name of the column containing text.
        id_column (str): The name of the column containing document IDs.

    Returns:
        pd.DataFrame: A new dataframe with each paragraph in a separate row.
    """
    new_rows = []
    for index, row in df.iterrows():
        doc_id = row[id_column]
        text = row[text_column]
        if isinstance(text, str):
            paragraphs = [para.strip() for para in re.split(r'\s*\n+\s*', text) if para.strip()]
            new_rows.append([(paragraph, doc_id) for paragraph in paragraphs])
        else:
            new_rows.append([("", doc_id)])


    flattened_rows = list(chain.from_iterable(new_rows))
    new_df = pd.DataFrame(flattened_rows, columns=['paragraph', id_column])
    return new_df

# Example usage
data = {'document_id': [1, 2, 3],
        'text': ["This is the first paragraph.\n\nThis is the second.  \n",
                "Paragraph one\nParagraph two.\r\nParagraph three",
                123]
       }
df = pd.DataFrame(data)
result_df = split_dataframe_paragraphs_3(df, 'text', 'document_id')
print(result_df)

```

In this method, the structure is similar to the first example, using `for` loops to process each row. The difference here is how we gather the paragraphs. The paragraphs are appended as a list of tuples with paragraph and document id. `itertools.chain.from_iterable` flattens this list. Then, the resulting list of paragraph/doc_id pairs is converted into a DataFrame. While it may look more complex initially, in some cases, this can have less overhead because it processes lists of items in a streaming manner. It is more suitable for very large dataframes compared to repeated appending operations using list comprehensions.

**Recommendations**

For a deep understanding of regular expressions, I recommend "Mastering Regular Expressions" by Jeffrey Friedl. This is the definitive book for mastering regex. It covers all the intricacies, from basic syntax to advanced techniques. To further improve your `pandas` knowledge, "Python for Data Analysis" by Wes McKinney (the creator of `pandas`) is the must-have resource. It covers the framework extensively.

These three examples showcase various approaches to achieve the same goal. The best approach will often depend on your preferences, data size, and whether you need custom functions for particular scenarios. For many use-cases, I find the list comprehension approach to be optimal balance of readability and performance. However, having these alternatives is helpful when dealing with diverse situations and data. Hopefully, these examples provide a solid foundation for your work.
