---
title: "How can pandas profiling be used for text data analysis?"
date: "2025-01-30"
id: "how-can-pandas-profiling-be-used-for-text"
---
Profiling text data with pandas, while not its primary strength, can be highly valuable when used strategically. I've found that the commonly cited functions for numerical data, such as `describe()` or `value_counts()`, need adaptation to reveal meaningful patterns in textual columns. Instead of direct statistical summaries, profiling text often involves characterising length distributions, identifying frequent tokens, and discerning patterns that might suggest the need for further cleaning or preprocessing steps.

Initially, one might attempt to directly apply typical pandas profiling functions. However, running `df['text_column'].describe()` will provide basic statistics on the string column, like the number of entries, unique values, most frequent value (which is rarely insightful), and the length of the longest string. This is hardly useful for nuanced analysis. My experience has shown that the key is to transform the text into representations that lend themselves to statistical analysis, which is where the true power of pandas in combination with other libraries shines through.

The first fundamental step involves looking at the distribution of text lengths. This can give immediate clues about the nature of your data. Are there exceptionally long entries that might require truncation or more specific handling? Or are most of the entries concentrated around a specific length, suggesting uniform data collection patterns? We achieve this by calculating and plotting the length of each string:

```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_string_length_distribution(df, column_name, bins=50):
    """
    Plots a histogram of the lengths of strings in a pandas DataFrame column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing string data.
        bins (int): The number of bins for the histogram.
    """
    df['text_length'] = df[column_name].str.len()
    plt.figure(figsize=(10, 6))
    plt.hist(df['text_length'], bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of String Lengths in {column_name}')
    plt.xlabel('String Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    df.drop(columns=['text_length'], inplace=True)


# Example usage:
data = {'text': ['This is a short sentence.', 'A somewhat longer example.',
                'This is a very very very very long sentence with many words.',
                 'Short one.']}
df = pd.DataFrame(data)
plot_string_length_distribution(df, 'text', bins=10)
```

In this function, I create a temporary column to store the length of the text entries, using the `.str.len()` method. The use of `matplotlib` here allows for a rapid visual assessment of the lengths. After plotting, the temporary column is removed. The `bins` argument is adjustable, which is useful for controlling the level of detail. This visual analysis often reveals outliers or unexpected length patterns that might be obscured by simply looking at numerical statistics.

The second key profiling strategy concerns identifying frequent words or tokens. This step helps in assessing content richness and might indicate areas for cleaning like handling stopwords or non-alphanumeric characters. Libraries like `NLTK` or `spaCy` are ideal for this, but the frequency calculation is conveniently handled within `pandas` combined with `collections.Counter`:

```python
import pandas as pd
from collections import Counter
import re

def calculate_token_frequencies(df, column_name, top_n=10):
    """
    Calculates and returns the top N most frequent tokens in a pandas DataFrame column,
    after cleaning.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing string data.
        top_n (int): The number of top tokens to return.

    Returns:
        collections.Counter: A Counter object with the top N tokens and their counts.
    """

    def preprocess_text(text):
      text = str(text).lower() # Convert to lowercase
      text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphanumeric
      return text.split()

    all_tokens = df[column_name].apply(preprocess_text).explode().tolist()
    token_counts = Counter(all_tokens)
    return token_counts.most_common(top_n)


# Example Usage:
data = {'text': ['This is a simple sentence, and another.', 'And, this is another example with repeated words', 'this is a test, This test is simple', 'One more']}
df = pd.DataFrame(data)
top_tokens = calculate_token_frequencies(df, 'text', top_n=5)
print(f"Top 5 tokens: {top_tokens}")
```

Here, I employ a function to preprocess the text by converting it to lowercase and removing non-alphanumeric characters via regular expressions, ensuring consistency. I then split each string into individual words (`tokens`), creating a list of all tokens from the entire column. Finally, I use `collections.Counter` to get the frequency count of each token. This reveals the most common terms. If the top tokens are largely stop words (like “the”, “is”, “and”), then the need for stop word removal before further text analysis becomes very apparent. The `explode()` method is useful to convert lists of tokens from each text entry into one single list containing all tokens. This pattern, combining string methods, `apply()`, `explode()`, and `Counter` is highly effective.

A third aspect of text profiling involves identifying specific patterns using regular expressions. For example, you might want to locate email addresses, phone numbers, or URLs. This goes beyond simple word frequencies to a form of pattern analysis, helping identify potential data quality issues or specific features in the text.

```python
import pandas as pd
import re


def identify_patterns(df, column_name, regex_pattern, pattern_name):
    """
    Identifies and counts occurrences of a specific pattern in a DataFrame column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing string data.
        regex_pattern (str): The regular expression pattern to search for.
        pattern_name (str): A descriptive name for the identified pattern.

    Returns:
       pd.DataFrame: A new DataFrame with the identified pattern counts.
    """
    df[f'{pattern_name}_count'] = df[column_name].str.findall(regex_pattern).apply(len)
    return df

# Example usage:
data = {'text': ['Contact me at test@example.com or visit https://example.com', 'another email: user@domain.net', 'no patterns here', 'visit http://site.org']}
df = pd.DataFrame(data)
email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

df = identify_patterns(df, 'text', email_regex, 'email')
df = identify_patterns(df, 'text', url_regex, 'url')
print(df)
```

The function here uses `str.findall` and applies `len` to count occurrences. This transforms a text pattern search into a countable numerical feature. A detailed pattern can thus be profiled as a new numerical column, demonstrating how text data, through regular expressions, can become amenable to typical pandas-based numeric analyses. The output of the function adds columns with the identified counts, demonstrating how these features can be used in further analyses.

For comprehensive text analysis workflows, several resources are highly valuable. Books and online documentation for NLTK and spaCy provide a sound foundation for natural language processing techniques that go far beyond simple profiling. Regular expressions tutorials and online documentation are essential to develop precise patterns. Beyond that, resources on the theoretical foundations of vector space models (like tf-idf and word embeddings) are necessary if one wants to go from pure profiling into predictive modeling. A good starting point includes the `pandas` documentation itself. Although it does not focus on the application of text analysis, the fundamental data manipulation capabilities need to be mastered in order to build an effective profiling workflow.
