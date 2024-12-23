---
title: "How can NLTK be used to remove non-English words from a CSV file?"
date: "2024-12-23"
id: "how-can-nltk-be-used-to-remove-non-english-words-from-a-csv-file"
---

Alright,  I've actually bumped into this exact scenario a few times over the years, particularly when dealing with datasets that were cobbled together from various sources—think user-generated content aggregated from all corners of the internet. Cleaning up the text data became a crucial first step, and getting rid of non-English words, or rather, words that aren't likely to be English, is definitely part of that process.

The challenge, of course, isn't quite as straightforward as a simple dictionary lookup. Languages evolve, new words are constantly being introduced, and relying on a static list would leave us with significant gaps. We also have to consider things like proper nouns, which might not be in the standard nltk dictionaries, or words that have been 'english-ized', for lack of a better term, that started as loan words. Therefore, we need a nuanced approach that combines a few techniques. We'll be using nltk's language identification capabilities and then filtering based on that.

Let’s begin with the simplest case, where we just need to filter out entire rows based on the identified language. I encountered this initially when working on a project that aggregated product reviews from global marketplaces. Many of the reviews were submitted in languages other than English, and keeping those rows in the dataset would significantly skew the analysis.

Here’s how it looks using nltk and pandas:

```python
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk import download
from collections import Counter

# Download nltk data if not already present
try:
    words.words('en')
except LookupError:
    download('words')


def is_mostly_english(text, threshold=0.7):
    """
    Check if a text is predominantly English using word frequencies
    """
    english_words = set(words.words())
    tokens = word_tokenize(text.lower())
    english_count = sum(1 for token in tokens if token in english_words)
    total_count = len(tokens)
    if total_count == 0:
      return False
    return english_count / total_count >= threshold

def filter_csv_english(input_file, output_file, text_column):
    """
    Filters a CSV file, keeping only rows where the specified text column is predominantly English.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the filtered output CSV.
        text_column (str): Name of the column containing the text to check.
    """
    df = pd.read_csv(input_file)
    df['is_english'] = df[text_column].apply(is_mostly_english)
    filtered_df = df[df['is_english'] == True]
    filtered_df.drop('is_english', axis=1, inplace=True)
    filtered_df.to_csv(output_file, index=False)


# Example usage:
if __name__ == '__main__':
    # Assume we have a test.csv file with a 'review' column
    data = {'review': ["this is a test review", "Esto es una prueba", "Another english review.", "这是一个测试"]}
    df = pd.DataFrame(data)
    df.to_csv('test.csv', index=False)
    filter_csv_english('test.csv', 'filtered_english.csv', 'review')

```

In this code, `is_mostly_english` checks if the proportion of english words in a tokenized text is above the defined threshold. I’ve set it to 0.7 but depending on your data this might need adjustment. `filter_csv_english` function then loads the CSV, applies our language checking, filters to keep english text rows, drops the helper column, and finally saves to a new file. Notice that I load the words dictionary in a `try/except` block to manage first time runs without pre-downloaded dictionaries, a common issue. This approach uses the nltk's built-in english dictionary.

Now, this works pretty well for whole-row filtering, but what if we wanted to keep most of the text but get rid of the non-English words within the text itself? In a project where I was analyzing social media comments, many messages included a mix of languages, often using non-English words alongside English ones. It wouldn’t have been practical to filter entire comments, as it would have discarded too much valuable data. So, for this scenario, I took a slightly different path and looked at a token-by-token approach.

Here's how I tackled that:

```python
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk import download

# Ensure words are downloaded
try:
    words.words('en')
except LookupError:
    download('words')


def remove_non_english_words(text):
    """
    Removes words from a text that are not in the English dictionary
    """
    english_words = set(words.words())
    tokens = word_tokenize(text.lower())
    english_tokens = [token for token in tokens if token in english_words]
    return ' '.join(english_tokens)


def clean_csv_text_column(input_file, output_file, text_column):
    """
    Cleans a CSV file by removing non-english words from specified column.

    Args:
        input_file (str): Path to input CSV.
        output_file (str): Path to save the cleaned CSV
        text_column (str): Name of column to clean.
    """

    df = pd.read_csv(input_file)
    df[text_column] = df[text_column].apply(remove_non_english_words)
    df.to_csv(output_file, index=False)


# Example usage
if __name__ == '__main__':
    # Assume we have a test.csv file with a 'review' column
    data = {'review': ["this is a test review but there is some 'hola' and 'bonjour'", "This also has some 'guten tag' mixed in"]}
    df = pd.DataFrame(data)
    df.to_csv('test.csv', index=False)
    clean_csv_text_column('test.csv', 'cleaned_text.csv', 'review')
```

In this revised code, `remove_non_english_words` now tokenizes, and then only keeps the tokens that are present in the english word list, then rejoins it back into a string. `clean_csv_text_column` applies this function to the specified column. This technique retains the overall structure of the text, removing only the explicitly identifiable non-English words. It works well if you are  with some potential noise, like a low frequency of untracked words that didn't get dropped by the nltk dictionary.

There is also a third option which involves training your own language detector. I did this once when dealing with heavily domain-specific text, using a combination of a naive bayes model and a language corpus. For the sake of brevity, I'll sketch out the general idea rather than provide a complete code example, as training such a model is quite involved. We would use a library like scikit-learn to train a classifier using a dataset where the text language is labelled. This model would have the ability to learn patterns that aren't present in the standard dictionaries. After that, we could use this model to label the language of the word or sentence and then proceed as above with the filtering or removal. This is an advanced technique and only necessary when you're hitting limits with the existing methods.

For further study in language processing and text cleaning, I highly recommend diving into *Speech and Language Processing* by Daniel Jurafsky and James H. Martin. It's a comprehensive text covering all the foundations of computational linguistics. Also *Natural Language Processing with Python* by Steven Bird, Ewan Klein, and Edward Loper is a highly readable and practical guide for getting hands-on with nltk. In general, research papers focusing on multi-lingual document processing and language identification techniques would provide more information on the specific challenges associated with this topic. The *Journal of Natural Language Engineering* often has very relevant publications.

In closing, dealing with non-English words in text data is a surprisingly complex problem. You need to take into account the specific use case, the data you are working with, and the tolerance for errors. Starting with the simple token dictionary lookup then advancing to language identification on the row/text or ultimately to training custom models should be approached sequentially based on the problem at hand. I hope these techniques and considerations are helpful to anyone needing to remove non-english text from their CSVs.
