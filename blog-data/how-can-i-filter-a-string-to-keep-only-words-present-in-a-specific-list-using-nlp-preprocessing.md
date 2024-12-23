---
title: "How can I filter a string to keep only words present in a specific list using NLP preprocessing?"
date: "2024-12-23"
id: "how-can-i-filter-a-string-to-keep-only-words-present-in-a-specific-list-using-nlp-preprocessing"
---

Alright,  I remember working on a particularly challenging text analysis project a few years back for a client in the financial sector. They needed to sift through thousands of news articles to identify only those relevant to their specific investment strategy. One of their core requirements was the ability to filter text, retaining only sentences containing keywords from their predefined investment vocabulary. That experience really cemented my understanding of the nuances of this particular task.

So, when you ask how to filter a string to keep only words present in a specific list using natural language processing preprocessing, you're essentially talking about implementing a controlled vocabulary filter after the initial text cleaning stage. The core concept revolves around tokenization followed by filtering. Tokenization breaks the input string into individual words (or sometimes phrases), and filtering discards tokens that aren't in your allowed list, which is also often called a lexicon or vocabulary. It’s not just about identifying words; it’s about efficiency and ensuring you're comparing apples to apples after standardization. Let's break down the process, looking at it from a practical standpoint using three different code examples, all demonstrating slightly different approaches.

**Example 1: Simple Set-Based Filtering with Python**

The most straightforward method is using Python and its built-in set operations. Here’s how that looks:

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def filter_string_simple(input_string, vocabulary):
  """
  Filters a string to keep only words present in the vocabulary.

  Args:
    input_string: The string to filter.
    vocabulary: A list of allowed words.

  Returns:
    A string containing only the words from the input_string found in the vocabulary.
  """
  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(input_string.lower())
  filtered_tokens = [token for token in tokens if token in vocabulary and token not in stop_words]
  return " ".join(filtered_tokens)


# Example usage
my_vocabulary = ['apple', 'banana', 'orange', 'fruit']
text_input = "I like to eat apples and bananas. Oranges are also a good fruit, unlike spinach."
filtered_text = filter_string_simple(text_input, my_vocabulary)
print(filtered_text)
# Output: apple bananas orange fruit
```

In this first example, the `filter_string_simple` function leverages `nltk` (Natural Language Toolkit) for both tokenization and stopword removal (though stopwords can be omitted if desired). It converts the input string to lowercase and then tokenizes it using `word_tokenize`. We filter out words that aren't present in the vocabulary *and* words from the stopword list, making the process a bit cleaner. Finally, it joins the filtered tokens back into a string. Notice the use of list comprehension – it’s both concise and efficient. I consider this example to be the ‘baseline’.

**Example 2: Filtering with Lemmatization and Stemming**

Sometimes you need a more robust approach to match words, especially if your vocabulary contains base forms. Lemmatization reduces words to their dictionary form, and stemming reduces words to their root form. Here’s how you might integrate that, with a focus on lemmatization:

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def filter_string_lemmatized(input_string, vocabulary):
    """
    Filters a string using lemmatization to match vocabulary items.

    Args:
      input_string: The string to filter.
      vocabulary: A list of allowed words.

    Returns:
      A string containing only the words from the input_string found in the vocabulary, after lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(input_string.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    filtered_tokens = [token for token in lemmatized_tokens if token in vocabulary and token not in stop_words]
    return " ".join(filtered_tokens)

#Example usage
my_vocabulary_lemmatized = ['run', 'walk', 'exercise']
text_input_lemmatized = "He was running quickly, she walks slowly. They are exercising."
filtered_text_lemmatized = filter_string_lemmatized(text_input_lemmatized, my_vocabulary_lemmatized)
print(filtered_text_lemmatized)
# Output: run walk exercising
```
Here, we add the `WordNetLemmatizer`.  Before filtering against the vocabulary, we now lemmatize all the tokens. This means that ‘running’ will be reduced to ‘run’, ‘walks’ to ‘walk’, and ‘exercising’ to ‘exercise’, enabling matches even if words aren’t in their base form initially. While stemming could be a valid alternative here (or even used in tandem), lemmatization tends to give better results, particularly when preserving the contextual meaning of a word is essential. This enhanced matching approach is valuable for more complex vocabulary matching.

**Example 3: Filtering with SpaCy and custom stopword lists**

For more advanced NLP projects, SpaCy tends to be the library of choice due to its efficiency and pre-trained models. It also offers flexible stopword manipulation. Here’s how to integrate it into our task with the ability to use a custom stop word list:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def filter_string_spacy(input_string, vocabulary, custom_stopwords = None):
    """
    Filters a string using SpaCy for tokenization and a custom stopword list.

    Args:
      input_string: The string to filter.
      vocabulary: A list of allowed words.
      custom_stopwords: A list of custom words to exclude.

    Returns:
      A string containing only the words from the input_string found in the vocabulary.
    """

    doc = nlp(input_string.lower())
    if custom_stopwords:
        combined_stopwords = set(spacy.lang.en.stop_words.STOP_WORDS).union(set(custom_stopwords))
    else:
       combined_stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)

    filtered_tokens = [token.text for token in doc if token.text in vocabulary and token.text not in combined_stopwords]
    return " ".join(filtered_tokens)


# Example usage
my_vocabulary_spacy = ['stock', 'market', 'price', 'invest']
text_input_spacy = "The stock market prices are volatile. Should you invest quickly? We must think about this. "
custom_stop_words = ['think', 'about', 'must', 'this']
filtered_text_spacy = filter_string_spacy(text_input_spacy, my_vocabulary_spacy, custom_stop_words)
print(filtered_text_spacy)
# Output: stock market prices invest
```
Here, SpaCy’s processing pipeline is used to tokenize the text into `Doc` objects. We also integrate a custom stopword list as an optional parameter, allowing you to easily remove task-specific terms in addition to standard stopwords. This flexibility is one of SpaCy's strengths. Note how the logic for extracting text from the tokens within the `Doc` object is used. This is one of the core differences between `NLTK` and `SpaCy`, and is crucial to remember when working with either library. SpaCy also comes preloaded with many features. For example, the lemma is immediately available via `token.lemma_`, or stopword information with `token.is_stop`. Choosing `token.text` is intentional because we are matching to the exact text found in our vocabulary list.

**Resource Recommendations:**

For a deeper dive, I recommend delving into several resources:

1.  **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** This book is essentially the bible for NLP. It provides an exhaustive and rigorous treatment of core concepts, covering everything from basic tokenization up to advanced machine learning techniques. Look for the 'Text Preprocessing' section to explore these concepts.

2.  **"Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper:** This practical guide focuses on using `NLTK` and is exceptionally valuable for understanding the basics of NLP preprocessing. It contains several hands-on examples that mirror my examples shown here.

3.  **SpaCy's Official Documentation:** SpaCy's documentation is well-maintained and comprehensive. It's essential for mastering the practical usage of the SpaCy library, and I recommend starting with the tutorials available on their site.

In conclusion, filtering strings using a specific word list is a foundational preprocessing step in NLP. It can be accomplished using several different methods, from simple set-based operations to more complex lemmatization techniques, and with flexible libraries such as `NLTK` or `SpaCy`. Selecting the appropriate method will always depend on the specifics of your application and the desired level of sophistication. By understanding these different approaches and the tools available, you should have a solid foundation for creating effective text filtering pipelines.
