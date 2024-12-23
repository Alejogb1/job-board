---
title: "How can I find the bigram vocabulary in Python?"
date: "2024-12-23"
id: "how-can-i-find-the-bigram-vocabulary-in-python"
---

Alright, let's tackle this. It's a common task, and one I've bumped into more than a few times over the years, particularly when dealing with natural language processing pipelines. I recall working on a sentiment analysis project where understanding the context provided by bigrams was crucial for achieving acceptable accuracy. So, finding that bigram vocabulary in a Pythonic way? It's definitely achievable and straightforward with the right approach.

Essentially, you're looking to identify all two-word sequences present in a given text corpus. These pairs are known as bigrams, and they form the basis for many text analysis techniques. Instead of just counting individual words, you get a feel for how often word combinations appear together, giving you more context-rich information. The trick is to efficiently process your text and extract these pairs.

There are primarily two methods that I find myself using in these scenarios. One revolves around simple iteration with string manipulation, and the other uses more powerful tools from the `nltk` (Natural Language Toolkit) library. I'll walk you through both with examples.

**Method 1: Iteration and Basic String Handling**

This approach is quite fundamental but offers a solid understanding of the process. It doesn't require any extra external packages, which can be handy for smaller, isolated projects. It breaks down the text into individual words, then goes through them, creating the bigrams on the fly using a moving window approach.

```python
def find_bigrams_manual(text):
    words = text.split()
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i+1]))
    return bigrams

sample_text = "the quick brown fox jumps over the lazy dog"
bigram_list = find_bigrams_manual(sample_text)
print(bigram_list)
```

This `find_bigrams_manual` function takes a string of text as input. First, it splits the text into individual words using `text.split()`. This assumes that spaces are good delimiters, which works for a lot of plain text scenarios. Then, it iterates over the list of words using a `for` loop. The important bit is that we iterate up to `len(words) - 1` to avoid an "index out of bounds" error at the end of the list, because we are always looking one word ahead in each step. It takes each word and the following word to form the bigram. The result is a list of tuples, where each tuple is a bigram (two words). For the example sentence, the output should be: `[('the', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumps'), ('jumps', 'over'), ('over', 'the'), ('the', 'lazy'), ('lazy', 'dog')]`

This method is simple, but it's also case-sensitive and doesn't handle punctuation well. In practice, you would probably want to perform additional preprocessing steps. Things like converting text to lowercase and stripping out punctuation would greatly improve the results in terms of generalization.

**Method 2: Using the `nltk` Library**

The `nltk` library is more robust and is what I typically lean on in my more serious projects because it handles many of the pre-processing complexities for me. It provides a `bigrams` function that works on tokenized text, making it much cleaner and efficient. `nltk` expects you to have the text in a tokenized form, which can be achieved using its `word_tokenize` function.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams

nltk.download('punkt')  # Necessary for tokenization if not already downloaded

def find_bigrams_nltk(text):
  tokens = word_tokenize(text)
  bigram_list = list(ngrams(tokens, 2))
  return bigram_list

sample_text = "The quick, brown fox. Jumps over the lazy dog!"
bigram_list = find_bigrams_nltk(sample_text)
print(bigram_list)

```
In this second code snippet, we begin by importing the required modules from `nltk`: `word_tokenize`, used to split the text into tokens (usually words or punctuation marks), and `ngrams`, a more general function that we specify to make bigrams (2-grams). I also download the `punkt` resource if you do not have it already; it's required for the `word_tokenize` function to work correctly. The `find_bigrams_nltk` function first tokenizes the text using `word_tokenize`. Afterwards, `ngrams(tokens, 2)` creates an iterator yielding bigrams, which we then convert to a list. It includes the punctuation as tokens, which you might want to filter out later in your data analysis. This `nltk` approach makes it much easier to handle punctuation. You could further use `nltk` features like stemming or lemmatization to refine the vocabulary even more if needed. For our example sentence, the output is `[('The', 'quick'), ('quick', ','), (',', 'brown'), ('brown', 'fox'), ('fox', '.'), ('.', 'Jumps'), ('Jumps', 'over'), ('over', 'the'), ('the', 'lazy'), ('lazy', 'dog'), ('dog', '!')]`.

**Frequency Analysis**
If you’re interested in seeing the most common bigrams, which is often what happens next, you can use the `collections.Counter` object. This is extremely helpful to prioritize more frequent bigrams. Here's an example:

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
from collections import Counter

nltk.download('punkt') # necessary for tokenization if not already downloaded

def find_and_count_bigrams(text):
  tokens = word_tokenize(text.lower())  # Tokenize and make lowercase
  bigram_list = list(ngrams(tokens, 2))
  bigram_counts = Counter(bigram_list)
  return bigram_counts

sample_text = "The quick, brown fox. The quick fox jumps over the lazy dog!"
bigram_counts = find_and_count_bigrams(sample_text)
print(bigram_counts.most_common(5)) # shows the 5 most frequent bigrams

```

Here, I added the step to make the text lowercase (`text.lower()`) to reduce the variations of the same words. We use `Counter` to find how frequently each bigram appears in our text, and the `most_common(5)` command returns the five most frequent bigrams and their counts in a list. The output shows the bigrams and the number of times they appeared; for example: `[(('the', 'quick'), 2), (('quick', 'brown'), 1), ((',', 'brown'), 1), (('brown', 'fox'), 1), (('fox', '.'), 1)]`. Note that `('the', 'quick')` shows up twice as it is in the sample text, demonstrating the usefulness of `Counter`.

**Further Reading and Considerations**

For more in-depth knowledge of NLP, I'd strongly suggest checking out *Speech and Language Processing* by Daniel Jurafsky and James H. Martin. This textbook is a classic and provides a thorough understanding of the techniques and mathematics involved. Additionally, the documentation for `nltk` is excellent and a must-read if you plan to dive deep. For practical aspects, explore *Natural Language Processing with Python*, also by the `nltk` authors, which will guide you through implementation.

In real-world scenarios, you’d likely have much more text than these examples. The `nltk` approach generally scales better and is more robust when the text has multiple formats and nuances. Remember that preprocessing, such as removing stop words (common words like "the," "a," "is"), can enhance your bigram analysis by focusing on words that contribute more to the meaning of your text. You would typically iterate over a corpus of documents (instead of single sentences like in my examples) and add the counts from each document. Finally, remember, selecting the right technique depends on your specific application and how you plan to use this bigram vocabulary.
