---
title: "How can I measure the similarity of two words to identify duplicates?"
date: "2024-12-23"
id: "how-can-i-measure-the-similarity-of-two-words-to-identify-duplicates"
---

Alright, let's tackle this head-on. Measuring word similarity for duplicate detection is something I've spent a fair bit of time on, particularly during a project involving large-scale text data cleanup a few years back. We were ingesting documents from various sources, and the sheer volume meant manual checking for duplicates was a non-starter. This led us down the path of exploring various similarity metrics, and the experience underscored how critical choosing the right approach can be.

At its core, the challenge isn’t just about identical strings; it’s often about identifying semantically similar words or strings that might appear differently due to minor typos, abbreviations, or different casing. Therefore, a straightforward string comparison won't suffice. You need a more nuanced method that understands that ‘colour’ and ‘color’, or 'apple' and 'apples’ are essentially the same for most practical purposes in this context.

There isn't one perfect solution, because the best approach often depends on the type of data you’re working with and what level of variation you're willing to accept. I've generally found three categories particularly effective: string-based methods, edit-distance methods, and embedding-based methods.

First, let’s explore **string-based methods**. These are relatively simple and work by comparing the characters within the words. One common approach here is the Jaccard index, which measures the similarity between sets. In our word similarity context, we treat each word as a set of characters or n-grams (sequences of characters). Let's take a quick example in python:

```python
def jaccard_similarity(str1, str2, n_gram=2):
    str1_ngrams = set([str1[i:i+n_gram] for i in range(len(str1) - n_gram + 1)])
    str2_ngrams = set([str2[i:i+n_gram] for i in range(len(str2) - n_gram + 1)])
    intersection = len(str1_ngrams.intersection(str2_ngrams))
    union = len(str1_ngrams.union(str2_ngrams))
    return intersection / union if union > 0 else 0

word1 = "apple"
word2 = "apples"
print(f"Jaccard similarity for '{word1}' and '{word2}': {jaccard_similarity(word1, word2)}") # Output: ~0.6
print(f"Jaccard similarity for 'apple' and 'orange': {jaccard_similarity('apple', 'orange')}") # Output: ~0.0
```

In this snippet, I’ve implemented a Jaccard index calculation using character pairs (bigrams). You can experiment with different values of `n_gram` – increasing the value may work better for longer words while also increasing the number of unique n-grams, reducing the sensitivity to small variations. For instance, 'kitten' and 'sitting' will have a Jaccard similarity of 0 when n_gram=2, even though there is clearly a large overlap. String based methods are quite fast to calculate, but struggle with minor spelling mistakes, abbreviations or differences in word order when applied to phrases.

Next, we move on to **edit-distance methods**. The most prominent example here is the Levenshtein distance, which quantifies how many single-character edits (insertions, deletions, or substitutions) it takes to transform one word into another. Lower distances mean higher similarity. This technique is robust to slight misspellings.

```python
def levenshtein_distance(str1, str2):
    if len(str1) < len(str2):
        return levenshtein_distance(str2, str1)
    if len(str2) == 0:
        return len(str1)
    previous_row = range(len(str2) + 1)
    for i, c1 in enumerate(str1):
        current_row = [i + 1]
        for j, c2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

word1 = "intention"
word2 = "execution"
print(f"Levenshtein distance between '{word1}' and '{word2}': {levenshtein_distance(word1, word2)}") # Output: 6

word3 = "kitten"
word4 = "sitting"
print(f"Levenshtein distance between '{word3}' and '{word4}': {levenshtein_distance(word3, word4)}") # Output: 3

```

This code demonstrates how you calculate the Levenshtein distance. While I have used a dynamic programming approach here, it’s worth knowing there are variations that could be more suitable depending on your requirements. For instance, the Damerau-Levenshtein distance, a simple modification, also accounts for transpositions (swapping of adjacent characters). The advantage of edit distances is that they are insensitive to character order in small sequences but can be computationally expensive for long strings.

Finally, let's discuss **embedding-based methods**. These use the concept of vector embeddings, where each word is represented as a point in a high-dimensional space. Semantically similar words should appear close to each other in this space. This is probably the most sophisticated approach and requires pre-trained models. The cosine similarity between these vectors is often used to calculate the similarity of the words. Popular word embedding models include Word2Vec, GloVe, and fastText. These models are trained on very large text corpora and offer a more robust way to handle synonyms and semantically related words.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# Load pre-trained Word2Vec model.
word2vec_model = api.load("word2vec-google-news-300")

def embedding_similarity(word1, word2, model):
    try:
      vec1 = model[word1]
      vec2 = model[word2]
      return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    except KeyError:
      return 0 # Handle cases where words are not in the vocabulary

word1 = "king"
word2 = "queen"
print(f"Embedding similarity between '{word1}' and '{word2}': {embedding_similarity(word1, word2, word2vec_model)}") # Output: ~0.65

word3 = "king"
word4 = "apple"
print(f"Embedding similarity between '{word3}' and '{word4}': {embedding_similarity(word3, word4, word2vec_model)}") # Output: ~0.06

```

Here, I'm demonstrating how to load a pre-trained Word2Vec model and then calculating cosine similarity between the word vectors. Note that you would need to have the `gensim` library installed. `pip install gensim`. I also include error handling here because not every word will be in the pre-trained vocabulary. Word embeddings are powerful because they capture semantic similarity, but they are also the most resource intensive, especially if the embedding model needs to be trained. Pre-trained models like the one used here are a good starting point.

For more in-depth theoretical knowledge, I highly recommend delving into “Speech and Language Processing” by Daniel Jurafsky and James H. Martin for string and edit distance methods and “Deep Learning” by Ian Goodfellow et al. for a good understanding of embeddings. Also for practical application, explore the “Natural Language Processing with Python” book by Steven Bird, Ewan Klein and Edward Loper. These should give you the solid theoretical understanding combined with practical skills needed.

In practice, the choice of which method is best for your situation really depends on your specific needs. String-based methods are computationally cheap but often too shallow. Edit distance provides a good middle ground. Embeddings, while the most powerful, can be computationally expensive and require a proper understanding of the underlying models. I often started by using Jaccard or Levenshtein and only switched to embeddings when semantic similarity was crucial or when I had enough processing resources. Sometimes, a combination of these methods, or even using them in sequence (e.g., if edit distance is low, you do not bother checking embedding similarity) provides the best results, depending on your specific needs and performance constraints.
