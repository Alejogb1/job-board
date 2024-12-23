---
title: "How can negative samples be obtained for a Gensim Word2Vec model?"
date: "2024-12-23"
id: "how-can-negative-samples-be-obtained-for-a-gensim-word2vec-model"
---

Let’s dive right into it, shall we? I recall a particularly thorny project a few years back involving sentiment analysis on a colossal corpus of customer reviews. We were using Word2Vec, and like many, we hit that inevitable wall: effectively generating negative samples. The challenge, as always, wasn't about *if* we could do it, but *how efficiently and effectively* we could do it while keeping model performance robust.

Word2Vec, at its core, relies on the concept of learning word representations by analyzing their contextual co-occurrence within a text. The skip-gram and continuous bag-of-words (cbow) architectures aim to predict either context words from a target word or a target word from its context, respectively. The problem arises because for each positive training example—say, "the cat sat on"—where "cat" is the target word and "the," "sat," and "on" are context words—we need corresponding *negative* examples. These negative examples are crucial. They teach the model what is *not* a valid association, preventing it from learning useless or even detrimental relationships.

The naive approach, and one I initially experimented with, is to randomly sample words from the vocabulary. While straightforward, this method often results in poor negative samples, frequently drawing words that are semantically similar or commonly co-occurring. The model struggles to differentiate these artificially constructed negative associations, resulting in less informative word embeddings. It’s like trying to train a dog by only telling it what *not* to do, without ever clearly showing it what *to* do.

So, how do we do better? The most common, and generally most effective, method employed by gensim, and the one I ended up relying on extensively, is based on *unigram distribution with a power function*. It’s a rather simple trick, but its impact is significant. The frequency with which a word is sampled as a negative sample is directly proportional to its unigram frequency raised to a power, generally around 0.75. This approach addresses the problem of common words dominating negative sampling and focuses the model on distinguishing between less frequent, thus more informative, word pairs. Common words like "the," "a," and "is" would otherwise be selected too often as negative examples and, well, they just aren’t very helpful in building better embeddings.

Let’s break that down with some code. Here's how you would generally configure a Word2Vec model in gensim to utilize negative sampling:

```python
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [["the", "quick", "brown", "fox"], ["jumps", "over", "the", "lazy", "dog"], ["a", "cat", "sits", "calmly"]]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1, negative=5, sample=1e-3)

print(model.wv["fox"])

```

In this snippet, `negative=5` specifies that for every positive training example, five negative samples will be generated. Crucially, gensim defaults to using that frequency-based sampling method which gives the power function. The `sample` parameter can also reduce the impact of common words by randomly dropping high-frequency words from input data. I found this to be particularly useful for dealing with text containing a large number of common words (stop words for example).

Now, let's explore how this sampling actually works conceptually using some contrived data. Although it is important to note that gensim doesn't expose the explicit computation of the distribution from which the negative samples are picked and it does the sampling internally, we can emulate that process here to better understand what's going on behind the curtains. Let's assume, for the purposes of illustration, that we have the words and their frequencies in our training corpus as follows:

```python
import numpy as np

word_frequencies = {
    "the": 20,
    "quick": 5,
    "brown": 3,
    "fox": 2,
    "jumps": 4,
    "over": 3,
    "lazy": 1,
    "dog": 1,
    "a": 10,
    "cat": 2,
    "sits": 2,
    "calmly": 1
}

def calculate_negative_sampling_distribution(word_frequencies, power=0.75):
  """Calculates probabilities for negative sampling based on word frequencies. """
  total_freq = sum(freq**power for freq in word_frequencies.values())
  probabilities = {word: (freq**power) / total_freq for word, freq in word_frequencies.items()}
  return probabilities

probabilities = calculate_negative_sampling_distribution(word_frequencies)

# Print probabilities.
for word, prob in probabilities.items():
    print(f"{word}: {prob:.4f}")


```

This snippet showcases how the distribution for negative sampling is skewed by the power factor and word frequencies. Common words will have higher probabilities compared to infrequent ones, but the power factor will smooth the difference between probabilities of common and less common words compared to a raw unigram distribution. We are not sampling based on raw frequency, we are sampling based on this adjusted probability.

While the frequency-based approach is generally reliable, sometimes, if you need more control, or if you’re working with a highly specialized dataset, you might want to implement custom sampling logic. It isn't common, but when I worked on a project using domain-specific language from legal contracts, this kind of customization was essential. The standard techniques often struggled with terms that had highly specific meanings.

Here's an example of how you might create a custom sampling function, although this is generally not the preferred method, as gensim implementations are optimized for efficiency:

```python
import random

def custom_negative_sampling(word, vocabulary, num_negative_samples, custom_probabilities):
  """Custom negative sampling based on a given probability distribution. """

  negative_samples = []
  while len(negative_samples) < num_negative_samples:
     sampled_word = random.choices(list(vocabulary), weights=list(custom_probabilities.values()))[0]
     if sampled_word != word:
         negative_samples.append(sampled_word)

  return negative_samples

# assume custom_probabilities is computed as above

vocabulary = list(word_frequencies.keys())
target_word = "cat"
negative_samples = custom_negative_sampling(target_word, vocabulary, 5, probabilities)

print(f"Negative samples for '{target_word}': {negative_samples}")

```

This example shows a very basic implementation of custom sampling. It's just for demonstration, because in practice, using this approach directly with large datasets and deep learning models would be computationally inefficient compared to the highly optimized techniques implemented within gensim's Word2Vec class. But the principle is the same: use a probability distribution to select negative samples.

In conclusion, obtaining effective negative samples is crucial for training robust Word2Vec models. While the standard frequency-based sampling with a power of 0.75 (or close to it) works remarkably well in most scenarios, it’s valuable to understand the underlying principles and know when you might need to implement a custom solution, even if it's for a specialized use case.

For further reading, I recommend the original Word2Vec paper by Mikolov et al., "Efficient Estimation of Word Representations in Vector Space," which provides the foundational ideas. Also, "Distributed Representations of Words and Phrases and their Compositionality," also by Mikolov et al., expands on this, detailing advanced sampling techniques. A good textbook that covers Word2Vec (and related topics) is “Speech and Language Processing” by Daniel Jurafsky and James H. Martin. Understanding the theory combined with practical experience, that's the key to getting the most out of these kinds of models.
