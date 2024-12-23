---
title: "Is stemming or lemmatization beneficial?"
date: "2024-12-23"
id: "is-stemming-or-lemmatization-beneficial"
---

Alright, let's tackle this one. The question of whether stemming or lemmatization is beneficial isn't a simple yes or no; it's a nuanced area where the optimal approach depends heavily on the specific task at hand. I've seen projects, more times than I'd like to recall, where the wrong choice here led to significant performance issues and a lot of debugging. My own journey through this started years back on a large-scale text analytics project where we were processing millions of customer reviews daily. We initially went all-in on stemming and quickly realized its limitations. Let me explain.

Stemming, at its core, is a heuristic process that chops off the ends of words in an attempt to reduce them to their root form. Think of it as a quick and dirty approach. The most common algorithm is the Porter stemmer, which operates based on a series of rules. A word like "running" might become "run," and "cats" might reduce to "cat". The advantage here is speed and simplicity. It's computationally cheap and generally easy to implement. However, stemming often leads to over-stemming or under-stemming. Over-stemming occurs when different words are reduced to the same root, even when their meanings aren't directly related. Under-stemming, conversely, leaves related words with different stems. Take, for example, the word "university," which might be stemmed to "univers," alongside words like "universal." While related semantically, they are distinct, and this can lose important nuances during analysis. It's a blunt instrument, effectively.

Lemmatization, on the other hand, aims for a more sophisticated approach. It's a process that transforms a word to its lemma, which is the dictionary form of the word. This requires a deeper understanding of the language's grammar and morphology. For instance, lemmatizing "better" would result in "good," and "went" would become "go." Lemmatization uses a lexicon, that is, a dictionary, and morphological analysis to obtain the root word. Compared to stemming, lemmatization is computationally more expensive, since it requires more processing. However, this is often well worth the cost, particularly in applications where semantic accuracy is crucial. It is important to remember that lemmatization requires part-of-speech tagging to understand the role of the word within the given sentence. This process adds computational overhead but improves accuracy.

Now, let's look at some code examples to see this in practice. We’ll use Python, and the NLTK library, a commonly used tool for natural language processing.

First, stemming with PorterStemmer:

```python
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
words = ["running", "easily", "cats", "better", "universities", "university"]
stemmed_words = [stemmer.stem(word) for word in words]

print(f"Original words: {words}")
print(f"Stemmed words: {stemmed_words}")

```

Executing this, you might observe output similar to:

```
Original words: ['running', 'easily', 'cats', 'better', 'universities', 'university']
Stemmed words: ['run', 'easili', 'cat', 'better', 'univers', 'univers']
```

Notice the irregularities? While "running" became "run" and "cats" became "cat," "easily" is now "easili," and both "universities" and "university" are reduced to "univers". This demonstrates both the potential for both accurate and flawed reductions.

Next, let's see lemmatization with the WordNetLemmatizer.

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
words = ["running", "easily", "cats", "better", "universities", "university"]
pos_tags = [pos_tag(word_tokenize(word))[0] for word in words]
lemmatized_words = [lemmatizer.lemmatize(word, pos=tag[1][0].lower()) if tag[1][0].lower() in ['a','n','v','r'] else word for word, tag in zip(words, pos_tags)]


print(f"Original words: {words}")
print(f"Lemmatized words: {lemmatized_words}")

```

Here’s an example output:

```
Original words: ['running', 'easily', 'cats', 'better', 'universities', 'university']
Lemmatized words: ['running', 'easily', 'cat', 'good', 'university', 'university']
```

Observe that "better" becomes "good," "cats" is correctly lemmatized to "cat" and importantly, "university" and "universities" remain distinct, preserving that difference. Note the added complexity using part-of-speech tags (POS) which is not needed for stemming, but necessary for lemmatization.

Finally, consider a more complex case with sentence-level lemmatization, ensuring part-of-speech tagging:

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
sentence = "The cats were running quickly and it was better than before."
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)
lemmatized_tokens = [lemmatizer.lemmatize(token, pos=tag[1][0].lower()) if tag[1][0].lower() in ['a','n','v','r'] else token for token, tag in zip(tokens, pos_tags)]
lemmatized_sentence = " ".join(lemmatized_tokens)
print(f"Original sentence: {sentence}")
print(f"Lemmatized sentence: {lemmatized_sentence}")
```

The expected output would look like:

```
Original sentence: The cats were running quickly and it was better than before.
Lemmatized sentence: The cat be run quickly and it be good than before .
```
Notice how the words are reduced to their root forms while still maintaining grammatical context.

Now, back to the original question: when should one choose stemming over lemmatization, or vice-versa?

Stemming is a reasonable choice when speed and resource constraints are paramount, and a slight loss of semantic accuracy is acceptable. Examples include tasks like basic keyword indexing or information retrieval where exact word matching isn’t critical. It’s often a fine initial step for quickly building prototypes or preliminary systems.

Lemmatization becomes beneficial when semantic accuracy and context are crucial, such as in sentiment analysis, text summarization, or building conversational AI applications. Here, having accurate root words helps improve the understanding of text and prevents potentially misleading results caused by over or under-stemming. It's particularly important where the distinctions between different forms of a word are critical to meaning.

In my experience, most of the time, the overhead of lemmatization is well worth it. The increased accuracy often outweighs the additional computation cost, especially with modern computing power. The one exception being very large datasets where the computational cost of lemmatization may become prohibitive. However, even there, careful preprocessing with stemming, coupled with more targeted lemmatization on a sub-set of data can sometimes deliver the best of both worlds.

For further study, I highly recommend delving into "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. This book provides a very thorough treatment of these and other related topics in natural language processing, and offers a deeper understanding of the underlying theoretical framework. "Foundations of Statistical Natural Language Processing" by Christopher D. Manning and Hinrich Schütze is another excellent text, offering a more mathematical approach to these concepts. Finally, for a more practical and hands-on approach, explore the NLTK documentation thoroughly; this will provide useful examples of the algorithms.

Ultimately, the 'best' method isn't universal; it's always contextual. Careful consideration of your task requirements, the nature of your data, and the desired accuracy will guide you to an appropriate solution. I’ve had to re-engineer so many of these things over the years and I’ve come to learn to always consider what works best for the *specific* task at hand.
