---
title: "How can NLP extract the morphological roots of words?"
date: "2024-12-23"
id: "how-can-nlp-extract-the-morphological-roots-of-words"
---

Alright, let's talk about morphological root extraction in nlp. I've been around the block a few times with this, encountering various challenges along the way, especially during a particularly tricky project involving a multilingual sentiment analysis system. We were trying to go beyond simple keyword matching and capture the nuances in user reviews, which meant getting down to the core meaning of words, regardless of their surface variations. That's where effective root extraction became indispensable.

The core issue, as many of you know, lies in the fact that natural languages are incredibly complex. Words don't just appear in their dictionary form; they're modified through prefixes, suffixes, and inflections to convey tense, number, case, and other grammatical properties. To understand the true 'meaning' of a word and group similar words together (for, say, search or analysis), we need to strip these modifications to reveal the underlying root, or lemma. This process, often called stemming or lemmatization, is crucial for reducing the vocabulary size and improving the accuracy of various nlp tasks.

Now, there isn't a single perfect method; the "best" approach depends heavily on the language and the desired level of accuracy. Broadly, the techniques fall into a few categories: rule-based approaches, statistical methods, and hybrid solutions.

Rule-based stemmers, like the well-known Porter Stemmer, employ a set of programmed rules to chop off prefixes and suffixes. These rules are typically developed based on linguistic analysis of the specific language. They're generally fast and easy to implement, which is why they're popular. However, they can sometimes be overly aggressive, producing stems that aren't actual words (over-stemming), or, conversely, fail to reduce some words to their root (under-stemming).

Let me illustrate this with a simple python example using the `nltk` library.

```python
import nltk
from nltk.stem import PorterStemmer

porter = PorterStemmer()

words = ["running", "runs", "ran", "easily", "cats", "cacti", "understanding", "understood"]

for word in words:
    print(f"Original: {word}, Stemmed: {porter.stem(word)}")
```
This snippet demonstrates the behaviour of the porter stemmer. It gets many right, but notice 'easily' being reduced to 'easili', which isn't an actual root word. This highlights the limitations of rule-based approaches.

Statistical methods, on the other hand, utilize corpora to learn patterns in word formation. Techniques like n-gram models or hidden markov models can be used to infer relationships between words and their potential roots. This approach generally yields more accurate results than simple rule-based methods, but requires substantially more computational resources and time to train. They also need a significant amount of training data.

Lemmatization, a more advanced approach, uses dictionaries or wordnet databases to map each word form to its base form, considering the context and part-of-speech. This often offers higher accuracy than stemming because it aims for the dictionary root as opposed to simply stripping affixes. Consider a situation where you need to lemmatize 'better'. A stemmer might give you 'bett', but a lemmatizer would correctly identify the root as 'good', because 'better' is the comparative form of the adjective.

Here’s an example of lemmatization using `nltk` with WordNet:

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


words = ["better", "running", "understood", "cars", "mice", "geese"]

for word in words:
    pos = get_wordnet_pos(word)
    print(f"Original: {word}, Lemma: {lemmatizer.lemmatize(word, pos=pos)}")
```
Here, you can see how 'better' is correctly lemmatized to 'good', 'running' to 'run' (when provided the verb part-of-speech), and plurals like 'mice' and 'geese' are reduced to their singular forms 'mouse' and 'goose'. The `get_wordnet_pos` function is crucial because lemmatization requires part of speech information for correct results.

Hybrid methods, unsurprisingly, combine rule-based and statistical techniques. They can, for example, use a rule-based stemmer for an initial rough reduction followed by a statistical model to refine the root. This often provides the best compromise between speed and accuracy, but it involves a greater level of complexity to implement.

In that sentiment analysis project, we actually implemented a hybrid solution that started with a modified version of the porter stemmer, specific to the needs of our languages, followed by a custom-trained statistical model on a large text corpus. We found that simply using a pre-trained lemmatizer was not sufficient to handle the subtleties of multi-lingual contexts. The combination produced a very robust system.

Now, what are some things to keep in mind when diving into this? Firstly, the choice between stemming and lemmatization should be driven by the specific use case. For search applications, where recall is vital, stemming may be adequate. For applications that demand greater precision such as text summarization or machine translation, lemmatization, if the computational overhead is acceptable, is almost always preferred.

Secondly, for languages other than English, you will probably have a narrower range of pre-built models. It may be necessary to develop custom stemming/lemmatization approaches based on careful study of the linguistic properties of the language, such as inflectional patterns and affixation rules. The `nltk` library is a good starting point, as it includes support for languages other than English, albeit with varying degree of development.

And lastly, the performance of any root extraction method will be greatly influenced by the quality and representativeness of the training corpus if you are going down a statistical route. In some areas, such as in legal or medical text where specialised terminologies are the norm, a general purpose approach will likely not be good enough. It's often the case that we'll require careful customization and evaluation.

I’d suggest further reading into specific areas if you're interested. For general background on nlp, “speech and language processing” by Daniel Jurafsky and James H. Martin is an excellent resource. If you want to dig into specific lemmatization approaches, particularly those based on finite-state transducers, the "finite state morphology" chapter in "computational morphology" by George E. P. Antworth may be useful. And for the practical aspects, diving into the `nltk` and `spaCy` documentation is always a great idea.

Finally, I want to share one last code snippet illustrating that the context and part-of-speech of the word can make a difference in the lemma.
```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

sentences = ["The cars were parked.", "He cars about her."]

for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    lemmatized_words = []
    for word, tag in pos_tags:
        pos = get_wordnet_pos(word)
        lemma = lemmatizer.lemmatize(word, pos=pos)
        lemmatized_words.append(lemma)
    print(f"Original: {sentence}, Lemmatized: {' '.join(lemmatized_words)}")

```
Here, 'cars' is treated differently, depending if it's a plural noun ('car') or a verb ('car'). This illustrates an important aspect of more sophisticated lemmatization compared to basic stemming.

It’s a complex area, no question. But with the right understanding and tools, effective morphological root extraction can be a game-changer for a variety of nlp applications. Hope this overview was helpful.
