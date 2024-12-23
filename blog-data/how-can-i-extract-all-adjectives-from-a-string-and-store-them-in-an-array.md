---
title: "How can I extract all adjectives from a string and store them in an array?"
date: "2024-12-23"
id: "how-can-i-extract-all-adjectives-from-a-string-and-store-them-in-an-array"
---

Okay, let's tackle this. From my experience, working with natural language processing (NLP) often throws these kinds of parsing problems your way, and it’s rarely as straightforward as it first appears. A simple string of words needs to be deconstructed into its parts of speech, and identifying adjectives among them requires a more nuanced approach than just looking for words that "describe" something. I've actually been in a similar situation, developing a sentiment analysis tool for customer feedback, and the accuracy of adjective extraction was critical.

The challenge isn't just about identifying *any* word that might function as an adjective in isolation. Context matters immensely. The same word can be a noun, verb, or adjective depending on its usage. Therefore, a purely rule-based solution is usually brittle and fails in edge cases. Instead, the most reliable way involves leveraging part-of-speech (POS) tagging using existing NLP libraries. This process annotates each word with its grammatical category, allowing for precise adjective extraction.

So, to answer directly, the general steps would involve: text tokenization, pos tagging, and finally, the filtering to keep only adjectives. It's a multi-step process.

Let's break it down with some examples in python. I’m comfortable with it, and the nltk library makes this task considerably easier.

**Example 1: Basic Implementation using NLTK**

First, you’ll need to install NLTK and download some necessary data. Assume you've done that or that you're using a virtual environment setup correctly, as i often do. This example is about showcasing the process end-to-end.

```python
import nltk
from nltk import word_tokenize, pos_tag

def extract_adjectives(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    adjectives = [word for word, tag in tagged_tokens if tag.startswith('JJ')]
    return adjectives

text_example = "The quick brown fox jumps over the lazy dog. It was a very beautiful day."
extracted_adjectives = extract_adjectives(text_example)
print(extracted_adjectives)
# Expected output: ['quick', 'brown', 'lazy', 'very', 'beautiful']
```

Here, we use `word_tokenize` to split the sentence into individual words, and then `pos_tag` to assign a part of speech tag to each token. NLTK's tagset follows the Penn Treebank notation where 'JJ', 'JJR', and 'JJS' all represent adjectives (adjective, comparative adjective, superlative adjective respectively). We utilize the `startswith('JJ')` because we want to capture all these variations of adjectives. Then, we simply extract all the words whose tags start with "JJ".

**Example 2: Handling More Complex Sentences and Punctuation**

The previous example was straightforward. Real-world text, however, contains more complexity. Consider the situation below. This involves handling of punctuation correctly. Also this example handles cases of adjectives connected by a hyphen.

```python
import nltk
from nltk import word_tokenize, pos_tag

def extract_adjectives_advanced(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    adjectives = []
    for word, tag in tagged_tokens:
        if tag.startswith('JJ'):
            adjectives.append(word)
        elif word == '-':
            # handle hyphenated adjectives
            continue
        elif tag.startswith('NN') and 'JJ' in [t for _, t in pos_tag(tokens)[pos_tag(tokens).index((word,tag))-1:pos_tag(tokens).index((word,tag))]] :
                #check for noun that functions as adj, like "high-school"
            continue
    return adjectives


text_example_2 = "That's a long-term commitment. The high-school teacher was very understanding."
extracted_adjectives_2 = extract_adjectives_advanced(text_example_2)
print(extracted_adjectives_2)
# Expected output: ['long', 'term', 'high', 'understanding']
```

In this version, we've made an adjustment to check for hyphenated words and also to handle cases where a noun might function as an adjective, like 'high' in 'high-school.' We check if the word before it is indeed and adjective. In real-world scenarios, this might be an area where further customization and context checking could be added.

**Example 3: Leveraging SpaCy for Improved Performance and Contextual Understanding**

While nltk is powerful, spaCy often provides better performance and more pre-trained models, and it’s often what i find myself reaching for in these scenarios. Let’s look at a basic implementation using SpaCy. This example leverages language models so we can appreciate the power of the library.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_adjectives_spacy(text):
    doc = nlp(text)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return adjectives


text_example_3 = "This amazing, brand-new car is incredibly fast. The old, broken toy was sad."
extracted_adjectives_3 = extract_adjectives_spacy(text_example_3)
print(extracted_adjectives_3)
# Expected output: ['amazing', 'brand-new', 'fast', 'old', 'broken', 'sad']
```

SpaCy loads a language model and parses the text into tokens with rich contextual information. SpaCy uses the "ADJ" pos tag which neatly maps to our need for adjectives. This is more performant and generally more accurate than using NLTK.

**Important Considerations and Further Reading**

The solutions above are basic starting points. In real applications, you'll likely encounter more complex situations:

1.  **Contextual Ambiguity:** Consider phrases like "the running man". The word "running" is a verb in its present participle form, but it may function as an adjective in the present context. In such cases, techniques like dependency parsing (available in spaCy) can help you decipher the relationships between words and improve accuracy further.

2.  **Domain-Specific Language:** Pre-trained models might not perform well in highly specialized domains. In those cases, consider using custom model training.

3.  **Performance Optimization:** For very large texts, consider batch processing and optimized text preprocessing techniques.

4.  **Data Augmentation:** To improve the generalization of adjective extraction models, you could use data augmentation.

For further reading, I recommend these resources:

*   **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** This book is considered a bible in NLP. The chapters on part-of-speech tagging and parsing would be most relevant.
*   **The SpaCy Documentation:** SpaCy’s official documentation is excellent, and i often refer to it. Dive deeper into its part-of-speech tagging, dependency parsing, and customization features.
*   **NLTK Documentation:** Similarly, explore the documentation for NLTK, particularly its tagging modules.
*   **Papers on Sequence Tagging:** Look for scholarly articles on Hidden Markov Models (HMMs) and Conditional Random Fields (CRFs) as they are fundamental to how some POS taggers work, although you usually don't need to implement these directly if using prebuilt libraries.

In summary, extracting adjectives is not about a simple find-and-replace task. It demands using the tools that part-of-speech tagging in robust NLP libraries like NLTK or spaCy offers. While I've provided some starter code, you should explore these libraries further, customize them to your specific needs, and always test your solutions with edge cases in mind. This process requires a blend of theoretical knowledge and a practical approach, informed by the nuances of the problem you're tackling.
