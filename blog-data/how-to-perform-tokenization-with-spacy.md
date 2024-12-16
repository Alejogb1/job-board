---
title: "How to perform tokenization with spaCy?"
date: "2024-12-16"
id: "how-to-perform-tokenization-with-spacy"
---

Alright, let’s talk about tokenization in spaCy. I've spent a fair bit of time working with natural language processing, and tokenization is always the initial, crucial step. Getting it wrong can snowball into all sorts of downstream issues. I recall a project a few years back where the client's sentiment analysis was consistently off – turns out we were missing edge cases in how the text was initially broken down into tokens, specifically concerning emoticons mixed with punctuation. So, let’s delve into how spaCy handles this, and what makes it work.

Fundamentally, tokenization is the process of segmenting text into its constituent units, often words, punctuation marks, or even sub-word units. These units are called tokens. spaCy offers a sophisticated and highly configurable approach to tokenization that goes beyond simple whitespace splitting, which, as I'm sure you appreciate, has limited utility with real-world text.

spaCy's tokenization pipeline consists of several components operating on the text in sequence, the most important of which is the tokenizer. The `Tokenizer` object in spaCy employs a set of rules, special cases, and exception handling to determine how text is split. This allows it to deal with complex linguistic phenomena such as contractions ('can't'), hyphenated words ('state-of-the-art'), and URLs. The underlying algorithms are quite nuanced, considering, for instance, how surrounding characters influence how a punctuation mark should be treated.

Here's how it basically works: spaCy takes your raw text input, and the `Tokenizer` processes it to generate a `Doc` object. This `Doc` object is more than just a sequence of strings; it encapsulates token metadata such as starting and ending indices in the original string, as well as language-specific attributes, such as parts of speech. Crucially, the `Doc` object retains references back to the original text, crucial for later processing steps. The tokenizer is generally fast and efficient, a characteristic I’ve found incredibly beneficial when working with large datasets.

Now, let's look at some code examples.

**Example 1: Basic Tokenization**

This snippet demonstrates the most common usage of spaCy for tokenization.

```python
import spacy

nlp = spacy.load("en_core_web_sm")  # Loading a pre-trained model

text = "This is a sentence. It's tokenized with spaCy! How cool is that?"
doc = nlp(text)

for token in doc:
    print(token.text, token.idx)
```
In this example, we load a smaller English language model. We then feed our example text into the model. The `nlp()` function applies the entire processing pipeline, including the tokenizer, and returns a `Doc` object. We can then iterate through this `Doc` object and access each token’s text representation (`token.text`) and its starting character index in the original string (`token.idx`). This output clearly demonstrates how even though our input text includes various sentence types and punctuation, spaCy handles these seamlessly.

**Example 2: Handling Special Cases and Rules**

This example illustrates spaCy's capacity to handle more complex scenarios beyond simple whitespace segmentation.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Let's visit google.com or read about state-of-the-art AI."
doc = nlp(text)

for token in doc:
    print(token.text, token.is_punct, token.is_alpha, token.like_url)

```

Here, you can see how spaCy correctly tokenizes "Let's" into two tokens, "Let" and "'s", because it recognizes the apostrophe as a token boundary. Similarly, it recognizes "state-of-the-art" as a hyphenated word and keeps it together instead of breaking it apart. It also correctly identifies "google.com" as something URL-like. The `token.is_punct` flag tells us if a token is punctuation and `token.is_alpha` tells us if it is alphabetic. `token.like_url` identifies if the token looks like a URL. These token-level attributes are very helpful for fine-grained analysis and are essential when creating custom pipelines.

**Example 3: Customizing Tokenization**

In some instances, pre-trained models might not precisely meet your requirements. spaCy allows you to customize the tokenization process using custom rules. In one project I was on, we had a specific identifier format that the standard tokenizer would not handle correctly, forcing us to resort to this customization approach. While complex, it's very helpful for niche requirements.

```python
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")

prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
infix_re = spacy.util.compile_infix_regex(nlp.Defaults.infixes)

# Custom rule
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab,
                 prefix_search=prefix_re.search,
                 suffix_search=suffix_re.search,
                 infix_finditer=infix_re.finditer,
                 token_match=None,
                 rules={"my_custom_identifier": [{ "ORTH": "id123-456" }]})

nlp.tokenizer = custom_tokenizer(nlp)

text = "Here is my id123-456, and another regular word."
doc = nlp(text)

for token in doc:
    print(token.text)
```

Here, we replaced the default tokenizer with a new tokenizer that has additional rules defined. Specifically, we are instructing it to recognize "id123-456" as a single token. This customization is done through a `rules` argument passed to the `Tokenizer` constructor. In this instance, we're just using a simple dictionary rule to add that token, but there are many more advanced ways to handle tokenization, including using regular expressions for more complex patterns.

It is essential to understand that the `Tokenizer` is only one component within the larger NLP pipeline. After tokenization, spaCy typically proceeds with steps such as part-of-speech tagging, dependency parsing, and named entity recognition. Proper tokenization is a prerequisite for the accurate and efficient functioning of these downstream tasks.

For a deeper dive into the mechanisms underlying spaCy's tokenization capabilities, I would recommend reviewing the detailed documentation within the spaCy library itself. You can find this in the “Advanced tokenization” section, and specifically the section detailing “tokenizer exceptions, prefixes, suffixes, and infixes.” Specifically, for understanding the theoretical underpinnings, the work on finite-state transducers, which are related to how some tokenizers operate, is useful. Also, keep an eye out for recent papers related to pre-trained language model tokenizers, such as SentencePiece and Byte Pair Encoding (BPE), since even though spaCy does not rely directly on these for the tokenizer, a solid understanding of their mechanisms can help you comprehend the design choices behind many tokenizers. A good overview is found in “Neural Machine Translation of Rare Words with Subword Units,” which describes BPE. “SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing” details the workings of SentencePiece.

In practice, if you are working with languages or text domains not covered by the core spaCy models, you might need to create your own custom models and tokenizers, which could be quite complex but can provide large gains in accuracy. I've personally found that starting simple and gradually refining the tokenization approach is the most effective way to handle these more advanced challenges. Pay attention to error messages generated during tokenization, they often indicate where there may be unexpected behavior.
