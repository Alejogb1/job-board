---
title: "How to add a span as an entity in a spaCy doc (Python)?"
date: "2024-12-16"
id: "how-to-add-a-span-as-an-entity-in-a-spacy-doc-python"
---

Alright, let's dive into this. I remember tackling a particularly thorny NLP project a few years back, involving a large corpus of legal documents. We needed to identify very specific contractual clauses—things like 'liability limitation' or 'force majeure'—which weren’t always neatly identifiable by simple token matching. We found ourselves needing more control over how spaCy parsed and represented these phrases; we needed, essentially, to create custom entities spanning multiple tokens. That's exactly what we're discussing here: adding a span as an entity in a spaCy `doc`.

The core issue is that spaCy's default entity recognition usually relies on models trained on specific datasets. These models are fantastic, but they might not perfectly capture every domain-specific entity we're interested in. Sometimes, we need to manually define an entity that consists of a sequence of tokens. SpaCy, thankfully, provides us the mechanisms to do just that. Let's break it down.

First, it’s crucial to understand that a spaCy `doc` object is essentially a sequence of `token` objects. When we talk about a 'span', we're referring to a contiguous subsequence of those tokens. These spans can then be tagged as entities. We don't directly 'add' an entity *to* the document, we 'add' a span *and mark it as an entity*. This crucial distinction clarifies the mechanics.

The primary method to achieve this is via the `doc.set_ents` method, which lets us directly modify the document’s entities. However, to construct a span that represents our target entity, we first have to construct it. We usually do this by explicitly defining the start and end indices of the tokens that constitute the span. This isn’t as complex as it sounds.

Here’s a simplified example to illustrate. Let’s say we have the text "The quick brown fox jumps over the lazy dog". and we want to make the span "quick brown fox" as an entity with label "ANIMAL_PHRASE".

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# Define the start and end token indices.
start_index = 1  # 'quick' is the second token (0-indexed)
end_index = 4    # up to (but not including) 'jumps'

# Create a Span object.
span = Span(doc, start_index, end_index, label="ANIMAL_PHRASE")

# Set the document's entities. Note that we need to provide a list.
doc.set_ents([span])

# Now let's verify.
for ent in doc.ents:
  print(ent.text, ent.label_)
```

Executing this snippet will print “quick brown fox ANIMAL_PHRASE”. Notice, importantly, that `end_index` is *exclusive*. That’s a common point of confusion. We are specifying the index of the token that immediately follows the end of our desired span. This ensures we accurately capture the sequence of tokens. Also, `doc.set_ents` expects a *list* of `Span` objects, even when we only add one.

Now let’s consider a slightly more involved case. Often, the tokens for our desired span aren't fixed, but derived from some sort of pattern matching or lookup within the text. Imagine we need to identify currency amounts in the form "$100", "$1000.50", or "€500.20". We’d need to find those patterns first.

```python
import spacy
import re
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

text = "The product costs $100, while shipping is another $15.50. This other item costs €250.00."
doc = nlp(text)

spans = []
for match in re.finditer(r'[\$\€]\d+\.?\d*', text):
    start, end = match.span()
    start_token_index = len(doc[:start])
    end_token_index = len(doc[:end])

    # Create the span, and append to spans.
    span = Span(doc, start_token_index, end_token_index, label="CURRENCY_AMOUNT")
    spans.append(span)

doc.set_ents(spans)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

Here, we're using a regular expression to identify the potential currency strings. The crucial part is finding the correct token indices corresponding to the start and end character indices of the match found by the regex engine. We do this using slicing of `doc`, and checking the length of the token list to determine the corresponding token indices.

One last practical example that highlights an area I’ve seen trip people up. Let's say we are processing a text where mentions of “New York City” are quite important and should be tagged as a single entity. However, sometimes the data comes with minor variations (like "new york city," "New York City"). We may need to account for such case variants and create a single, unified entity for all cases.

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
texts = ["I love New York City.", "We're going to new york city.", "Visiting NEW YORK CITY is great!"]

for text in texts:
    doc = nlp(text)
    normalized_text = text.lower() # Normalize to handle variations.
    start_char_index = normalized_text.find("new york city")
    if start_char_index != -1:
        end_char_index = start_char_index + len("new york city")

        start_token_index = len(doc[:start_char_index])
        end_token_index = len(doc[:end_char_index])

        span = Span(doc, start_token_index, end_token_index, label="CITY")
        doc.set_ents([span])
    for ent in doc.ents:
        print(ent.text, ent.label_)

```

In this example, I’m using `text.lower()` to normalize my search string and ensuring I can identify the location even if it’s case-variant. This approach helps in creating robust rules that handle slight data inconsistencies.

A point worth noting is that when setting custom entities, it's advisable to set them after you've performed any preprocessing or tokenization tasks that might shift token boundaries. It is also important to note that if the character indices we calculate based on the *string*, do not perfectly align with the start and end of the corresponding tokens after processing the string in spaCy, your custom span may end up not aligning to what you had in mind.

To deepen your understanding, I recommend looking into spaCy's documentation—it's wonderfully detailed and provides numerous examples. Specifically, read the sections on the `Doc` object, `Span` object, and the entity ruler (though the ruler might be overkill for what you're tackling). For a more theoretical background in information extraction, consider the book "Speech and Language Processing" by Jurafsky and Martin. While broad, it provides essential context. Another resource worth exploring is the "Handbook of Natural Language Processing" by Indurkhya and Damerau; this is a more applied resource that gives good insight into a range of NLP topics, including information extraction.

In summary, adding a span as an entity in spaCy involves carefully crafting a `Span` object by specifying the start and end token indices, and then setting the document's entities with `doc.set_ents`. This is a very powerful and frequently used mechanism for tailoring spaCy to very specific tasks.
