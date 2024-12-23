---
title: "How can I add spans as entities in a SpaCy document using Python?"
date: "2024-12-23"
id: "how-can-i-add-spans-as-entities-in-a-spacy-document-using-python"
---

Okay, let’s tackle this. I've certainly encountered this situation before, particularly when dealing with specialized text corpora that require custom entity recognition. Adding spans as entities in a spaCy document programmatically is a fundamental task if you're aiming to go beyond spaCy’s pre-trained models or if you have unique domain-specific entities. It’s not just about annotation; it’s about enriching your document representation for downstream tasks.

The key lies in manipulating the `doc.ents` attribute and leveraging the `Span` object, not simply trying to inject raw text. The `doc.ents` attribute is where spaCy stores the recognized entities within a document, and it's a tuple of `Span` objects. Let me elaborate on how to create these spans and integrate them properly.

Fundamentally, a `Span` object in spaCy represents a slice of text within a document. It’s defined by a start and end character index, as well as the `label` which designates the entity type. Importantly, these spans need to align with the tokenization spaCy has already performed. You can’t arbitrarily specify any random start and end indices, or you'll run into errors, usually relating to overlapping spans or invalid boundaries. My experience in the past has taught me that this is a common source of frustration, particularly when dealing with complex or pre-tokenized data.

To illustrate, let’s start with a very basic example using python:

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm") # ensure you have this model installed. `python -m spacy download en_core_web_sm` if not
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# Example 1: Creating a simple Span object and adding it to doc.ents
span = Span(doc, 0, 3, label="ANIMAL") # "The quick brown" gets the animal tag
doc.ents = list(doc.ents) + [span] # list is needed for concatenation, then convert back to a tuple.

print([(ent.text, ent.label_) for ent in doc.ents]) # display entities.
```

Here, I’ve loaded the `en_core_web_sm` model, processed a text string, and then created a `Span` representing the first three tokens: “The quick brown”. I've labeled it as an "ANIMAL". Note the usage of `list(doc.ents) + [span]` to add it, and then implicitly converting back to a tuple by assigning back to `doc.ents`. This is crucial because `doc.ents` is a tuple which is immutable, hence the modification through a list. This ensures that all existing entities (if any) are retained and that we're adding our new entity correctly.

Now, imagine a slightly more complex scenario, one where you might be receiving indices from an external source, say, a database or some other annotation tool. Let’s say you have character start and end indices, and you need to convert them to spaCy token indices:

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
text = "Apple Inc. is based in Cupertino, California."
doc = nlp(text)

# Example 2: Adding an entity based on character indices.
char_start_index = 0
char_end_index = 9 # "Apple Inc"

# Finding the token start and end indices
start_token_index = 0
end_token_index = 0
for i, token in enumerate(doc):
  if token.idx == char_start_index:
    start_token_index = i
  if token.idx + len(token.text) == char_end_index:
    end_token_index = i + 1

span = Span(doc, start_token_index, end_token_index, label="ORGANIZATION")
doc.ents = list(doc.ents) + [span]

print([(ent.text, ent.label_) for ent in doc.ents])
```

Here, the character indices of "Apple Inc." are used to infer the corresponding token indices. The loop iterates through each token, comparing their `idx` attribute, which is the character offset, and the token's length to align with the desired text. I've included this explicit loop as it's a common scenario where your entity boundaries might not match up perfectly with spaCy tokenization. This method provides finer control and is essential when dealing with pre-annotated data.

One particularly useful technique that you’ll inevitably come across, especially when dealing with multiple entities and pre-existing annotations, is to use the `Matcher` to create or modify spans based on matching patterns. This combines string matching with more complex logic for identifying entities:

```python
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
text = "I need to buy some bananas and oranges from the supermarket"
doc = nlp(text)

# Example 3: Adding entities using a Matcher
matcher = Matcher(nlp.vocab)
pattern = [
    [{"LOWER": "bananas"}],
    [{"LOWER": "oranges"}]
]
matcher.add("FRUIT_ENTITY", pattern)

matches = matcher(doc)
new_entities = []

for match_id, start, end in matches:
    span = Span(doc, start, end, label="FRUIT")
    new_entities.append(span)

doc.ents = list(doc.ents) + new_entities

print([(ent.text, ent.label_) for ent in doc.ents])
```

In this example, I've created a `Matcher` and defined a pattern for "bananas" and "oranges". The `Matcher` finds all occurrences of those words, and it iterates over them to create `Span` objects. I append those into `new_entities` before concatenating it with the existing `doc.ents`. This provides a structured approach and allows for flexibility with more complex patterns. My experience shows the matcher to be a powerful tool for both simple and complex entity extraction pipelines.

Remember, for a more in-depth theoretical understanding of spaCy's tokenization, text processing pipelines, and the `Span` object, consult the spaCy documentation itself; it's meticulously detailed. Beyond that, "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper provides an excellent foundation for understanding core NLP concepts, which helps in understanding why and how spaCy works as it does. Additionally, to really understand the underlying algorithms for things like tokenization and model training, delving into papers from the Association for Computational Linguistics (ACL) would be beneficial - look for work on named entity recognition architectures (NER). These resources have been invaluable to me in understanding these processes at a deep level.

In summary, adding spans as entities involves understanding spaCy’s tokenization model and leveraging `Span` objects along with `doc.ents`. Direct manipulation, index-based adjustments, and using the `Matcher` are the key tools. By mastering these, you can greatly enhance the effectiveness and accuracy of your NLP pipeline when dealing with complex, custom data.
