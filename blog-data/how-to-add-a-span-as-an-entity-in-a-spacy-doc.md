---
title: "How to add a span as an entity in a spaCy doc?"
date: "2024-12-16"
id: "how-to-add-a-span-as-an-entity-in-a-spacy-doc"
---

Okay, let's talk about spans in spaCy. I’ve grappled with this exact issue plenty of times, particularly when working on custom named entity recognition pipelines. It's not always straightforward, especially when you're trying to extend spaCy's capabilities beyond its built-in entity types. The core problem lies in understanding that `Doc` objects in spaCy are immutable, and we need to use a specific process to introduce these new, customized span entities.

The primary approach involves using the `Span` object and its integration with the `Doc.spans` container. You can't just randomly add a `Span` to a `Doc`. Instead, you add them to a specifically named span group within the `Doc.spans` dictionary. If that group doesn't exist, you typically create it. Let's delve into the details.

First off, the `Span` itself represents a slice of the original `Doc`. It’s defined by character start and end indices, or by token-based start and end indices. Remember, the character-based method is less robust because it is susceptible to errors if tokenization changes, whereas token indices are relative to token sequences. This means, if you change the tokenizer settings, character indices might be off. I've learned that the hard way on a project involving user-submitted text where pre-tokenization was mandatory. A small change to the tokenizer, and all my previously identified character spans were broken. Thus, I prefer token-based indexing.

Let's start with the most basic way to add a span entity, using token indices:

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

# Define the span with token indices:
start_token_index = 2
end_token_index = 5
new_entity_span = Span(doc, start_token_index, end_token_index, label="ANIMAL")

# Add the span to the 'custom_entities' span group
if "custom_entities" not in doc.spans:
    doc.spans["custom_entities"] = []
doc.spans["custom_entities"].append(new_entity_span)

# Verification:
print([(ent.text, ent.label_) for ent in doc.spans["custom_entities"]])
```

In the preceding code, I create a `Span` object with start index 2 and end index 5, which corresponds to "brown fox jumps." Notice that spaCy’s indexing is exclusive of the end index. Crucially, we check for the existence of "custom_entities" in `doc.spans` before appending. This prevents potential errors if you try to add to a non-existent list. This approach was a lifesaver when I was building a modular entity recognition pipeline where different modules could add entities independently without conflicts.

Now, let's look at adding multiple spans of varying types. This is a common use case, especially if you’re building a system that needs to recognize different types of custom entities:

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple released the new iPhone last year. Microsoft is doing good.")

# Define two spans:
span1 = Span(doc, 0, 1, label="COMPANY")  # "Apple"
span2 = Span(doc, 5, 7, label="PRODUCT")  # "new iPhone"
span3 = Span(doc, 8, 9, label="TIME")  # "last year"
span4 = Span(doc, 10, 11, label="COMPANY") # "Microsoft"

# Add them to the 'custom_entities' span group:
if "custom_entities" not in doc.spans:
    doc.spans["custom_entities"] = []
doc.spans["custom_entities"].extend([span1, span2, span3, span4])

# Verification:
print([(ent.text, ent.label_) for ent in doc.spans["custom_entities"]])
```

In this example, we create several `Span` objects with differing labels and then extend the `doc.spans` entry using `extend`. This was vital in my work involving financial news analysis, where I needed to extract company names, products, and time references, often in the same document, with varying context. Doing it this way ensured a structured and maintainable way to add diverse entities.

Finally, let’s consider a scenario where you might have a list of pre-defined entity locations based on some external lookup. I encountered this issue when integrating a knowledge graph, which provided spans, but they were in the form of character offsets:

```python
import spacy
from spacy.tokens import Span
from spacy.util import filter_spans


nlp = spacy.load("en_core_web_sm")
text = "The University of California is a good place, and Stanford University is not too far."
doc = nlp(text)

# Suppose you have the following character spans:
char_spans = [ (4, 28, "ORGANIZATION"), (48,66, "ORGANIZATION")]  # "University of California", "Stanford University"
spans = []
for start, end, label in char_spans:
    span = doc.char_span(start, end, label=label)
    if span is not None:
        spans.append(span)


if "custom_entities" not in doc.spans:
    doc.spans["custom_entities"] = []

filtered_spans = filter_spans(spans)
doc.spans["custom_entities"].extend(filtered_spans)

print([(ent.text, ent.label_) for ent in doc.spans["custom_entities"]])
```

Here, we begin with character spans from some external source and use the `doc.char_span()` method, which attempts to convert these to token-based spans. The `filter_spans` function is critical here. Character spans can lead to overlaps when tokenized, and the filtering function resolves such conflicts, guaranteeing no overlaps in the final spans. In my past project dealing with legal text, overlapping named entities was a very common problem due to inconsistent annotation styles, making the use of `filter_spans` indispensable.

Key things to remember: always use a specific span group name (e.g., "custom_entities") instead of the default one, be consistent in either using character or token indices, and consider using token-based indices due to their robustness. If working with character indices, ensure to validate the spans using `doc.char_span()` which converts them to token-based spans, and then you'll likely need to `filter_spans` to prevent overlapping entities in the case of ambiguous spans. Also, remember, the `Doc` object is immutable, hence all updates should be made using the provided interfaces. It's not about changing the `Doc` directly, but adding to specific span groups within it.

For deeper understanding, I highly recommend reading "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper, which offers a strong foundational understanding of language processing concepts. Also, "Speech and Language Processing" by Daniel Jurafsky and James H. Martin is invaluable for a more advanced treatment of NLP and entity recognition. Finally, reviewing spaCy's official documentation, especially the sections on `Doc`, `Span`, and extension attributes, will be very helpful. These references should cover the necessary background for mastering these concepts.
