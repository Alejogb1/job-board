---
title: "How do I Add a span (doc'a:b') as an entity in a spaCy doc (python)?"
date: "2024-12-23"
id: "how-do-i-add-a-span-docab-as-an-entity-in-a-spacy-doc-python"
---

,  I’ve been down this road more times than I care to recall, and it usually involves custom entity recognition workflows, which can get tricky if you’re not careful about how you’re handling the span creation and integration into spaCy's doc object. I remember once, working on a legal text processing pipeline, we spent a whole sprint just refining how we were adding custom entities for specific clause types. It wasn't a pretty sight, but it taught me a few things that I’ll share.

Fundamentally, the issue you're facing is how to represent a continuous sequence of tokens (a span) as a distinct entity within a `spaCy` `Doc` object. This is crucial because many NLP tasks, such as information extraction, relation extraction, and custom named entity recognition (ner), rely on this representation. `spaCy` has a fairly elegant way of handling this, but understanding the underlying mechanisms is key to avoiding common pitfalls.

The core idea is to modify the `doc.ents` attribute. However, you can't directly assign new values like you might with a simple list; you need to use `doc.set_ents()`, and to create your spans, we'll be using `Span` objects and the `doc` itself as the base object. This means you’re not just adding a string or list; you’re creating an entity tied to the text structure of the document. The `Span` object stores information like the start and end token indices, the parent `Doc` and the entity label you want to assign. Let’s get into some specifics:

Firstly, you don’t directly modify the existing spans, rather, you overwrite them in most cases if you want to change the entities. The `doc.ents` attribute contains a tuple and is not a mutable list, which is quite deliberate. This design decision prevents unintentional modification that might invalidate the consistency of the tokenization and entity alignment.

Let’s examine a simple scenario. Assume we have a `Doc` and an arbitrary span that we want to label as an entity.

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
text = "The quick brown fox jumped over the lazy dog."
doc = nlp(text)

start_index = 2  # Index of "brown"
end_index = 5  # Index after "fox" - Remember the Pythonic exclusive end index
label = "ANIMAL"

span = Span(doc, start_index, end_index, label=label)
doc.set_ents([span], overwrite=True)


# Verifying results
for ent in doc.ents:
    print(ent.text, ent.label_)
```

In this first example, I've created a simple span for "brown fox jumped". Notice that the `end_index` parameter is exclusive, meaning the span includes all tokens *up to* that index. Also, the `overwrite=True` parameter is important when you are adding new entities. It clears previously detected entities in order to add the new entity. This is useful when you are manually overriding entity recognizers. If you have other entities, that you do not want to override, you can take them from the `doc.ents` attribute.

Now, let's move on to a slightly more complex case: suppose you have a function that is responsible for generating spans based on some external analysis or pattern-matching rule.

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

def find_custom_spans(doc):
    spans = []
    for i, token in enumerate(doc):
        if token.text.lower() == "lazy":
            start_index = i
            end_index = i + 2 # "lazy dog"
            if end_index <= len(doc):
              spans.append(Span(doc, start_index, end_index, label="ADJECTIVE_PHRASE"))
    return spans

text = "The quick brown fox jumped over the lazy dog. It was really lazy dog."
doc = nlp(text)
spans = find_custom_spans(doc)
doc.set_ents(spans, overwrite=True)
for ent in doc.ents:
    print(ent.text, ent.label_)

```

Here, `find_custom_spans` demonstrates how we can programmatically locate relevant portions of the document to turn into `Span` objects. This pattern is extremely common when you’re building a custom NER system and dealing with domain-specific terms, for example.

Now, let’s address a potential challenge: working with pre-existing entities. You might want to keep some of spaCy’s entities while adding your own custom ones. This needs a slightly different approach:

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
text = "Apple is a tech company based in California."
doc = nlp(text)

original_entities = list(doc.ents) # Convert to a list, so it's mutable
start_index = doc[5].i # 'California'
end_index = len(doc) # end of the sentence
span = Span(doc, start_index, end_index, label="LOCATION_PHRASE")

updated_entities = original_entities + [span]
doc.set_ents(updated_entities, overwrite=True)

for ent in doc.ents:
    print(ent.text, ent.label_)

```

This is the best approach if you want to mix SpaCy's default entities with your own customized ones. First, create a list with the entities of the document. Secondly, you can create a new entity. And finally, add the two lists together before passing them into the `set_ents` method. This retains the originally detected entities and adds a new custom one.

In terms of resources to further refine your understanding, I'd highly recommend the official `spaCy` documentation, which is extraordinarily comprehensive and continuously updated. Specifically, familiarize yourself with the `spaCy.tokens.Span` class and the methods of the `Doc` object, such as `set_ents` and `has_annotation`. For a broader perspective on custom NER techniques, “Natural Language Processing with Python” by Steven Bird, Ewan Klein, and Edward Loper is an invaluable resource. Another good book is "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. These delve deeper into the theoretical and practical aspects of the task.

Furthermore, keep in mind the impact of tokenization. Ensure the start and end indices you're using are aligned with the actual token boundaries generated by `spaCy`, as slight mismatches can lead to incorrect spans. When creating a custom `Span` the token indices are extremely important.

Finally, always double-check that the entity labels you are using are appropriate for your use case. If you are going to use a custom model, you need to ensure that the labels have been correctly used in the training.

In closing, adding spans as entities in `spaCy` is straightforward once you grasp the concept of `Span` objects and how `set_ents` works with them, and always be careful with how you manage existing entities. With a bit of care and the right resources, you can build robust and accurate custom entity recognition pipelines. Remember that clear code, a clear understanding of the `spaCy` documentation, and a strong grasp of the principles of NLP are key to your success.
