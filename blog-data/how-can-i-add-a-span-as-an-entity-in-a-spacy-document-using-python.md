---
title: "How can I add a span as an entity in a spaCy document using Python?"
date: "2024-12-23"
id: "how-can-i-add-a-span-as-an-entity-in-a-spacy-document-using-python"
---

Alright,  I recall a project back in '18 involving large-scale text annotation for a legal document processing system where this specific task became quite central. We needed to tag specific phrases as entities, not just single words, and spaCy's flexibility allowed us to achieve this quite elegantly.

You're asking about adding spans as entities in a spaCy document, which essentially means taking a portion of the text and labeling it as a recognized entity. It's crucial for things like named entity recognition (ner) pipelines and more complex information extraction processes where understanding contiguous phrases as units is vital. Let's delve into how you can do this in Python using spaCy.

First, it's important to understand how spaCy structures its documents. A `Doc` object is essentially a sequence of `Token` objects. When you initially process text with a spaCy language model, these tokens are automatically created, alongside linguistic features like part-of-speech tags. However, spaCy doesn’t inherently know which sequences of tokens should be treated as entities; that’s something you have to instruct it on.

The primary mechanism for creating and adding these span entities is through spaCy’s `Span` object. A `Span` is, well, a span of tokens within a `Doc`. You create a `Span` by providing the `Doc` object and start and end token indices, and a label. Once you have this `Span` object, you can then add it to the `Doc`'s `ents` collection (short for entities).

Let’s solidify this with some code.

```python
import spacy

# Load a spaCy model, for example 'en_core_web_sm'
nlp = spacy.load("en_core_web_sm")

# Example text
text = "The quick brown fox jumped over the lazy dog near the old oak tree."

# Process the text
doc = nlp(text)

# Define the start and end indices for the entity "quick brown fox"
start_index = 1  # index of "quick"
end_index = 4 # index of the first token after "fox"

# Create a Span object
span = doc[start_index:end_index]

# Add the Span as an entity. The label can be arbitrary, but here we use "ANIMAL"
span_ent = doc.char_span(span.start_char, span.end_char, label="ANIMAL")

if span_ent:
  doc.ents = list(doc.ents) + [span_ent]

# Print out all recognized entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

In this example, we've manually identified "quick brown fox" as an animal, based on its tokens (indices 1 through 3, not inclusive of index 4), and added it as an entity with the label 'ANIMAL'. We use `doc.char_span` to create a new span, and add it to `doc.ents` *after* checking if `span_ent` exists (char-based spans can return None if the given parameters are invalid). This is particularly important, you can’t just assign the span directly; the object needs to be part of the `ents` collection within the `Doc`.

Now, let's consider a slightly more complex case. What if you want to create multiple entities in the document? It's more efficient to use a function to handle the span creation and addition.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def add_custom_entity(doc, start_char, end_char, label):
  """Adds a custom span entity to a doc given start and end indices."""
  span = doc.char_span(start_char, end_char, label=label)
  if span:
    doc.ents = list(doc.ents) + [span] # Add the new span to the doc
  return doc

text = "The company 'Acme Corp' is located in 'New York City'. It's also opening an office in 'San Francisco'."
doc = nlp(text)


doc = add_custom_entity(doc, text.find("Acme Corp"), text.find("Acme Corp") + len("Acme Corp"), 'ORGANIZATION')
doc = add_custom_entity(doc, text.find("New York City"), text.find("New York City") + len("New York City"), 'LOCATION')
doc = add_custom_entity(doc, text.find("San Francisco"), text.find("San Francisco") + len("San Francisco"), 'LOCATION')


for ent in doc.ents:
  print(ent.text, ent.label_)
```

This snippet shows a more programmatic way to identify and add named entities based on text positions using `text.find()`. The `add_custom_entity` function handles the process, allowing us to efficiently create and insert spans without repetitive code. It takes character start and end positions as input, which makes it convenient for working directly with strings. Note the usage of character-based spans, which are often preferable for accurate span creation. This is because it allows for more robust handling of tokenization nuances and edge cases, such as where tokenization might break up phrases that you want to treat as a single entity.

Lastly, let’s consider a situation where we need to modify the pre-existing entities that spaCy has recognized. While spaCy’s NER is excellent, sometimes you’ll want to add or modify existing ones based on domain-specific rules.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion"

doc = nlp(text)

# Print initial entities
print("Initial Entities:")
for ent in doc.ents:
  print(ent.text, ent.label_)

# Find the existing 'GPE' entity 'U.K.'
for ent in doc.ents:
  if ent.text == "U.K." and ent.label_ == "GPE":
    # Create a new, more inclusive span over 'U.K. startup' and set it to ORG
    new_span = doc.char_span(ent.start_char, text.find("startup") + len("startup"), label="ORG")
    if new_span:
      doc.ents = tuple([e for e in doc.ents if e != ent]) + (new_span,)
    break

print("\nModified Entities:")
for ent in doc.ents:
  print(ent.text, ent.label_)
```

Here, we modify spaCy's standard entity output. The example redefines the ‘GPE’ for “U.K.” to an ‘ORG’ entity by extending it to “U.K. startup”. This is a common use case where pre-trained models provide a starting point, but domain expertise requires you to adapt the results. We removed the old entity and added a new one that includes more context.

For delving deeper into spaCy and its capabilities, I'd strongly recommend reading the official spaCy documentation—it's exceptionally well-written and thorough. Specifically, the sections on `Doc`, `Span`, and `EntityRecognizer` are essential. Furthermore, "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper is a classic text that can give you foundational knowledge. And if you’re looking to expand your knowledge of information extraction, the book "Information Extraction" by Isabelle Moulinier and Marie-France D'Halluin offers comprehensive strategies for tasks much like this.

Working with text is rarely straightforward, but spaCy gives us the needed control to create precisely the entities we require. The key takeaways here are: use spans created from char positions, the `ents` property is how you access and modify entities, and finally, programmatically creating and handling your entities using functions can significantly simplify complex annotation tasks. These techniques should form a strong base for effectively working with named entities in your natural language processing projects.
