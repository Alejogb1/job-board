---
title: "How do I import the Example class from spaCy's training module?"
date: "2024-12-23"
id: "how-do-i-import-the-example-class-from-spacys-training-module"
---

Alright,  I've certainly been down this road a few times, often encountering similar import challenges, especially when diving into the intricacies of spacy's training pipeline. Importing the `Example` class from spaCy's training module isn't always as straightforward as one might expect, primarily because it's not directly exposed at the top level of the `spacy` package. It lives within the training specific submodules. Understanding how spaCy structures its internal modules and the intended use case for `Example` objects helps clarify why the import path is as it is.

In my experience, the main confusion usually stems from a misunderstanding of spaCy's architecture. We typically import functionalities like `nlp` or `Doc` from the root of the package (`import spacy`), which might lead one to expect the `Example` class to be accessible similarly. However, the training components, including `Example`, are kept separate to maintain a cleaner api for general usage. The `Example` class, as you likely know, is crucial for building custom training data and, consequently, for modifying spaCy models to fit domain-specific needs. It serves as the fundamental data structure that pairs a `Doc` object (the input text) with its corresponding annotations, facilitating the learning process.

Now, let's get to the core of it. You'll find the `Example` class within the `spacy.training` module (specifically, often within `spacy.training.example`). Therefore, the correct import statement is typically:

```python
from spacy.training import Example
```

That’s the most direct route. However, sometimes depending on spaCy version, you might find that you also can import it from `spacy.training.example` module directly:

```python
from spacy.training.example import Example
```

Both are usually acceptable, but the first form is generally considered to be more stable between versions. The important takeaway here is the `spacy.training` part of the import path. This module houses the classes, functions, and utilities necessary for model training and evaluation.

Let’s illustrate this with some concrete examples, keeping in mind a typical training workflow scenario. Imagine you've processed some raw text data and have the associated gold-standard annotations (for instance, part-of-speech tags, named entities). You now need to format this data into `Example` objects before using them to update your spaCy model.

**Example 1: Creating an Example from raw text and its annotations**

Let's say you have a raw text string and some pre-annotated entities, which are usually provided in the form of character start and end indices:

```python
import spacy
from spacy.training import Example

nlp = spacy.blank("en")  # or your trained model
text = "Apple is looking at buying U.K. startup for $1 billion"
annotations = {"entities": [(0, 5, "ORG"), (27, 31, "GPE"), (36, 46, "MONEY")]}

doc = nlp(text)

example = Example.from_dict(doc, annotations)

print(example.reference)  # Prints the Doc object corresponding to annotations
print(example.predicted)  # prints the doc object that you passed.
print(example.reference.ents) # Prints span of entities as defined in annotation

```
Here, `Example.from_dict` facilitates constructing a training data point from a `Doc` and a dictionary containing the gold standard annotations. This demonstrates a common way `Example` objects get created.

**Example 2: Using the Example for model update**

Now that we know how to create these, let’s see them in use during training. In this scenario, I am using dummy training data and updating an existing model (not shown).

```python
import spacy
from spacy.training import Example
import random

# Load a pre-trained model or create a blank one
nlp = spacy.load("en_core_web_sm") # or spacy.blank("en") if starting from scratch

# Dummy training data
train_data = [
    ("This is a sentence about Apple.", {"entities": [(24, 29, "ORG")]}),
    ("Microsoft is a technology company.", {"entities": [(0, 9, "ORG")]}),
    ("I love programming in Python.", {"entities": [(21,27, "LANGUAGE")]}),
]

optimizer = nlp.initialize() # Setup the optimizer for backpropagation
losses = {}

# Actual Training loop
for i in range(10): # Run for a small number of epochs

    random.shuffle(train_data)

    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)

        nlp.update([example], losses=losses, sgd=optimizer)

    print(f"epoch: {i}, losses {losses}")

```
This snippet shows how the `Example` object is the crucial input during the `nlp.update` phase, facilitating the backward pass and model parameter adjustments based on your training data. The `losses` object accumulates the errors produced during the model update.

**Example 3: Loading and saving Examples in spacy's 'DocBin' format.**

In situations where large amounts of data are involved, loading them every time or working with python objects might slow things down. You can save this data to disk, and load them back later using the `DocBin` class. You can save the `Example` objects as docs and then load them again to use with the training loop. Here's how:

```python
import spacy
from spacy.training import Example
from spacy.tokens import DocBin

nlp = spacy.blank("en") # Or an existing nlp object

train_data = [
        ("This is an Example sentence.", {"entities": [(11,18, "EXAMPLE")]}),
        ("Second example", {"entities": []}),
        ("Third Example.", {"entities": [(0,5, "EXAMPLE")]}),
    ]

db = DocBin()
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)

    db.add(example.reference) # save reference doc of example objects

db.to_disk("./examples.spacy")

new_db = DocBin().from_disk("./examples.spacy")

loaded_docs = list(new_db.get_docs(nlp.vocab)) # Load the saved doc objects

# You will need to recreate the examples again
examples = []
for i, (text, annotations) in enumerate(train_data):
    example = Example.from_dict(loaded_docs[i], annotations)
    examples.append(example)

print(len(examples)) # prints 3
```
This example shows that for more complex tasks, and for more efficient reading/writing of documents, the `DocBin` class is handy. Notice that we save the reference doc of the `Example` objects to disk. This is a common pattern used.

In conclusion, the `Example` class is essential when training or fine-tuning spaCy models with your own specific dataset. The key is to import it correctly from `spacy.training`. I find the official spaCy documentation particularly helpful for staying up-to-date with the most recent versions. I also recommend delving into "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper for a broader understanding of NLP and data preparation techniques and "Speech and Language Processing" by Daniel Jurafsky and James H. Martin for a more theoretical understanding of NLP and machine learning as applied to NLP tasks. These resources provide the proper context required to use these classes efficiently.
Remember, successful training hinges not only on model architecture, but also on the proper preparation of data, and these insights should help you navigate that.
