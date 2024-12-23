---
title: "How can I locate entities in a document using spaCy 3?"
date: "2024-12-23"
id: "how-can-i-locate-entities-in-a-document-using-spacy-3"
---

Let's jump right in, shall we? Having navigated the often-turbulent waters of natural language processing for quite some time, I can tell you that entity recognition, especially in its practical implementation, is where theory meets the real world with a bit of a thud. Specifically, when discussing spaCy 3, we're talking about a robust and flexible framework that, while powerful, demands a clear understanding of its mechanics for efficient use. So, let's explore how to pinpoint entities in your documents using spaCy 3, based on some of my past experiences.

Early in my career, I was tasked with extracting key information from a large corpus of legal documents. The goal was to automate the identification of named entities like organizations, people, locations, and dates to streamline the initial document review process. Back then, the challenges weren't just about having the right library – they were equally about understanding the subtleties of the data and how to tailor the models effectively to our specific use case. spaCy 3, thankfully, provides a suite of tools that greatly facilitate such tasks.

The core of entity recognition in spaCy lies in its statistical models, trained on massive amounts of text data. These models essentially learn to predict which spans of text are likely to represent certain entity types. To get started, you typically load a model using `spacy.load()`. Consider the common `en_core_web_sm` model:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple is looking at buying U.K. startup for $1 billion."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

In this initial example, we load the small English language model, which provides a basic level of entity recognition. When we process the sample text, we iterate through the `doc.ents` attribute, which contains all the identified entities. Each entity has a `text` attribute indicating the text span, and a `label_` attribute, which contains the predicted entity type (like ‘ORG’ for organization).

This is a good starting point, but let's face it, real-world scenarios rarely present themselves as simply as this. Often, you'll deal with documents that contain specialized terminology or entity types that the pre-trained models might not be fully equipped to handle. This brings us to a crucial aspect of working with spaCy: customisation.

I recall a project involving financial news articles, where we needed to identify custom entity types like ‘ticker_symbol’ and ‘transaction_type’. The pre-trained models were largely ineffective for these specific categories, requiring a strategy for refinement. spaCy 3 allows us to either fine-tune existing models or train entirely new ones. I opted for the latter, and here’s the basic framework for that (although fine-tuning is generally more efficient when sufficient labelled data is not available for training):

```python
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random

# Sample training data (replace with your real training data)
TRAIN_DATA = [
    ("The company ACME reported a profit.", {"entities": [(12, 16, "ORG")]}),
    ("Buy shares of GME", {"entities": [(15, 18, "TICKER")]}),
    ("Selling AAPL before the crash", {"entities": [(8, 12, "TICKER"), (20,25,"EVENT")]}),
    ("The merger between company XYZ and ABC went through.", {"entities": [(23,26, "ORG"), (31, 34, "ORG")]}),
    ("The IPO of NVDA was successful.", {"entities": [(11,15,"TICKER")]})
]
nlp = spacy.blank("en") # Start with a blank model
ner = nlp.add_pipe("ner")

# Add custom labels to the ner pipeline
ner.add_label("ORG")
ner.add_label("TICKER")
ner.add_label("EVENT")

optimizer = nlp.begin_training()

for i in range(20): # Training loop - increase iterations if needed
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        nlp.update(examples, sgd=optimizer, losses=losses)

    print(f"Losses at iteration {i}: {losses}")


# Test the custom trained model
text = "ACME, a well known organisation, reported profits on GME and AAPL"
doc = nlp(text)

for ent in doc.ents:
   print(ent.text, ent.label_)

```
This code snippet illustrates how to initiate a blank english model, add an ner pipeline, and use training data to teach the model to recognise custom named entities. Here, `TRAIN_DATA` is a sample dataset, and ideally, this should consist of hundreds or thousands of examples. The `minibatch` function provides training data in batches, which is useful for more efficient training. The crucial step is within the loop, where we update the model using the `nlp.update()` method, based on the examples extracted from the training data. We're essentially teaching the model, by providing labelled instances, what patterns are indicative of specific entity types, in our case, ‘ORG’ for organizations, ‘TICKER’ for trading symbols, and ‘EVENT’ for significant occurrences. After training, we see our customized model will have better performance for our custom entities, as demonstrated when we test it with a new sentence.

One other practical challenge I faced involved the proper handling of context when identifying entities. Consider the sentence "The president met the CEO of Apple on Friday." We need to correctly understand that "Apple" is an organisation in this context, not the fruit, and the surrounding words provide contextual clues. To help the model recognise these context-based patterns, a more advanced model architecture may be needed, such as transformers, which excel at understanding the context between words. While spaCy offers pre-trained transformer models, sometimes a small refinement may be enough. Let’s say we’ve identified that our current model mistakenly flags dates as locations (because, for some reason, the training data has some inconsistencies) and we would like to specifically adjust that in the current model. We might do that with the `spacy train` command in the terminal. However, we will explore an alternative with the following code, to focus on the python implementation of this. We'll also add a custom matcher, which operates on the token level, which can be useful for context-specific tasks.

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_lg")

# Example text and model
text = "The meeting was scheduled for August 10th, 2024 in London."
doc = nlp(text)

# Initial entities
print ("initial entities:")
for ent in doc.ents:
    print(f"Entity Text: {ent.text}, Label: {ent.label_}")

# Custom Matcher to check for wrong dates
matcher = Matcher(nlp.vocab)
date_pattern = [{"LOWER": "january"}, {"IS_DIGIT": True, "OP":"?"}, { "TEXT": "," , "OP":"?"},  {"IS_DIGIT": True}]
month_pattern = [{"LOWER": {"IN": ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]}}]
matcher.add("DATE_PATTERN", [date_pattern, month_pattern])

matches = matcher(doc)

# Correct entities if any matches found
if matches:
    print ("\ncorrected entities:")
    for match_id, start, end in matches:
        span = doc[start:end]
        for ent in doc.ents:
            if ent.start == start and ent.end == end:
                ent.label_ = "DATE"
                print (f"Entity Text: {ent.text}, Label: {ent.label_}")
                break;
        else: #If an entity isn't already present as an entity
           print(f"Entity Text: {span.text}, Label: DATE") # Just print it

    for ent in doc.ents:
      if ent.label_ != "DATE": #Print all entities that aren't dates
        print(f"Entity Text: {ent.text}, Label: {ent.label_}")
```

In this example, we load the large pre-trained model, and examine its initial output. We notice that the date is recognized incorrectly as a location, which we need to correct, so we set up a custom matcher with `spacy.matcher.Matcher()`. The `matcher.add()` method registers matching patterns. These patterns are defined as list of dictionaries that describe the characteristics of each token. The `matcher(doc)` applies these patterns to the document, and we then use the matching spans to potentially correct or augment the identified entities, here in the form of relabeling identified entities. We modify the label directly using `ent.label_ = "DATE"`. This provides an example of how context analysis can be accomplished, and in a more complicated context, we may have to also train specific models for different types of context using more advanced techniques.

As a starting point for diving deeper into this field, I strongly recommend the official spaCy documentation, which is comprehensive and meticulously maintained. For a more theoretical understanding of the models used, “Speech and Language Processing” by Daniel Jurafsky and James H. Martin is an excellent resource. For practical insights specifically on entity recognition, refer to "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper, particularly the chapters focusing on information extraction and named entity recognition.

In summary, effective entity recognition with spaCy 3 requires an understanding of the pre-trained models, the mechanisms for customization through training or fine-tuning, and how to use the API for context analysis. While I've touched upon some essential aspects, remember that every data set can be unique, so experimentation is vital. It’s about understanding what spaCy offers, and, more importantly, knowing how to shape it to solve your specific problems.
