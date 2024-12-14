---
title: "How to Identify documents after processing in a spaCy pipeline?"
date: "2024-12-14"
id: "how-to-identify-documents-after-processing-in-a-spacy-pipeline"
---

hey there,

i've seen this question pop up a bunch, and it's something i've definitely battled with in the past. when you’re churning through a mountain of text with spacy, it’s super easy to lose track of which document is which, especially when you start parallelizing things or doing more complex operations. so, let's get down to the nitty-gritty of how to keep your documents identified after a spacy pipeline. it's mostly about planning your data structures carefully before you even call spacy.

my first encounter with this was back when i was building a news article summarizer. i was using spacy to extract entities and key phrases, and things were flying through the pipeline. i quickly realized that all my outputs were just lists of extracted stuff, and i had no clue where they came from, making it useless for further processing. i had to rewrite a significant portion of the code. i learned my lesson: keep track of identifiers at the start, that's like the golden rule.

the core problem is that a basic spacy pipeline transforms a string of text into a spacy `doc` object. that doc object itself doesn't inherently remember the original identifier. so, we need to attach the identifier information ourselves. one way to handle this is to create a custom data structure that links the identifier to the text before it goes through the pipeline, and to the processed `doc` object after it returns from the pipeline. this is what I usually do: a simple dictionary works wonders.

here's an example of how i typically structure this before the spacy pipeline:

```python
documents = {
    "doc_1": "this is the first document.",
    "doc_2": "and here is the second document, with more text.",
    "doc_3": "a shorter one."
}
```
this basic dictionary gives me something to work with, the keys here "doc_1", "doc_2" etc are the ids, and the values are the string to process.

now, let’s write a piece of code that uses this dict and adds a custom attribute to each doc element, i like to name the custom attributes doc_id to differentiate it from the other ids spacy uses. this also includes storing results in a new structured dict, so that you can keep the id associated with the processed data.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def process_documents(docs, nlp):
  processed_docs = {}
  for doc_id, text in docs.items():
        doc = nlp(text)
        doc.set_attribute("doc_id", doc_id)
        processed_docs[doc_id] = doc
  return processed_docs


documents = {
    "doc_1": "this is the first document.",
    "doc_2": "and here is the second document, with more text.",
    "doc_3": "a shorter one."
}


processed = process_documents(documents, nlp)
for doc_id, doc in processed.items():
    print(f"processing doc with id: {doc.get_attribute('doc_id')} ")
    print(f"first token: {doc[0].text}")

```
this code iterates through the dictionary, passes the text to the spacy pipeline and stores the return in a new dictionary with the original ids as keys. the `set_attribute("doc_id", doc_id)` part is where the magic happens, associating the original id with the doc object. notice that we are using the method `doc.get_attribute` to fetch that id when we are iterating through the results.

this is great for when you just want to attach one id but what if you wanted to attach multiple ids, or multiple types of data. we can do this by creating the docs array with tuples of data: the id, the text, and some additional info. in this example we add a category. the same logic applies but we just use different data structures.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def process_documents_tuples(docs_tuples, nlp):
  processed_docs = {}
  for doc_id, text, category in docs_tuples:
    doc = nlp(text)
    doc.set_attribute("doc_id", doc_id)
    doc.set_attribute("category", category)
    processed_docs[doc_id] = doc
  return processed_docs


documents_with_categories = [
  ("doc_1", "this is a tech document about spacy.", "tech"),
  ("doc_2", "this is a business document about sales.", "business"),
  ("doc_3", "this is a short sports document", "sports")
]

processed_with_cat = process_documents_tuples(documents_with_categories, nlp)
for doc_id, doc in processed_with_cat.items():
    print(f"processing doc with id: {doc.get_attribute('doc_id')}, category: {doc.get_attribute('category')}")
    print(f"first token: {doc[0].text}")
```

in this example, the logic remains the same but we use a list of tuples instead of a dictionary, and we are attaching the `doc_id` and the `category` to the `doc` object. it’s really versatile, you can add as many pieces of identifying info or metadata as your heart desires. the most important thing is to set them before the data is processed, that's the most critical point. the thing is that doc objects are mutable, and we can use that to add custom metadata.

there's a different way to approach this that may be more efficient if you are going to modify the doc objects themselves, using the `Doc.set_extension` method, spacy documentation recommends using this to set custom attributes at the Doc level, instead of using the `set_attribute` method. this is a good practice when you know you will be changing attributes during the life of the doc.

```python
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")

Doc.set_extension("doc_id", default=None)
Doc.set_extension("category", default=None)


def process_documents_extensions(docs_tuples, nlp):
  processed_docs = {}
  for doc_id, text, category in docs_tuples:
    doc = nlp(text)
    doc._.doc_id = doc_id
    doc._.category = category
    processed_docs[doc_id] = doc
  return processed_docs

documents_with_categories = [
    ("doc_1", "this is a tech document about spacy.", "tech"),
    ("doc_2", "this is a business document about sales.", "business"),
    ("doc_3", "this is a short sports document", "sports")
]

processed_with_ext = process_documents_extensions(documents_with_categories, nlp)
for doc_id, doc in processed_with_ext.items():
  print(f"processing doc with id: {doc._.doc_id}, category: {doc._.category}")
  print(f"first token: {doc[0].text}")


```

in this code we are using `Doc.set_extension` to declare new attributes to the spacy doc object. we can access it using `doc._.doc_id` and `doc._.category` this is generally what spacy recommends. the code has no practical functional change, but is a good practice to have.

it is important to know that the order of operations is important, as setting the custom attribute should happen after the spacy processing but before you extract the data.

in essence, the trick is to not rely on implicit document tracking, which is non-existent, and create an explicit system to keep tabs on each text unit you are processing using the available mechanisms, like dictionaries, tuples or doc extensions, there are probably other mechanisms but i usually stick to the three ones provided here. remember you can always create more complex dictionaries to keep all types of metadata.

it's funny how simple it seems now, but i remember spending way too long trying to debug a script, only to realize my identifiers had gone missing in the pipeline. it was like a cosmic game of hide-and-seek but with data, and i was definitely not winning.

if you're interested in learning more, i'd recommend checking out the spacy documentation, which has good examples on how to use the methods we used. there's also a great book i can point you to, "natural language processing with python" by steven bird, ewan klein, and edward loper it’s a classic, and covers these concepts in more detail. and maybe read some papers on building robust nlp pipelines, that may help you understand some of the best practices out there.
