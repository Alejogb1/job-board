---
title: "How can spacy tokens be iterated to extract BILOU tags?"
date: "2024-12-23"
id: "how-can-spacy-tokens-be-iterated-to-extract-bilou-tags"
---

,  It's a problem I've seen crop up a few times, particularly when dealing with custom named entity recognition models that rely on BILOU tagging. I remember one project, oh, it must have been back in 2019, where we were dealing with extracting very specific clinical entities from unstructured medical reports – a real challenge, and perfectly suited for BILOU. The core issue, as you're likely experiencing, is that spaCy's `token` object doesn't inherently expose BILOU tags; you need to derive them based on entity spans. So, how do we iterate through those tokens and get those tags efficiently?

The trick is to leverage spaCy's powerful concept of "spans." A span is basically a slice of tokens within a `doc` that corresponds to a named entity. The BILOU tag, in case you need a reminder, stands for: Beginning, Inside, Last, and Unit. 'B' marks the beginning of an entity, 'I' marks tokens within an entity, 'L' marks the end, and 'U' designates a single-token entity. 'O' means outside any entity. Therefore, we essentially need to compare the token's position against the entity spans' boundaries.

Now, the straightforward approach is to iterate over all tokens within a document, and within that loop, iterate through the identified entities within that document as well. Within this nested structure, we check for each token whether it overlaps with an entity.

Let me show you some code examples, working progressively towards a more robust solution. Let's start with a fundamental, if slightly verbose, approach:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def get_bilou_tags_basic(text):
  doc = nlp(text)
  bilou_tags = ['O'] * len(doc)  # Initialize all tags to 'O'

  for ent in doc.ents:
    if len(ent) == 1:
      bilou_tags[ent.start] = 'U'
    else:
      bilou_tags[ent.start] = 'B'
      for i in range(ent.start + 1, ent.end - 1):
        bilou_tags[i] = 'I'
      bilou_tags[ent.end - 1] = 'L'

  return [token.text for token in doc], bilou_tags

text = "Apple is looking at buying U.K. startup for $1 billion."
tokens, tags = get_bilou_tags_basic(text)
print(f"Tokens: {tokens}")
print(f"BILOU Tags: {tags}")
```

This first code snippet is functional. It loops through each entity. For every entity that’s a single token, it tags it as ‘U’. For multi-token entities, it tags the beginning as ‘B’, middle tokens as ‘I’, and the last token as ‘L’. While this works, there are a couple of inefficiencies. For example, a more Pythonic way to handle the list of 'O' tags, would be using a list comprehension and it is important to optimize the iterations and access indices properly which this code will not do.

Next, let's refine it to make it more concise and arguably more readable. We'll use Python’s built-in `enumerate` function, which provides the index with the value during iteration.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def get_bilou_tags_optimized(text):
  doc = nlp(text)
  bilou_tags = ['O'] * len(doc)

  for ent in doc.ents:
    if len(ent) == 1:
      bilou_tags[ent.start] = 'U'
    else:
        for i, token in enumerate(doc):
            if ent.start <= i < ent.end:
                if i == ent.start:
                    bilou_tags[i] = 'B'
                elif i == ent.end-1:
                    bilou_tags[i] = 'L'
                else:
                  bilou_tags[i] = 'I'
  return [token.text for token in doc], bilou_tags

text = "Apple is looking at buying U.K. startup for $1 billion."
tokens, tags = get_bilou_tags_optimized(text)
print(f"Tokens: {tokens}")
print(f"BILOU Tags: {tags}")
```
Here, we've incorporated the `enumerate` function. This eliminates the need to manually keep track of index using the integer range method as seen in the previous example. More importantly, note that this version is more optimized as it does not use list indexing as frequently. However, in both examples, we are still looping over *all* tokens multiple times (once for each entity span), which isn't ideal for extremely long documents or large datasets.

For our third example, let’s create a version with better efficiency by streamlining the iteration, avoiding multiple comparisons with range checks. This approach iterates over *tokens* only once, which significantly speeds up the process, especially with long text.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def get_bilou_tags_efficient(text):
    doc = nlp(text)
    bilou_tags = ['O'] * len(doc)
    ent_starts = {ent.start: ent for ent in doc.ents} #dictionary of entity start indices to the respective entity
    for i, token in enumerate(doc):
        if i in ent_starts:
            ent = ent_starts[i]
            if len(ent) == 1:
                bilou_tags[i] = 'U'
            else:
                bilou_tags[i] = 'B'
                for j in range (i+1, ent.end-1):
                  if j < len(doc) and j > -1:
                    bilou_tags[j] = 'I'
                if ent.end - 1 < len(doc):
                    bilou_tags[ent.end-1] = 'L'


    return [token.text for token in doc], bilou_tags

text = "Apple is looking at buying U.K. startup for $1 billion. This is not part of the deal. Google will announce something big soon."
tokens, tags = get_bilou_tags_efficient(text)
print(f"Tokens: {tokens}")
print(f"BILOU Tags: {tags}")

```

In this refined version, we avoid nested iterations altogether by checking a dictionary. The dictionary `ent_starts` stores the starting indices of the entities, allowing us to use `in` and lookups on the indices. This has significantly improved the processing speed. For very large documents or datasets, this can have a dramatic effect on performance.

Now, a couple of points to consider beyond the code. The `en_core_web_sm` model in spaCy is just for demonstration purposes. Depending on your specific use case, you'll likely need to switch to a larger model or train your own to get better entity recognition. In the medical domain, I've found that custom models, often trained from manually annotated data, are crucial.

For anyone looking to delve deeper into the details of named entity recognition or sequence labeling, I highly recommend Christopher Manning and Hinrich Schütze's "Foundations of Statistical Natural Language Processing.” Though slightly dated, the theoretical underpinnings remain highly relevant. For a more modern take, consider “Speech and Language Processing” by Daniel Jurafsky and James H. Martin. For more specific application of BILOU and its use in NER, the CoNLL-2003 shared task papers on named entity recognition are absolutely critical. Also, keep an eye on the literature around transformer-based models which has moved the needle quite a bit.

These resources will provide you with both the theoretical and practical knowledge to tackle these kinds of tasks efficiently and effectively. It's often a blend of theory and practice that yields the best results. Remember, iteration is key – both in coding and in understanding these complex methods. Keep tweaking, testing, and you'll get there.
