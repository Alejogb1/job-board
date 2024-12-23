---
title: "How can I integrate a YIELD generator into a spaCy pipeline using `add_pipe`?"
date: "2024-12-23"
id: "how-can-i-integrate-a-yield-generator-into-a-spacy-pipeline-using-addpipe"
---

,  Integrating a generator function into a spaCy pipeline via `add_pipe` isn't immediately obvious, and I recall quite a few situations where I initially stumbled with this myself. It primarily stems from the way spaCy pipelines are designed to work with `Doc` objects, and generators operate on potentially different data structures. The key is understanding how to bridge that gap. I’ve spent considerable time troubleshooting this, particularly when handling large document collections and wanting to stream the processing rather than loading everything into memory. The benefits in those scenarios are tremendous, specifically when we're talking about resource constraints.

The core issue comes down to the fact that `add_pipe` expects a callable that accepts a `Doc` object as input and returns a modified `Doc` object, or simply returns `None` if its function is to merely modify the doc in place without generating a new one. Generators, on the other hand, generally yield individual items, not necessarily full `Doc` objects. So, we need to build an intermediary that can take the output from our generator and package it as a series of `Doc` objects that spaCy can understand.

The crucial part involves constructing what I call a "generator wrapper." This wrapper consumes the elements yielded by your generator, and converts them into usable `Doc` objects. Then, it can be used with a component that you then add to the spaCy pipeline.

Let's get specific. I'll provide three different examples, each slightly nuanced, to illustrate how you might approach this:

**Example 1: Simple Text Processing Generator**

Imagine you have a text file where each line represents a document, and you want to tokenize these documents without loading the whole file into memory. The generator would read the file line by line. Our pipeline component needs to transform these lines into `Doc` objects.

```python
import spacy

def text_line_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip() # Removing any extra whitespace

def create_doc_component(nlp):
    def component(docs):
        for doc_text in docs:
            doc = nlp(doc_text)
            yield doc
    return component

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    file_path = "my_text_data.txt"

    # Assuming my_text_data.txt exists
    with open(file_path, "w") as f:
        f.write("This is the first document.\n")
        f.write("Here is the second one.\n")

    doc_generator = text_line_generator(file_path)
    component_fn = create_doc_component(nlp)
    
    nlp.add_pipe(component_fn, name='custom_doc_generator', last = True)

    for doc in nlp.pipe(doc_generator):
        print([token.text for token in doc]) # Show tokens

    # Cleanup for example purposes
    import os
    os.remove(file_path)
```
Here, `text_line_generator` is our generator. `create_doc_component` creates a function that consumes the string yield, creates a `Doc` and yields the `Doc`. We then create the pipeline component with this function and add it using `add_pipe`.  This approach shows how to turn any string generator into a source for spaCy.

**Example 2: Generator Yielding Pre-Tokenized Text**

Now, let’s assume that your generator doesn’t yield raw text but rather pre-tokenized data. Perhaps you have text in JSON format, or have applied some preprocessing already.  In this case, you need to utilize `nlp.tokenizer.tokens_from_list`.

```python
import spacy

def pre_tokenized_generator():
    data = [
        (["This", "is", "a", "pre-tokenized", "sentence", "."], ["NOUN", "VERB", "DET", "ADJ", "NOUN", "PUNCT"]),
        (["Another", "one", "here", "."], ["DET", "NUM", "ADV", "PUNCT"])
    ]
    for tokens, tags in data:
        yield tokens, tags

def create_doc_pretokenized_component(nlp):
    def component(docs):
         for tokens,tags in docs:
             doc = nlp.tokenizer.tokens_from_list(tokens)
             for i, token in enumerate(doc):
                 token.tag_ = tags[i]
             yield doc

    return component
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner']) # Disable default components
    token_tag_gen = pre_tokenized_generator()
    component_fn = create_doc_pretokenized_component(nlp)

    nlp.add_pipe(component_fn, name='pretokenized_component', last=True)

    for doc in nlp.pipe(token_tag_gen):
      for token in doc:
         print(f"{token.text} {token.tag_}")
```

In this example, the generator yields tuples of tokens and tags. In the component, I specifically use `nlp.tokenizer.tokens_from_list`, as well as manually setting the part-of-speech tags. This is crucial because we are not relying on spaCy's default tokenization or tagging process. I disable the 'parser' and 'ner' components since they are not needed for this specific use case and may result in unexpected errors from pre-tokenized input.

**Example 3: Generator Yielding Dictionaries with Metadata**

Finally, let's consider a scenario where your generator provides not only the text but also some metadata along with it, such as document ids, categories, or timestamps, and you'd like to include this in the `Doc` object via custom extensions.

```python
import spacy
from spacy.tokens import Doc

def metadata_generator():
    data = [
        {"id": 1, "category": "news", "text": "Breaking news today."},
        {"id": 2, "category": "opinion", "text": "This is my take on the matter."}
    ]
    for item in data:
        yield item

def create_doc_metadata_component(nlp):
    Doc.set_extension("doc_id", default=None)
    Doc.set_extension("category", default=None)

    def component(docs):
        for data in docs:
            doc = nlp(data['text'])
            doc._.doc_id = data['id']
            doc._.category = data['category']
            yield doc
    return component

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    meta_gen = metadata_generator()
    component_fn = create_doc_metadata_component(nlp)

    nlp.add_pipe(component_fn, name='metadata_component', last=True)

    for doc in nlp.pipe(meta_gen):
        print(f"Doc ID: {doc._.doc_id}, Category: {doc._.category}, Text: {[token.text for token in doc]}")
```

Here, we are yielding a dictionary containing the text along with other information. I then define two custom extensions to `Doc`, "doc_id" and "category", and set them within the pipeline component. This shows how a generator can provide more than just the raw text for processing. The additional data is then stored directly on the `Doc` object.

These three examples demonstrate various strategies for wrapping your generators. The common theme is that you have to create an intermediary component that iterates through the yielded objects and convert those objects into spaCy `Doc` objects.

For more in-depth theoretical understanding of pipelining in NLP systems, I strongly recommend looking into the original publications by spaCy’s core team. Specifically, the work by Matthew Honnibal and Ines Montani on efficient pipelining and the design philosophy behind spaCy's architecture is invaluable. Consider also the work by Yoav Goldberg on neural networks for NLP, particularly regarding the efficiency of text processing frameworks. The official spaCy documentation is a great resource, obviously, but diving deeper into the theoretical underpinnings will clarify *why* certain approaches work and provide a more robust understanding.

One thing to be cautious about, especially with more complex generators, is to make sure your generator logic doesn't have hidden side effects, like inadvertently changing global state. This might not always cause an error, but it can lead to subtle bugs. Careful testing is paramount here.

Remember, the precise implementation will vary according to your generator's specific output and processing needs. The core concept remains the same: bridging the gap between your generator and spaCy’s expected `Doc` input. I’ve often found that clearly defining the input and output types of your generator, as well as the exact steps of transformation needed, often provides a clearer path to a solution. It might seem slightly more verbose initially, but it certainly leads to more robust and understandable code.
