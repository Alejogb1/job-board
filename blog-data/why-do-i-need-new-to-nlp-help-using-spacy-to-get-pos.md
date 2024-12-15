---
title: "Why do I need New to NLP help using spacy to get POS?"
date: "2024-12-15"
id: "why-do-i-need-new-to-nlp-help-using-spacy-to-get-pos"
---

so, you're asking about why you need `new` when working with spaCy and part-of-speech (pos) tagging, specifically. i get it, it's one of those things that can feel kinda opaque at first, but let's unpack it a bit, i've been there myself. trust me.

basically, when you're using spaCy, you're not just working with raw text strings. spaCy is designed to handle text as a sequence of tokens, each with its own set of attributes. this lets you do really powerful things, like pos tagging, dependency parsing, named entity recognition and more all efficiently. it doesn't do it magically, though.

the core issue here revolves around spaCy's processing pipeline. when you load a spaCy model, you're essentially loading a series of pre-trained components that perform different tasks, such as tokenization, pos tagging and parsing. if you have tried loading a model like `en_core_web_sm` or similar, you have already been interacting with this machinery. spaCy models aren't just glorified dictionaries; they're complex statistical models trained on vast amounts of text data. they learn patterns in language that enable them to make informed predictions about the structure and meaning of new texts.

now, the `new` keyword, or at least the concept of creating a new `doc` object, is crucial because you need to actually *apply* these models to the text that you're working with. spaCy does not automatically parse or apply processing to raw text strings, that wouldn't make much sense. a `doc` object is spaCy's container for holding the output of this processing pipeline.

let me give you an analogy. think about a factory assembly line, you wouldn't just place the raw material randomly on the floor. you put it on the conveyor belt where the different machines do their part to transform it. spaCy pipeline is that conveyor belt and your raw text string is the raw material. the `doc` object is that finished product, and only when you have this finished product, a `doc` object, can you start to inspect its parts, in your case, the pos tags.

so, when you have your text loaded and you need pos tagging (or other operations), you feed your text into the spacy pipeline, creating a `doc` object. that's what i mean by `new`. in code it looks like `doc = nlp(your_text)`.

let’s say you’re trying to get pos tags from a string, something like this:

```python
import spacy

# let's pretend this is my model i loaded from somewhere
nlp = spacy.load("en_core_web_sm")

text = "the quick brown fox jumps over the lazy dog"
```

if you stop there, you just have a string called `text`. you haven’t actually asked spacy to *do* anything with it. if you were to start asking for pos information on `text` it would be like asking your washing machine what it thinks about quantum physics, it just wouldn’t make sense. the key is to create the `doc` object:

```python
doc = nlp(text)

for token in doc:
    print(token.text, token.pos_)
```

now, the crucial part, `doc = nlp(text)`, is what triggers spacy’s pipeline. it’s what goes and uses all the machinery of the model to process the text. each token is no longer just part of a string, it's a `token` object with a `pos_` attribute. that `pos_` attribute is spacy’s output of its pos tagging.

i remember this one project i was working on, it was ages ago, where i made the mistake of trying to just iterate directly through a string and expect to get pos tags. that's how i really learned this the hard way. i was looking at the spacy documentation for what felt like hours, thinking there was some kind of bug, and wondering why the pos tags were not showing up. it turned out i was just missing the whole `doc` object part. it's funny how often this happens when you're coding, isn't it? it always ends up being a tiny detail that completely throws you off.

ok, let’s say you have a more complex situation, for example, a text with multiple sentences:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "this is the first sentence. this is the second sentence. and this is the third."

doc = nlp(text)

for sentence in doc.sents:
    print(sentence)
    for token in sentence:
        print(f'\t{token.text}, {token.pos_}')
```

notice how the spacy `doc` object is what enabled to split the sentences in the text and then look into each token of each sentence. now we are using another powerful tool that is enabled by the `doc`.

one more example and perhaps a useful one for you, let’s say you have a text file and want to load it and analyze each line. i mean, sure, you could load the file into a variable but you wouldn’t be using spacy pipeline so again you wouldn’t have any of the advantages of spacy, it would be just a plain string again. here is a more realistic approach:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# pretend that text.txt is a file in your folder with 3 sentences, one on each line
with open("text.txt", "r") as file:
    for line in file:
        doc = nlp(line.strip())
        for token in doc:
            print(f'{token.text}, {token.pos_}')
```

this is a very common approach, it takes a lot of code for just a few things but it is a very good example to show why you need that new `doc` object. the pipeline of spacy can’t do anything if you don’t have a `doc` object.

so, to reiterate, the "need" for a `new` document (or in practice the `doc` object) is rooted in the way that spacy is architected: you need to explicitly run your text through spacy's pipeline to create a structured, tokenized, and annotated document (the `doc` object) that contains all of the information you require, pos tags included. you cannot ask for pos tags from a raw string, that’s it. it needs to be a `doc` object. that’s how spacy works.

if you want to really understand more of these design decisions, i'd recommend delving into some resources like the paper “natural language understanding with distributed representations” by andrea zandona, it will give you a solid background on nlp concepts and how tools like spacy are implemented. another resource that i found useful are the books from “speech and language processing” by jurafsky and martin. they go way more deeper into specific subjects like pos tagging. it will definitely make your spacy understanding way better.

i hope this is helpful. it's not an uncommon point of confusion, so don't worry about it. good luck with your nlp adventures.
