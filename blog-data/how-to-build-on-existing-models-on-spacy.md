---
title: "How to build on existing models on spacy?"
date: "2024-12-15"
id: "how-to-build-on-existing-models-on-spacy"
---

so, you want to build on existing spacy models right? i get it. been there, done that, got the t-shirt – probably a dozen of them actually. i've spent a good chunk of my career elbow deep in natural language processing, and spacy is a tool i keep coming back to. it's just so clean and flexible, once you understand its quirks. and trust me, it has them.

let's talk about fine-tuning, because that’s usually what people mean when they say "build on". you've got a pre-trained spacy model, probably one of their `en_core_web_*` variants, maybe something else you found, and you need it to do something more specific. the stock models are great for general language tasks, but if you're dealing with, say, very specific terminology in the medical field or social media slang, well, they need a bit of a nudge in the right direction.

my first encounter with this was back in the early 2010s. i was working on a project that involved analyzing technical documents for a telecom company. the spacy models of that time weren’t exactly fluent in jargon like ‘multiplexing’ or ‘qam modulation’. i spent days getting frustrated, trying to get them to correctly identify those terms, to not treat them like random words in a sentence. it was a mess. eventually i had to look in a completely different direction and start thinking about transfer learning with spacy. not only did it work, but it was so much faster than what i was trying before.

so, what are the basic steps? first, you'll need data. and not just any data – labeled data, the type where you’ve manually gone through and annotated the text with what you want the model to learn. this is often the biggest hurdle, but good data, clean and with consistent labels, is the absolute foundation for any successful fine-tuning endeavor. think of it like this: you wouldn't expect a student to pass an exam without giving them material to study, would you?

spacy makes this relatively straightforward via its `spacy.training.example` object which can wrap up texts and their labels. you'll basically create instances of this for all of your training data. the labels are usually in the form of lists of tuples, indicating the character span of an entity, along with its type.

here’s a quick example of how you might structure your training data:

```python
import spacy
from spacy.training import Example

nlp = spacy.load("en_core_web_sm") # or any other model you choose

training_data = [
    (
        "the quick brown fox jumped over the lazy dog.",
        {"entities": [(0, 3, "ANIMAL"), (10, 13, "ANIMAL"), (26, 29, "ANIMAL")]},
    ),
    (
        "a cat sits on the mat.",
        {"entities": [(2, 5, "ANIMAL")]},
    )
     (
        "this is a text with some random words",
        {"entities": []}
    )

]

examples = []
for text, annotations in training_data:
   examples.append(Example.from_dict(nlp.make_doc(text), annotations))


```

this creates a set of examples based on some training data about animals. the `examples` variable now holds the list ready to be used to update the model.

now that you have your training data, it’s time to update the model. and this is where it gets a little more involved. you need to disable the parts of the spacy pipeline that you’re not training, to prevent them from being unnecessarily affected during the fine-tuning process. for example, if you are only looking to improve the entity recognition, then you disable everything else like the part-of-speech tagger. you’ll use the `nlp.disable_pipes()` context manager for this.

here’s an example using the example data from above, showing how to perform the training loop with the entity recognizer, also known as the `ner` component.

```python
import random
import spacy
from spacy.training import Example

nlp = spacy.load("en_core_web_sm") #or use a different model


training_data = [
    (
        "the quick brown fox jumped over the lazy dog.",
        {"entities": [(0, 3, "ANIMAL"), (10, 13, "ANIMAL"), (26, 29, "ANIMAL")]},
    ),
    (
        "a cat sits on the mat.",
        {"entities": [(2, 5, "ANIMAL")]},
    ),
    (
         "this is a text with some random words",
        {"entities": []}
    )
]

examples = []
for text, annotations in training_data:
   examples.append(Example.from_dict(nlp.make_doc(text), annotations))

# disable other pipes
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.create_optimizer()
    for i in range(100): # training iterations
        losses = {}
        random.shuffle(examples)
        for example in examples:
            nlp.update([example], losses=losses, sgd=optimizer)
        if i % 10 == 0:
            print(f"Losses at iteration {i}: {losses}")
```

this code snippet disables every pipeline component except the named entity recognizer, creates an optimizer, runs for 100 iterations through the data, updating the named entity component weights every time, and printing the losses every 10 iterations. you usually need a lot more data and iterations to get a proper model, but this illustrates the basic idea.

after you run this code, your `nlp` object has now been changed with the new entity annotations. notice that the code is only retraining the ner and leaving the rest as is.

also, the loss value is an important part of the training. it’s a metric showing the difference between the prediction of the model, and the real value. it’s good to see that this loss goes down with each iteration. but if it goes down very fast and stays low, you're probably overfitting to your small data.

another important point: it's often better to use `spacy train` which uses a config file, rather than a pure python code like this, because it allows to have all the necessary configuration and keep track of the results. this function takes care of the whole training routine and can load the configuration from a file, which usually includes not only the optimizer but other parameters related to the model that you will need in more complex scenarios. it also offers more robust control for larger datasets.

finally, once you’ve trained your model, you probably want to save it and use it elsewhere. to do this, you use the `nlp.to_disk()` method, and you can reload the model using `spacy.load()` as usual.

```python
nlp.to_disk("custom_ner_model")

loaded_nlp = spacy.load("custom_ner_model")

text_to_test = "the big black cat was very sleepy"
doc = loaded_nlp(text_to_test)

for ent in doc.ents:
   print(ent.text, ent.label_)

```
this will save the model under the folder “custom_ner_model” and then later load it to use it on new text, and printing the found entities and their labels, which should show the model predicting the cat as an animal.

now, as for resources, i'd recommend going through the official spacy documentation, as that's always your first port of call. they have detailed explanations about how the pipeline works, how to create training data, and how to train the models. also, if you really need to understand the underlying principles of these model architectures, i'd suggest taking a look at the original papers where the models are explained. something like "attention is all you need" if you want to understand how the transformers works inside the models, or perhaps "efficient estimation of word representations in vector space" if you want to understand how the word vectors are trained.

a general book recommendation would be "speech and language processing" by daniel jurafsky and james h. martin, which offers an extensive overview of natural language processing and language models, but be warned that it's a heavy read and it covers more than just what you will need for working with spacy. also, looking into blogs and tutorials online also help, but make sure you are getting the information from reputable sources. there are many resources available, so take your time to sift through them to find the ones that match your experience and what you want to do.

there was a time i thought i would never understand this, but practice and repetition made me get it in the end. fine-tuning models is something that needs dedication, because if you overtrain it, your model is going to be useless. so be sure to spend time getting to know your data, your problem, and the specifics of spacy. oh and one tip, don't be afraid to experiment and fail, it’s part of the learning process, after all, that's how i learned most things in my career. someone told me that my bugs are so complex, that they require their own bugs, and it is very funny to me. but you got this!
