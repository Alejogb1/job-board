---
title: "How to do Named Entity Recognition Model Training Data Creation?"
date: "2024-12-15"
id: "how-to-do-named-entity-recognition-model-training-data-creation"
---

well, alright, so you're looking at the classic ner data prep situation, yeah? i've been there, trust me. countless times. it's like the unsung hero of any ner project. you can have the fanciest model architecture, a gpu farm humming away, but if your training data is a mess, it's all going to fall apart. i'm talking from experience, here. i remember back in my early days, i thought i could get away with just tossing some semi-random tagged text at a model. boy, was i wrong. the results were, shall we say, less than optimal. think, like, entity types being assigned to the wrong things, even to random parts of words. it was… painful.

so, how do you actually do it? well, it’s not exactly rocket science but it needs structure and a systematic approach. at its heart, it's about teaching the model what to look for and where, and also, importantly, what *not* to look for. the main idea is to create a dataset made of text and annotations that accurately reflects the real-world text your model will eventually see. you can't just feed it perfect, textbook-style examples and then expect it to handle the chaos of the real world.

the process, at the high level, looks like this: first, you need the source text. that could be anything: news articles, social media posts, medical reports, legal documents, product reviews, whatever domain you are targeting. next, you annotate the text. by this i mean that you have to identify and label the named entities. these are things like names of people, organizations, locations, dates, products, and so on. what entities you need depends on what information the model needs to extract. and, finally, you validate the annotated data, so it is correct and consistent. and, yes, this process is, for the most part, manual. sorry. there isn't a magic wand here, yet, at least, but we have tools that can make your work less painful.

now, let's get into the weeds a bit with a few code-like examples for illustration. imagine you're working with a set of movie reviews, and you want to identify the names of the actors. the classic format for annotation would be the iob (inside, outside, beginning) scheme. this scheme labels each word with one of three tags: 'b-' followed by the entity type for the first token of the entity, 'i-' followed by the entity type for all other tokens of the entity, and 'o' for any token that is not part of an entity.

```python
# sample iob format
sentence = "john doe and jane smith starred in the movie inception"
iob_tags = [
    "b-person", # john
    "i-person", # doe
    "o", # and
    "b-person", # jane
    "i-person", # smith
    "o", # starred
    "o", # in
    "o", # the
    "o", # movie
    "o", # inception
]

# a more complex example with different entity types
sentence2 = "apple acquired beats electronics on may 28, 2014, for $3 billion."
iob_tags2 = [
    "b-org", # apple
    "o", # acquired
    "b-org", # beats
    "i-org", # electronics
    "o", # on
    "b-date", # may
    "i-date", # 28
    "o", #,
    "i-date", # 2014
    "o", #,
    "o", # for
    "b-money", # $3
    "i-money", # billion
    "o", #.
]
```

this is pretty straightforward for simple sentences, but it gets a lot more difficult when you deal with real-world, complex sentences. that’s why consistency is key. if you’re annotating the person 'john doe', you always need to label 'john' as 'b-person' and 'doe' as 'i-person'. if you change that pattern mid-annotation, your model will be confused, and if your model is confused, the predictions are going to be confused as well. like, if you sometimes label 'john' as 'b-person' and other times 'i-person', the model won't be able to learn the pattern. it’s all about being consistent and not letting chaos creep into your data. it’s like cooking, right? if you suddenly swap ingredients mid-recipe, things are not going to end up as expected.

the next important thing is the tools you use. you’ve probably heard about different annotation platforms. i've used some in the past, from simple custom scripts i made in my early days that were incredibly inefficient to the cloud-based solutions. the advantage of the cloud solutions is, that they offer a visual interface, which makes the whole process more intuitive. they also often have collaboration features. this can be really useful when you are working in teams. some even have built-in quality control tools to spot inconsistencies in annotations that helps a lot too.

here’s a simple example of how you might represent this annotated data in a python dictionary:

```python
data = [
  {
    "text": "john doe and jane smith starred in the movie inception",
    "entities": [
      {"start": 0, "end": 8, "label": "person"},
      {"start": 13, "end": 23, "label": "person"}
    ]
  },
  {
    "text": "apple acquired beats electronics on may 28, 2014, for $3 billion.",
     "entities": [
        {"start": 0, "end": 5, "label": "org"},
        {"start": 15, "end": 31, "label": "org"},
        {"start": 35, "end": 46, "label": "date"},
        {"start": 52, "end": 62, "label": "money"}
    ]
  }
]
```

in this format, the 'text' key stores the actual text, and the 'entities' key contains a list of dictionaries for each entity. each of these entity dictionaries has a ‘start’ key for the starting position of the entity, an ‘end’ key for the end position, and a ‘label’ key for the entity type.

there is one more important point. don’t underestimate the power of having a good annotation guideline. this is something i didn't really understand early on, but now i swear by it. it’s a document that explicitly states what each entity should be, how it should be tagged, and rules for tricky cases, also known as edge cases. for example, if you have a case where the name of a company and the product share the same name, how do you handle that?. these guidelines must be precise and comprehensive so the annotators work consistently with the same definitions and this will ultimately result in better quality data.

one thing i learned from the trenches is that data is iterative. you do your first annotation. train the first model. and it starts getting things wrong. you then have to go back, review the errors, update your annotations, and then train again. it's a cycle. the whole process of data creation and model training can become something of a dance. this is very common and everyone goes through that, so don't beat yourself up about it.

also, please try to stay away from very unbalanced datasets. for example, if you have 90% of examples labeled as person entities and only 10% labeled as organisation entities, this will impact your model’s performance on the latter entity type. try to have representation of all the entity types so your model learns them properly. if you have 200 examples for one entity and only 5 for other, this will create some problems. this is why having an annotation guideline is very important, it forces you to think about all the different cases of entities beforehand.

as for resources, i wouldn't recommend just blindly grabbing stuff off the web. rather, look for academic papers and books. for a deep dive into ner, 'natural language processing with python' by bird, klein and loper is a solid start, it covers the fundamentals. and for a broader understanding of machine learning, 'pattern recognition and machine learning' by christopher bishop remains a classic. if you really want to know the state-of-the-art in data annotation for ner i would recommend trying to find papers on 'active learning' applied to annotation, this is one of the areas that is constantly progressing nowadays. you can check google scholar and search for those keywords.

in conclusion, creating quality data for ner models is a non-trivial task. it's time-consuming and requires careful planning and execution. but with the proper tools, guidelines, and a little experience, you’ll have more success. also, don't worry, everyone makes a mistake or two. i once forgot to change the entity type in one annotation and labelled a bunch of movie titles as locations. my model started to predict theaters showing up in random cities all over the world. that was a fun one.
