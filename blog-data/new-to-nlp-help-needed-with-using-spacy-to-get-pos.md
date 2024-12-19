---
title: "New to NLP help needed with using spacy to get POS?"
date: "2024-12-15"
id: "new-to-nlp-help-needed-with-using-spacy-to-get-pos"
---

alright, so you're diving into nlp and want to grab part-of-speech tags using spacy, right? been there, done that, got the t-shirt (and a few scars from debugging unicode issues). it's a fairly common starting point, and spacy is definitely a solid tool for this. i remember when i first touched nlp stuff back in the day. i was trying to build a rudimentary chatbot for my college project using nlp, and honestly, getting the basics of pos tagging correct was surprisingly tricky. i spent an entire weekend just staring at error messages and feeling very, very confused until i slowly got the hang of it with the help of some online guides (before stackoverflow was a thing, yeah i'm *that* old).

anyway, let's get down to business. pos tagging, if you're unfamiliar, is basically the process of marking each word in a sentence with its grammatical role. nouns, verbs, adjectives, adverbs, you get the idea. spacy makes this quite easy, but there are always a few things that can trip you up.

first, you'll need spacy installed. if you haven't already, open up your terminal or command prompt and do this:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

that second command downloads the english language model. spacy models are essentially pre-trained neural networks that have learned all sorts of linguistic tricks. `en_core_web_sm` is a small model, which is usually good enough for simple tasks like pos tagging. if you're working with more specific text domain like medical text, a larger model may be better but will cost more in computational time. we'll start with the basics though.

once you have spacy installed and the model downloaded, let’s look at some code. here is a basic snippet to get you going:

```python
import spacy

# load the model
nlp = spacy.load("en_core_web_sm")

# example text
text = "the quick brown fox jumps over the lazy dog."

# process the text
doc = nlp(text)

# loop through the tokens and print the text and the pos
for token in doc:
    print(f"{token.text}: {token.pos_}")
```

this should output each word with its pos tag. notice how we access the pos tag using `token.pos_`. this underscore `_` is used because without it you would get an integer code for the pos tag, and `pos_` gives the string description of the tag. spacy also has finer grained tags that are accessible with `token.tag_` if you want more detail, but that's usually not needed if all you want is to get the core pos.

i remember one time, i had this weird issue with some french text i was working on and for some reason, spacy was not picking up the articles and prepositions properly. i had installed all language modules correctly, i made sure of that. after lots of debugging it turned out i had loaded a wrong model or model versions were incompatible. it's usually some silly detail like that that causes issues. it is also important to note that if the `_` is omitted you get an integer code for the tag. something to keep in mind while debugging for the future.

now, about handling more complex sentences. sometimes sentences will have ambiguity. like the word 'bank' can be a noun (the financial institution) or a verb (to tilt). the model is going to try to select the one based on the context.

let’s see another example to demonstrate this:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text_list = [
  "i saw her duck.",
  "i saw her duck in the river.",
  "i will go to the bank to deposit the money.",
  "i will bank on that happening.",
  "the man was wearing a suit"
  ]

for text in text_list:
  doc = nlp(text)
  print(f"\ntext: {text}")
  for token in doc:
        print(f"  {token.text}: {token.pos_}")
```

here, you will notice that ‘duck’ is tagged as a noun and a verb based on the context. also 'bank' is tagged as a noun or verb depending on the context it is used. this is pretty cool, and it's where the power of these pre-trained models really shows.

something else you might run into is punctuation. sometimes you don't want to deal with punctuation tokens. here is how you can filter them out:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "this is a sentence, with some punctuation! and some more..."

doc = nlp(text)

print(f"text: {text}")
for token in doc:
    if not token.is_punct:
        print(f" {token.text}: {token.pos_}")
```

we use `token.is_punct` to check if a token is punctuation. that will help you filter them out. we should also note other methods available like `token.is_alpha` to check if the token is alphabetical and `token.is_digit` to check if it's a number. this can help you preprocess the text if needed, before you do any work on the pos tags. spacy is very fast and efficient in handling that.

the code snippets should get you started with the core concepts. but, to *really* understand nlp and pos tagging, you should also check out some material in literature. instead of specific links i would recommend for example:

*   "speech and language processing" by daniel jurafsky and james h. martin: this is kind of the standard book on nlp, and the sections on pos tagging go into a lot of depth. it gets very technical, but that is very helpful for any person trying to understand better the theory behind the tools.

*   "natural language processing with python" by steven bird, ewan klein, and edward loper: this book is very practical, and it uses nltk, which is an alternative to spacy, but the core concepts are the same. understanding the basics of nlp is helpful no matter what library you end up using. in the long run, understanding nltk will complement your spacy knowledge.

*   research papers: many research papers are always being released in the nlp field. try searching google scholar for pos tagging. that should give you a vast array of technical papers explaining new approaches to pos tagging. this is important to keep up to date with the latest progress in the field.

also, before i forget, there was this one time where i was debugging a spacy based pipeline and it kept telling me that 'the' was an adverb, i thought i was losing my mind, turns out i was using a pre-trained model for a different language (it was portuguese). the joys of coding, eh?

anyway, i hope that this helps. pos tagging is one of the fundamental steps in many nlp tasks so getting a good handle on it early will pay off. it may seem frustrating at first, but it will start making sense after a while. keep experimenting and remember the pos tags are not always perfect, no model is. but they are very helpful and you will get better with time. good luck.
