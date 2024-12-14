---
title: "How does a spacy matcher work?"
date: "2024-12-14"
id: "how-does-a-spacy-matcher-work"
---

alright, so you want to get into how spacy matchers tick, huh? i get it. it’s one of those things that seems simple on the surface but can get real deep real quick when you start needing it to do more complex stuff. i remember back in the day, probably around 2018, when i first started using spacy for a project involving extracting specific information from customer service chat logs. i thought i could just brute force it with regex, but man, that was a disaster. the patterns i was trying to create were becoming unreadable and unbelievably fragile. i'm talking about massive, complex, spaghetti regex, where one character change could break the entire thing. and that's when i stumbled upon spacy's matcher. it was like someone had thrown me a lifeline.

the basic idea behind a spacy matcher is pretty straightforward. you're basically defining a sequence of tokens, or "patterns," that you want to find within a text. instead of having to write elaborate regular expressions which are a pain in the ass to maintain, spacy's matcher allows you to specify these patterns using dictionaries. each dictionary represents a single token and defines the properties you're looking for. you can specify things like the token's text, its part-of-speech tag, its dependency label, or even custom attributes. think of it like a more structured and readable way of doing regex, tailored for natural language processing.

let's break down how it works under the hood a little more. spacy, at its core, tokenizes your text, turning it into a sequence of `token` objects. these token objects carry all this information about each individual word or punctuation mark in the sentence. when you create a matcher, you’re essentially saying: hey, spacy, look for sequences of tokens that match the patterns i’m about to give you, and when you find a match, let me know.

here is an example of how i would create a simple matcher to find any mention of “apple pie” in a text:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [{"LOWER": "apple"}, {"LOWER": "pie"}]
matcher.add("apple_pie", [pattern])

doc = nlp("i love apple pie, i eat it a lot!")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(f"found match: {matched_span}") # prints "found match: apple pie"

```

in this code snippet, we load the small english model, create a `matcher` instance from the nlp pipeline and define a pattern list. our pattern consists of two dictionaries, each representing a token, and specifying that we want tokens that are "apple" and "pie" in lowercase. when we apply our `matcher` to a `doc`, it returns all matches in the text, and in this case we just loop through the results printing the matched text `span`.

the beauty of the matcher comes in its versatility. you are not limited to just looking for exact text matches, you can use different keys in your pattern dictionaries to make it more flexible. for example, you can check for parts of speech tags or lemmas. this is really helpful if you want to find phrases with similar meanings or grammatical structures without having to enumerate all of its possible variations. it's also faster than regex in most cases as spacy already has parsed and annotated your text, the matcher doesn’t have to go back and re-analyze the text structure with patterns.

let’s say, i needed to extract all phrases that are of the form “a [adjective] [noun]”. i could accomplish that with the following matcher definition:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [
    {"ORTH": "a"},
    {"POS": "ADJ"},
    {"POS": "NOUN"}
]
matcher.add("adj_noun", [pattern])

doc = nlp("i saw a blue car and a big house")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(f"found match: {matched_span}") # prints "found match: a blue car" and "found match: a big house"

```

here we are saying, find a token with the text "a", followed by a token that’s an adjective and another token that’s a noun. the `ORTH` key checks for the exact text of the token, and `POS` key does a part-of-speech tag lookup, giving a level of abstraction that regex can’t easily achieve. that’s way better than trying to match it using regex, and it’s way easier to understand too, if you have worked with `spacy` before.

you can also introduce more complex logic in your pattern. things like quantifiers (e.g., zero or more, one or more), which is extremely useful for dealing with variations in text structure. the matcher also allows you to introduce constraints like token dependency parsing and custom attributes. you can add `OP` keys for these, like `?`, `*`, or `+` within a dictionary to look for optional, zero or more, or one or more of that token or group of tokens, respectively. this provides incredible flexibility in pattern definition, in contrast to regex where complex logic becomes very hard to read.

when i was dealing with those chat logs i mentioned earlier, i had some logs that involved numbers followed by measurement units and i needed to extract those values and units as a pair. it was a headache dealing with different types of unit names, abbreviations, and pluralisations. so i leveraged this functionality, and i remember having a pattern like this:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [
    {"IS_DIGIT": True},
    {"LOWER": {"IN": ["meter", "meters", "m", "mile", "miles", "km", "kilometer", "kilometers"]}}
]
matcher.add("distance_measurement", [pattern])


doc = nlp("the road is 100 miles long, i ran 5 km, and it's over 10000 meters.")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(f"found match: {matched_span}") # prints "found match: 100 miles" "found match: 5 km" and "found match: 10000 meters"

```

here, we are telling the matcher to find a digit and then an unit of length. notice, the `IN` operator for the `LOWER` key is performing a kind of "or" operation on multiple values. the matcher allows for these kinds of complex matching expressions. its a lot more expressive than doing the same thing with raw regex. it helped me alot with the chat log analysis, i can't stress that enough. and all of this is in part thanks to spacy's internal tokenization and parsing pipeline. it is important that you have a good quality parsing pipeline before using a matcher, or you'll be having a hard time.

i find myself using matchers daily. the important thing is to understand how your text is being tokenized by spacy and to think of the patterns you want to find in terms of the token’s features. once you wrap your head around this, its very simple and powerful. it is way easier to maintain and extend than more conventional regular expressions.

if you are looking for some resources to further your understanding of spacy, i recommend the book "natural language processing with python" by steven bird, ewan klein, and edward loper. it provides a comprehensive foundation in the area of nlp, and although it does not focus specifically on spacy, it will give you the background needed to understand why things work the way they do. also, you can check out the documentation of the `spacy` library; they have very extensive examples of matchers and other important features in the library. i'm pretty sure even a goldfish could understand it. well, maybe not.
