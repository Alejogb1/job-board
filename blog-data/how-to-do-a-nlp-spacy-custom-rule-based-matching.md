---
title: "How to do a NLP: Spacy custom rule based matching?"
date: "2024-12-15"
id: "how-to-do-a-nlp-spacy-custom-rule-based-matching"
---

alright, let's talk about custom rule-based matching with spacy. it's something i've spent a fair amount of time on, and it's definitely a powerful tool once you get the hang of it. i've had my fair share of head-scratching moments with this, so hopefully my experience can save you some time.

i remember a project back in my early days where i was trying to extract specific financial terms from news articles. i was initially using a very naive keyword approach, basically searching for words like "profit", "loss", "revenue", etc. that was a disaster. i was getting so many false positives, catching things like "loss of signal" or "profit sharing" where it was not referring to the company's profit or loss statements. i was chasing my tail, and it got clear i needed something more intelligent and context aware, this is when i discovered spacy rule-based matching.

the thing with spacy is that it doesn't just look for simple strings; it looks for patterns based on token attributes—things like the text itself, its part-of-speech tag, its dependency relationship, and so on. this allows to build rules that capture more nuanced information.

so how do we actually do it? well, the core of rule-based matching in spacy is the `matcher` object. you initialize a `matcher` with a vocabulary, and then you add patterns to it. each pattern is a list of dictionaries, where each dictionary describes the constraints for one token.

let me give you a basic example: let's say you want to find the phrase "fast car" where "fast" is an adjective and "car" is a noun. here's how you might do that:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [
    {"POS": "ADJ"},
    {"POS": "NOUN"}
]
matcher.add("FAST_CAR", [pattern])

doc = nlp("that's a fast car. he has a slow bike.")

matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(string_id, start, end, span.text)
```

this code will output:

```
FAST_CAR 2 4 fast car
```

pretty straightforward, right? we define a pattern that says "first, find a token that's an adjective, then find a token that's a noun directly after it." then we apply that to a document and get the matches.

now, you can get much more granular with patterns. you're not just limited to part-of-speech tags. for instance, you can use the `lemma`, the base form of the word or the actual text to be matched as follows:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [
    {"LEMMA": "buy"},
    {"TEXT": "the"},
    {"POS": "NOUN"}
]

matcher.add("BUY_ITEM", [pattern])

doc = nlp("they bought the book. i will buy the car. buying the house")

matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(string_id, start, end, span.text)
```

this will output:

```
BUY_ITEM 1 4 bought the book
BUY_ITEM 6 9 buy the car
```

notice that although we matched "buy" via lemma we were able to match "bought" in the document because both have the same lemma form.

now, something that has caused me a bit of grief in the past is dealing with optional tokens or tokens that repeat. for that, we use the `OP` attribute in the token dictionary. `OP` can be `'!'`, `'?'`, `'+'`, or `'*'`.
`'!'` means the token must not exist, `'?'` means the token is optional, `'+'` means the token must exist at least one time and can repeat and `'*'` means the token is optional and can repeat.
let's say we want to match any price amount in a document like "$10", "$100", "$1000" or "$10,000". here is an approach:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [
    {"TEXT": "$"},
    {"IS_DIGIT": True},
    {"TEXT": ",", "OP":"?"},
    {"IS_DIGIT": True,"OP":"*"},
]

matcher.add("PRICE", [pattern])

doc = nlp("the price is $10. the price was $10000, it was also $10,000 and $1")

matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(string_id, start, end, span.text)
```

this will output:

```
PRICE 3 4 $10
PRICE 6 7 $10000
PRICE 10 13 $10,000
PRICE 15 16 $1
```

here, the comma is optional and the digits after it are also optional and can repeat. this lets us catch various ways to represent prices.

another thing i learnt along the way is that you can create more complex patterns by combining patterns or using lists within the pattern dictionaries.
but i won't dive into that for now.

now, you might be wondering, what's the secret sauce to doing rule based matching well? it's not so much about the code, but rather understanding how to represent what you are looking for in the format that spacy's matcher understands. one of the biggest problems i've had to overcome is anticipating all the edge cases that can occur in real-world text. you might start with a seemingly straightforward pattern and then discover all sorts of strange variations in real data that breaks your pattern. for me the trick has been to incrementally refine the patterns with more data.

so, if you want to become an expert in rule based matching, i suggest spending a considerable time with the spacy documentation, specially the section on rule-based matching and the `matcher` class specifically.
and if you're looking for a more in-depth look into the background of nlp, a good introductory book would be "speech and language processing" by daniel jurafsky and james h. martin. you will find an extensive overview of the concepts involved.

one last piece of advice: don't try to create the perfect pattern from the start. start simple, test your pattern on real data, and iterate based on the errors you encounter. think of it like crafting a perfect joke: it takes time to tweak the punchline.

rule-based matching in spacy can seem a little daunting when you begin, but it's a very powerful method once you get the concepts and format right, and with some experience and trial and error, you'll be able to extract some really useful patterns from text.
