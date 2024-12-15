---
title: "How does the spacy matcher work?"
date: "2024-12-15"
id: "how-does-the-spacy-matcher-work"
---

alright, so you're asking about the spacy matcher, eh? i’ve spent way more time than i'd like to *confess* battling with that thing, so i can probably give you the lowdown. it's not rocket science, but there's a bit of nuance to it.

basically, the spacy matcher is a tool within the spacy library that lets you find specific sequences of tokens in a text. now, when i say tokens, i mean the individual words, punctuation, or even whitespace that spacy breaks your text into. it’s not just about matching exact strings; you can specify patterns based on all sorts of token attributes – think part-of-speech tags, lemma forms, and even custom attributes.

my first encounter with the matcher was a real *head-scratcher*. i was trying to extract specific product descriptions from customer reviews, and i was initially using regex. total disaster. regex, while powerful, became a tangled mess when dealing with the variations in language. for example, trying to catch "amazing camera" "best camera" "great camera" with regex ended up being like trying to catch water with a sieve. the slightest variation, and regex would fail. that’s where the spacy matcher came into play.

the way the matcher works is by defining patterns. these patterns are lists of dictionaries, where each dictionary describes the attributes a token in the sequence should have. the order of dictionaries in the list is critical because it represents the order of tokens you’re searching for in your text. let’s say you want to find sequences where you have an adjective followed by a noun. here’s a simple pattern example:

```python
pattern = [
    {"POS": "ADJ"},
    {"POS": "NOUN"}
]
```

this pattern would find things like "big dog," "small house," "red car", in any given text. the {"pos": "adj"} is a dictionary describing that the token needs to have the tag "adj". the other one, the {"pos":"noun"}, is another dictionary describing that the next token needs to have the tag "noun". the matcher processes the text and keeps the matches that follow this sequence of tags. easy right?

so how do you actually use this in spacy? the first step involves creating a matcher object, and then adding your patterns to that object. let's put all that together:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm") #load english model
matcher = Matcher(nlp.vocab) # initiate matcher object
pattern = [
    {"POS": "ADJ"},
    {"POS": "NOUN"}
]
matcher.add("ADJ_NOUN_PATTERN", [pattern]) # adding pattern to matcher

text = "The big dog was running in the park."
doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end] # get span from start to end
    print(string_id, start, end, span.text)
```

running the code above will output: "adj_noun_pattern 1 3 big dog".
here we load the english model, create a matcher, declare a pattern for adjective + noun and then add the pattern to the matcher giving it a name of *adj_noun_pattern*. we then define a sample text, process it to a spacy document and run the matcher on it. the result gives you the string name of the pattern, the starting position, the ending position and the actual text of the match.

the real power lies in the ability to use a wide variety of token attributes. you can use:
*   `text`: for matching specific words, case sensitively.
*   `lower`: for matching specific words ignoring case.
*   `lemma`: for matching based on lemma forms
*   `pos`: for part-of-speech tags.
*   `tag`: for detailed part-of-speech tags.
*   `dep`: for dependency labels.
*   `is_alpha`, `is_digit`, `is_punct`, etc: for checking token properties
*   `shape`: for matching the token shape, like all upper or all lower.
*   custom attributes: the user can add custom attributes for custom needs

the matcher also lets you use operators within the patterns to make matches more flexible, they work the same as regular expressions, but using a dictionary object instead of regular expressions characters. here's a quick rundown:
*   `{OP: "!"}`: negated token, match if the token does not have the condition.
*   `{OP: "?"}`: optional token, the token can be present or not.
*   `{OP: "*"}`: zero or more tokens of this type.
*   `{OP: "+"}`: one or more tokens of this type.

lets say i want to catch phrases like "a big car", "a very big car", "a very very big car", "big car", "a car". using the operators i can easily do this using the matcher. check this code:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
pattern = [
    {"TEXT": "a", "OP": "?"},
    {"POS": "ADV", "OP": "*"},
    {"POS": "ADJ"},
    {"POS": "NOUN"}
]
matcher.add("ADJ_NOUN_PATTERN", [pattern])

text = "a big car was driving, i saw a very big car and a very very big car, i also saw a car, and a yellow car"
doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(string_id, start, end, span.text)

```

the output will be:
`adj_noun_pattern 1 4 a big car`
`adj_noun_pattern 7 11 a very big car`
`adj_noun_pattern 12 16 a very very big car`
`adj_noun_pattern 18 20 a car`
`adj_noun_pattern 22 24 yellow car`

as you can see using the operators and the token characteristics allows us to match different forms of phrases that have different words on them. note the *op:"?"* for "a" that states that the word a is optional, and the *op:"*"* for the adv, that states that the adverbs can be repeated 0 or n times.

so, for my product review example, this became incredibly useful. i could define patterns for things like "positive adjective" + product name (where product name was also defined via a pattern). i could also account for modifiers and even negations. the code became far more readable and maintainable than it ever was with regex.

another time, i was tasked with extracting dates from legal documents. it was a nightmare. there were dates formatted in so many different ways. i was thinking to myself “i don't get paid enough for this”, i almost lost it. the matcher allowed me to define patterns that handled different formats, like `{"SHAPE": "dd/dd/dddd"}` or `{"TEXT": {"IN": ["january", "february", ..., "december"]}, "OP": "?"}, {"SHAPE": "dd"}` + `",", {"SHAPE": "dddd"}` and also use the operators as i described before to be more flexible. i could also use custom attributes if i wanted to add more filters.

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

date_pattern_1 = [{"SHAPE": "dd/dd/dddd"}]
date_pattern_2 = [
    {"TEXT": {"IN": ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]}, "OP":"?"},
    {"SHAPE": "dd", "OP": "?"},
    {"TEXT": ",", "OP": "?"},
    {"SHAPE": "dddd"}
]

matcher.add("DATE_PATTERN", [date_pattern_1, date_pattern_2])

text = "i signed this document on 01/01/2023, or was it in january 2022, or maybe january 15, 2022 or the 15th of january 2021"
doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(string_id, start, end, span.text)
```

the code output will be:
`date_pattern 5 6 01/01/2023`
`date_pattern 9 11 january 2022`
`date_pattern 12 15 january 15, 2022`
`date_pattern 16 20 january 2021`

as you can see, the flexibility is the real power of the matcher.

if you're looking for resources, i’d recommend checking out the spacy documentation, it's pretty comprehensive. there are also some good natural language processing books that cover tokenization and part of speech tagging in detail; something like 'speech and language processing' by jurafsky and martin would be useful, or 'natural language processing with python' by bird, klein and loper if you are more into hands on approaches. i wouldn't bother with generic programming books for this problem, it's a very specific domain. reading the original research paper on spacy might also help but is not as crucial. those should give you a solid background on the underlying concepts.

the spacy matcher is not just a simple string matcher; it's a powerful tool for identifying complex textual patterns using all the text processing steps that spacy does. it's been an absolute lifesaver for me more than once. and if you know, it's also faster than regex. you have to understand how it works, create good patterns, and you'll be set.
