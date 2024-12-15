---
title: "Why is a spaCy Matcher Rule not finding a phrase in the text?"
date: "2024-12-15"
id: "why-is-a-spacy-matcher-rule-not-finding-a-phrase-in-the-text"
---

alright, so you're banging your head against the wall with a spacy matcher, and it's not picking up the phrase you *swear* should be there, i get it. i've been there, many times. it's a classic case of "it should just work!" meets "wait, what?". let's break it down because there's a few gotchas that can sneak up on you.

first off, the thing to always check, *always*, is tokenization. spacy does a pretty good job by default, but sometimes, things aren't tokenized the way you expect, and that's going to throw off your matcher rules faster than a misconfigured regex. think of it, spacy takes your string and splits it into chunks, based on punctuation, spaces, and a whole lot of language-specific rules. if your phrase spans multiple tokens but your rule is expecting a single token or token sequence to match exactly, well that's your problem.

let me give you an example, i was working on a project years ago, probably back in the 2.x spacy days (wow, time flies), dealing with extracting product names from user reviews. i had this rule:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [{"LOWER": "super"}, {"LOWER": "duper"}, {"LOWER": "laptop"}]
matcher.add("laptop_name", [pattern])

text = "this is a superduper laptop, and it's great!"

doc = nlp(text)

matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(f"found a match: {matched_span.text}")
```

so my brain thought that 'superduper laptop' was what was there right? but my code was not. and guess what? it was not catching "super duper laptop" or any other form. because in my text that phrase was split in 3 tokens "super", "duper", and "laptop". spacy tokenized "superduper" as a single token. the fix there? my pattern should have matched the different tokens:  `[{"LOWER": "superduper"}, {"LOWER": "laptop"}]` would have worked, or even better `[{"LOWER": "super"}, {"LOWER": "duper"}, {"LOWER": "laptop"}]` will work also, but i should check first my tokens, and the text.

i learned that hard way (and many other similar cases). so rule number one: inspect your tokens. you can loop through your `doc` and print `token.text`, `token.lemma_`, `token.pos_`, `token.tag_`, etc, that's your best friend when you need to debug these things.

next, be careful with casing. unless you're specifically using a `{"TEXT": ...}` rule, which does a case-sensitive match, the `{"LOWER": ...}` rule will do a lowercase comparison. but you might have a text that is all caps, or with weird capitalization, and you should deal with that.
here's a simplified example that trips people quite a bit:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [{"LOWER": "apple"}, {"LOWER": "pie"}]
matcher.add("food_item", [pattern])

text = "I love Apple pie, and apple Pie!"

doc = nlp(text)

matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(f"found a match: {matched_span.text}")

```

that code will find both "Apple pie" and "apple Pie!" without issue. but if you change the pattern to `[{"TEXT": "apple"}, {"TEXT": "pie"}]`, no matches will be found at all (unless the text was exactly "apple pie" in lowercase). my point is, using `{"LOWER": ...}` is generally a better approach unless you *need* case-sensitive matching for very particular reasons.

another thing, be aware of *lemma* matching. by default `{"LEMMA": ...}` will try to match based on the base form of the word, not its surface form. for example the lemma of 'running' is 'run'. which can be very useful. or very confusing if you are not aware of that.
take a look at that:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [{"LEMMA": "run"}, {"LOWER": "track"}]
matcher.add("race", [pattern])

text = "she is running on the track"
text2 = "i run on the track"

doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(f"found a match: {matched_span.text}")

doc2 = nlp(text2)
matches = matcher(doc2)

for match_id, start, end in matches:
    matched_span = doc2[start:end]
    print(f"found a match: {matched_span.text}")

```

that will find "running on the track" and "run on the track" as the lemma of running is "run". that is why it matched in both sentences. that could or could not be what you wanted. this is a good example of spacy's power and also a great example of how easy it is to make a small mistake.

also watch out for whitespace and punctuation. spacy tokenizes those as well, and that might create some issues. think a text like `"...some words...".` and you wanted to match `some words`. spacy might tokenize the first three dots as one token, `some` as one, `words` as another, and the last three dots as another one. if your pattern does not match, it will simply not find it. my usual approach is to use `{"IS_PUNCT": False}` to skip punctuation tokens or `{"IS_SPACE": False}` to skip spaces, depending on what i need. this can be very useful for a more resilient code. it depends on your use case, but i find this useful in a lot of cases.

another thing, and this one's bit subtle, check the *order* of your pattern components. it matters. `[{"LOWER": "a"}, {"LOWER": "b"}]` won't match `b a`, for that you will need `[{"LOWER": "b"}, {"LOWER": "a"}]`. that is simple enough but easy to overlook at the beginning.

finally, and this is less about the code and more about the process, always simplify your problem. if you're not getting a match on a large corpus, start with the smallest example that's failing. that is also useful if you have really complex patterns, start with the simplest pattern you can think of, and build your way up. if that doesn't work, you know something is wrong at the very beginning. that approach saved me hours of debugging on several projects that i worked.

to find more about spacy matcher there are a few resources i would recommend. the spacy documentation is always a good start. there are tutorials and examples that are very useful for begginers, and also useful even for experimented users. but also i would recommend the book "natural language processing with python" by steven bird, ewan klein, and edward loper, that is a great resource to understand the basics of nlp and how it works behind the curtains. "speech and language processing" by dan jurafsky and james h. martin it's also a great resource for more in depth knowledge of the nlp field. and also i would advise to check a couple of research papers. for example "spaCy: Industrial-Strength Natural Language Processing in Python" by Matthew Honnibal, Ines Montani. It's a deep dive into the design and implementation of spacy.

oh, and a random joke: why did the programmer quit his job? because he didn’t get arrays! yeah i know, i know… it was a bad joke. anyway.

so, check your tokenization, pay attention to casing, be aware of lemma matching, manage your punctuation and spacing, and simplify your problem. those are some of the most common pitfalls. if your rule still isn't catching, go back to step one and recheck your tokens. i promise, the answer is usually there, staring you in the face and sometimes it's so simple you simply overlooked it. it always takes time, a lot of it sometimes. keep at it, and don't forget to get some sleep.
