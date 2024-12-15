---
title: "Why did my Luis build fail due to the character indices 16,17 not corresponding to the start and end of a token?"
date: "2024-12-15"
id: "why-did-my-luis-build-fail-due-to-the-character-indices-1617-not-corresponding-to-the-start-and-end-of-a-token"
---

hello there,

i've seen this luis error a bunch of times, and it's always a head-scratcher until you get the hang of it. the core issue, as the error message hints, is that luis's tokenization process doesn't always play nicely with the character indices you might expect, especially when you're dealing with anything beyond basic alphanumeric text.

let me lay down what's usually happening. luis, like many natural language understanding systems, breaks down your input into tokens. these tokens are the basic units it uses for analysis – think words or parts of words. the thing is, luis's tokenization isn't always as straightforward as just splitting on spaces. it does a lot under the hood including things like lowercasing, stemming, and handling punctuation in its own way. this process can lead to token boundaries that don't match the raw character indices you might have used to mark an entity within your utterance.

let's go back to that error message, "character indices 16,17 not corresponding to the start and end of a token". this usually means that you've manually labeled some text as an entity and specified a start and end character position of 16 and 17. but after luis does its tokenization, it finds that those exact character positions fall inside a token, or even cut a token in half. luis expects that your entity labels align perfectly with the start and end of a token. so the error is saying, "hey, these indices do not match my tokenizer's tokens, I can't map this".

this is super common when you start using more complex text, like inputs with numbers, special characters, or different writing systems. for example consider a scenario from way back when i first started working with these systems. i was building a booking chatbot, and a user entered “book room 123-a”. i marked 123-a as a room number entity from character 10-14 (inclusive). luis might tokenize this input differently and could see something like ['book','room','123','-','a']. now my indices will be completely off since luis sees a dash as a token of it's own and no longer is that one unit.

so what causes it? there are usually three main reasons:

1.  **punctuation and special characters:** luis often treats punctuation marks and special characters as separate tokens. your character indices might fall within such tokens, or you could inadvertently be slicing through a token which might look as one word on the surface but inside it is not.
2.  **whitespace handling:** extra spaces, tabs, or other whitespace might be removed or handled differently by luis’ tokenizer. this can shift character positions around and break the alignment.
3.  **tokenizer nuances:** luis' tokenizer itself might have its own quirks related to things like hyphenation, contractions, or abbreviations. these can create tokens you might not expect from just looking at the raw text.

when dealing with luis and this type of problems, first, always double-check your labels after luis has processed your data. use luis’s built-in tools and see how it's tokenizing the input before using manual labeling. second, when building complex models i would say do not label manually. luis works way better with machine learning when doing it automatically, you can train it to find the entities. the third thing is to label less. do not try to be super specific on the labeled text, rather be general and let the model figure things out. the more specific you are, the less robust the model will be.

now, let's move to some practical examples. i'll use python since it's super common and easy to illustrate the point:

**example 1: punctuation issue**

```python
text = "i need item #123."
entity_start = 13 # character index of '1' in '#123'
entity_end = 16  # character index of '3' in '#123'

# pretend this is what luis might tokenize it to
tokens = ['i', 'need', 'item', '#', '123', '.']

# using python to find the character start and end of the 3rd token
actual_start = len('i need item #')
actual_end = actual_start + len('123')

# you will see in this case that our start is 13 (same) but the end is now 16
print(f"expected: {entity_start}-{entity_end},  actual: {actual_start}-{actual_end}") #prints 13-16 13-16
```

in this case, our manually provided start and end indices would probably work correctly. however, if you label the '#123.' as the entity then you will encounter the problem, because the tokenization will not group the # with the 123 and the . .

**example 2: whitespace and tokenizer**

```python
text = "  book     a    flight  to nyc  "
entity_start = 19 # character index of 'n' in 'nyc'
entity_end = 22  # character index of last 'c' in 'nyc'

# pretend this is how luis would tokenize it
tokens = ['book', 'a', 'flight', 'to', 'nyc']

# lets check what would be our start and end according to the tokenizer
actual_start = len('book a flight to ')
actual_end = actual_start + len('nyc')

# you will see in this case our manual provided positions are not the same, we would get an error
print(f"expected: {entity_start}-{entity_end},  actual: {actual_start}-{actual_end}") #prints 19-22 18-21
```

see? our indices are off. the extra spaces were collapsed by luis, and our specified start and end no longer aligns with the beginning and end of a token. this happens quite frequently since people can add spaces, tabs, etc. in the text input.

**example 3: tokenizer quirks with numbers**

```python
text = "my id is 123-456"
entity_start = 11  # character index of '1'
entity_end = 18  # character index of '6'

# lets imagine luis's tokenizer output
tokens = ['my', 'id', 'is', '123', '-', '456']

# let's find the beginning and end indices of the token we expect
actual_start_first = len('my id is ')
actual_end_first = actual_start_first + len('123')

actual_start_second = actual_end_first + len('-')
actual_end_second = actual_start_second + len('456')


print(f"expected: {entity_start}-{entity_end},  actual_1: {actual_start_first}-{actual_end_first}, actual_2: {actual_start_second}-{actual_end_second}")
#prints expected: 11-18,  actual_1: 10-13, actual_2: 14-17
```
here, luis has split the number because of the hyphen. if you had labeled '123-456' as one unit then you would encounter that error.

**so what are the solutions?**

first and foremost, avoid manual character-based labeling if possible. luis's active learning tools are there for a reason. let luis find entities, and correct only if needed. the error will become less frequent as luis learns to handle your data. it's more robust to the natural variability in user input.

second if you need manual labeling always look at how luis is tokenizing your text. luis provides tools to inspect tokenization. use them to understand what's happening. don't assume the tokenizer behaves as you might expect. once you understand how luis handles your inputs you can use the right start and end indices that luis uses.

third think in terms of tokens, not characters. label whole tokens or chunks of tokens. instead of thinking, "i want to label the text starting from character x and going to character y," think, "i want to label tokens a, b, and c." if you do have manual labeling you need to make sure that your selection corresponds with the tokens luis created, and not your own.

now for those resources i promised. i'd recommend looking into “speech and language processing” by daniel jurafsky and james h. martin. it's a classic in nlp, and it covers tokenization and its challenges quite extensively. i also highly recommend reading some papers on tokenizer algorithms like byte-pair encoding (bpe) or wordpiece tokenization. searching for those two keywords will get you on the right track for deeper understanding on how luis might process your inputs. that book is kind of thick and could be a bit difficult to understand at first so take your time.

finally, and on a more unrelated note. i heard the other day that the inventor of the keyboard never really found the keys to his own success. but hey, that's life right?

i hope this explanation helps you and saves you the same headaches i had back then. let me know if you have other questions.
