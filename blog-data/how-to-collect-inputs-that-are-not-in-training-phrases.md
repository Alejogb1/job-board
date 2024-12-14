---
title: "How to collect inputs that are not in training phrases?"
date: "2024-12-14"
id: "how-to-collect-inputs-that-are-not-in-training-phrases"
---

alright, so you're facing the classic "how do i handle stuff my model hasn't seen before" problem. i've been down this road more times than i care to remember. it's like, you meticulously train this fantastic natural language processing (nlp) model, feeding it gigabytes of perfectly crafted data, and then it encounters a totally new phrase and just stares back blankly. it's frustrating, i get it.

let's break down how i typically approach this, keeping it simple and, as much as possible, away from the mystical black box side of machine learning.

first off, let's be clear: no model, no matter how advanced, will perfectly understand *everything*. that’s not how these things work. we're talking about statistical approximations, pattern recognition, not some form of actual comprehension. therefore, a key concept is understanding we need ways to handle the *unknown*, gracefully.

my first instinct is not to throw more data at it right away, that's usually the knee-jerk reaction, instead, let's evaluate how our pipeline is dealing with out-of-vocabulary (oov) inputs. usually, this involves a few techniques i've found effective in my past escapades in building conversational interfaces.

one of the most straightforward steps is to check how our tokenizer is handling things. the tokenizer is the first step, that transforms raw text into numerical data that our model can chew on. if it's treating unseen words as single unknown tokens, like `<unk>`, well, that's a dead end for our model. it has no way to understand that. that's why you should definitely check how many of your input tokens are of `<unk>` in the logs. i remember when i was working on a sentiment analyzer for customer reviews i was seeing about 40% `<unk>` tokens, it was a disaster. we needed to really evaluate our tokenizer.

so, instead of single unk tokens, we can use subword tokenization, think of byte-pair encoding (bpe), or wordpiece. this means splitting words into smaller units, like 'un-expect-ed-ly', which can enable our model to infer a relationship between similar words, even if it hasn't seen them together before. it allows for a degree of generalization. for instance, if the model sees 'play' but not 'played', the subword 'ed' might give a clue about past tense.

here's a simple python code snippet using `tokenizers` library, to show how this works. imagine we are using a pretrained bert model:

```python
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    vocab_file="path/to/your/vocab.txt",  # path to bert vocab
    lowercase=True
)

text1 = "this is a new phrase."
text2 = "this is an unseenphrase"

output1 = tokenizer.encode(text1).tokens
output2 = tokenizer.encode(text2).tokens

print(f"tokenized text 1: {output1}")
print(f"tokenized text 2: {output2}")


```
output could be something like
```
tokenized text 1: ['this', 'is', 'a', 'new', 'phrase', '.']
tokenized text 2: ['this', 'is', 'an', 'un', '##se', '##en', '##phrase']
```

notice that it has splitted into subwords like `un`, `##se`, `##en`, `##phrase` , this helps the model.

after tokenization, if the model still struggles, we can look into ways to create an environment where these unknowns can be "tolerated" by it, this can be done at different levels.

one technique is to implement some kind of fuzzy matching at the input level. before even sending the input to the model, check if the input is similar to any known training phrase. this can be done with string similarity algorithms like levenshtein distance or cosine similarity with sentence embeddings. this allows us to "redirect" the model to the closest known training phrase. this adds a layer of robustness. in essence, if the user says something similar to what the model understands, then at least the response is "almost" correct and the user experience does not collapse.

for this fuzzy matching, we can use a pre-trained sentence embedding model, like those from sentence transformers. this provides vectors that captures the semantic meaning of the input. a simple example using `sentence-transformers` library:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')  # a general purpose model

known_phrases = ["how do i get a refund", "what is the shipping cost?", "where is my order?"]
unknown_phrase = "i want to be reimburse of my money"

known_embeddings = model.encode(known_phrases)
unknown_embedding = model.encode([unknown_phrase])

similarities = cosine_similarity(unknown_embedding, known_embeddings)
most_similar_index = np.argmax(similarities)

print(f"closest phrase : {known_phrases[most_similar_index]}")

```
the output here is
```
closest phrase : how do i get a refund
```
it means that the model considers that the phrase: "i want to be reimburse of my money" is similar to "how do i get a refund" so we can redirect the user to this action.

now, after we do the first matching process, another approach is to incorporate a feedback loop, or a user in the loop system. when a user enters a phrase the model doesn't understand, don’t simply return an "i don't understand" message. instead, ask the user to clarify, or provide a list of possible options based on the fuzzy matching (that we discussed before). this not only helps resolve the immediate issue but also gives us data to add to the training set later. think of it as a learning opportunity. we actually want this to happen. the best is to create an entire system, that logs the unknown queries, ask the user to choose from a list of most closest queries, then review the data, create more data and re-train, so that this new data is incorporated in the system.

here's a code snipped where we give a couple of options to the user.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')  # or other suitable model

known_phrases = ["how do i get a refund", "what is the shipping cost?", "where is my order?","track my order"]
unknown_phrase = "where is my package"

known_embeddings = model.encode(known_phrases)
unknown_embedding = model.encode([unknown_phrase])

similarities = cosine_similarity(unknown_embedding, known_embeddings)
most_similar_indices = np.argsort(similarities[0])[-3:][::-1] # get top 3 indices

print("Did you mean any of these:")
for i,index in enumerate(most_similar_indices):
  print(f"{i+1}. {known_phrases[index]}")
```
output example

```
Did you mean any of these:
1. where is my order?
2. track my order
3. what is the shipping cost?
```

this allows the user to choose, and learn from the possible options. its a win-win for everyone.

finally, remember that nlp is an active field, so it is important to keep up-to-date with the latest advances. books like "speech and language processing" by daniel jurafsky and james h. martin, or "natural language processing with transformers" by lewis tunstall, leandro von werra and thomas wolf. can be very useful. also, research papers on nlp are key. the website paperswithcode can be of help to find interesting papers for a certain specific area.

also, let's be honest, sometimes you'll add a new phrase to your model, train it for hours and it still doesn't understand, and you wonder "what am i even doing?", but hey, that's tech for ya! (i guess that's the joke, maybe not that funny, but is what it is).

in short, you want to move away from the "it works, or it doesn't work" mentality and think more like: how i can make my system gracefully handle the unknown, learn from those unknowns, and continuously improve. tokenization, fuzzy matching, user feedback, continuous review and improvement. these are some of the tools that i use in my day-to-day that make the difference between a usable system, and a frustrating one. it's not a magic bullet, but a more pragmatic approach. hope this helps and feel free to ask more if something is not clear.
