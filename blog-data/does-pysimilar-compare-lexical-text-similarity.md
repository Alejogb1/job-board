---
title: "Does pysimilar compare lexical text similarity?"
date: "2024-12-14"
id: "does-pysimilar-compare-lexical-text-similarity"
---

let's tackle this. you're asking if pysimilar checks for lexical similarity between texts, and the short answer is: yes, but with some caveats i’ve learned the hard way.

pysimilar isn't a library i've used extensively lately. i remember a project back in '16, it was something about trying to build a quick and dirty plagiarism checker. the idea was to flag similar submissions in student assignments. we had a deadline, and like any good last minute programmer, i opted for the quickest solution available. it was naive, i know, but that's how we learn. we were using pysimilar on a bunch of essays and short code snippets. i didn't grasp the nuances back then, just threw it at the data and called it a day. the results were, let's say, less than optimal. it flagged some things as super similar when they weren't, and missed others that were.

anyway, pysimilar works by calculating the similarity between two text strings based on a particular algorithm that computes the distance between them. under the hood it frequently uses algorithms like the n-gram approach, or variations of the levenshtein distance. in essence, it’s looking at how many changes, additions or removals are needed to convert one string into another, or the number of shared n-grams, and then comes up with a similarity score.

lexical similarity, in this context, means how similar the actual words and characters are, disregarding context or meaning. it is completely superficial. think of it this way: "the cat sat on the mat" and "the bat sat on the mat" are very lexically similar, although the meaning is different. pysimilar, in general, focuses primarily on this superficial similarity. it does not try to understand the semantics of the text, the deep meaning, only the sequence of characters or words. it’s purely mechanical. if you changed a few letters in a document it will notice it. if you rephrased a complete paragraph but kept the underlying ideas, chances are that it will not pick up on the connection. i have found this a pain at times.

this is where things can get tricky with pysimilar. it’s great at catching exact matches or very close variations, but it won’t understand if two sentences are paraphrases of the same idea. that was a significant issue in our plagiarism detector attempt. students, being the clever problem solvers they are, found ways to change words and rephrase sentences while maintaining the original content. pysimilar was, in my specific situation, easily fooled. it is a tool that gives you a similarity score. it does not try to understand.

let me show you a basic example of how you might use pysimilar:

```python
from pysimilar import compare

text1 = "this is the first example string."
text2 = "this is the second string example."

similarity_score = compare(text1, text2)
print(f"the similarity score is: {similarity_score}")

text3 = "this is almost exactly the same."
text4 = "this is almost exactly the same"

similarity_score = compare(text3, text4)
print(f"the similarity score is: {similarity_score}")
```

this should show you a quick demo, where similar phrases get a higher score. the output of the first `compare` will be lower than the second, the strings are almost identical in the second example. you will find this a general trend. the higher the score the more the texts are similar.

now, there are other libraries that might be more robust when you need a broader understanding of text similarity, especially in terms of semantics and not just character comparisons. one example is the 'sentence transformers' library. it encodes text into vector embeddings and measures similarity based on semantic meaning. its a whole other ball game but that is what i should had used back then, it would had saved me a lot of time.

for example, the below example will probably show a higher similarity score between text1 and text3 than the previous example, despite the fact that the words are completely different. this is because the underlying meaning is very close.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

text1 = "the quick brown fox jumps over the lazy dog."
text2 = "a fast brown fox leaps over a sleeping canine."
text3 = "this is a completely unrelated sentence."

embeddings = model.encode([text1, text2, text3])

similarity12 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
similarity13 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"similarity between text1 and text2: {similarity12}")
print(f"similarity between text1 and text3: {similarity13}")

```

i would highly recommend diving into 'natural language processing with python' by steven bird, ewan klein, and edward loper if you want to get a deeper grasp on the fundamental algorithms and ideas behind similarity measures and text processing. this particular book provides a good foundation that will be useful in almost any project you may encounter when dealing with natural language. it is something that i had on my shelf, and if i had consulted it back then, i would have had much more success on that project.

also, i can point you to the research paper “natural language processing (almost) from scratch” by collobert et al. it was a game changer in how we perceive similarity in text, where words were no longer one hot encoded but mapped to dense vectors based on the context where they appear. i found it very informative and a good source to get up to speed with the modern approaches, instead of just purely using algorithms like the levenshtein distance which are based on character or string changes.

now a word of caution: when you're using similarity scores, interpret the results cautiously. no single metric is perfect. it depends heavily on the context and the kind of similarity that you’re trying to capture. it is important to remember that these are just tools and they require a deep understanding of how they work so that they can be used effectively.

and, since you’re curious, i recall another project where i was working on automatic summarization. we tried to use pysimilar to identify redundant sentences, but it turned out that phrases that were very similar in meaning could be expressed in very different ways, making our approach less than perfect again. it is almost like using a hammer to screw a nail, it is the wrong tool for the job, and it’s never a good idea to force a specific library that is not appropriate. we ended up abandoning pysimilar and moved to techniques that employed word embeddings and cosine similarity. i learned that sometimes, you have to give up on a pet project to move forward and learn from your mistakes. a bit like when you try to fix a bug and it just gets worse, sometimes the best option is to just revert the changes and start again. that has happened more times than i can count.

one common trick for doing quick checks is to use string matching like the following:

```python
text_a = "this is a text example"
text_b = "this is also a text example"

matches = 0
words_a = text_a.split()
words_b = text_b.split()

for word in words_a:
   if word in words_b:
        matches+=1
similarity = matches / max(len(words_a), len(words_b))

print(f"quick similarity check : {similarity}")

```
this can sometimes help to gauge quick similarities without relying on external libraries. but this approach is quite basic. it ignores the order of the words.

in summary, pysimilar does compare lexical text similarity, but it’s important to know its limits. it looks at superficial similarities, not semantic understanding. for simple comparisons like detecting minor typos or very similar strings, it can be useful. but for more complex text analysis, where you need to capture the meaning and intent behind text, other libraries and techniques may be much more suitable. just make sure you’re choosing the right tools for the job.

and now, for my one joke: why did the programmer quit his job? because he didn’t get arrays. i will see myself out.
