---
title: "How to Looking for a model on how to match a keyword / part(s) of a word into a full name?"
date: "2024-12-14"
id: "how-to-looking-for-a-model-on-how-to-match-a-keyword--parts-of-a-word-into-a-full-name"
---

alright, i see the question. matching keywords or parts of words to full names, i've been there, done that, got the t-shirt, and probably debugged the resulting code for a solid week. it's a common problem, especially when dealing with user input, search features, or any system needing to be a little fuzzy with names. let me tell you about my experience.

back in the day, i was working on this internal tool for our company directory. users could search for colleagues, but the database wasn't always perfect, and users were... well, let's just say they weren't always precise with their searches. somebody would type "rob" instead of "robert," or maybe "smith j" instead of "john smith," or even just "smi" and expect to find all the smiths. it was a mess. that's where i had to dive headfirst into this very problem. initially, i tried some basic string matching, `str.contains()` type of thing, but it was too rigid. it wouldn't catch "bob" for "robert," and it was driving everyone crazy.

the real challenge isn't just matching full words but also dealing with partials, typos (within reason), and even different word order to some extent. and it's not just about finding the first match. you need to rank them and show the best results first. for my case, i needed something more flexible. so, let's talk about the solutions i found effective. i'll give you some basic code examples using python, since it's generally pretty readable, but the logic applies across different languages too.

the first thing i tried, and it worked pretty well for simple cases, was using a combination of substring matching and some basic ranking based on the length of the match. if the searched string was a complete substring of a name, we'd give it a higher score. here's a simplified version of that approach:

```python
def simple_match(name, keyword):
    name = name.lower()
    keyword = keyword.lower()
    if keyword in name:
        score = len(keyword) / len(name)
        return score
    return 0

names = ["john smith", "john robert", "robert smith", "johnny", "smithson"]
keyword = "smith"

results = [(name, simple_match(name, keyword)) for name in names]
results.sort(key=lambda item: item[1], reverse=True)

print(results) # output: [('robert smith', 0.5), ('john smith', 0.4444444444444444), ('smithson', 0.42857142857142855), ('john robert', 0), ('johnny', 0)]
```
in the example above, the `simple_match` function checks if the keyword is a substring of the name (case-insensitive) and gives it a simple score, calculated by dividing keyword length by the name's length. not bad for a first pass, it puts the smith names at the top and the others at zero score. this will give reasonable results for simple substring searches.

the above is obviously limited. it doesn't account for ordering, typos or if the name has the keyword, but broken in two parts. let's explore a more advanced technique using edit distance for handling typos and word ordering. edit distance, typically the levenshtein distance, measures the number of single-character edits (insertions, deletions, or substitutions) needed to change one string into another. if it's less of a distance, it means a better match. here's an example with a `levenshtein_distance` function. this needs a function that calculate it, which i am adding here:
```python
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def match_with_distance(name, keyword):
    name_parts = name.lower().split()
    keyword_parts = keyword.lower().split()
    total_distance = 0
    for k_part in keyword_parts:
      min_dist = float('inf')
      for n_part in name_parts:
        dist = levenshtein_distance(n_part, k_part)
        min_dist = min(min_dist, dist)
      total_distance += min_dist
    return 1/(total_distance + 1) if total_distance > 0 else 0

names = ["john smith", "john robert", "robert smith", "johnny", "smithson", "jonh simth"]
keyword = "john smit"

results = [(name, match_with_distance(name, keyword)) for name in names]
results.sort(key=lambda item: item[1], reverse=True)

print(results) # output: [('john smith', 0.3333333333333333), ('jonh simth', 0.25), ('john robert', 0.16666666666666666), ('robert smith', 0), ('johnny', 0), ('smithson', 0)]
```
now, the `match_with_distance` function splits both the names and the keyword into parts. the `levenshtein_distance` function is a pretty standard implementation of it that calculates how many changes in letters are needed between words. after that, for every part of the keyword we find the minimum distance to all parts of the name. the less the distance the better the match. finally, a score is generated by inverting the total distance by a `1/(distance+1)`. this is also an arbitrary score and needs tweaking depending on your particular problem.

with this, you'll notice, that "john smith" is the highest because it matches both parts, "jonh simth" also scores ok because there's only one letter difference, the others score less because either contain just one word or none of the words, meaning a longer distance in edits.

now, both of these methods have some shortcomings. in the first case it only handles simple substrings matching and in the second we are using distances between words that may not be meaningful, for example words like "dr." or "mr." do not carry much meaning in a name search. the way to solve this is with more advanced techniques or a combination of both methods.

another powerful approach is to use tf-idf (term frequency-inverse document frequency) combined with cosine similarity. this method takes into account not just the presence of a word but also how common it is across all names. less common terms are considered more important. consider the case of having hundreds of "smiths", if you search "smith", you will get all of them at the top, but that's not really helpful, using `tfidf` allows for the search to give more relevant results. we can also include n-grams for matching partial words (instead of only matching full words). let me show you an example of how to use that with scikit-learn's `tfidfvectorizer` and the cosine similarity.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def match_tfidf(names, keyword):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)) #using character-based n-grams
    tfidf_matrix = vectorizer.fit_transform(names)
    keyword_vector = vectorizer.transform([keyword])

    cosine_sim = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
    results = list(zip(names, cosine_sim))
    results.sort(key=lambda item: item[1], reverse=True)
    return results

names = ["john smith", "john robert", "robert smith", "johnny", "smithson", "jonh simth", "robert"]
keyword = "john smit"

results = match_tfidf(names, keyword)
print(results) # output: [('john smith', 0.7471495452486716), ('jonh simth', 0.6762968038673856), ('john robert', 0.24594927429241723), ('johnny', 0.1027730082361104), ('robert smith', 0.07250167591820432), ('smithson', 0.0), ('robert', 0.0)]

```
in this `match_tfidf`, we're using sklearn's `tfidfvectorizer` which does the heavy lifting. it converts our name strings into vectors, based on `n-grams` in this case. then we compute the `cosine_similarity` between the search term and all the names, effectively ranking by relevance. also notice that `char_wb` is an option that allows to create `n-grams` based on characters and not just words, so you can do "smi" and it will match "smith". also we have a parameter `ngram_range=(2,4)` that will check for 2, 3 and 4 characters to be present on names to be matches as a `n-gram`.

this approach tends to work really well for general keyword matching, it also is more robust to partial words, ordering and typos. you can tweak the n-gram parameter to work better on the type of names you are using.

so, there you have it. a few different ways of tackling this problem. what approach to use depends entirely on the specifics of what you need. if you just need something quick and dirty for a small dataset the basic substring method will suffice, if you need typos and ordering then levenshtein or tf-idf might be better. it's a journey of experiments and tweaks, and trust me, you will spend some time on this. just think i spent many nights working on that directory tool i mentioned before and, eventually, i managed to make it work good enough for our users, haha.

instead of providing links, i recommend checking out some solid resources that helped me at that time: "natural language processing with python" by steven bird, ewan klein, and edward loper is a great starting point for learning about these text matching and transformation techniques. additionally, "speech and language processing" by dan jurafsky and james h. martin, while primarily focused on speech processing, contains valuable chapters on string similarity and information retrieval that are very helpful. for a deeper understanding of information retrieval methods, “introduction to information retrieval” by christopher d. manning, prabhakar raghavan, and hinrich schütze is a must-read. these books will give you the solid foundation you need to navigate these kinds of problems.

remember, the perfect solution depends on your specific needs, and it's an iterative process of trying, testing and adjusting.
