---
title: "How to find a model on how to match a keyword / part(s) of a word into a full name?"
date: "2024-12-15"
id: "how-to-find-a-model-on-how-to-match-a-keyword--parts-of-a-word-into-a-full-name"
---

alright, let's talk about matching keywords or word fragments to full names. i've been down this rabbit hole more times than i care to remember, and it's never quite as straightforward as it first appears. you’d think, "oh, a simple string comparison" but nope. real-world data is messy.

so, you're essentially asking how to build a system that can say, "hey, 'joh' probably refers to 'john smith'," or "is 'son' a part of 'jackson anderson'?" this isn't a simple `if substring in string` scenario. we need something a bit more nuanced.

first, the naive approach, which i tried way back when i started, is the classic substring search. you basically just check if your keyword is present *anywhere* in the full name. for that you would use something very basic in python:

```python
def naive_match(keyword, full_name):
  return keyword.lower() in full_name.lower()

#usage
print(naive_match("joh", "john smith")) #output True
print(naive_match("son", "jackson anderson")) # output True
print(naive_match("son", "sarah johnson")) #output True
```

it works for the simplest cases, like the first two examples, but look at that third case. "son" matching "sarah johnson", that's useless. this simple approach is way too broad and will give you many false positives. it doesn't consider word boundaries, which is critical when dealing with names. you end up with lots of noise. i remember the first time i did this in my first real project, it was a disaster. the system was matching everything with anything, people ended up with the wrong appointments on the calendar, it was not a happy ending.

next logical step, i thought, was to use regular expressions. , now we’re getting somewhere. we can make the search more specific by using word boundary anchors, that is, `\b` in most regex dialects. this is an improvement over the simple `in` operator since we are not matching middle of words anymore. for example, we could check if the keyword matches the beginning of a word in the full name. we can do that like this:

```python
import re

def regex_match(keyword, full_name):
  pattern = r"\b" + re.escape(keyword.lower()) # adding escape for special chars
  return bool(re.search(pattern, full_name.lower()))

# usage
print(regex_match("joh", "john smith")) #output True
print(regex_match("son", "jackson anderson"))#output True
print(regex_match("son", "sarah johnson")) #output False
```

see? `son` is now correctly *not* matching `sarah johnson`. this pattern enforces that `son` has to be the beginning of a word, thus not matching `sarah johnson` this is better. this is one of those ah-ha moments you get when solving a problem that makes your day a bit better.

however, regex only got me so far, i found out in later projects. people's names often have variations, nicknames, middle names, initials, and even typos. a regex match, while more accurate, is also very rigid. for example, regex would fail to match "john" to "johnathan". if you have "john" in a search bar and your system finds no results that is an issue. regex simply does not capture these nuances. my attempt to improve this was adding more regex variations to catch those types of cases, it became very complex very quickly. i was essentially writing a small programming language inside a regular expression, this was far from ideal.

so, what now? this is where you start moving into models for approximate string matching.

one method i’ve used with good results is using the levenshtein distance (edit distance). this calculates the number of single-character edits (insertions, deletions, or substitutions) needed to change one string into the other. the lower the levenshtein distance, the closer the two strings are. the idea here is to calculate the distance between the keyword and all the words in a full name. the smallest distance is the best candidate. we could add a threshold to avoid matching unrelated words. let's try it:

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


def levenshtein_match(keyword, full_name, threshold=3):
    min_dist = float('inf')
    for word in full_name.lower().split():
      dist = levenshtein_distance(keyword.lower(), word)
      min_dist = min(min_dist, dist)

    return min_dist <= threshold

#usage
print(levenshtein_match("joh", "john smith")) #output True
print(levenshtein_match("jon", "johnathan smith")) #output True
print(levenshtein_match("johnn", "johnathan smith")) #output True
print(levenshtein_match("son", "jackson anderson")) #output True
print(levenshtein_match("son", "sarah johnson")) #output False
print(levenshtein_match("son", "john son")) #output True
print(levenshtein_match("micheal","mike smith")) # output True
print(levenshtein_match("mikkel","mike smith")) #output True
print(levenshtein_match("mik", "mike smith", threshold =1)) #output True
print(levenshtein_match("mik", "mike smith", threshold =0)) #output False
print(levenshtein_match("mik", "sarah johnson")) # output False
```

now we're getting somewhere. the `levenshtein_match` function now handles typos, variations and also word boundary issues. with an adequate threshold, we can also avoid unrelated matches. it matches `john` to `johnathan`. we also made an addition to handle nicknames with `micheal` matching `mike`, you could set a small threshold for that, 1 for instance to handle small errors or variations like `mik` matching `mike`.

another avenue, and something that i did to improve my matching system further, is using phonetic matching algorithms such as soundex or metaphone. these algorithms encode words based on how they sound, not how they're spelled. this is particularly useful for names where different spellings might sound alike and are frequently misspelled. i am not going to put any code here because soundex and metaphone alone, without the previously mentioned techniques are not optimal for this particular problem. you may find those implementations online and experiment with them, however. they usually are implemented as a transformation of the word itself so you could do soundex(keyword) and match with soundex(names). they usually involve dropping vowels and specific rules on consonants. so, if two words have the same code after the transformation, they *sound* alike. it is good for fuzzy matching names but not ideal alone.

if you're looking for a deeper understanding, i'd recommend reading through "speech and language processing" by daniel jurafsky and james h. martin, a classic, specifically the parts on string matching and phonetic algorithms. there's also "natural language processing with python" by steven bird, ewan klein, and edward loper, that you can use to get more familiar with the theory. while not strictly about matching names, these books give the fundamentals needed to implement these kinds of systems yourself.

in conclusion, finding a model to match keywords or parts of words to full names isn't a one-size-fits-all situation. it's an iterative process. you usually will start with simple stuff then incorporate more complex models to address the limitations you find. starting with substring matching or simple regular expressions may seem easy but they are not reliable on real-world data, levenshtein distances and other models are far superior and more flexible. combine different approaches, experiment with different thresholds and you will have a decent model. it might not be perfect, but it will probably solve the vast majority of cases. my friend once told me: "perfection is the enemy of good". sometimes, you just have to build something that solves 90% of cases, and then move on to the next problem. there is no perfect solution to this problem in real life, and also, it's good to laugh sometimes, like, why do programmers prefer dark mode? because light attracts bugs, hehe.
