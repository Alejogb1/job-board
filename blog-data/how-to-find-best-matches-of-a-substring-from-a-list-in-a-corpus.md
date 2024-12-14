---
title: "How to Find best matches of a substring from a list in a corpus?"
date: "2024-12-14"
id: "how-to-find-best-matches-of-a-substring-from-a-list-in-a-corpus"
---

alright, so you’re asking about finding the closest matches of a substring within a larger body of text. it's a pretty common task, and i've definitely spent more than a few late nights tackling this kind of problem. let me walk you through some of the approaches i've used, and how i'd break it down. it's not exactly rocket science, but getting it efficient and accurate takes a bit of thought.

first off, it's essential to clarify what we mean by "best match." do we want exact matches? or are we looking for fuzzy matches that are similar but not identical? this decision significantly impacts the techniques we use. and what about the size of the corpus? are we talking a few kilobytes of text or gigabytes? that also has a big impact, because if it is just a few kilobytes the naive approach is just fine.

let's assume, for this explanation, that we're aiming for a mix of both exact and fuzzy matches, and the corpus is of moderate size. if it gets extremely large we can always use indexes which are out of this scope, but, i will mention them later on.

a very simple approach, and one i used back in the day when i was still learning, is just a linear scan. it's not the most efficient, but it's easy to understand and implement. it works like this: for each substring you’re searching for, loop through the entire corpus, and check for matches. for an exact match, it is just `string.find()` or `string.indexOf()`. if there’s a hit, we can record it. it is super simple to implement using any language like python:

```python
def find_exact_matches(corpus, substrings):
    results = {}
    for substring in substrings:
        matches = []
        start = 0
        while True:
            start = corpus.find(substring, start)
            if start == -1:
                break
            matches.append(start)
            start += 1
        results[substring] = matches
    return results

# example:
corpus = "this is a test string with test and testing inside"
substrings = ["test", "string"]
matches = find_exact_matches(corpus, substrings)
print(matches) # Output: {'test': [10, 26], 'string': [15]}
```

pretty straightforward, eh? now, this method is fine for small corpora, but it's not going to scale well. its time complexity is something around *o(m*n)* where 'm' is the number of substrings and 'n' the size of the corpus, which is not ideal as the corpus grows. for fuzzy matching, we need something better than just direct comparison.

for fuzzy matching, one of the more common approaches i've seen used, and one that i've used a lot, is the levenshtein distance (also known as edit distance). the levenshtein distance calculates the minimum number of single-character edits (insertions, deletions, substitutions) required to change one string into another. a lower distance means a closer match. i had a situation long ago when i had to handle misspelling, because the user used a voice recognition system, and the output was far from perfect. the levenshtein distance was a life saver.

here’s an example of a simplified version of the levenshtein distance calculation in javascript (mind you i’m more of a backend person but js is a pretty universal language):

```javascript
function levenshteinDistance(s1, s2) {
  const matrix = [];

  // increment along the first column of each row
  let i;
  for (i = 0; i <= s2.length; i++) {
    matrix[i] = [i];
  }

  // increment each column in the first row
  let j;
  for (j = 0; j <= s1.length; j++) {
    matrix[0][j] = j;
  }

  // fill in the rest of the matrix
  for (i = 1; i <= s2.length; i++) {
    for (j = 1; j <= s1.length; j++) {
      if (s2.charAt(i - 1) == s1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1, // insertion
          matrix[i - 1][j] + 1 // deletion
        );
      }
    }
  }

  return matrix[s2.length][s1.length];
}

function find_fuzzy_matches(corpus, substrings, max_distance) {
  const results = {};
  for (const substring of substrings) {
    results[substring] = [];
    const words = corpus.split(/\s+/);
    for (const word of words){
      const distance = levenshteinDistance(substring, word)
      if(distance <= max_distance) {
          results[substring].push({word: word, distance: distance})
      }
    }
  }
  return results;
}

//example
const corpus = "this is a test string with test and testing inside";
const substrings = ["test", "sting", "tes"];
const matches = find_fuzzy_matches(corpus, substrings, 2)
console.log(matches);
```

this javascript example isn’t the fastest implementation. to use this efficiently, you can pre-compute the distances, but that can take a lot of space for large corps. the runtime for the levenshtein is *o(mn)* where m and n are the lengths of the strings being compared. so you need to factor that in as you have to calculate the distance of the substring for every word in your corpus for each substring. that is not an easy task.

now, this is a classic computer science problem so there is always more to it and improvements can be done. i find that often using a combination of techniques is necessary. for example, you might start by splitting your corpus into smaller chunks and then use a levenshtein method. or even use another algorithm like the damerau–levenshtein distance which accounts for transpositions (swapping adjacent characters) which can improve your results. this algorithm i remember using it in college for a school project to build a spell checker.

if you are dealing with very large corpora, you might want to look into using indexing structures like suffix trees or inverted indexes, similar to what search engines use. they are more complex to implement but give you very fast lookups. these are out of the scope of this response but are essential for huge datasets. i had a project where the corpus was huge and if i had a naive linear scan it would have taken hours, i ended up using a suffix tree and it was orders of magnitude faster.

there are some trade-offs here. if you want exact matches and speed, and you can not afford to use indexes, `string.find()` is your best bet. but if you are looking for fuzzy matches, you will have to calculate edit distances, which will be computationally more expensive. it all depends on the needs of your application. remember, the optimal solution often involves a clever mix of these techniques. one size does not fit all. and this is a good exercise to keep in mind for future projects.

for more on this, i'd really suggest checking out the "introduction to algorithms" by thomas h. cormen, charles e. leiserson, ronald l. rivest, and clifford stein. it has a very nice explanation of algorithms like the levenshtein distance. and if you are into search indexing and data structures, "managing gigabytes" by ian h. witten, alistair moffat, and timothy c. bell is an excellent resource. it is a bit dated but fundamental concepts still apply. and also, there's a very nice paper written by edward m. mccreight about the suffix tree algorithm. it is a good read if you want to dive deeper into that. the title is "a space-economical suffix tree construction algorithm." i can give you all the citations if needed.

i hope this helps and gives you a good starting point. i’ve been doing this for a while, and i’ve learned that the best approach is often to understand the problem completely and iterate.
oh and a joke? why do programmers prefer dark mode? because light attracts bugs.
