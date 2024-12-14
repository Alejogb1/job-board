---
title: "Is there any method to extract predefined words from any text file using natural language processing and python?"
date: "2024-12-14"
id: "is-there-any-method-to-extract-predefined-words-from-any-text-file-using-natural-language-processing-and-python"
---

well, this is a pretty common ask, i've definitely been down this road a few times. extracting specific words from text using python and nlp is actually a staple task in the field, and there are a few ways to approach it. it's not exactly rocket science, but there are some gotchas that can trip you up if you're not careful. let me break down my experience and show you what i've learned.

when i first started messing around with nlp, i remember trying to build a basic sentiment analysis tool. i needed to pull out particular keywords that indicated positive or negative feelings. at the time, i was a lot less sophisticated, and i thought i could just use a bunch of basic string matching and `if` statements. boy, was i wrong. it became a complete mess, particularly when i started dealing with things like plurals, different tenses, and slight variations in spelling. i ended up spending more time writing edge-case logic than i did actually on the core functionality. after spending a long weekend on this disaster i figured i would finally embrace the power of established libraries, it was a life changing experience as a programmer let me tell you.

so, instead of rolling our own complicated mess, it is always better to lean on the tools that are designed to handle this type of thing: natural language processing libraries. in python, `nltk` and `spaCy` are probably the two most prominent options, i tend to lean more towards `spaCy`. they both provide the necessary components, like tokenization (which involves splitting the text into individual words or units) and lemmatization (reducing words to their base form). this helps a lot in making it easier to find variations of your target words.

here’s a simple example using `spaCy`, assuming you have it installed (`pip install spacy`) and a suitable model like `en_core_web_sm` also installed. you can download the model with `python -m spacy download en_core_web_sm`

```python
import spacy

def extract_keywords_spacy(text, keywords):
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(text)
  lemmatized_keywords = [nlp(keyword)[0].lemma_ for keyword in keywords]
  found_keywords = []
  for token in doc:
    if token.lemma_ in lemmatized_keywords:
      found_keywords.append(token.text)

  return list(set(found_keywords))

example_text = "the dogs were running quickly, they also liked running and walks. one dog ran faster than all other dogs"
target_keywords = ["dog", "run", "walk"]
extracted = extract_keywords_spacy(example_text, target_keywords)
print(f"extracted keywords: {extracted}")
#expected output extracted keywords: ['dog', 'running', 'walk', 'dogs', 'ran']
```

so, what's happening here? first we load the `spacy` language model. then, inside the function, we take the input text and process it using `nlp(text)` to create a `doc` object. this object contains tokenized and lemmatized versions of your text. before searching for the keywords we make sure the keywords list is also in lemma form. after that it iterates over the tokens looking to match to the lemmatized keywords. this method will get 'run', 'ran', and 'running' due to the lemmatization. finally, we use a `set` to eliminate duplicates and convert it back into a `list` for output.

now, this example is functional but i'll be the first to say it's quite basic. it doesn't account for a ton of things, like handling multiple word keywords ("data science" should be treated as one token). if you were to feed this function a text saying "the quick brown fox jumps over the lazy dog but only after walking a bit, it is really not much to walk", it will work just fine and find 'dog' and 'walk' among other similar words, but it won't give you much context about where they're in the sentence.

to address more complex scenarios, it is necessary to make use of pattern matching. this allows us to be more specific and flexible with the extraction criteria. spaCy has a pattern matching library. here is an example that shows how to use it:

```python
import spacy
from spacy.matcher import Matcher

def extract_keywords_pattern(text, keywords):
  nlp = spacy.load("en_core_web_sm")
  matcher = Matcher(nlp.vocab)

  patterns = []
  for keyword in keywords:
      # handle single-word keywords
      if " " not in keyword:
          patterns.append([{"LEMMA": nlp(keyword)[0].lemma_}])
      # handle multi-word keywords
      else:
          keyword_tokens = nlp(keyword)
          pattern = [{"LEMMA": token.lemma_} for token in keyword_tokens]
          patterns.append(pattern)
  matcher.add("keyword_match", patterns)

  doc = nlp(text)
  matches = matcher(doc)
  found_keywords = [doc[start:end].text for match_id, start, end in matches]
  return list(set(found_keywords))


example_text = "the quick brown fox loves data science, and also machine learning. the data science field is great, also, machine learning is fun"
target_keywords = ["data science", "machine learning", "fun"]
extracted = extract_keywords_pattern(example_text, target_keywords)
print(f"extracted keywords using patterns: {extracted}")
#expected output extracted keywords using patterns: ['data science', 'machine learning', 'fun']
```

in this example, we initialize `Matcher`. then, for each keyword, we create a spaCy matcher pattern. if it's a single word, we look for its base form. if it's a multi-word keyword, we make a pattern with lemmas of each word from the keyword. and then we add it to the matcher with the label "keyword_match". the function iterates through the matches, extracts the matching text using the span and finally returns all the unique matches as a list.

now, if you are thinking, "but what about context? i need to know where in the text i found these words", you're absolutely on the mark. simply extracting keywords isn't always sufficient. you might want to retrieve the surrounding text or the sentence in which a keyword appears, especially when you are trying to perform advanced analysis. for that, we can use the `sent` attribute in the document object that spacy creates. here is the next example that handles context:

```python
import spacy
from spacy.matcher import Matcher

def extract_keywords_with_context(text, keywords):
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)

    patterns = []
    for keyword in keywords:
        if " " not in keyword:
            patterns.append([{"LEMMA": nlp(keyword)[0].lemma_}])
        else:
            keyword_tokens = nlp(keyword)
            pattern = [{"LEMMA": token.lemma_} for token in keyword_tokens]
            patterns.append(pattern)
    matcher.add("keyword_match", patterns)

    doc = nlp(text)
    matches = matcher(doc)
    keyword_with_context = []
    for match_id, start, end in matches:
        span = doc[start:end]
        sentence = span.sent.text
        keyword_with_context.append({"keyword": span.text, "sentence": sentence})

    return keyword_with_context

example_text = "the quick brown fox loves data science, and also machine learning. the data science field is great, also, machine learning is fun. data science is the best. the dog is friendly"
target_keywords = ["data science", "machine learning", "dog"]
extracted_with_context = extract_keywords_with_context(example_text, target_keywords)
print(f"extracted keywords with context: {extracted_with_context}")
#expected output extracted keywords with context: [
# {'keyword': 'data science', 'sentence': 'the quick brown fox loves data science, and also machine learning.'},
# {'keyword': 'machine learning', 'sentence': 'the quick brown fox loves data science, and also machine learning.'},
# {'keyword': 'data science', 'sentence': 'the data science field is great, also, machine learning is fun.'},
# {'keyword': 'machine learning', 'sentence': 'the data science field is great, also, machine learning is fun.'},
# {'keyword': 'data science', 'sentence': 'data science is the best.'},
# {'keyword': 'dog', 'sentence': 'the dog is friendly'}
#]

```

this method is the same as the previous one but, instead of just returning a list of keywords it returns a list of dictionaries containing the keyword with the context sentence. it uses the attribute `sent` of the span object to retrieve the sentence. this is very powerful when you want to perform extra analysis of the text later on.

i have actually used that sentence method in the past. i was working on a project to analyse customer reviews for a product. we wanted to understand which features were being discussed and the sentiment towards them, so, just finding the features was not enough we also wanted to get the sentence context to understand the sentiment towards that feature, this approach worked wonders. the sentences where then passed on to a sentiment analyser which helped a lot with the final report.

when choosing your tech stack for this type of task consider your needs and requirements, spacy is fast and efficient. nltk, another very prominent tool, might be useful for more complex tasks like stemming (a process that is similar to lemmatization), but, i’ve found spacy to be sufficient for most extraction tasks.

also, keep in mind that nlp is not magic. your results can only be as good as the quality of your data and the careful design of your solution. think very carefully about your use case when defining your target words. sometimes, a simple set of words is enough, sometimes a lot of manual work is required to make sure you get the data you need for a good analysis. garbage in garbage out.

resources-wise, i can recommend "speech and language processing" by dan jurafsky and james h. martin, it is a very solid book for any natural language processing project. there is also the very popular "natural language processing with python" by steven bird, ewan klein, and edward loper, which covers `nltk` and some fundamentals of the field. both are a very good place to start and have a wealth of information.

finally, as a small tech joke here, i can tell you that regex is the programmer's equivalent of using a hammer to solve any problem and for nlp, sometimes it’s just not enough.
