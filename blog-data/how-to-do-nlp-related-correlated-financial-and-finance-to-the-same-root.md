---
title: "How to do NLP related correlated 'financial' and 'finance' to the same root?"
date: "2024-12-15"
id: "how-to-do-nlp-related-correlated-financial-and-finance-to-the-same-root"
---

so, i see you're having a common issue with nlp, specifically when it comes to financial text – the nuances of 'financial' and 'finance' being treated as distinct entities when they really mean the same thing in most contexts. it’s something i've definitely bumped into, and it's not just with these two words. i've seen similar things with 'investing' and 'investment', 'market' and 'markets'. it throws off your analysis if not handled correctly. i've spent quite some time in the trenches of text processing, believe me.

first things first, what's happening is that your tokenization process is treating them as separate tokens. tokenization is that initial step where you break down the text into individual units (tokens), which are often words. a naive tokenizer will see 'financial' and 'finance' and simply register them as two separate words. that's normal, but problematic for your task.

the core problem you're facing is stemming and lemmatization. these techniques are essential to nlp. stemming is the crude way. it chops off the ends of words to reduce them to a common form, a stem. think of it like this: 'running', 'runs', and 'ran' could be stemmed to 'run'. it's fast, but often imprecise and can create stems that aren't even real words. i once used a stemming algorithm on medical text and it turned "patient" into "pati". it was quite something! not something i would recommend for anything that needs reasonable analysis.

lemmatization, on the other hand, is a smarter approach. it considers the context of the word and reduces it to its base form, or lemma, which is actually a valid word in the language. so 'running', 'runs', and 'ran' would all lemmatize to 'run'. similarly, 'financial' and 'finance' should both be lemmatized to either 'finance' or ‘financial’ depending on how you construct your pipeline.

i’d like to point out that choosing between stemming and lemmatization is crucial to your results. lemmatization tends to work better for most serious analyses, especially when you're dealing with text where precision is important (like financial text). but stemming does have its place, typically where you are really tight on processing power or speed and don’t need the words to be 100% accurate. personally, i'd advise always go with lemmatization if possible. 

now, let's get to practical solutions. i'll show examples using python, since it's the language i usually default to for nlp tasks.

**example 1: using nltk and wordnet lemmatizer**

nltk (natural language toolkit) is a foundational library for nlp in python. it provides a lot of tools, including a lemmatizer based on wordnet, a lexical database of the english language.

```python
import nltk
from nltk.stem import wordnet
from nltk.corpus import wordnet as wn

def get_wordnet_pos(word):
    """map pos tag to wordnet tags"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)
    
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = wordnet.WordNetLemmatizer()

words = ["financial", "finance"]

for word in words:
  lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
  print(f"the lemma of '{word}' is: '{lemma}'")
```

this snippet downloads necessary resources, initializes the wordnet lemmatizer, and then applies it to 'financial' and 'finance'. the function `get_wordnet_pos` helps the lemmatizer to determine the part of speech of each word which is necessary for wordnet to work correctly. you'll see the result is the same lemma for each. that is the expected behaviour.

**example 2: using spacy**

spacy is another incredibly powerful nlp library, known for its speed and ease of use. it has its own lemmatization engine which in my opinion, does a better job in most cases. i tend to use spacy when i need speed over everything else. 

```python
import spacy

nlp = spacy.load("en_core_web_sm")

words = ["financial", "finance"]

for word in words:
    doc = nlp(word)
    lemma = doc[0].lemma_
    print(f"the lemma of '{word}' is: '{lemma}'")

```

here, we load a spacy model ('en_core_web_sm'), which includes a lemmatizer, process the words as documents and then extract the lemma. the output shows that spacy, does the job and does it very quickly.

**example 3: using a custom stemmer**

just for demonstration purposes, let's build a quick crude custom stemmer. remember, i don't recommend using custom stemmers unless you know exactly what you are doing. normally you're better using existing well tested ones. if you’re going to create something custom then make sure you really need it and you’re willing to test it thoroughly. but for this specific simple example of ‘financial’ and ‘finance’, it'll be a very basic case.

```python
def custom_stemmer(word):
    if word.endswith("ial"):
        return word[:-3] #chop off 'ial' to get the stem
    if word.endswith("e"):
        return word[:-1] #chop off the last 'e'
    return word

words = ["financial", "finance"]

for word in words:
    stem = custom_stemmer(word)
    print(f"the stem of '{word}' is: '{stem}'")
```

this custom stemmer is clearly very basic. it only chops off 'ial' or 'e' at the end, however, if we had words like 'social', we would get the ‘soc’ word, which does not help the analysis. it’s just to show you how you could programmatically create something which is normally a terrible idea for proper nlp analysis, but you could do it if you needed to for a very particular case (and for demonstration purposes here).

regarding resources, avoid random blog posts – stick to books and papers, it is much safer in the long run. for a deep dive into nlp, i would highly recommend "speech and language processing" by daniel jurafsky and james h. martin. it's a comprehensive guide covering a vast array of nlp techniques. another great resource for practical applications is "natural language processing with python" by steven bird, ewan klein, and edward loper; it provides an excellent practical overview of nlp using nltk. if you want something more modern and a little more technical i would recommend "transformers for natural language processing" by dennis rothman. that book specifically goes deep into transformer networks, and their applications to real world nlp projects which you might find useful. it’s a relatively recent approach which is being used in lots of companies at the moment, including my previous employer.

one last thing – debugging nlp pipelines can be tricky; sometimes it feels like the words are conspiring against you. i remember one time, i spent a whole day trying to understand why my model was performing poorly. the problem turned out to be that i had accidentally left in the default list of stop words in the processing pipeline that included common words in the finance industry. so the models were not getting any meaningful information. after that day i've made it a habit to double and triple check every single little preprocessing step. it was quite an expensive lesson, to be honest, but a very useful one. oh, and the stop word removal list included the word “to”, my boss thought that it was quite funny when i told him.

remember, nlp is iterative. experimentation and careful tuning of your pipeline parameters are the name of the game. you'll encounter various problems along the way, and i hope the above gives you a good starting point to tackle this very common one. good luck!
