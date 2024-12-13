---
title: "pos tagging using brown tag set in nltk?"
date: "2024-12-13"
id: "pos-tagging-using-brown-tag-set-in-nltk"
---

Okay so you're asking about part-of-speech tagging using the Brown tagset in NLTK yeah I've been there done that got the t-shirt probably had more than a few late nights debugging taggers back in the day trust me it's a common road to walk down for any natural language processing enthusiast

Let's break this down real simple and see what we're talking about

First off NLTK is your friend if you're dipping your toes into NLP it's like the Swiss army knife but for text processing If you haven't got it installed well just google it its super easy you should be able to install it with pip no biggie once you have that you're good to go for this particular problem

The Brown tagset is a specific way of labeling words in text with their grammatical role think nouns verbs adjectives adverbs etc It's more granular than your typical simplified tag sets which is both a blessing and a curse depending on what you’re trying to do. The Brown corpus a super old corpus used the Brown tagset and so its kinda become the gold standard for certain use cases even though more modern ones exist

Now when you want to use NLTK to tag text using the Brown tagset well NLTK has pre trained taggers that do most of the heavy lifting for you that is a very big plus they're trained on tagged data like the Brown corpus which makes them quite accurate for most common English text. You don’t have to train your own model unless you are doing some super niche stuff I would not recommend that unless you absolutely know what you are doing

Let’s get to the code bit it is the part I personally prefer the most anyway First up is how to load the nltk and how to initialize the tagger that we will use for our use case

```python
import nltk
from nltk.tag import UnigramTagger
from nltk.corpus import brown
nltk.download('brown')
nltk.download('punkt') # You need this for tokenization
nltk.download('averaged_perceptron_tagger') # Alternative POS tagger model


def train_brown_tagger():

    brown_tagged_sents = brown.tagged_sents(categories='news')
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]

    unigram_tagger = UnigramTagger(train_sents)
    accuracy = unigram_tagger.evaluate(test_sents)

    print(f"Unigram tagger accuracy: {accuracy}")

    return unigram_tagger


brown_tagger = train_brown_tagger()
```
Okay so this first snippet trains the unigram tagger which is the baseline for many POS tagging tasks. I've seen some folk try to start with RNNs or transformers for this problem but they overcomplicate it unigram taggers are a good start they are fast and work well most of the time anyway So we import the necessary modules train a basic model and evaluate it and then return the tagger itself You can also try other taggers. There are a bunch available in NLTK its like a candy store you have the `averaged_perceptron_tagger` which is generally better so I included a download for it and the `UnigramTagger` like the one we used above for some reason people like the UnigramTagger as it is easy to understand

Next up let's see how to use the tagger to actually tag some text. This is the fun part where the magic happens

```python
def tag_text(text, tagger):

  tokens = nltk.word_tokenize(text)
  tagged_tokens = tagger.tag(tokens)

  return tagged_tokens

# Example usage
sample_text = "The quick brown fox jumps over the lazy dog."
tagged_text = tag_text(sample_text, brown_tagger)
print(tagged_text)

```
Simple right so we define a function called `tag_text` that takes some text and the tagger we created before we then tokenize the text into words using `nltk.word_tokenize`. I know some purists are gonna tell you to use your own tokenizers but just lets use this for simplicity sake its faster you just have to have all of your nltk downloads ready and then use the tagger to tag those tokens

This is where you would use the code like this for your application. But just as an aside the real world is super messy here. Sometimes a word can have more than one meaning so sometimes you will encounter the word “bank” as a noun as in `river bank` or as a verb `to bank`. Now with this tagger the tagger will use the most common use of that word in the training dataset so it is not always the most accurate for the specific case you need. I mean for that to be more accurate you would need context from the sentence

Let's do another example that uses a different tagger and also shows some slightly more advanced usage so you can see the versatility in the tool we are using

```python
def advanced_tag_text(text):
    tokens = nltk.word_tokenize(text)
    # Using the NLTK's default tagger which is generally more accurate than a unigram tagger
    tagged_tokens = nltk.pos_tag(tokens)
    # Mapping Penn Treebank tags to Brown tags
    brown_tags = [(word, map_penn_to_brown(tag)) for word, tag in tagged_tokens]
    return brown_tags

def map_penn_to_brown(penn_tag):
    """Maps Penn Treebank tags to Brown tags."""
    mapping = {
      'CC': 'CC',    # Coordinating conjunction
      'CD': 'CD',    # Cardinal number
      'DT': 'DT',    # Determiner
      'EX': 'EX',    # Existential there
      'FW': 'FW',    # Foreign word
      'IN': 'IN',    # Preposition or subordinating conjunction
      'JJ': 'JJ',    # Adjective
      'JJR': 'JJR',  # Adjective, comparative
      'JJS': 'JJS',  # Adjective, superlative
      'LS': 'LS',    # List item marker
      'MD': 'MD',    # Modal
      'NN': 'NN',    # Noun, singular or mass
      'NNS': 'NNS',  # Noun, plural
      'NNP': 'NP',   # Proper noun, singular
      'NNPS': 'NPS', # Proper noun, plural
      'PDT': 'PDT',  # Predeterminer
      'POS': 'POS',  # Possessive ending
      'PRP': 'PP$',  # Personal pronoun
      'PRP$': 'PP$', # Possessive pronoun
      'RB': 'RB',    # Adverb
      'RBR': 'RBR',  # Adverb, comparative
      'RBS': 'RBS',  # Adverb, superlative
      'RP': 'RP',    # Particle
      'SYM': 'SYM',  # Symbol
      'TO': 'TO',    # to
      'UH': 'UH',    # Interjection
      'VB': 'VB',    # Verb, base form
      'VBD': 'VBD',  # Verb, past tense
      'VBG': 'VBG',  # Verb, gerund or present participle
      'VBN': 'VBN',  # Verb, past participle
      'VBP': 'VBP',  # Verb, non-3rd person singular present
      'VBZ': 'VBZ',  # Verb, 3rd person singular present
      'WDT': 'WDT',  # Wh-determiner
      'WP': 'WP',    # Wh-pronoun
      'WP$': 'WP$',  # Possessive wh-pronoun
      'WRB': 'WRB',  # Wh-adverb
        '.': '.',
         ',': ',',
         ':': ':',
         '(': '(',
         ')': ')',
         '\'\'':'"',
         '``':'"'
    }
    return mapping.get(penn_tag, penn_tag) # default to no mapping if needed

sample_text = "This is an advanced example, you see the tags are mapped differently!"
tagged_text_advanced = advanced_tag_text(sample_text)
print(tagged_text_advanced)
```

So this example is a bit more involved. We first use the `nltk.pos_tag` which uses the Penn Treebank tagset now you might say hey I wanted Brown tags I thought you’re an experienced stackoverflow user well here is the gotcha, the default more accurate tagger in NLTK uses the Penn Treebank tagset so what you have to do to get Brown tag set is that you need to map them. I did that for you in that `map_penn_to_brown` function. Now the mapping might not be 100% perfect but most of the common tags are mapped this is a fairly standard approach in NLP when you need a different tag set then what is given

Now for the gotcha part that I am sure many of you probably would be wondering about and I definitely have experienced in the past. Training your own tagger takes time and a good chunk of annotated data. That is why using the pre trained ones is generally preferred unless you have a very specific domain which requires your own tagger. Now how you train your own tagger from scratch is a very long process that requires a lot more than what we are talking about here but I can provide some material for you to read.

I am a big fan of the “Speech and Language Processing” by Dan Jurafsky and James H. Martin, it is a great resource if you really want to understand the theory behind these algorithms there is also a free draft of it online just look for it it is not that hard.

Another great source would be Christopher D. Manning's work he has published a lot about the use of machine learning in NLP and he is a very famous professor. He also has many publicly available courses you can look at just google his name you should be able to find his work very easily.

Now for some practical tips when doing POS tagging. Always clean your text before you tag it. Remove unwanted characters like random symbols or HTML tags this can seriously mess up your tagging. Also remember that text data is super messy it will always have typos and other weird stuff always try to understand the data you are working with before doing any heavy processing. And always evaluate your taggers carefully. Just because you have an accuracy score does not mean your model will work well for your use case

And lastly remember to have fun while doing NLP. I know it can get annoying sometimes but it is a super interesting field. I once spend a whole week debugging a really silly issue in one of my taggers turned out that some encoding issues in the text that I was trying to tag I mean it was that simple. But that is how coding goes sometimes. It is just sometimes the weirdest things pop up just when you least expect it and make you wonder why this was not as easy as it should have been. It's like trying to debug a Javascript web page it just never ends.

I hope this helps and you are able to tag your data more effectively now. Remember if you have a problem dont bang your head on the keyboard for too long sometimes taking a break can really help you with your issues. Good luck with all of your POS tagging endeavors!
