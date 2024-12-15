---
title: "How to clean up ngrams?"
date: "2024-12-15"
id: "how-to-clean-up-ngrams"
---

hey, i've been there, staring at a pile of ngrams that look like they've been through a blender. it’s a common problem, especially when you're dealing with messy text data. let me share what i've learned from a few scrapes i’ve gotten into, specifically how i tackled the cleanup.

first off, when you say "clean up," i assume we're talking about getting rid of those ngrams that are just noise, the ones that won't actually help with your analysis. maybe they're too common, too short, contain punctuation or unwanted symbols, or even are just plain irrelevant to what you are doing. i once worked on a project where we were trying to do sentiment analysis of tweets about a new phone launch, and believe me, the uncleaned ngrams were... creative. things like "the the the," "!!!," and a whole lot of URLs were clogging up the system. not pretty. 

the first thing i always do is establish a baseline. what are the most common ngrams, without any cleaning? this gives a very clear picture of how much work is ahead. i always use python for this because it is my go-to and for me it is the fastest, so i’ll give you an example code snippet to illustrate. it uses the `nltk` library which you might be familiar with. if not, it’s worth getting acquainted with as it has a lot of text processing tools. you might want to check the “natural language processing with python” book, by steven bird, ewan klein and edward loper, if you plan to delve more into nlp (natural language processing).

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

def get_top_ngrams(text, n, top_k):
    tokens = word_tokenize(text.lower())
    n_grams = ngrams(tokens, n)
    ngram_counts = Counter(n_grams)
    return ngram_counts.most_common(top_k)

text = "this is a test text. this is another test text! and this test again. this, this, this! test test test."
top_2grams = get_top_ngrams(text, 2, 10)
print(f"top 2-grams without cleaning: {top_2grams}")

```

this code will generate uncleaned ngram frequency and is useful to see what needs cleaning. the code simply tokenizes the text, creates ngrams of the size that you want (here, it's bigrams), and then counts them. the `.most_common()` part at the end gives you the top `k` most frequent ones. if you run it, you will quickly see that there are issues that you need to address, like punctuation or words that are too common.

now, regarding the actual cleaning, there are a few stages i usually go through, and they usually solve most problems.

1.  **lowercasing:** like in the example code, i always convert everything to lowercase. otherwise, “the” and “The” are going to be treated as different ngrams, which is usually not what you want. this is a quick win that can dramatically reduce the size of your ngram set.

2.  **punctuation and special character removal:** i often use regular expressions for this. anything that isn't a letter or number usually gets tossed out. this also helps with cleaning symbols that might be useless to you. i have also seen, in some occasions, people use regex to separate words that are glued, example: "helloworld" becoming "hello world" before generating the ngrams.

3.  **stop word removal:** this is crucial. stop words are common words like "the," "is," and "a" that don't carry much meaning on their own. they can completely skew your ngram analysis if they are not eliminated. `nltk` has a built-in list of stop words, which i often use but you can use other libraries or construct your own if you need it. you should see the documentation of `nltk` it is extremely well written and can help you a lot if you are not very used to the library.

4.  **min and max length filter:** it is very common that ngrams that are too short or too long are not important to you. usually ngrams of size one are too noisy, while ngrams of 6,7, 8 or more may be so specific that don't actually have a meaning in the context you are studying. depending on your task you might want to filter them to the appropriate size. 

here is an example of code with some of the cleaning steps applied:

```python
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def get_top_ngrams_cleaned(text, n, top_k, min_len=2, max_len=10):
    tokens = clean_text(text)
    n_grams = ngrams(tokens, n)
    ngram_counts = Counter(ngrams for ngrams in n_grams if min_len <= len(ngrams) <= max_len)
    return ngram_counts.most_common(top_k)


text = "this is a test text. this is another test text! and this test again. this, this, this! test test test."
top_2grams_cleaned = get_top_ngrams_cleaned(text, 2, 10)
print(f"top 2-grams after cleaning: {top_2grams_cleaned}")
```

notice i imported the `stopwords` dataset from `nltk` and used it to filter out stop words from the tokens after cleaning the text. now, the ngrams should be cleaner. the regular expression, `re.sub(r'[^a-z0-9\s]', '', text)` clears anything that is not a number, a letter or a space.

i did once worked on a project where we were extracting keywords from product reviews. we had this ridiculous ngram "this product not good," and we needed to understand why. turns out, people were using that kind of "not good" negative construction a lot, but without proper cleaning, it would get lost in the noise. the moment we implemented proper cleaning techniques, our analysis became way more useful.

another problem i have faced in the past is when you have different forms of the same word. for example, "running" and "ran". ideally you want to consider those two words the same so the ngrams are aggregated, that is why is so useful to use a process named lemmatization, that reduces a word to its base form. `nltk` also provides lemmatization tools, and it should be used as a pre-processing step before ngram extraction. also you should study the "speech and language processing" book by daniel jurafsky and james h. martin if you want to know more about natural language processing.

now, regarding more advanced cleaning strategies i always recommend to have in mind:

1.  **frequency based filtering:** sometimes, very frequent ngrams are not useful (as shown in the code example with stop words) and very infrequent ngrams might also not be. if your corpus is large, it might make sense to filter ngrams based on their frequency counts. i have worked on systems that used to calculate the tf-idf scores for each ngrams and then filter by these scores. it can be very useful and improve performance a lot, specially when dealing with big corpora of text. tf-idf, if you don't know it, is a statistic that can measure how relevant a ngram is in a document comparing with a collection of documents.

2.  **domain specific cleaning:** if you are working on a very specific domain, you might need to add more specific cleaning rules. for example, in the twitter sentiment analysis project, we had to remove hashtags, usernames and also had to do some basic slang to english conversion. so there is no one size fits all cleaning strategy.

3.  **handling misspellings:** when i had to work with social media data, it was a nightmare. some people can't spell properly. we used a spell checker library that corrected simple misspellings, but this process has limits since it depends of the dictionary used. you can also train language models to fix typos, but this is a very advanced topic.

4.  **context-aware cleaning:** this is another advanced topic that can be very useful, specially if the context of the ngrams matters. there are some recent deep learning based models that can encode the context of the ngrams and remove the ones that are not relevant. i would recommend you to start simpler and experiment more with what is already available, but keep this in mind as you progress.

let's get to another example code snippet with all of this:

```python
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

def get_top_ngrams_advanced(text, n, top_k, min_len=2, max_len=10, min_freq=2):
    tokens = clean_text_advanced(text)
    n_grams = ngrams(tokens, n)
    ngram_counts = Counter(ngrams for ngrams in n_grams if min_len <= len(ngrams) <= max_len)
    filtered_ngrams = [ngram for ngram, count in ngram_counts.items() if count >= min_freq]
    return Counter(filtered_ngrams).most_common(top_k)

text = "this is a test text. this is another test text! and this test again. this, this, this! test test test. running runs run"
top_2grams_advanced = get_top_ngrams_advanced(text, 2, 10, min_freq=2)
print(f"top 2-grams after advanced cleaning: {top_2grams_advanced}")
```

notice that now i also added the lemmatizer, and included a minimum frequency filter. now running, runs and run will all be lemmatized as "run". also the frequency filtering makes sure you only get ngrams that appears at least 2 times in your corpus. i believe that it is always good to have frequency filters in place.

finally, always remember to iterate. you should not start with all the cleaning steps all at once. start simple, analyze the results, and then add more complex techniques as needed. it is common that the perfect strategy depends on the specific dataset and task at hand, and only experience can lead you to the best solution. also, there is a lot of theory in the background so studying is always a good idea if you want to master the subject. so, if you are doing nlp, i hope that my explanation and experience will save you some headaches, and that you find this helpful. have a great day.
