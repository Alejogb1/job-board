---
title: "How would I train a dataset to find synonyms of words that are related to extracurricular activities in Python?"
date: "2024-12-15"
id: "how-would-i-train-a-dataset-to-find-synonyms-of-words-that-are-related-to-extracurricular-activities-in-python"
---

alright, so you're looking at building a model to find synonyms for words related to extracurricular activities, that's a pretty neat challenge. i've actually bumped into something similar back when i was trying to categorize user interests for this tiny app i was working on, think tinder, but for board game players. yeah, that was a fun project that went nowhere. it's tougher than it sounds when users start listing things like 'd&d,' 'miniature painting,' and 'competitive cheese tasting' - trying to group that stuff without explicit knowledge, it's not a walk in the park. so let me break down how i’d approach this problem, focusing on practical steps and some gotchas along the way.

first off, we need to decide what "synonym" means in your context. are we talking about words with very similar meaning like "soccer" and "football" or related terms like "guitar" and "band practice" that indicate a connection without being exactly the same? the approach will differ a little depending on the goal. i'll assume you want to capture semantic relations, meaning it doesn't necessarily need to be strict synonyms in a lexical sense like 'bad' and 'terrible'. rather, related words in a context of extracurricular activities.

we're in the realm of natural language processing (nlp) here, and in python there are a few packages that make this task way more manageable than it used to be. most notably, i recommend diving into `spacy` and `gensim`. they’re like the bread and butter for this kind of stuff.

let’s start by building a basic word embedding model with `gensim`. word embeddings are representations of words in a vector space, where words with similar meanings or usage are located closer together. it's like mapping the concepts in a high-dimensional space. cool stuff.

```python
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import string

# some sample text data containing your extracurricular activity related words
text_data = """
i love playing soccer with my friends in the park.
basketball is another sport i enjoy.
i also enjoy reading books in my free time.
i also attend choir practices on fridays.
volunteering at the local animal shelter is rewarding.
i am a member of the drama club and love acting.
my brother likes to do art and painting.
i love to attend coding clubs and learn new things.
i have a collection of stamps and enjoy philately.
i practice playing guitar daily.
i also play chess every weekend.
i like to write poems in my journal.
i also love going hiking.
"""

# preprocess the text to remove punctuation and lowercase
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    return tokens

# tokenize the sample data
tokenized_data = [preprocess(sentence) for sentence in text_data.split('\n') if sentence]

# train a word2vec model
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# get the top synonyms for a given word
def get_synonyms(word, n=5):
  try:
    synonyms = model.wv.most_similar(word, topn=n)
    return synonyms
  except KeyError:
    return f"the word '{word}' is not in the vocabulary."

# print the results
print(get_synonyms('soccer')) # find synonyms for soccer
print(get_synonyms('guitar')) # find synonyms for guitar
print(get_synonyms('coding')) # find synonyms for coding
```

this code snippet uses `gensim` to train a `word2vec` model on some fabricated data. notice the `preprocess` function, we remove all the punctuation and lowercase all the words, otherwise, you could have issues with matching words due to capitalization or punctuation. the `vector_size` controls the dimensionality of your word vectors, `window` refers to the number of words before and after a given word that will be considered when building the embedding, and `min_count` makes sure a word appears at least once in our vocabulary, it can be increased to ignore very rare words. the `get_synonyms` function does what it says on the tin: looks up the most similar words according to our vector space model. keep in mind the output depends on the training data provided. the more data you have, the better the model will perform. it is important to create a data set that is representative of your target domain, this means if you want to know the words related to "music," you need to have plenty of sentences about music-related extracurriculars.

however, this naive approach has a limitation. it only considers the words in its context within the data. for example, it doesn't really understand that "coding" is related to "programming" or that "stamp collecting" is related to "philately" unless they appear together often in your sample data. the model is only as good as the data. that's where more advanced techniques come in, like using pre-trained models and potentially fine-tuning them.

now, let's move into `spacy` for more advanced functionalities and a slightly different approach. spaCy has powerful tools for nlp tasks, including pre-trained models that understand word relationships pretty well out of the box.

```python
import spacy

nlp = spacy.load('en_core_web_lg')

def get_spacy_synonyms(word, n=5):
  doc = nlp(word)
  if not doc:
    return f"the word '{word}' is not in the vocabulary"

  word_vector = doc.vector
  if not word_vector.any():
        return f"the word '{word}' does not have a vector representation in this model"


  similar_words = []
  for token in nlp.vocab:
        if token.has_vector:
            similarity = word_vector.dot(token.vector)
            similar_words.append((token.text, similarity))

  similar_words = sorted(similar_words, key=lambda item: item[1], reverse=True)
  return [w for w,s in similar_words[1:n+1]]

# print the results
print(get_spacy_synonyms('soccer')) # find synonyms for soccer
print(get_spacy_synonyms('guitar')) # find synonyms for guitar
print(get_spacy_synonyms('coding')) # find synonyms for coding
```

here we’re using `spacy` and its large English model (`en_core_web_lg`). this model has been trained on massive amounts of text data so it usually has a good grasp of word semantics even if it hasn't seen the exact combination of words before. `spacy` does not return the values that generated the synonyms, but you can sort them yourself, as this example shows. i was trying to get into competitive coding a while back and i was using this model to expand my vocabulary. i was really trying to optimize all my algorithm implementations, it's crazy how hard you have to try to remove a single line in some of the problems. funny how sometimes the best optimization is not adding code.

but you don’t need to stop there. you can combine the strengths of both models or even go further and fine-tune a transformer model on your custom data, this is where things get very interesting. transformer models like bert or roberta are the state-of-the-art when dealing with natural language problems.

for instance you could use the `sentence-transformers` library, which provides pre-trained transformer models for sentence embeddings. these embeddings capture sentence-level meaning, which can be helpful if you want to find related activities that are described in multiple words. fine-tuning these large models, however, requires more data, specialized hardware like gpus and more expertise. it’s a much more complicated setup than the previous examples. but it's worth looking into if you are serious about it.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')

# sample extracurricular activities
extracurriculars = [
    "playing soccer",
    "playing football",
    "reading books",
    "choir practice",
    "volunteering at animal shelter",
    "drama club acting",
    "art and painting",
    "coding clubs",
    "stamp collecting",
    "playing guitar",
    "playing chess",
    "writing poems",
    "hiking",
    "doing pottery",
    "knitting scarves",
    "doing yoga"
]


def get_sentence_synonyms(query, n=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    embeddings = model.encode(extracurriculars, convert_to_tensor=True)

    # compute cosine similarity
    cos_sim = util.cos_sim(query_embedding, embeddings)

    # get the index of most similar sentences
    similar_indices = cos_sim.argsort(descending=True)[0][:n]

    return [extracurriculars[idx] for idx in similar_indices]

# print the results
print(get_sentence_synonyms('sports', n=3))
print(get_sentence_synonyms('music', n=3))
print(get_sentence_synonyms('creative writing', n=3))
```

this last example shows a small taste of what's possible with sentence transformers. `sentence-transformers` creates embeddings for the entire sentence instead of just the individual words. this is helpful when the query word is not a noun. in this example, i create embeddings for the sentences in the variable `extracurriculars` and then the `get_sentence_synonyms` finds the ones that are most similar to the query provided. i recommend going down this route if your query is not just a single word, but instead a sentence. keep in mind this is an example and may need more tuning and configuration to perform best.

if you want to go further, i suggest taking a look at some nlp books, like "speech and language processing" by daniel jurafsky and james h. martin or “natural language processing with python” by steven bird, ewan klein, and edward loper. they explain the core concepts in a more rigorous way than a blog post or a quick tutorial can.

in short, you have several options to build a system like this. if you are starting you can get a decent working model with `gensim` or `spacy`, and for more accurate and complex results, `sentence-transformers` should be your starting point. it all depends on what you really need. choose the tool that suits best your requirements and capabilities. just start coding, explore and see what works best for you.
