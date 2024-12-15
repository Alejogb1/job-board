---
title: "How to select the best topic model based on the coherence score?"
date: "2024-12-15"
id: "how-to-select-the-best-topic-model-based-on-the-coherence-score"
---

so, you're looking into picking the best topic model using coherence scores, cool. i've been down that rabbit hole more times than i care to count. let me walk you through what i've learned, mostly from banging my head against the wall until something finally worked, haha.

first, let's get this straight. we're talking about topic models, which means we're working with unstructured text data, probably a bunch of documents that we're trying to make sense of. models like latent dirichlet allocation (lda) and non-negative matrix factorization (nmf) are the usual suspects. the goal is to find underlying topics in that text, grouping words together that tend to appear in similar contexts.

now, coherence is a metric that tries to quantify how well these discovered topics hang together. a high coherence score means that the words within a topic are semantically related, and thus that topic is likely to be meaningful to us humans. a low score, on the other hand, suggests a topic that's just a random jumble of words, probably not something useful.

so, how do we actually use coherence to pick a good model? it's not a magic bullet, and it requires careful experimentation. you see the coherence score by itself has limits, because it depends on other parameters and data.

the typical process i've found that works is:

1. **model training with variations:** train multiple topic models, varying hyperparameters like the number of topics and parameters specific to the particular topic model algorithm. for lda, things like `alpha` and `beta` are important, for nmf, the number of components and the initialization method, and the parameters of other topic models like hdp (hierarchical dirichlet process), are relevant. the key is to generate a wide variety of results to analyze and compare. don’t be afraid to try a big grid of parameter combinations, even ones that seem a little out there. i once thought the results were not affected by initialization for one topic model, but after a proper grid search it became obvious it is important. it depends on the specific type of initialization method, and this is how you discover these things by exploring all the options.

2. **calculate coherence for each model:** for each trained model, calculate the coherence score for each identified topic, and then calculate the average coherence score across all topics. this usually involves using libraries like `gensim` or `scikit-learn` (with some additional calculations or external libraries for coherence).

3. **compare and select:** finally, compare the average coherence scores of all models. the model with the highest score is usually, but not always, a good candidate for being the 'best'. it also depends on your specific objective and what you are trying to achieve. sometimes a slightly lower score with more interpretable topics is preferred.

here's how this would translate to python code, using `gensim` for lda, and some other standard libraries.

```python
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# let's imagine you have your documents in a list called 'documents'
# assume your documents are already pre-processed text in lowercase strings
# example: documents = ["this is a first document", "second doc here"]

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

def train_lda_model(documents, num_topics, passes=10, iterations=50, random_state=100):
  processed_docs = [preprocess_text(doc) for doc in documents]
  dictionary = Dictionary(processed_docs)
  corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
  lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, iterations=iterations, random_state=random_state)
  return lda_model, dictionary, corpus


def calculate_coherence(lda_model, dictionary, corpus, processed_docs):
  coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  return coherence_lda


# some example data
documents = [
    "this is a document about machine learning algorithms",
    "another document about deep neural networks and backpropagation",
    "natural language processing is used for text classification",
    "computer vision algorithms deal with image recognition",
    "how do machine learning models get trained?",
    "the fundamentals of statistical inference",
    "bayesian methods are useful in many cases",
    "reinforcement learning agents learn through interaction",
    "this book is about artificial intelligence and its applications",
    "the future of ai is a hot topic in tech industry"
]

# Example 1: Grid search with number of topics, for LDA
num_topics_values = [3, 5, 7, 10]
coherence_scores = {}
processed_docs = [preprocess_text(doc) for doc in documents] # needed only once

for num_topics in num_topics_values:
  lda_model, dictionary, corpus = train_lda_model(documents, num_topics)
  coherence = calculate_coherence(lda_model, dictionary, corpus, processed_docs)
  coherence_scores[num_topics] = coherence
  print(f'number of topics {num_topics}, coherence score is {coherence}')

best_num_topics = max(coherence_scores, key=coherence_scores.get)
print(f'best number of topics based on coherence is {best_num_topics}')
```

in this first example we do a simple grid search, varying just the number of topics for lda. we initialize the lda model, calculate coherence, and select the number of topics that yields the best score. you would have to run this example to see which parameter works best, because it depends on the data.

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel

def train_nmf_model(documents, n_components, init='nndsvda', solver='cd', random_state=100):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    nmf_model = NMF(n_components=n_components, init=init, solver=solver, random_state=random_state)
    nmf_model.fit(tfidf_matrix)
    return nmf_model, vectorizer

def calculate_nmf_coherence(nmf_model, vectorizer, documents):
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[-10:][::-1] # we take the top 10 words
        topic_words.append([feature_names[i] for i in top_words_idx])

    dictionary = Dictionary(topic_words)
    corpus = [dictionary.doc2bow(doc) for doc in topic_words]
    coherence_model_nmf = CoherenceModel(topics=topic_words, dictionary=dictionary, texts=topic_words, coherence='c_v')
    return coherence_model_nmf.get_coherence()


# Example 2: grid search for nmf
n_components_values = [3, 5, 7, 10]
init_values = ['nndsvda', 'nndsvd', 'random']
coherence_scores_nmf = {}
for n_components in n_components_values:
  for init in init_values:
    nmf_model, vectorizer = train_nmf_model(documents, n_components, init=init)
    coherence = calculate_nmf_coherence(nmf_model, vectorizer, documents)
    key = f'{n_components}_{init}'
    coherence_scores_nmf[key] = coherence
    print(f'number of components: {n_components}, init: {init}, coherence score is {coherence}')


best_combination = max(coherence_scores_nmf, key=coherence_scores_nmf.get)
print(f'best hyperparameter combination based on coherence is {best_combination}')
```
this second code snippet does grid search for nmf, varying the number of components and the initialization method. we are calculating the coherence using `gensim` as we did in the first example.

```python
# Example 3: using umap for dimensionality reduction, then clustering and selecting topics
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


def train_umap_clustering(documents, n_neighbors=15, min_dist=0.1, min_cluster_size=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(tfidf_matrix.toarray())

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)
    return cluster_labels, vectorizer, tfidf_matrix

def calculate_cluster_coherence(cluster_labels, vectorizer, documents):
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label == -1: #skip noise label
            continue
        cluster_docs = [documents[i] for i, l in enumerate(cluster_labels) if l==label]
        vectorizer_cluster = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix_cluster = vectorizer_cluster.fit_transform(cluster_docs)
        feature_names_cluster = vectorizer_cluster.get_feature_names_out()
        sum_tfidf = tfidf_matrix_cluster.sum(axis=0)
        top_words_idx = np.argsort(sum_tfidf.A1)[-10:][::-1]
        topic_words.append([feature_names_cluster[i] for i in top_words_idx])
    if not topic_words: # if there are no topics
        return 0
    dictionary = Dictionary(topic_words)
    corpus = [dictionary.doc2bow(doc) for doc in topic_words]
    coherence_model = CoherenceModel(topics=topic_words, dictionary=dictionary, texts=topic_words, coherence='c_v')
    return coherence_model.get_coherence()



# Example 3: umap + hdbscan
n_neighbors_values = [10, 15, 20]
min_cluster_size_values = [3, 5, 7]
coherence_scores_umap = {}
for n_neighbors in n_neighbors_values:
  for min_cluster_size in min_cluster_size_values:
      cluster_labels, vectorizer, tfidf_matrix = train_umap_clustering(documents, n_neighbors, min_cluster_size=min_cluster_size)
      coherence = calculate_cluster_coherence(cluster_labels, vectorizer, documents)
      key = f'{n_neighbors}_{min_cluster_size}'
      coherence_scores_umap[key] = coherence
      print(f'n_neighbors {n_neighbors}, min_cluster_size {min_cluster_size}, coherence score is {coherence}')
best_combination_umap = max(coherence_scores_umap, key=coherence_scores_umap.get)
print(f'best hyperparameter combination based on coherence for umap clustering is {best_combination_umap}')
```
finally, in this third code snippet, we use a different approach: we first perform dimensionality reduction with `umap`, then use `hdbscan` to cluster the reduced data, treating the clusters as topics. we calculate the coherence score as in the other examples. this is a less common approach compared to lda and nmf but it is another technique used in the field.

now a few extra tips i picked up over the years:

*   **pre-processing is key:** the quality of your topic models and their coherence scores depend heavily on how you clean and pre-process your text. remove punctuation, lowercase everything, handle stop words, and maybe even use stemming or lemmatization. i cannot emphasize this enough; i have spent way too much time trying to optimize my models, just to find out my raw data was terrible.

*   **not all coherence scores are created equal:** the `c_v` score, which i used above, is common. but there are other options like `u_mass`. try a few of them, and see which aligns better with your specific data and the way you understand topic quality.
*   **human evaluation is crucial:** coherence scores are useful, but they aren’t the whole story. at some point, you need to actually *look* at the topics and see if they make sense. sometimes a slightly lower coherence score will produce topics that are more interpretable and useful. i've had cases where the highest score was just a topic made of nonsense, and i ended up choosing a model with a lower coherence score, but more meaningful topics.
*   **experiment with different algorithms:** lda and nmf are common, but they're not the only options. consider other algorithms like hdp, or even approaches based on dimensionality reduction like the umap plus clustering method i showed. sometimes, the standard techniques just don't cut it and you need something a bit different.

*   **document everything** i'm serious! keep track of all the different models you trained, the parameters you used, and their coherence scores. it will save you a lot of time and headaches in the long run. a proper notebook is always helpful in the process.

for further reading, i'd suggest checking out "probabilistic topic models" by david m. blei. it gives you the theoretical basis of the common topic models used. also, read the original papers for lda, and nmf; and for umap and hdbscan as well. they’ll give you the nuts and bolts of how the algorithms work. and finally for coherence look into the original papers. don't just rely on online articles, go to the source material.

and always remember, the 'best' model depends on what you need it for. it's a combination of quantitative metrics like coherence, but also a qualitative evaluation of the topics themselves. don’t get trapped in a number, there is no magic score that tells you everything.
