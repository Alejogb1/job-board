---
title: "How can I use similarity functions with words not in my vocabulary?"
date: "2024-12-23"
id: "how-can-i-use-similarity-functions-with-words-not-in-my-vocabulary"
---

Alright, let's tackle this one. I’ve definitely seen this challenge play out in a number of projects over the years. The scenario of having to compute similarity between words, some of which your system hasn't encountered before, is surprisingly common, especially when dealing with user-generated text or rapidly evolving vocabularies. It's not a showstopper, though. There are several well-established techniques we can leverage to get around this limitation.

The core problem, of course, stems from how we typically represent words in computational systems. If you're using a traditional vocabulary-based approach, such as one-hot encoding or even a simple integer index, an 'out-of-vocabulary' (oov) word simply doesn't have a vector representation. Therefore, a similarity computation becomes undefined. We need to move beyond this direct mapping approach and explore methods that offer more flexibility.

One effective strategy is to use character-level or subword-level embeddings. The idea here is to break words down into smaller, more fundamental units. Instead of treating a word like 'unconventional' as a single atomic unit, we might treat it as a sequence of characters ('u', 'n', 'c', 'o', 'n', ... ) or subwords (like 'un', 'con', 'vent', 'ion', 'al'). These sub-units are far more likely to be present in your training data, even if the entire word is novel. Then, the representation of an unseen word can be constructed from the learned embeddings of its constituent parts.

For example, let's assume we’re working with a character-level model. A word like ‘flabbergasted’ (assuming it's oov) can be decomposed into a sequence of character vectors. We can sum, average, or more sophisticatedly use recurrent neural networks (rnns) like lstms or grus to combine these embeddings into a single representation for the unseen word. Similarity can then be calculated using cosine similarity, or other appropriate metrics, on these constructed vectors.

Here's a conceptual illustration in python, using a simplified example:

```python
import numpy as np

# Assume a pre-trained character embedding model exists (simplified for demonstration)
# In reality, these would be large, learned vectors from actual training data.
character_embeddings = {
    'a': np.array([0.1, 0.2]),
    'b': np.array([0.3, 0.4]),
    'c': np.array([0.5, 0.6]),
    'd': np.array([0.7, 0.8]),
    'e': np.array([0.9, 0.1]),
    'f': np.array([0.2, 0.3]),
    'g': np.array([0.4, 0.5]),
}


def get_word_embedding_character_level(word, char_embeddings, aggregation='average'):
    char_vecs = [char_embeddings.get(char, np.zeros(2)) for char in word]  # Default to zeros for unknown chars
    if not char_vecs:
        return np.zeros(2)
    
    if aggregation == 'average':
        return np.mean(char_vecs, axis=0)
    elif aggregation == 'sum':
      return np.sum(char_vecs, axis =0)
    #In a real scenario, consider using a more sophisticated approach with RNN or similar.
    else:
      raise ValueError("Unsupported aggregation method")



def cosine_similarity(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

# Example usage
word1 = "cat"
word2 = "dog"
oov_word = "fog"

word1_vec = get_word_embedding_character_level(word1, character_embeddings)
word2_vec = get_word_embedding_character_level(word2, character_embeddings)
oov_word_vec = get_word_embedding_character_level(oov_word, character_embeddings)

sim_word1_word2 = cosine_similarity(word1_vec, word2_vec)
sim_word1_oov = cosine_similarity(word1_vec, oov_word_vec)
sim_word2_oov = cosine_similarity(word2_vec, oov_word_vec)

print(f"Similarity between '{word1}' and '{word2}': {sim_word1_word2:.2f}")
print(f"Similarity between '{word1}' and '{oov_word}': {sim_word1_oov:.2f}")
print(f"Similarity between '{word2}' and '{oov_word}': {sim_word2_oov:.2f}")

```

This code is for demonstration purposes, of course. In practice, your embeddings would be learned from large datasets and be of much higher dimensionality (hundreds or thousands). The aggregation strategy for characters into a word representation would likely use rnn's or similar architectures, not just simple averaging or summation.

Another powerful technique for handling oov words involves using pre-trained contextual word embeddings, like those from bert, roberta, or similar models. These models generate embeddings that are contextual, meaning that the vector representation of a word depends on its surrounding words. This is incredibly beneficial as words often have different meanings depending on the context they are in. If you have a sentence containing an oov word, the surrounding words will help the model produce an adequate representation for it even if the word itself was not part of the training vocabulary. The contextual embedding captures its meaning through its relations with known words in the given context.

Here’s a simplified example showcasing how contextual embeddings can generate a representation for oov words using a hypothetical scenario:

```python
import numpy as np
# Hypothetical contextual embedding generation
def get_contextual_embedding(sentence, vocab_embeddings):
    # This is a simplified hypothetical representation of a contextual model
    # In reality you would be using a pretrained transformer model.
    words = sentence.split()
    embeddings = []

    for word in words:
      embedding = vocab_embeddings.get(word, np.random.rand(10)) # hypothetical precomputed embeddings and random initialization for unknown words.
      embeddings.append(embedding)
    return np.mean(embeddings,axis = 0)

#Assume that the contextual model also outputs a word embedding based on context of each word.
def get_oov_word_embedding_contextual(sentence, oov_word_index, vocab_embeddings):
  # In reality you would use a transformer model. This is a highly simplified representation.
    words = sentence.split()
    if oov_word_index >= len(words) or oov_word_index <0:
        return None
    
    #This would be computed by a model in real scenario
    embedding = vocab_embeddings.get(words[oov_word_index], np.random.rand(10))  # hypothetical precomputed embeddings and random initialization for unknown words.
    return embedding



vocab = {
    "the": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]),
    "quick": np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2]),
    "brown": np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3]),
    "fox": np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4]),
    "jumps": np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "lazy": np.array([0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    "dog": np.array([0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
}

sentence1 = "the quick brown fox jumps"
sentence2 = "the quick lazy dog"
sentence3 = "a fribbulous animal appears" # fribbulous is an oov word

sentence1_vec = get_contextual_embedding(sentence1, vocab)
sentence2_vec = get_contextual_embedding(sentence2, vocab)
oov_word_index = 1
oov_word_vec = get_oov_word_embedding_contextual(sentence3, oov_word_index, vocab)

sim_s1_s2 = cosine_similarity(sentence1_vec, sentence2_vec)

#Similarity of oov word to sentence 1 and 2 is very simplified since contextual model also provides word embeddings in context
sim_oov_s1 = cosine_similarity(oov_word_vec, sentence1_vec)
sim_oov_s2 = cosine_similarity(oov_word_vec, sentence2_vec)



print(f"Similarity between sentence1 and sentence2: {sim_s1_s2:.2f}")
print(f"Similarity between '{sentence3.split()[oov_word_index]}' in sentence3 and sentence1: {sim_oov_s1:.2f}")
print(f"Similarity between '{sentence3.split()[oov_word_index]}' in sentence3 and sentence2: {sim_oov_s2:.2f}")


```

In a genuine setup, ‘get_contextual_embedding’ and 'get_oov_word_embedding_contextual' would utilize pre-trained transformer models (like bert, roberta, or similar) to encode the input sentences and word embedding. The primary advantage is that even if 'fribbulous' is oov, its context in the sentence 'a fribbulous animal appears' allows for generation of an appropriate vector which may be similar to existing words in the sentence. The word vector for 'fribbulous' will depend on the other words in the sentence. This allows for contextualization of the meaning.

Finally, another crucial approach to consider is to integrate external knowledge sources. If your domain is somewhat specialized, you may find value in incorporating lexical resources or ontologies such as wordnet. These resources capture semantic relationships between words and could allow you to infer meaning and compute similarity even when a word is missing in your original vocabulary. Specifically, you can calculate the distance between word meanings based on the structure of the ontology. A word not found in the original vocabulary can be mapped to a suitable node in the ontology based on its character or subword representations and thus compute similarity.

Here's an example of how WordNet could be utilized. Please note that while this example uses nltk's wordnet, it is a very shallow illustration. Real implementation might involve other ontology structures and reasoning logic to infer word relationships.

```python
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
# Download WordNet data if you haven't already
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()


def wordnet_similarity(word1, word2):
    #Lemmatize words.
    word1 = lemmatizer.lemmatize(word1)
    word2 = lemmatizer.lemmatize(word2)

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    if not synsets1 or not synsets2:
        return 0 #No synsets found
    
    max_sim = 0
    for syn1 in synsets1:
      for syn2 in synsets2:
        sim = syn1.path_similarity(syn2)
        if sim is not None and sim > max_sim:
            max_sim = sim

    return max_sim
# Example Usage

word1 = "cat"
word2 = "dog"
oov_word = "feline"

sim_word1_word2_wn = wordnet_similarity(word1,word2)
sim_word1_oov_wn = wordnet_similarity(word1,oov_word)
sim_word2_oov_wn = wordnet_similarity(word2,oov_word)


print(f"Similarity between '{word1}' and '{word2}' using Wordnet: {sim_word1_word2_wn:.2f}")
print(f"Similarity between '{word1}' and '{oov_word}' using Wordnet: {sim_word1_oov_wn:.2f}")
print(f"Similarity between '{word2}' and '{oov_word}' using Wordnet: {sim_word2_oov_wn:.2f}")


```

In real world scenario the ontology mapping can be much more complex and require more complex reasoning algorithms and might even involve mapping to multiple synsets and computing the aggregate similarity. This example serves as a very shallow demonstration of the concept.

For anyone looking to dive deeper into the theory and implementation of these techniques, I recommend these resources:

*   **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** A comprehensive textbook covering a broad range of natural language processing topics, including word embeddings, contextual models, and semantic analysis. This is practically essential reading for anyone working with NLP.
*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:** This book provides a deep dive into the theoretical foundations of deep learning, including the architectures that underpin most of the advanced word embedding techniques used today. Understanding these foundations will allow you to adapt these techniques to the specific constraints and needs of your project.
*   **Research papers on specific contextual embedding models:** Search for the original papers on models such as bert, roberta, elmo on venues like arxiv or nips. Reading the original papers gives you an in-depth understanding of these models and how they are architected, which is very important when using them to solve problems.

Ultimately, the ‘best’ approach often depends on your specific application, the computational resources available, and the characteristics of your data. By experimenting with the methods described and understanding their underlying mechanisms, you’ll be well-equipped to handle the challenge of computing similarity with out-of-vocabulary words.
