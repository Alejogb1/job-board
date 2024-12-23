---
title: "How can I use a similarity function with words that are outside the vocabulary?"
date: "2024-12-23"
id: "how-can-i-use-a-similarity-function-with-words-that-are-outside-the-vocabulary"
---

Alright,  It's a problem I've bumped into more than a few times, particularly back when I was working on that initial iteration of our semantic search engine for a client a few years ago. We were dealing with a relatively limited vocabulary initially, and seeing how the system would respond to out-of-vocabulary (oov) words was crucial. The core challenge, as you've likely discovered, is that similarity functions, especially those based on techniques like word embeddings, rely on having a vector representation for each word. If a word doesn't exist in the pre-trained model's vocabulary, you essentially have a missing piece, throwing off any direct calculation of similarity.

The most straightforward approach, and one I've seen used widely, involves some form of fall-back strategy. Instead of simply disregarding the oov word, you try to approximate a representation for it. There are several techniques that I've found useful over the years.

**1. Character-Based Embeddings**

One effective way to handle oov words is to derive a representation based on the characters that compose them. This leverages the fact that, while a word might not be in the vocabulary, its constituent parts (letters, letter groups) might contain useful semantic information. For instance, the prefix "un-" often implies negation, and the suffix "-ing" usually signifies a verb.

Here, rather than relying on whole words alone, we compute embeddings for characters or n-grams of characters within the word. You could take the average of these character embeddings or use a Recurrent Neural Network (RNN), like a bidirectional LSTM, to process the character sequence and generate a single, fixed-size embedding representing the oov word.

Consider the word "unbelievable." The embeddings might not have "unbelievable" but would certainly have "un," "believ," and "able" as character-based representations or as n-grams within the vocabulary. We can then combine these. It's a form of compositional semantics, where we compose the meaning of the word from its sub-parts. This approach allows a similarity function to be applied, even when a full-word vector is unavailable. This approach is often termed subword tokenization.

Here’s a Python example using basic averaging (for demonstration purposes, real implementations often use more advanced models):

```python
import numpy as np

def char_embedding(word, char_to_vec, default_vec):
    """
    Calculates a word embedding by averaging its character embeddings.
    """
    embeddings = []
    for char in word:
      if char in char_to_vec:
        embeddings.append(char_to_vec[char])
      else:
        embeddings.append(default_vec)
    if embeddings:
      return np.mean(embeddings, axis=0)
    return default_vec

# Example usage (assuming you have character vectors already)
char_to_vec = {'u': np.array([0.1, 0.2]),
                'n': np.array([0.3, 0.4]),
                'b': np.array([0.5, 0.6]),
                'e': np.array([0.7, 0.8]),
                'l': np.array([0.9, 1.0]),
                'i': np.array([1.1, 1.2]),
                'v': np.array([1.3, 1.4]),
                'a': np.array([1.5, 1.6])}
default_vec = np.array([0,0])


word1_vec = char_embedding("unbelievable", char_to_vec, default_vec)
word2_vec = char_embedding("believe", char_to_vec, default_vec)
# Calculate cosine similarity (you'd replace this with your preferred method)
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


similarity = cosine_similarity(word1_vec, word2_vec)
print(f"Similarity between 'unbelievable' and 'believe': {similarity}")

```

**2. Subword Modeling with Byte-Pair Encoding (BPE)**

Another effective approach, and one that I favor for its practicality, is byte-pair encoding (BPE). BPE learns a set of subword units from the training data. This avoids the issue of seeing "unseen" words entirely. The idea is that frequent character sequences are merged into single units. This approach has demonstrated considerable effectiveness, especially in scenarios where morphological variation is high.

With BPE, words are broken down into the most frequently occurring sequences, so that your overall vocabulary is significantly reduced but can accurately represent most words. Instead of handling words at the character level, you operate at a learned subword level that tends to reflect meaningful segments. If a word is not in the vocabulary, it would usually be split into common subwords, many of which would be in the vocabulary.

Here's an illustration of how you'd use a BPE tokenizer from a common library, in this case `sentencepiece`:

```python
import sentencepiece as spm
import numpy as np


# Assume you have trained a BPE model (e.g. using sentencepiece library)
# model.model file will exist after training
# Example: train sentencepiece model.
#spm.SentencePieceTrainer.train('--input=text.txt --model_prefix=bpe_model --vocab_size=1000 --model_type=bpe')
sp = spm.SentencePieceProcessor(model_file='bpe_model.model')

def bpe_embedding(word, sp, model_to_vec, default_vec):
    """
    Calculates a word embedding by averaging BPE subword embeddings.
    """
    subwords = sp.encode(word, out_type=str)
    embeddings = []
    for subword in subwords:
        if subword in model_to_vec:
            embeddings.append(model_to_vec[subword])
        else:
            embeddings.append(default_vec)

    if embeddings:
        return np.mean(embeddings, axis=0)
    return default_vec



# Example usage (assuming you have a pretrained embedding map)
model_to_vec = {"un" : np.array([0.1, 0.2]),
                "believ" : np.array([0.3, 0.4]),
                "able" : np.array([0.5, 0.6]),
                "play": np.array([0.7, 0.8])
               }
default_vec = np.array([0,0])

word1_vec = bpe_embedding("unbelievable", sp, model_to_vec, default_vec)
word2_vec = bpe_embedding("playable", sp, model_to_vec, default_vec)
word3_vec = bpe_embedding("play", sp, model_to_vec, default_vec)
# Calculate cosine similarity (you'd replace this with your preferred method)

from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))
similarity = cosine_similarity(word1_vec, word2_vec)
similarity2 = cosine_similarity(word2_vec, word3_vec)
print(f"Similarity between 'unbelievable' and 'playable': {similarity}")
print(f"Similarity between 'playable' and 'play': {similarity2}")

```
**3. Contextualized Word Embeddings**

Lastly, contextualized embeddings, as generated by models like BERT, RoBERTa, or ELMo, are more advanced, but they are worth considering if you have more computational resources and a large dataset. These models process words in their specific context, creating unique vector representations that account for the surrounding text. While these aren't *specifically* about *generating* embeddings for out-of-vocabulary words in an isolated manner, they can handle it. They do so by using a subword approach as described above or generate unique embedding representations from an existing subword model for the out-of-vocabulary word based on the context, effectively making the oOV problem practically go away. They don't *average* subword representations like the other techniques I've shown; instead, their architecture and training process generate a contextualized representation from any sequence of words and/or subwords.

Here's a simplified illustration of utilizing a pre-trained model from Hugging Face Transformers:

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def contextualized_embedding(text, tokenizer, model):
    """
    Gets the contextualized embedding for a word in a given text using BERT.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
      outputs = model(**inputs)
    # take embedding for the first word in the sentence
    return outputs.last_hidden_state[0][1].numpy() # 0th batch, 1st token in sentence (assuming the first word)



# Example usage:
word1_vec = contextualized_embedding("the unbelievable experience", tokenizer, model)
word2_vec = contextualized_embedding("the believe experience", tokenizer, model)

from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

similarity = cosine_similarity(word1_vec, word2_vec)
print(f"Similarity between 'unbelievable' and 'believe' (contextual): {similarity}")

```
Each approach has its merits and drawbacks. Character-based embeddings and BPE are more computationally inexpensive and useful when you're dealing with morphologically rich languages or resource constraints. Contextualized embeddings are far more powerful, but also much more demanding in terms of computational power.

I recommend consulting "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. Also, a great paper is Mikolov et al. (2013) "Distributed Representations of Words and Phrases and their Compositionality," which discusses the concept of word embeddings, although the methods now available to process words have evolved. You also should look at SentencePiece paper, “SentencePiece: A simple and language independent subword tokenizer for neural text processing.” For BPE, consider “Neural Machine Translation of Rare Words with Subword Units,” Sennrich, et al (2015). For contextual embedding papers look for papers that mention BERT, RoBERTa, or ELMo models for examples.

The best method, in my experience, usually depends on your specific use case, the resources you have, and the level of accuracy you require. It often involves a good deal of experimentation to see what works best for your task.
