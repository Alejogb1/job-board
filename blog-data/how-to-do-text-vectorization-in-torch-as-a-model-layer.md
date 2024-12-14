---
title: "How to do text vectorization in Torch as a model layer?"
date: "2024-12-14"
id: "how-to-do-text-vectorization-in-torch-as-a-model-layer"
---

alright, so you're looking to slot text vectorization directly into your pytorch model as a layer, huh? i've been down that rabbit hole myself a few times, so i can definitely share some insights and, most importantly, code that actually works. believe me, i’ve had my share of headaches getting this piece to play nice with the rest of the network.

the issue most people run into is that typical text preprocessing steps – tokenization, vocabulary creation, integer encoding – they are often done as separate, upfront operations. which means your model is only getting preprocessed number sequences as input. it's fine for experimentation, but when you're pushing for a full, end-to-end system, it's way cleaner and faster to have that vectorization step baked right into your model architecture. it simplifies the data pipeline a lot.

let’s start with the basics. we want to go from raw text to a tensor of numerical vectors, and we want to do it inside a torch module. the most common method is to use bag-of-words or tf-idf, but that won’t cut it if you’re dealing with sequences and sequence models like rnns and transformers. for those cases, i find an embedding layer with a vocabulary mapping makes the most sense as part of a larger model.

the main challenge here is not about the math, which is mostly simple lookup tables, it's really about proper design and handling out of vocabulary tokens. because in a real world setup, your model will always see words it hasn't encountered before. so here’s how i usually approach it, and a simplified version of what i use as part of one of my old models that is handling customer support tickets.

first, we will start with a basic vocabulary mapping for creating embedding sequences.

```python
import torch
import torch.nn as nn
from collections import Counter
import re

class TextVectorizer(nn.Module):
    def __init__(self, vocab, embedding_dim=128):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.unk_token = '<unk>' #unknown token
        self.pad_token = '<pad>' #padding token
        self.unk_idx = self.vocab.get(self.unk_token, 0) #default index of zero if not found
        self.pad_idx = self.vocab.get(self.pad_token, 1) #default index of one if not found

    def _tokenize(self, text):
        text = text.lower() #simple lower casing text
        text = re.sub(r'[^a-z0-9\s]', '', text) #removing punctuation.
        return text.split()

    def _text_to_indices(self, text):
        tokens = self._tokenize(text)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        return indices

    def forward(self, texts):
        indices = [self._text_to_indices(text) for text in texts]
        max_len = max(len(seq) for seq in indices)
        padded_indices = [seq + [self.pad_idx] * (max_len - len(seq)) for seq in indices]
        return torch.tensor(padded_indices)

def build_vocab(texts, min_freq=2):
    tokens = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens.extend(text.split())
    
    token_counts = Counter(tokens)
    vocab = {'<unk>': 0, '<pad>': 1}
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

if __name__ == '__main__':
    texts = [
        "this is a test sentence",
        "another test example with more words",
        "yet another example",
        "unseen word here", #this will be converted to the <unk> token
        "simple case"
    ]
    vocab = build_vocab(texts)
    print(f'vocabulary: {vocab}')
    vectorizer = TextVectorizer(vocab)
    text_batch = [
       "this is a test sentence",
        "new unseen word",
        "simple case"
    ]
    vectorized_batch = vectorizer(text_batch)
    print(f"vectorized tensor batch: {vectorized_batch}")
    embedded_tensor = vectorizer.embedding(vectorized_batch)
    print(f'embedded tensor shape: {embedded_tensor.shape}')
```

in this snippet, the `textvectorizer` class inherits from `nn.module`. it takes a vocabulary during initialization. the `forward` method tokenizes, converts words into numerical indices, pads the sequences so they have the same length, and returns a padded tensor of indices. before the `forward` method is called, the `build_vocab` method builds a vocabulary by iterating through the texts, tokenizing, and counting word frequencies to ignore rare ones and assigning them to unknown. in the main part of the code, you see that we can build a vocabulary, then use that vocabulary to create a `textvectorizer`, pass a text list to get the numerical vectors. after that, we see how the embedding layer turns these numerical indices into embedding sequences.

but what happens when you need more control, like pre-trained embeddings and handling different types of text preprocessing? in my experience with customer support, it turned out that the vocabulary was not enough, sometimes some very specific jargon was used and it was difficult to properly map. so i switched to using pre-trained embeddings like glove, and i was able to improve the vectorization of the text quite a lot.

so let’s create an extended example with pre-trained glove embeddings.

```python
import torch
import torch.nn as nn
from collections import Counter
import re
import numpy as np
import os

class TextVectorizerPretrained(nn.Module):
    def __init__(self, vocab, embedding_dim=100, embeddings_matrix=None, freeze_embeddings=True):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.unk_idx = self.vocab.get(self.unk_token, 0)
        self.pad_idx = self.vocab.get(self.pad_token, 1)

        if embeddings_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embeddings_matrix, dtype=torch.float32))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False #if we want to freeze the embedding layer
        else:
            nn.init.uniform_(self.embedding.weight, -0.25, 0.25) # if no embedding matrix use uniform init


    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def _text_to_indices(self, text):
        tokens = self._tokenize(text)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        return indices

    def forward(self, texts):
        indices = [self._text_to_indices(text) for text in texts]
        max_len = max(len(seq) for seq in indices)
        padded_indices = [seq + [self.pad_idx] * (max_len - len(seq)) for seq in indices]
        return torch.tensor(padded_indices)


def load_glove_embeddings(file_path, vocab, embedding_dim):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    embedding_matrix = np.random.uniform(-0.25, 0.25, size=(len(vocab), embedding_dim))
    for word, index in vocab.items():
        if word in embeddings:
            embedding_matrix[index] = embeddings[word]
    return embedding_matrix


def build_vocab_advanced(texts, min_freq=2):
    tokens = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens.extend(text.split())
    
    token_counts = Counter(tokens)
    vocab = {'<unk>': 0, '<pad>': 1}
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


if __name__ == '__main__':
    texts = [
       "this is a test sentence",
        "another test example with more words",
        "yet another example",
        "unseen word here",
        "simple case"
    ]
    vocab = build_vocab_advanced(texts)
    print(f'vocabulary: {vocab}')
    embedding_dim = 100
    glove_file_path = 'glove.6b.100d.txt'  #replace this with the actual file path, make sure the file is in the same folder of this code
    
    if not os.path.exists(glove_file_path):
        print(f"Error: {glove_file_path} not found. Please download it.")
    else:
        embeddings_matrix = load_glove_embeddings(glove_file_path, vocab, embedding_dim)
        vectorizer = TextVectorizerPretrained(vocab, embedding_dim, embeddings_matrix)
        text_batch = [
            "this is a test sentence",
            "new unseen word",
            "simple case"
        ]
        vectorized_batch = vectorizer(text_batch)
        print(f"vectorized tensor batch: {vectorized_batch}")
        embedded_tensor = vectorizer.embedding(vectorized_batch)
        print(f'embedded tensor shape: {embedded_tensor.shape}')
```

here, the `textvectorizerpretrained` class now accepts an optional `embeddings_matrix` during initialization and a `freeze_embeddings` flag. the `load_glove_embeddings` function reads glove vectors and create the matrix. it’s important to check that file exists. if you don't have it you might want to go to the stanford nlp website and download it (that’s what i did). the rest of the forward propagation is the same, and you can now see that the main code is loading the glove embeddings if the file exists and constructing the vectorizer using those embeddings. when you do that the embedding layer will be initialized with those vectors.

now, if your text input involves some numerical features, or you're working with some other structured data mixed with text, things get a little more complex. you'll likely want to combine the numerical features with embedding vectors, so you need to consider this as part of the forward method as well.

let’s add some numerical features and see how the forward method can handle this situation.

```python
import torch
import torch.nn as nn
from collections import Counter
import re
import numpy as np
import os

class TextAndNumericVectorizer(nn.Module):
    def __init__(self, vocab, embedding_dim=100, embeddings_matrix=None, num_numeric_features=2, freeze_embeddings=True):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.num_numeric_features = num_numeric_features
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.unk_idx = self.vocab.get(self.unk_token, 0)
        self.pad_idx = self.vocab.get(self.pad_token, 1)
        self.numeric_layer = nn.Linear(self.num_numeric_features, 10) #adding linear layer to handle numerical features
        self.dropout = nn.Dropout(0.2)

        if embeddings_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embeddings_matrix, dtype=torch.float32))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        else:
            nn.init.uniform_(self.embedding.weight, -0.25, 0.25)


    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def _text_to_indices(self, text):
        tokens = self._tokenize(text)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        return indices

    def forward(self, text_batch, numeric_batch):
        text_indices = [self._text_to_indices(text) for text in text_batch]
        max_len = max(len(seq) for seq in text_indices)
        padded_indices = [seq + [self.pad_idx] * (max_len - len(seq)) for seq in text_indices]

        padded_indices_tensor = torch.tensor(padded_indices)
        embedded_text = self.embedding(padded_indices_tensor)
        numeric_tensor = torch.tensor(numeric_batch, dtype=torch.float32) #convert to tensor
        embedded_numeric = self.numeric_layer(numeric_tensor)
        embedded_numeric = self.dropout(embedded_numeric)
        #concatenating embeddings of text and numerical features
        concatenated_tensor = torch.cat((embedded_text, embedded_numeric.unsqueeze(1).expand(-1, max_len, -1)), dim=-1)
        return concatenated_tensor


def load_glove_embeddings_advanced(file_path, vocab, embedding_dim):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    embedding_matrix = np.random.uniform(-0.25, 0.25, size=(len(vocab), embedding_dim))
    for word, index in vocab.items():
        if word in embeddings:
            embedding_matrix[index] = embeddings[word]
    return embedding_matrix


def build_vocab_advanced(texts, min_freq=2):
    tokens = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens.extend(text.split())
    
    token_counts = Counter(tokens)
    vocab = {'<unk>': 0, '<pad>': 1}
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


if __name__ == '__main__':
    texts = [
       "this is a test sentence",
        "another test example with more words",
        "yet another example",
        "unseen word here",
        "simple case"
    ]
    vocab = build_vocab_advanced(texts)
    print(f'vocabulary: {vocab}')
    embedding_dim = 100
    glove_file_path = 'glove.6b.100d.txt'
    if not os.path.exists(glove_file_path):
         print(f"Error: {glove_file_path} not found. Please download it.")
    else:
        embeddings_matrix = load_glove_embeddings_advanced(glove_file_path, vocab, embedding_dim)
        num_numeric_features = 2
        vectorizer = TextAndNumericVectorizer(vocab, embedding_dim, embeddings_matrix, num_numeric_features)
        text_batch = [
           "this is a test sentence",
            "new unseen word",
            "simple case"
        ]
        numeric_batch = [[1.2, 3.4], [4.5, 5.6], [7.8, 9.0]]
        concatenated_batch = vectorizer(text_batch, numeric_batch)
        print(f'concatenated tensor shape: {concatenated_batch.shape}')

```
in this final example, the `textandnumericvectorizer` class now includes a linear layer `numeric_layer`, a dropout layer, and takes the numeric batch as an input to the forward method. you can now see that in the main execution part, both text data and numeric are passed to the forward method to combine text embeddings with the numerical features. after passing through the numeric linear layer, both are concatenated and returned. this setup allows more complex model architectures.

it’s a bit more involved, but it gives you full control. you can handle pretty much any scenario you might encounter when processing text data.

to further improve text vectorization, i recommend exploring these topics: character-level embeddings, subword tokenization, attention mechanisms, and techniques for handling long-range dependencies. the papers on the bert architecture is a great start, and also the books deep learning with pytorch by eli stevens is a good general source. there are also papers in the nips and icml conferences with novel approaches to text vectorization. finally, remember, good vectorization is the first step to a great model, sometimes it's not about the fancy model you use, it's how you feed it.
and that’s all i can share from my own experience, hope this helps.
