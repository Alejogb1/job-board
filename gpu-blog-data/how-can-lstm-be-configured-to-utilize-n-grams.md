---
title: "How can LSTM be configured to utilize n-grams instead of fixed sequence length?"
date: "2025-01-30"
id: "how-can-lstm-be-configured-to-utilize-n-grams"
---
Recurrent Neural Networks, specifically LSTMs, inherently process sequences sequentially, making direct n-gram utilization seemingly counterintuitive. The core operation of an LSTM involves updating its hidden and cell states based on each element of the input sequence one at a time. Therefore, feeding it pre-segmented n-grams isn’t a conventional approach. However, I have found practical ways to achieve this effect, albeit indirectly. The goal isn’t to modify the LSTM’s fundamental operation, but to structure the input data and potentially introduce an embedding layer, enabling it to learn patterns at the n-gram level. In my past work building sentiment analysis models, I encountered this exact challenge when trying to capture nuanced phrases instead of relying solely on individual words.

The primary strategy involves preprocessing the input text to generate n-gram representations before feeding them into the LSTM. This means that instead of passing a sequence of words, you'd be passing a sequence of pre-computed n-grams. Critically, this doesn't mean the LSTM "sees" the n-grams as atomic units in the sequence. Instead, the model still operates sequentially across the n-grams, but because these are derived from larger contexts, the LSTM learns higher-order dependencies. The initial challenge is finding an appropriate method for generating and encoding these n-grams.

One common method is to use text tokenization libraries to create n-grams. We treat n-grams, regardless of `n`, as tokens themselves. Then we proceed by encoding these tokens before passing them to the LSTM. This method makes no fundamental changes to the LSTM, merely altering the input data’s pre-processing. The success of this approach lies in selecting suitable `n` values and the subsequent treatment of these n-grams, notably their representation as numerical inputs to the LSTM.

Another crucial step after n-gram generation is the encoding of these strings, typically using an embedding layer. One-hot encoding may prove impractical due to its high dimensionality, particularly with larger datasets. Embeddings map each n-gram token into a dense vector space where semantically similar tokens are close to each other. The embedding dimension becomes a hyperparameter that you can adjust based on your needs and computational constraints. I have seen empirically that embedding dimensions in the range of 50-300 typically provide a good trade-off between expressiveness and efficiency, depending on dataset size and complexity.

Finally, the LSTM processes these embedding vectors sequentially. Its temporal nature allows it to capture the dependency between these n-grams within the larger context of the sentence or text. It is important to note that while the LSTM operates on n-grams, it still treats each n-gram as an independent unit in the sequence. Therefore, there are no internal structural changes required for the LSTM.

Here are three code examples demonstrating these concepts using Python, Keras, and a simplified approach with `scikit-learn`.

**Example 1: N-gram generation using scikit-learn, and basic LSTM input preparation**

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample documents
documents = [
    "this is the first document",
    "this is the second document",
    "and this is the third one",
    "is this the first one?"
]

# Create 2-grams (bigrams)
vectorizer = CountVectorizer(ngram_range=(2, 2))
vectorizer.fit(documents)
bigrams = vectorizer.vocabulary_
vocab_size = len(bigrams) + 1 # +1 for padding

print(f"Vocabulary: {bigrams}")
print(f"Vocabulary size: {vocab_size}")


# Convert docs into n-gram sequences by mapping tokens to indices
def docs_to_indices(docs, vocab):
  indexed_docs = []
  for doc in docs:
    doc_bigrams = [str(x) for x in zip(doc.split()[:-1], doc.split()[1:])]
    indexed_doc = [vocab.get(bigram, 0) for bigram in doc_bigrams ]
    indexed_docs.append(indexed_doc)
  return indexed_docs

indexed_docs = docs_to_indices(documents, bigrams)
print(f"Indexed documents (before padding): {indexed_docs}")

# Pad sequences to ensure uniform input to the LSTM
max_length = max(len(doc) for doc in indexed_docs)
padded_docs = pad_sequences(indexed_docs, maxlen=max_length, padding='post')
print(f"Padded documents (ready for LSTM): {padded_docs}")


# Example Usage of the values for the LSTM layer
embedding_dim = 10 # Arbitrary embedding size
input_length = max_length # The maximum length of input after padding
```

This example showcases n-gram generation with scikit-learn's `CountVectorizer`, extracting bigrams from the documents. I create a vocabulary mapping each unique bigram to a unique integer index. I define a simple function `docs_to_indices` to map the documents to their corresponding bigram indices based on the extracted vocabulary. I pad the sequences using `pad_sequences` from Keras to ensure all sequences have the same length, preparing them for the LSTM. Finally I show a demonstration of how those values will likely be used to setup the embedding and LSTM input layers. The output shows the generated n-gram vocabulary, their integer index mapping, and the padded sequences, ready to be fed to an LSTM.

**Example 2: Keras embedding layer and simplified LSTM model**

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Same prep from previous example

documents = [
    "this is the first document",
    "this is the second document",
    "and this is the third one",
    "is this the first one?"
]
vectorizer = CountVectorizer(ngram_range=(2, 2))
vectorizer.fit(documents)
bigrams = vectorizer.vocabulary_
vocab_size = len(bigrams) + 1
def docs_to_indices(docs, vocab):
  indexed_docs = []
  for doc in docs:
    doc_bigrams = [str(x) for x in zip(doc.split()[:-1], doc.split()[1:])]
    indexed_doc = [vocab.get(bigram, 0) for bigram in doc_bigrams ]
    indexed_docs.append(indexed_doc)
  return indexed_docs
indexed_docs = docs_to_indices(documents, bigrams)
max_length = max(len(doc) for doc in indexed_docs)
padded_docs = pad_sequences(indexed_docs, maxlen=max_length, padding='post')


# Simple LSTM model
embedding_dim = 10
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=32))  # Example LSTM with 32 units
model.add(Dense(units=1, activation='sigmoid'))  # Binary classification (example)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Example training with random data
labels = np.random.randint(0, 2, size=(len(documents),1)) # Binary labels
model.fit(padded_docs, labels, epochs=5, verbose = 0) # Very simplified training loop

loss, accuracy = model.evaluate(padded_docs, labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")
```

This example illustrates building a basic LSTM model with an embedding layer.  The example reuses the n-gram generation and data preparation steps from the previous example. Crucially, it introduces an `Embedding` layer in Keras, which maps the integer indices of n-grams to dense vectors. The model then has an LSTM layer followed by a `Dense` layer to enable simple binary classification.  Note that in real-world scenarios this output would more commonly need to be a higher dimension, as well as the loss and metrics needed to be appropriate for the desired prediction task. I include an example training loop that uses random data to illustrate basic model fitting. The model summary shows the architecture and parameter counts. The evaluation step displays the results of training.

**Example 3:  Using different n-gram sizes and feature concatenation**

```python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
# Sample Documents, and labels
documents = [
    "this is the first document",
    "this is the second document",
    "and this is the third one",
    "is this the first one?"
]
labels = np.random.randint(0, 2, size=(len(documents),1))

# N-gram generation functions for unigrams, bigrams, and trigrams
def generate_ngrams(documents, ngram_range):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(documents)
    return vectorizer.vocabulary_, vectorizer

def documents_to_indices(docs, vocab, ngram_range):
    indexed_docs = []
    for doc in docs:
        doc_ngrams = []
        if ngram_range == (1, 1):
            doc_ngrams = doc.split()
        elif ngram_range == (2, 2):
            doc_ngrams = [str(x) for x in zip(doc.split()[:-1], doc.split()[1:])]
        elif ngram_range == (3, 3):
            doc_ngrams = [str(x) for x in zip(doc.split()[:-2], doc.split()[1:-1], doc.split()[2:])]

        indexed_doc = [vocab.get(ngram, 0) for ngram in doc_ngrams ]
        indexed_docs.append(indexed_doc)
    return indexed_docs

# Generate vocabularies for unigrams, bigrams, and trigrams
unigram_vocab, unigram_vectorizer  = generate_ngrams(documents, (1, 1))
bigram_vocab, bigram_vectorizer = generate_ngrams(documents, (2, 2))
trigram_vocab, trigram_vectorizer = generate_ngrams(documents, (3, 3))

unigram_vocab_size = len(unigram_vocab) + 1
bigram_vocab_size = len(bigram_vocab) + 1
trigram_vocab_size = len(trigram_vocab) + 1


# Convert documents into n-gram sequences by mapping tokens to indices
unigram_indexed_docs = documents_to_indices(documents, unigram_vocab, (1, 1))
bigram_indexed_docs = documents_to_indices(documents, bigram_vocab, (2, 2))
trigram_indexed_docs = documents_to_indices(documents, trigram_vocab, (3, 3))


# Pad sequences
max_length_uni = max(len(doc) for doc in unigram_indexed_docs)
max_length_bi = max(len(doc) for doc in bigram_indexed_docs)
max_length_tri = max(len(doc) for doc in trigram_indexed_docs)

padded_unigrams = pad_sequences(unigram_indexed_docs, maxlen=max_length_uni, padding='post')
padded_bigrams = pad_sequences(bigram_indexed_docs, maxlen=max_length_bi, padding='post')
padded_trigrams = pad_sequences(trigram_indexed_docs, maxlen=max_length_tri, padding='post')



# Model with multiple n-gram inputs
embedding_dim = 10
#Input layers
input_uni = Input(shape=(max_length_uni,))
input_bi = Input(shape=(max_length_bi,))
input_tri = Input(shape=(max_length_tri,))

# Embedding layers
embedding_uni = Embedding(input_dim=unigram_vocab_size, output_dim=embedding_dim)(input_uni)
embedding_bi = Embedding(input_dim=bigram_vocab_size, output_dim=embedding_dim)(input_bi)
embedding_tri = Embedding(input_dim=trigram_vocab_size, output_dim=embedding_dim)(input_tri)

# LSTM layers
lstm_uni = LSTM(units=32)(embedding_uni)
lstm_bi = LSTM(units=32)(embedding_bi)
lstm_tri = LSTM(units=32)(embedding_tri)

# Concatenate LSTM outputs
merged = concatenate([lstm_uni, lstm_bi, lstm_tri])

# Output layer
output = Dense(units=1, activation='sigmoid')(merged) # Binary classification (example)

# Define the model
model = Model(inputs=[input_uni, input_bi, input_tri], outputs=output)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit([padded_unigrams, padded_bigrams, padded_trigrams], labels, epochs=5, verbose = 0) # Very simplified training loop

loss, accuracy = model.evaluate([padded_unigrams, padded_bigrams, padded_trigrams], labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")
```

This more advanced example demonstrates how to use multiple n-gram representations within the same model. It generates unigrams, bigrams, and trigrams and prepares their respective embeddings and feeds them into parallel LSTM layers. The outputs from each LSTM are then concatenated before passing them to a final dense layer. This allows the model to learn from varying n-gram levels simultaneously. The model training loop is again kept simple for clarity, using random labels and a few epochs. However, this example shows a more complex and potentially powerful methodology.

For further learning, I recommend exploring resources related to natural language processing, specifically text embeddings and recurrent neural networks. Consider consulting books on deep learning with a focus on NLP, such as “Speech and Language Processing” by Daniel Jurafsky and James H. Martin. Another valuable resource is the official Keras documentation which is highly comprehensive and provides extensive examples. Research papers focusing on n-gram modeling and LSTM performance with varied input representations would also be beneficial, although these can often be very technical. Exploring implementations using other frameworks such as PyTorch can offer alternative perspectives and help solidify understanding. Finally, I would also study the practical application and best practices for data cleaning and pre-processing which can have a huge impact on performance of your model.
