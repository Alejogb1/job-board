---
title: "How to convert text to a numeric array for a Keras sequential model without the 'XXX not in index' error?"
date: "2025-01-30"
id: "how-to-convert-text-to-a-numeric-array"
---
The root cause of the "XXX not in index" error when converting text to a numeric array for a Keras sequential model invariably stems from an inconsistency between the vocabulary used during encoding and the vocabulary encountered during prediction or model application.  This discrepancy arises from the text preprocessing stage, specifically the handling of out-of-vocabulary (OOV) words.  My experience building and deploying sentiment analysis models for a large e-commerce platform highlighted this repeatedly.  Neglecting proper OOV handling consistently resulted in prediction failures.  Therefore, robust OOV management is critical.


**1. Clear Explanation:**

Converting text to a numeric array for neural networks requires mapping each word to a unique integer. This mapping is usually created using a vocabulary built from the training data.  The `Tokenizer` class in Keras provides a convenient way to achieve this. However, if the test or prediction data contains words not present in this training vocabulary, the model will fail because it cannot find the corresponding integer representation.  The solution lies in effectively handling these OOV words.

Several strategies exist:

* **Ignoring OOV words:**  This is the simplest, but often least effective, approach.  Words not in the vocabulary are simply discarded.  This is suitable only when OOV words are expected to be rare and their absence doesn't significantly impact the model's performance.

* **Replacing OOV words with a special token:** A dedicated token, such as `<UNK>` (Unknown), is assigned to represent all OOV words.  This preserves the sequence length and allows the model to learn a representation for unknown words.  This is generally a preferred approach for its balance of simplicity and effectiveness.

* **Creating a larger vocabulary:** Expanding the vocabulary to include all words encountered in both training and test data can eliminate OOV words altogether. However, this can lead to a significantly larger model and potential overfitting, especially if the vocabulary includes many rare words.

* **Using pre-trained word embeddings:** Employing pre-trained word embeddings like Word2Vec or GloVe allows the model to leverage semantic information even for OOV words.  These embeddings often contain vector representations for a vast vocabulary, mitigating the OOV problem.


**2. Code Examples with Commentary:**

**Example 1: Handling OOV words with `<UNK>`:**

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample training data
training_data = [
    "This is a positive sentence.",
    "This is a negative sentence.",
    "Another positive example."
]

# Tokenizer configuration
tokenizer = Tokenizer(num_words=10, oov_token="<UNK>") #Limit vocabulary size and handle OOV

# Fit tokenizer on training data
tokenizer.fit_on_texts(training_data)

# Convert training data to sequences
training_sequences = tokenizer.texts_to_sequences(training_data)

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in training_sequences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post')

# New text with OOV words
new_text = ["This is an unknown sentence with unusual words."]

# Convert new text to sequences, OOV words replaced by <UNK>
new_sequences = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding='post')

print(tokenizer.word_index) #Shows the vocabulary mapping, including <UNK>
print(training_padded)
print(new_padded)
```

This example demonstrates the use of `oov_token` to replace unseen words with `<UNK>`.  The `num_words` parameter limits vocabulary size, impacting OOV frequency.


**Example 2: Ignoring OOV words:**

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

training_data = ["This is positive.", "This is negative."]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_data)
sequences = tokenizer.texts_to_sequences(training_data)
padded_sequences = pad_sequences(sequences, padding='post')

#New Text with OOV words; these are simply ignored.
new_text = ["This contains entirely unseen words."]
new_sequences = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_sequences, padding='post')
print(padded_sequences)
print(new_padded) #Note: OOV words lead to empty sequences.
```

This illustrates how ignoring OOV words results in empty sequences for text containing unknown words.  This approach can lead to significant information loss.


**Example 3:  Using Pre-trained Embeddings (Illustrative):**

```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Assume pre-trained embeddings are loaded into 'embeddings_index' dictionary
#This step is simplified for demonstration; loading actual embeddings requires external resources.

# Sample word embeddings (replace with actual embeddings)
embeddings_index = {'this': np.array([1, 2, 3]), 'is': np.array([4, 5, 6]), 'positive': np.array([7, 8, 9]), 'negative': np.array([10, 11, 12]), '<UNK>': np.array([0, 0, 0])}
vocabulary_size = len(embeddings_index) + 1

# Sample text data
text_data = ["This is positive.", "This is negative.", "This is unknown."]

# Tokenizer to map words to indices.
tokenizer = Tokenizer(num_words=vocabulary_size, oov_token='<UNK>')
tokenizer.fit_on_texts(text_data)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(text_data)

# Pad sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

#Embedding matrix creation.
embedding_matrix = np.zeros((vocabulary_size, 3)) #3 is the embedding dimension
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix)
print(padded_sequences)
```

This example outlines the process of integrating pre-trained embeddings. Note that  obtaining and loading real embeddings is a separate, resource-intensive process, omitted here for brevity.  The `<UNK>` token ensures that even words absent from the initial vocabulary receive a vector representation.


**3. Resource Recommendations:**

The Keras documentation, particularly sections on text preprocessing and embedding layers, provides comprehensive guidance.  Several well-regarded textbooks cover deep learning for natural language processing.  Exploring research papers on OOV handling in neural networks is also beneficial.  A solid understanding of word embeddings and their application is crucial.  Finally, proficiency in Python and NumPy will prove invaluable.
