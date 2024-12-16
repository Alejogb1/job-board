---
title: "How is text classification performed in spaCy?"
date: "2024-12-16"
id: "how-is-text-classification-performed-in-spacy"
---

Alright, let's talk about text classification in spaCy. It's something I’ve tackled extensively over the years, often as part of larger projects dealing with unstructured data. Back when I was working on a system to automatically categorize customer feedback emails, this became a very practical skill – a lot more than just theoretical knowledge.

SpaCy, at its core, isn’t primarily a machine learning library. It’s an industrial-strength natural language processing toolkit. Its power in text classification stems from its robust preprocessing capabilities and how well it integrates with other machine learning libraries, primarily scikit-learn or TensorFlow/Keras. It doesn't provide classification algorithms directly but offers the crucial foundation for feature engineering and preparing data that we feed into those machine learning models.

The process, therefore, isn't a single step, but rather a carefully orchestrated pipeline. Think of it less like a black box and more like a series of interconnected modules. First, you feed your text data into spaCy's processing pipeline. This pipeline tokenizes the text – breaking it into individual words or sub-word units. It also performs part-of-speech tagging, identifies named entities (like organizations, locations, or people), and lemmatizes words, which reduces them to their base form (e.g., 'running' becomes 'run').

Next comes feature extraction. Now, machine learning models can't directly process raw text. They need numeric representations. Here, spaCy’s preprocessed output comes in handy. One common approach is to create a Bag-of-Words (BoW) representation or, more commonly, TF-IDF (Term Frequency-Inverse Document Frequency) vectors using scikit-learn. Each unique word in the corpus becomes a feature, and the corresponding value represents its frequency or importance within each document. Alternatively, you can utilize spaCy’s word embeddings (if using models with embeddings enabled) that capture the semantic meaning of words. Embeddings generally lead to better performance, particularly with smaller datasets.

Finally, you feed these extracted features, now in a numeric format, into a classification model. This step is where libraries like scikit-learn or TensorFlow/Keras really come into play. SpaCy’s role, at this stage, is primarily done; its task was to prepare the data in a way that the chosen model can understand and learn from.

Now, let's make it concrete with some code examples. I'll use scikit-learn for the model training since it's often the initial starting point:

**Example 1: Using TF-IDF for Classification with a Simple Linear Model**

```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data - replace with your own dataset
texts = [
    "This is a great movie, I loved it.",
    "I hated this movie. It was terrible.",
    "The food was amazing, highly recommend.",
    "Service was awful, never again.",
    "A very interesting book.",
    "This book is boring and dull."
]
labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

processed_texts = [preprocess_text(text) for text in texts]

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(processed_texts)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, labels, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(solver='liblinear') # Using 'liblinear' for small dataset efficiency
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

In this example, the `preprocess_text` function uses spaCy to lemmatize and remove stop words, yielding better feature representations. We then used `TfidfVectorizer` from scikit-learn to convert these preprocessed texts into numerical vectors before training a simple logistic regression classifier.

Now, let’s move on to a more advanced approach using pre-trained word embeddings:

**Example 2: Using pre-trained word embeddings with an average pooling approach.**

```python
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data - replace with your own dataset (same as above)
texts = [
    "This is a great movie, I loved it.",
    "I hated this movie. It was terrible.",
    "The food was amazing, highly recommend.",
    "Service was awful, never again.",
    "A very interesting book.",
    "This book is boring and dull."
]
labels = [1, 0, 1, 0, 1, 0]

# Load a spaCy model with embeddings
nlp = spacy.load("en_core_web_md") # Use 'md' or 'lg' models for embeddings

def get_doc_embedding(text):
    doc = nlp(text)
    #Average word embeddings for a document embedding
    return np.mean([token.vector for token in doc if not token.is_stop and not token.is_punct and token.has_vector], axis=0)


embeddings = [get_doc_embedding(text) for text in texts]
# Filter out any NaN results by taking care of zero length document
filtered_embeddings = [emb for emb in embeddings if not (np.all(np.isnan(emb)))]
filtered_labels = [label for index, label in enumerate(labels) if not np.all(np.isnan(embeddings[index]))]


X_train, X_test, y_train, y_test = train_test_split(filtered_embeddings, filtered_labels, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

Here, instead of TF-IDF, we use pre-trained word vectors from spaCy and create a document embedding by averaging the embeddings of all words in the document (excluding stop words and punctuation). This can capture the semantics and often improves classification. It's worth noting here that for larger datasets or more complex tasks, a weighted average of word embeddings or other more sophisticated techniques might be preferable.

Finally, for very large datasets or situations requiring more powerful models, you would often integrate with deep learning frameworks. The below snippet shows how you can achieve this:

**Example 3: Integration with a Neural Network (Conceptual)**

```python
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Sample data - replace with your own dataset
texts = [
    "This is a great movie, I loved it.",
    "I hated this movie. It was terrible.",
    "The food was amazing, highly recommend.",
    "Service was awful, never again.",
    "A very interesting book.",
    "This book is boring and dull."
]
labels = [1, 0, 1, 0, 1, 0]

nlp = spacy.load("en_core_web_sm")

def get_token_ids(text):
    doc = nlp(text)
    return [token.i for token in doc]  #returns the index of each word
tokenized_texts = [get_token_ids(text) for text in texts]

max_length = max(len(tokens) for tokens in tokenized_texts) #find the max length
padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokenized_texts, maxlen=max_length, padding='post') #pad to max length


vocab_size = len(set(word.i for doc in nlp.pipe(texts) for word in doc)) + 1 #vocab size must contain the unkonwn word id

X_train, X_test, y_train, y_test = train_test_split(padded_tokens, labels, test_size=0.2, random_state=42)

# Model creation
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=128, mask_zero = True)(input_layer) # embedding layer with masking
pooling_layer = GlobalAveragePooling1D()(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(pooling_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 10)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")
```
In this conceptual neural network example we're utilizing a basic neural network. We are tokenizing our input and passing that through an embedding layer before averaging it in the global pooling layer. This model is far more flexible for complex tasks, especially with the right architectural choices. Note that a more robust solution would include additional steps and layers such as regularization and hyperparameter tuning.

For further learning and a deeper understanding of the techniques, I would strongly recommend delving into Jurafsky and Martin's "Speech and Language Processing," a definitive guide to the field. Also, "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron is incredibly helpful for the practical side of model building. These resources, combined with your own hands-on practice, will equip you with the necessary skills to tackle a wide range of text classification tasks using spaCy and related technologies. The key takeaway is that spaCy acts as a powerful preprocessing engine, not the machine learning algorithm itself, and when used in conjunction with machine learning libraries it can achieve robust results.
