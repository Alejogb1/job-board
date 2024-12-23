---
title: "What is the best method for text classification using spaCy?"
date: "2024-12-16"
id: "what-is-the-best-method-for-text-classification-using-spacy"
---

Alright,  Text classification with spaCy is a frequent flyer on my projects, and I’ve seen it approached in a few ways over the years, some much more effective than others. It’s not so much about finding a single "best" method, but about understanding the strengths and limitations of different techniques and choosing what fits the particular data and classification problem. I'll break down the approaches that, in my experience, tend to produce the most reliable results, along with some example code to illuminate these points.

First, let's move past the naive methods. You might be tempted to jump straight into spaCy's built-in text categorizer, which uses a straightforward logistic regression model. While easy to implement, it often falls short for complex scenarios. For genuinely robust classification, we typically need more sophisticated models and better feature extraction.

The core principle here is to transform your text data into a suitable numerical representation that a machine learning model can interpret. SpaCy excels at pre-processing—tokenization, part-of-speech tagging, lemmatization, and dependency parsing—but it doesn’t inherently contain complex classification algorithms. We need to pair spaCy with robust machine learning libraries like scikit-learn or, for deep learning, frameworks like TensorFlow or PyTorch. Let’s dive into the three main approaches I generally favor.

**Approach 1: Feature Engineering with spaCy and Scikit-learn**

This method focuses on using spaCy to derive meaningful features and feeding those features into a traditional machine learning classifier from scikit-learn. Instead of using raw text, you extract linguistic properties, such as n-grams, term frequency-inverse document frequency (tf-idf), or even part-of-speech counts, to enhance your model's understanding. This is often the most efficient starting point.

Consider a scenario where I was working on categorizing customer reviews into sentiment labels (positive, negative, neutral). We started with raw text and had dismal results. Here’s how we approached it, using n-gram features:

```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd

# Assume you have a pandas DataFrame named 'df' with columns 'text' and 'label'
# For example: df = pd.DataFrame({'text': ['Great product!', 'Terrible service.', ...], 'label': ['positive', 'negative', ...] })
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Helper function to tokenize and lemmatize text
def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]

# Create a pipeline with tfidf and a classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2), max_features = 1000)),  # Including bigrams
    ('clf', LogisticRegression(random_state=42, solver='liblinear')) # A robust and well-suited classifier
])

# Assuming data is loaded into 'df'
df = pd.read_csv('reviews.csv') # Replace 'reviews.csv' with the path to your data.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
```

In this snippet, `TfidfVectorizer` converts tokenized and lemmatized text into tf-idf weighted n-grams (unigrams and bigrams here). We then use `LogisticRegression` as our classification algorithm within a scikit-learn pipeline.  This approach leverages both spaCy’s linguistic features and scikit-learn’s powerful machine learning capabilities.

**Approach 2: Using spaCy’s Embeddings and Neural Networks**

A more sophisticated approach uses spaCy's pre-trained word embeddings to represent text. These embeddings capture semantic relationships between words, allowing models to generalize more effectively.  While spaCy provides word vectors, for more complex classification tasks, we typically need to combine these into sentence or document embeddings and then feed them into a custom-built neural network, typically with either Tensorflow or PyTorch.

Imagine I was tasked with classifying technical articles into broad domains (e.g., networking, programming, security). Using averaged word embeddings often worked decently, but a custom CNN network improved results drastically.

```python
import spacy
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

# Load a medium or larger spaCy model for better embeddings
nlp = spacy.load("en_core_web_md") # or "en_core_web_lg"


#Function to get embedding for text
def get_doc_embedding(text):
    doc = nlp(text)
    #average vector of the tokens in the document
    embeddings = [token.vector for token in doc if not token.is_punct and not token.is_stop]
    if not embeddings: #handling the empty embeddings
        return np.zeros(nlp.vocab.vectors_length)
    return np.mean(embeddings, axis=0)

# Assume you have a pandas DataFrame named 'df' with columns 'text' and 'label'
# For example: df = pd.DataFrame({'text': ['Great article!', 'Terrible Explanation.', ...], 'label': ['networking', 'programming', ...] })
df = pd.read_csv('articles.csv') #replace 'articles.csv' with the path to your data.

#preprocess labels, ensuring numerical output
labels = df['label'].astype('category').cat.codes
num_classes = len(df['label'].astype('category').cat.categories)

# Convert texts to embeddings
embeddings = np.array([get_doc_embedding(text) for text in df['text']])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Build a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(nlp.vocab.vectors_length,)),
    Dense(num_classes, activation='softmax')  # Output layer with number of unique classes
])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")
```

This example showcases using spaCy’s word embeddings to create document-level embeddings, followed by training a basic neural network for classification using TensorFlow. This often captures more subtle relationships than feature engineering alone, at the cost of increased computational demands.

**Approach 3: Fine-tuning Pre-trained Language Models**

The most advanced and often best-performing approach involves fine-tuning a pre-trained language model directly on your classification task. Instead of relying on fixed word embeddings, we can modify the transformer-based models (such as BERT) for the target domain. SpaCy has integrations with libraries such as transformers for this. This approach can yield the best performance but requires significant data and computational resources.

In a previous job, I worked on classifying complex legal documents, where subtle nuances in language made classical methods perform poorly. Using fine-tuned models gave us much more accurate and robust outcomes.

```python
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.language import Language
from sklearn.model_selection import train_test_split
import random
import pandas as pd
# Load a pre-trained transformer-enabled spaCy model
nlp = spacy.load("en_core_web_trf")


def load_data_spacy(data, test_size=0.2):
    # Assume you have a pandas DataFrame named 'data' with columns 'text' and 'label'
    # For example: data = pd.DataFrame({'text': ['Legal Doc A', 'Legal Doc B', ...], 'label': ['contract', 'statute', ...] })

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_examples = []
    test_examples = []
    for _, row in train_data.iterrows():
        doc = nlp.make_doc(row["text"])
        example = Example.from_dict(doc, {"cats": {row["label"]: 1.0}})
        train_examples.append(example)
    for _, row in test_data.iterrows():
        doc = nlp.make_doc(row["text"])
        example = Example.from_dict(doc, {"cats": {row["label"]: 1.0}})
        test_examples.append(example)
    return train_examples, test_examples

# Assume you have a pandas DataFrame named 'df' with columns 'text' and 'label'
# For example: df = pd.DataFrame({'text': ['Legal Doc A', 'Legal Doc B', ...], 'label': ['contract', 'statute', ...] })
df = pd.read_csv('legal_docs.csv') # Replace 'legal_docs.csv' with the path to your data.


train_examples, test_examples = load_data_spacy(df, test_size=0.2)
# Train the text categorizer

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.initialize()
    for i in range(5): #epochs
        losses = {}
        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.05))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)
        print(f"Losses at iteration {i}: {losses}")


# Evaluate the model
def evaluate_model(nlp, test_examples):
    scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = 0
    fp = 0
    fn = 0
    for example in test_examples:
        doc = nlp(example.text)
        predicted_category = max(doc.cats, key=doc.cats.get)
        actual_category = list(example.get_aligned_spans(doc).values())[0].text
        if predicted_category == actual_category:
            tp +=1
        elif actual_category in doc.cats.keys():
            fp +=1
        else:
            fn +=1
    if (tp+fp) > 0:
        scores["precision"] = tp / (tp + fp)
    if (tp+fn) > 0:
         scores["recall"] = tp / (tp + fn)
    if (scores["precision"] + scores["recall"]) > 0:
        scores["f1"] = 2 * (scores["precision"] * scores["recall"]) / (scores["precision"] + scores["recall"])
    return scores

scores = evaluate_model(nlp, test_examples)
print(f"Evaluation: {scores}")

```

This example demonstrates fine-tuning a text categorization model using a transformer, leveraging spaCy’s tools for creating training examples and controlling the training process.

**Further Reading**

For a deep dive into these techniques, I recommend exploring the following resources:

*   *Natural Language Processing with Python* by Steven Bird, Ewan Klein, and Edward Loper: Excellent for a foundational understanding of NLP principles and classic techniques.
*   *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron: Provides a practical guide to machine learning with scikit-learn and TensorFlow, including in-depth sections on text data processing.
*  *Deep Learning with Python* by François Chollet: A clear and comprehensive overview of deep learning concepts and methods, including applications to NLP tasks.
*  *Attention is All You Need* by Ashish Vaswani et al: The seminal paper for transformers architecture. This can be complex for the novice but vital for more advanced applications.

In summary, the "best" method really depends on your specific problem. For relatively simple classifications, feature engineering with spaCy and scikit-learn is a great starting point. For more complex scenarios, using spaCy's word embeddings with custom neural networks can yield better results. If you're dealing with complex data or need maximal accuracy, fine-tuning a pre-trained transformer model is the way to go. The key is to experiment, iterate, and choose the method that aligns with your resources, data, and desired performance. It's not a one-size-fits-all solution, rather a toolkit you can draw from.
