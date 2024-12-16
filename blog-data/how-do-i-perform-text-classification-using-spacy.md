---
title: "How do I perform text classification using spaCy?"
date: "2024-12-16"
id: "how-do-i-perform-text-classification-using-spacy"
---

Alright, let’s tackle text classification with spaCy. It's a common task, and I've personally spent a good chunk of time implementing various solutions over the years. I'll walk you through the practical aspects, not just the theoretical, as that’s where things tend to get interesting, especially when you’re working with real-world data. I will offer some code snippets along the way and also mention places where you can further your understanding.

At its core, text classification is about assigning predefined categories or labels to a given text. SpaCy itself doesn't offer out-of-the-box classification algorithms, like a scikit-learn classifier. Instead, spaCy excels at the crucial preliminary steps: efficiently processing the raw text into a format suitable for machine learning algorithms. Think of it as the pre-processing powerhouse. You leverage spaCy’s tokenization, part-of-speech tagging, and named entity recognition to engineer features, which then get fed into a separate classifier. I've always found this separation of concerns to be beneficial when debugging and optimizing pipelines.

Let's get into it. The process generally looks like this: Load a spaCy model, process text, extract features, and use those features to train a classifier. Let's start with feature extraction.

**Feature Engineering with spaCy:**

The first step involves using spaCy’s capabilities to extract meaningful features from your text. The simplest feature set might consist of a bag-of-words representation, which just counts the occurrences of each token. However, this approach often lacks context. More sophisticated approaches include:

1.  **Token Counts and TF-IDF:** Counting the tokens and utilizing term frequency-inverse document frequency (TF-IDF) allows you to identify the important words in your corpus. I've used this with moderate success. SpaCy doesn't directly calculate TF-IDF, but we can combine it easily with libraries such as scikit-learn.

2.  **Part-of-Speech (POS) Tags:** The grammatical role of a word can be quite informative. Instead of just looking at words, you can look at the frequency of nouns, verbs, adjectives, and so on. I’ve found that including POS information can drastically improve performance in some scenarios, especially in sentiment analysis.

3.  **Named Entity Recognition (NER):** Identifying entities like people, organizations, and locations can help to categorize texts more accurately, particularly those that mention them. I once worked on a large news dataset; NER drastically helped categorize articles based on the entities they mentioned, enabling better analysis of the data.

Now, let's look at some code snippets to see these things in action.

**Code Snippet 1: Basic Tokenization and Feature Extraction**

```python
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct] # Remove stop words and punctuation
    pos_tags = [token.pos_ for token in doc]
    named_entities = [ent.label_ for ent in doc.ents]

    return {"tokens": tokens, "pos_tags": pos_tags, "named_entities": named_entities}


text_example = "Apple is a large technology company based in Cupertino. It has numerous products, such as the iPhone. I had a meeting with John last week. It was very productive."
features = extract_features(text_example)
print(features)

# Count most common tokens
token_counts = Counter(features['tokens'])
print(token_counts.most_common(5))

pos_tag_counts = Counter(features['pos_tags'])
print(pos_tag_counts.most_common(5))

ner_counts = Counter(features['named_entities'])
print(ner_counts.most_common(5))
```

This first snippet demonstrates simple token extraction, removal of stop words and punctuation, as well as extracting and counting part-of-speech tags and named entities. Notice how we collect these features into a dictionary for each text.

**Code Snippet 2: Combining spaCy with scikit-learn for TF-IDF**

```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha] # lemmatize, lowercase, remove non-alpha
    return " ".join(tokens) # Return as single string for vectorizer

# Sample Data, real data would be large
texts = [
    "This is a great movie, I enjoyed it immensely.",
    "The film was terrible and boring.",
    "I absolutely loved the performance of the actors.",
    "The plot was confusing and hard to follow.",
    "It's a must watch movie!"
]
labels = [1, 0, 1, 0, 1] # 1 for positive sentiment, 0 for negative sentiment

preprocessed_texts = [preprocess_text(text) for text in texts]

# Convert texts to tfidf
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed_texts)

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, labels, test_size=0.2, random_state=42)

# train the model (LogisticRegression)
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Here, we utilize scikit-learn’s TfidfVectorizer to convert our preprocessed texts (lemmatized, lowercased, and with stop words and punctuation removed) into TF-IDF vectors. We then use a simple logistic regression classifier to do text classification. This workflow shows how you can combine spaCy’s text processing capabilities with external libraries to perform a full text classification process.

**Code Snippet 3: Custom spaCy component for classification**

```python
import spacy
from spacy.language import Language
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@Language.factory("text_classifier")
class TextClassifier:
    def __init__(self, nlp, name):
        self.nlp = nlp
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(preprocessor=self.preprocess_text)),
            ('clf', LogisticRegression(random_state=42))
        ])
        self.is_trained = False

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return " ".join(tokens)

    def fit(self, texts, labels):
        self.model.fit(texts, labels)
        self.is_trained = True

    def predict(self, text):
        if not self.is_trained:
            raise Exception("Model needs to be trained first.")
        return self.model.predict([text])[0]

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("text_classifier", last=True)
classifier = nlp.get_pipe("text_classifier")

# Sample Data again
texts = [
    "This is a great movie, I enjoyed it immensely.",
    "The film was terrible and boring.",
    "I absolutely loved the performance of the actors.",
    "The plot was confusing and hard to follow.",
    "It's a must watch movie!"
]
labels = [1, 0, 1, 0, 1]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Fit the classifier
classifier.fit(X_train, y_train)

# Make predictions
y_pred = [classifier.predict(text) for text in X_test]
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# New Text Test
new_text = "This is an okay film."
prediction = classifier.predict(new_text)
print(f"Prediction for '{new_text}': {prediction}")
```
This snippet illustrates how to create a custom spaCy component, wrapping the model training and prediction process within spaCy’s pipeline. This is more integrated way to handle it and makes it usable within the existing spaCy framework. Here, I've created a custom `TextClassifier` that includes the same preprocessing steps, TF-IDF vectorization, and logistic regression. It adds flexibility, especially if you are aiming for a more end-to-end spaCy-centric workflow.

**Further Resources:**

To dive deeper into this topic, I highly recommend the following resources:

*   **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** A definitive textbook on natural language processing which goes into much greater depth about different machine learning algorithms and approaches, particularly in the chapter on text classification.
*   **"Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper:** A very good resource for foundational NLP concepts using Python and NLTK. Although it doesn't focus on spaCy, the conceptual basis will help you develop a solid foundation.
*   **The official spaCy documentation:** This is essential. Start with the tutorials and examples on their website. The quality of the documentation is exceptional. Pay special attention to their guides on custom pipelines, components, and models.

In conclusion, text classification with spaCy involves careful text processing, feature engineering, and using machine learning models. The approach I’ve shown demonstrates the practical integration of spaCy with scikit-learn and custom spaCy components. Remember to evaluate your model properly by splitting data sets into train, validation and test set, especially before putting it into production. Good luck!
