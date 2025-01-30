---
title: "How can text be classified into two categories?"
date: "2025-01-30"
id: "how-can-text-be-classified-into-two-categories"
---
Text classification, particularly into binary categories, is a foundational task in natural language processing (NLP). My experience building a spam detection system for an email service highlighted the practical challenges involved, moving beyond textbook examples to address nuanced real-world data.  The core idea involves transforming textual data into numerical representations that a machine learning model can interpret and then training that model to predict one of two predefined categories.

Fundamentally, the process unfolds in three primary stages: data preparation, feature extraction, and model training and evaluation. Data preparation is arguably the most crucial, often demanding the most effort. It begins with collecting a sufficient, labeled dataset where each text sample is paired with its corresponding category label (e.g., "spam" or "not spam"). The quality and balance of this data significantly impact the final model performance. This step might include dealing with imbalances—where one category has significantly more samples than the other—as well as handling missing values, inconsistencies in text format, and noise.

Next, the raw text needs to be converted into a numerical format that machine learning algorithms can use.  This process is called feature extraction. Simple approaches include Bag-of-Words (BoW), where each unique word in the dataset becomes a feature, and the value for each text sample is the word's frequency in the document.  While straightforward, BoW can result in very high-dimensional data, particularly with large vocabularies, and can also suffer from the 'curse of dimensionality' making it difficult to train an effective model. Term Frequency-Inverse Document Frequency (TF-IDF) is an improvement, weighing words based on their frequency in the document and inversely to their frequency across all documents. Words that are very common across most text samples, such as "the" or "a", are thus down weighted, giving more weight to discriminative terms. N-grams, which capture sequences of n words, can also improve performance, as they retain more contextual information than individual words.  More advanced techniques leverage word embeddings, where words are mapped to dense vectors in a lower-dimensional space, capturing semantic relationships between words, or contextual embeddings from transformer based language models.

Finally, a classification algorithm is chosen and trained using the numerical data. Logistic Regression, Support Vector Machines (SVMs), and tree based methods like Random Forests or Gradient Boosting are commonly employed for binary classification tasks. The choice depends on factors such as dataset size, dimensionality of the feature space, and required performance. The data is typically split into training, validation, and test sets. The model is trained using the training data, its performance is evaluated on the validation set to tune hyperparameters, and the final evaluation is conducted on the unseen test data to get an unbiased measure of its performance. Evaluation metrics often include accuracy, precision, recall, F1-score, and area under the receiver operating characteristic curve (AUC-ROC), which allows one to assess performance of the classifier while considering the tradeoff between precision and recall for different classification thresholds.

Here are some code examples demonstrating these concepts using Python, assuming an appropriately setup environment with `scikit-learn`.

**Example 1: Bag-of-Words with Logistic Regression**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your own)
documents = [
    "This is a great movie.",
    "I hated this film.",
    "The food was amazing.",
    "Terrible service. Would not recommend.",
    "A truly fantastic experience.",
    "This was the worst meal ever."
]
labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Initialize Bag of Words vectorizer
vectorizer = CountVectorizer()

# Fit and transform training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transform test data
X_test_vectorized = vectorizer.transform(X_test)


# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_vectorized, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_vectorized)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

This example demonstrates the most basic approach: converting text into numerical representations using Bag-of-Words (`CountVectorizer`), then training a logistic regression model. The `fit_transform` method is used on the training set to both learn the vocabulary and convert it to numerical feature vectors.  The `transform` method is applied to the testing data to apply the vocabulary learned from training data. The accuracy provides an overview of the classifier's performance on the test data. However, for small dataset like this, this score may not be indicative of generalization error on unseen data.

**Example 2: TF-IDF with Support Vector Machine**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data (replace with your own)
documents = [
    "This is a great movie.",
    "I hated this film.",
    "The food was amazing.",
    "Terrible service. Would not recommend.",
    "A truly fantastic experience.",
    "This was the worst meal ever."
]
labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)


# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transform test data
X_test_vectorized = vectorizer.transform(X_test)

# Initialize SVM model
model = SVC(kernel='linear') #linear kernel is often good for text data

# Train the model
model.fit(X_train_vectorized, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_vectorized)

# Print classification report
print(classification_report(y_test, y_pred))
```

Here, instead of Bag-of-Words, we use TF-IDF (`TfidfVectorizer`), which should improve upon plain word counts by weighing terms based on their frequency across documents. An SVM with a linear kernel (`SVC(kernel='linear')`) is chosen for classification, offering an effective approach particularly when high dimensional feature spaces are created from text data. The classification report gives a more detailed analysis, showing the precision, recall, f1-score, and support for each class.

**Example 3: Pre-trained Word Embeddings with Logistic Regression**

```python
import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load a pre-trained spaCy model
nlp = spacy.load("en_core_web_sm") # use a suitable model here, downloading if necessary

# Sample data (replace with your own)
documents = [
    "This is a great movie.",
    "I hated this film.",
    "The food was amazing.",
    "Terrible service. Would not recommend.",
    "A truly fantastic experience.",
    "This was the worst meal ever."
]
labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

# Function to get word embeddings
def get_embedding(text):
  doc = nlp(text)
  return np.mean([token.vector for token in doc], axis=0)

# Apply the embedding to each doc
X = [get_embedding(doc) for doc in documents]
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

This example uses pre-trained word embeddings from the `spaCy` library to represent text. Each text is processed, word vectors are extracted, and their mean is calculated to create a numerical vector representing the entire sentence. These embeddings capture contextual meanings and should result in a performance improvement over simple Bag-of-Words or TF-IDF.

For further exploration and skill development, I recommend focusing on texts covering various machine learning techniques with a specific focus on their application to NLP, especially those dealing with model evaluation, and exploring documentation from libraries such as `scikit-learn` and `spaCy`, alongside tutorials covering various text classification problems. Understanding the strengths and limitations of each technique allows one to tailor the method to a specific need. Furthermore, researching data preprocessing steps, dealing with imbalanced datasets, and utilizing hyperparameter tuning techniques will improve classification performance. Continuous experimentation with different parameters and feature extraction methods are key to successful text classification.
