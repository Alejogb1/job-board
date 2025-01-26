---
title: "What are the flaws in my dataset preprocessing?"
date: "2025-01-26"
id: "what-are-the-flaws-in-my-dataset-preprocessing"
---

Specifically, the dataset includes text data containing user reviews for various products, and you are experiencing inconsistent model performance during training.

The root cause of your inconsistent model performance likely stems from inadequate preprocessing of your text data, particularly in how you handle inconsistencies within user reviews. My experience working on similar sentiment analysis and product review projects indicates that seemingly minor variations in text representation can have a significant detrimental impact on model training, leading to fluctuating results.

A primary flaw often observed in text preprocessing pipelines revolves around the treatment of capitalization and punctuation. Machine learning models, by their nature, perceive distinct capitalization and punctuation patterns as unique tokens, thus differentiating between "Good" and "good," or "Fantastic!" and "Fantastic". While these variations are semantically similar for human understanding, they dramatically inflate the vocabulary size, dilute the representation of actual words, and can lead to sparse matrices. Moreover, the presence of inconsistent or improperly handled punctuation within tokens themselves (e.g., “co-star”, “co - star”) can further impede efficient learning by fragmenting words that should be treated as a single unit.

Another common mistake is the insufficient management of textual noise, specifically the inclusion of HTML tags, special characters, and irrelevant numerical data embedded within the text. While these elements are useful within the context of the web, they represent uninformative noise for the task of textual analysis and sentiment classification. Preserving HTML tags like `<br>` or `<div>` in the text unnecessarily expands the vocabulary and does not contribute meaningfully to the model's understanding of the sentiment. Similarly, maintaining numerical values that are unrelated to the semantic content of the text ("Rating: 4/5", "Order #12345") adds more uninformative features that compete with relevant signals.

Additionally, consider the impact of stop words. While stop word removal is commonly used to reduce noise, the default sets of stop words available in many natural language processing libraries may not be perfectly aligned with the nuances of your data, particularly when dealing with specific product reviews. Words considered stopwords in general, like “not” or “very,” might be crucial in determining the sentiment of a review, specifically in product descriptions. Overzealous stop word removal risks erasing valuable contextual information, thereby affecting the quality of features used in training. Finally, the absence of a stemmatization or lemmatization step also poses a challenge. Without these steps, related words with similar meanings, like "running," "ran," and "runs," are treated as completely different tokens. This again contributes to an increase in vocabulary size without a corresponding increase in semantic information and can hinder model generalization.

To address these issues, you'll need to implement a robust preprocessing pipeline. Here are examples based on my previous projects, illustrating some necessary techniques:

**Code Example 1: Normalizing Case and Punctuation**

This script segment demonstrates how to standardize casing and punctuation, which helps reduce feature sparsity. I use Python's regular expression library, re, along with string manipulation methods for the task.

```python
import re
import string

def normalize_text(text):
  """Normalizes text by converting to lowercase and removing punctuation."""
  text = text.lower()
  text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub(r'\s+', ' ', text).strip() # Remove multiple spaces
  return text

review1 = "This PRODUCT was AMAZING!!! I loved it!"
review2 = "this product was amazing. i loved it."

normalized_review1 = normalize_text(review1)
normalized_review2 = normalize_text(review2)

print(f"Original review 1: {review1}")
print(f"Normalized review 1: {normalized_review1}")
print(f"Original review 2: {review2}")
print(f"Normalized review 2: {normalized_review2}")
```

In this example, `normalize_text` function converts all characters to lowercase. It then uses a regular expression to remove all standard punctuation marks. The final step replaces multiple space occurrences with a single space and removes leading or trailing whitespace. The transformation converts both `review1` and `review2` to the same string thereby allowing the model to recognize their semantic equivalency.

**Code Example 2: Removing HTML and Numbers**

This section showcases how to strip HTML tags and numbers using regular expressions, another technique required for cleaning raw text data.

```python
import re

def remove_html_and_numbers(text):
  """Removes HTML tags and numbers from text."""
  text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
  text = re.sub(r'\d+', '', text)   # Remove numerical data
  text = re.sub(r'\s+', ' ', text).strip() # Remove multiple spaces
  return text

review3 = "<div>This is a great item!</div> Rating 4/5. Order #12345"
cleaned_review3 = remove_html_and_numbers(review3)

print(f"Original review 3: {review3}")
print(f"Cleaned review 3: {cleaned_review3}")
```

The `remove_html_and_numbers` function first removes HTML tags using the `r'<.*?>'` regex. It then removes all sequences of numerical digits using `r'\d+'` regex and normalizes whitespace as in the previous example. After applying the transformation, `review3` is stripped of all HTML content, as well as irrelevant numerical data, thereby retaining only the pertinent text.

**Code Example 3: Selective Stopword Removal and Lemmatization**

This example demonstrates stopword removal using NLTK and lemmatization using SpaCy. Notice that I am not removing the word "not", as it may be important for sentiment analysis tasks. Also, notice how lemmatization reduces words to their base form.

```python
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
stop_words.remove('not') # Removing 'not' from standard stop words.

def process_text(text):
    """Removes stopwords (excluding 'not') and performs lemmatization"""
    tokens = [token.lemma_ for token in nlp(text) if token.text.lower() not in stop_words]
    return " ".join(tokens)

review4 = "The product was not good, not at all. Running fast, run, runs, ran."
processed_review4 = process_text(review4)

print(f"Original review 4: {review4}")
print(f"Processed review 4: {processed_review4}")
```

In this example, I load the NLTK standard stopword list and then manually remove the word "not". I then load the SpaCy language model for lemmatization. I process text by tokenizing it using SpaCy, performing lemmatization, and excluding words that are present in my modified stopword list. Lemmatizing ‘Running’, ‘run’, ‘runs’, and ‘ran’ into the same form i.e ‘run’ and keeping the "not" from the original review, enables the model to better grasp the semantic meaning of the input.

To improve your preprocessing further, I would suggest investigating the following resources. First, explore texts on Information Retrieval, which discuss in detail strategies for building robust preprocessing pipelines. Second, consult documentation for natural language processing libraries like NLTK and SpaCy, paying special attention to best practices for tokenization, stop word removal, stemming/lemmatization. Finally, study research papers related to text preprocessing, particularly those that focus on sentiment analysis or product review analysis as these typically focus on common issues and effective techniques used in this context.

In summary, the inconsistent model performance you are observing is likely a consequence of insufficient preprocessing of the text data. A comprehensive pipeline that handles case, punctuation, noise, selective stopword removal, and lemmatization, like the examples I demonstrated above, should be implemented to improve model consistency. By addressing these shortcomings, you will be able to build models that are more robust and reliable.
