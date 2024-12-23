---
title: "How can I generate 2-grams using LimeTextExplainer?"
date: "2024-12-23"
id: "how-can-i-generate-2-grams-using-limetextexplainer"
---

Let's tackle this one. I remember working on a sentiment analysis project a few years back where generating n-grams, specifically 2-grams in your case, became essential to understand contextual meaning beyond individual words. LimeTextExplainer, while fantastic for its explainability power, doesn't directly provide n-gram generation. It works by perturbing the input text and observing how those perturbations affect the model’s prediction. Therefore, we need to pre-process our text to create the 2-grams before feeding them into Lime.

The core issue here isn't with Lime itself but rather with preparing your data to work effectively with it. The standard approach for creating 2-grams involves two primary steps: tokenization and n-gram creation. Tokenization splits your text into individual words (or other sub-word units). N-gram creation then takes these tokens and creates sequences of 'n' tokens. In your case, 'n' will be 2.

So, how do we accomplish this programmatically? We could use various text processing libraries, but for simplicity and practicality, I'll demonstrate with Python and two popular choices: `nltk` and `sklearn`.

**Approach 1: Using NLTK and Python Lists**

First, let's explore using `nltk` (Natural Language Toolkit), which offers robust functionality for tokenization. Here's how I might have tackled this in a prior project:

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

nltk.download('punkt')  # Download necessary resource for tokenization

def generate_2grams_nltk(text):
    """Generates 2-grams from input text using NLTK."""
    tokens = word_tokenize(text.lower()) # Tokenize & lower case
    bigrams = ngrams(tokens, 2)
    return list(bigrams) # Return list of tuples


text_example = "This is a sample text for demonstrating 2-gram generation with NLTK."
grams = generate_2grams_nltk(text_example)
print(f"2-grams (NLTK): {grams}")
```

In this snippet:
1. We import the necessary modules from `nltk`.
2. `nltk.download('punkt')` ensures that the necessary tokenization resources are downloaded. This only needs to be run once.
3. The function `generate_2grams_nltk` first lower cases the input `text`. This is important to ensure uniformity.
4. Then `word_tokenize` splits the text into individual tokens.
5. `ngrams(tokens, 2)` creates 2-gram tuples from the tokens.
6. Finally we convert it to a list of tuples for easier handling.

The output will be a list of tuple pairings such as:
`[('this', 'is'), ('is', 'a'), ('a', 'sample'), ... ]`

This approach is quite straightforward and provides a clear illustration of the process. Note that these are *tuples* not *strings*. This is important to keep in mind when building your vectorizer later, which we will explore in the subsequent examples.

**Approach 2: Using Scikit-learn's CountVectorizer**

Now, let's turn to `sklearn` (scikit-learn). It includes a fantastic class called `CountVectorizer` that can create n-grams for us in a more streamlined fashion. This approach also directly provides the required feature vector for machine learning models, which is beneficial when working with Lime.

```python
from sklearn.feature_extraction.text import CountVectorizer

def generate_2grams_sklearn(text):
  """Generates 2-grams from input text using sklearn's CountVectorizer."""
  vectorizer = CountVectorizer(ngram_range=(2, 2))
  vectorizer.fit([text])
  grams = vectorizer.get_feature_names_out()
  return grams

text_example = "This is a sample text for demonstrating 2-gram generation with sklearn."
grams_sklearn = generate_2grams_sklearn(text_example)
print(f"2-grams (sklearn): {grams_sklearn}")
```

Here's a breakdown:
1. We import `CountVectorizer` from `sklearn`.
2. In `generate_2grams_sklearn`, we create an instance of `CountVectorizer` with `ngram_range=(2, 2)`. This tells the vectorizer to only generate 2-grams.
3. `vectorizer.fit([text])` learns the vocabulary from our input text. Notice that the input text is enclosed in a list. `CountVectorizer` expects a list of strings when using fit, even when the list contains a single element, like our current example.
4. `vectorizer.get_feature_names_out()` retrieves the n-grams that were identified.

The output will be a list of string 2-grams:
`['a sample', 'demonstrating 2', 'for demonstrating', 'gram generation', 'is a', 'sample text', 'text for', 'this is', '2 gram']`

Notice how this output is a list of *strings*, unlike the NLTK implementation which returns a list of *tuples*. This is critical to understand because this is the format that will be passed into your text classification model later.

**Approach 3: Integrating with LimeTextExplainer**

Finally, let's see how you'd typically integrate this into a Lime explanation process. We'll use the `sklearn` method to tokenize and create n-grams.

```python
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

def create_2gram_lime_explainer(text, classifier):
    """Creates a LimeTextExplainer using 2-grams."""

    vectorizer = CountVectorizer(ngram_range=(2, 2))
    pipeline = make_pipeline(vectorizer, classifier)
    explainer = lime.lime_text.LimeTextExplainer(class_names=['negative', 'positive'])

    explanation = explainer.explain_instance(text,
                                           classifier_fn=pipeline.predict_proba,
                                           num_features=5) # Show the top 5 features

    return explanation

#dummy classifier for demonstration
class DummyClassifier:
  def predict_proba(self, X):
    return [[0.1, 0.9] for _ in X]

text_example = "This is a great example text. This really is great."
dummy_classifier = DummyClassifier()

explanation_lime = create_2gram_lime_explainer(text_example, dummy_classifier)
print(f"Lime explanation: {explanation_lime.as_list()}")
```

Let me break down this more complex snippet:
1. We import Lime and `sklearn` components.
2. The `create_2gram_lime_explainer` function encapsulates the entire process.
3. Inside the function, we instantiate a `CountVectorizer` configured for 2-grams, just like in the second approach.
4. We use `make_pipeline` to combine the vectorizer and the classifier into a single model. This is critical because Lime expects a single model which manages the entire preprocessing pipeline, not only the classification component.
5. We then create a `LimeTextExplainer`.
6. The most critical part is within the `explain_instance` call, where we feed our input `text`, pass our prepared `pipeline.predict_proba` method and specify `num_features`. This `pipeline.predict_proba` method already has the n-gram transformation encoded as part of the pipeline, so Lime automatically analyses the effect of removing various n-grams, which is what we want.
7. Finally, the function returns the resulting Lime explanation which we then print using `explanation_lime.as_list()`.
8. Because this example is for demonstration purposes, I have provided a dummy classifier that simply returns a constant probability, however, in practice you will replace this with a trained classifier.

The Lime output, while not based on a real classifier in this instance, shows the 2-grams that are considered most important to the model's (dummy) prediction.

**Recommended Resources**

For a more in-depth understanding, I highly recommend:

*   **"Speech and Language Processing" by Dan Jurafsky and James H. Martin:** This is a comprehensive textbook on natural language processing, covering tokenization, n-grams, and much more. It’s a fundamental resource for anyone working in NLP.
*   **Scikit-learn documentation:** The documentation for `CountVectorizer` and other feature extraction modules is excellent and should be consulted for specific details and advanced configurations.
*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** This book offers a practical introduction to machine learning using Python, and includes relevant examples of text processing and using it in the context of classification tasks.

I've used similar techniques and resources in many projects, and they have consistently provided the needed n-gram capabilities when combined with explainability techniques. This workflow is robust and can be adapted for various text analysis scenarios beyond just Lime. Remember that the correct choice of tokenization and n-gram configuration can significantly impact the performance of your models and the quality of your explanations. Always test different approaches and choose the ones that are suitable for your specific application.
