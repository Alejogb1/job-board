---
title: "How can multi-class classifier predictions be improved by filtering classes beforehand?"
date: "2025-01-30"
id: "how-can-multi-class-classifier-predictions-be-improved-by"
---
A common performance bottleneck in multi-class classification arises when certain classes, while present in the training data, are extremely unlikely given the specific input. Pre-filtering these improbable classes can significantly enhance prediction accuracy and efficiency. I've encountered this issue repeatedly, particularly in natural language processing tasks involving large vocabularies. In essence, by narrowing the potential class space, the classifier can focus on the most relevant options, avoiding misclassifications stemming from sparse feature representations for rarely seen classes.

The core idea behind pre-filtering is to apply a separate, simpler model or set of rules prior to the main multi-class classification stage. This pre-filter acts as a gating mechanism, reducing the set of candidate classes to a more manageable subset. The main classifier then only operates on this reduced set. This decoupling allows for optimizations that would be impractical if all classes were considered concurrently.

Consider an application where we're classifying product reviews into fine-grained categories: "Electronics/Smartphones/Apple," "Electronics/Smartphones/Samsung," "Books/Fiction/Science Fiction," "Books/Non-Fiction/Biography," and so on. In such a setup, if a review contains words like "battery," "screen," and "processor," it is extremely improbable that it falls under "Books/Non-Fiction/Biography."  Rather than making the primary classifier sift through *all* categories, including the highly improbable one, we can pre-filter using simpler keyword matching or a coarse-grained classifier to eliminate irrelevant categories.

Let's explore three concrete examples showcasing different pre-filtering approaches.

**Example 1: Keyword-Based Pre-Filtering**

This method relies on identifying keywords associated with each broad class. It is straightforward to implement and computationally inexpensive, making it suitable as an initial step.  I often use this when resources are limited or when the data has a clear keyword-category correlation.

```python
def keyword_prefilter(text, keyword_map):
    """
    Filters classes based on keywords found in the text.

    Args:
    text (str): Input text to be classified.
    keyword_map (dict): A dictionary mapping broad categories to keywords.

    Returns:
    list: List of potentially relevant broad categories.
    """
    relevant_categories = []
    text = text.lower() # Normalize the text
    for category, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in text:
                relevant_categories.append(category)
                break # Move to the next category once a keyword is found
    return list(set(relevant_categories)) # Remove duplicates


# Example Usage:
keyword_map = {
    "Electronics": ["battery", "screen", "processor", "camera"],
    "Books": ["book", "novel", "author", "page"],
    "Food": ["restaurant", "taste", "dish", "recipe"]
}

review1 = "The phone's battery life is impressive."
review2 = "This novel was written by a famous author."
review3 = "I recommend this restaurant, their dish was amazing."

prefiltered_classes1 = keyword_prefilter(review1, keyword_map)
prefiltered_classes2 = keyword_prefilter(review2, keyword_map)
prefiltered_classes3 = keyword_prefilter(review3, keyword_map)

print(f"Prefiltered classes for review 1: {prefiltered_classes1}") # Output: ['Electronics']
print(f"Prefiltered classes for review 2: {prefiltered_classes2}") # Output: ['Books']
print(f"Prefiltered classes for review 3: {prefiltered_classes3}") # Output: ['Food']
```

In this example, `keyword_prefilter` function examines the input text and identifies broad categories based on predefined keywords. The main classifier would then operate on the classes `Electronics`, `Books`, or `Food` respectively, rather than the full set of fine-grained classes. This significantly reduces the classifier's computational load and minimizes the chances of assigning a "Book" category to a smartphone review, for example. Note that normalization by lowercasing ensures consistent matching.

**Example 2: Coarse-Grained Classifier as Pre-Filter**

When keyword matching is insufficient or prone to error, a simpler, coarse-grained classifier can provide a more accurate pre-filtering mechanism.  In my experience, training a basic classifier using a subset of the overall categories works surprisingly well.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# Fictional training data
train_data = [
    ("This phone has a great display", "Electronics"),
    ("The plot of the novel is fantastic.", "Books"),
    ("I enjoyed the tasty pizza at the restaurant", "Food"),
    ("The processor speed is incredible", "Electronics"),
    ("I just finished reading a biography", "Books"),
    ("The restaurant is excellent", "Food"),
     ("The camera quality is amazing", "Electronics"),
    ("The author's writing style is captivating", "Books"),
    ("I will order that food again", "Food")
]

texts = [item[0] for item in train_data]
labels = [item[1] for item in train_data]

# Create a simple classifier pipeline
coarse_grained_classifier = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(solver='liblinear')) #  'liblinear' works well for smaller datasets
])

# Split for training/testing
X_train, _, y_train, _ = train_test_split(texts, labels, test_size = 0.2, random_state=42)

# Train the model
coarse_grained_classifier.fit(X_train, y_train)


def coarse_grained_prefilter(text, classifier):
    """
    Filters classes based on predictions of coarse-grained classifier.

    Args:
    text (str): Input text to be classified.
    classifier: A trained coarse-grained classifier object.

    Returns:
    list: List of potentially relevant broad categories.
    """
    predicted_class = classifier.predict([text])[0]
    return [predicted_class] # Return the most likely class

# Example Usage:
review4 = "The phone's camera takes amazing pictures."
review5 = "I couldn't put down the book."
review6 = "The food was delicious and cheap."

prefiltered_classes4 = coarse_grained_prefilter(review4, coarse_grained_classifier)
prefiltered_classes5 = coarse_grained_prefilter(review5, coarse_grained_classifier)
prefiltered_classes6 = coarse_grained_prefilter(review6, coarse_grained_classifier)


print(f"Prefiltered classes for review 4: {prefiltered_classes4}") # Output: ['Electronics']
print(f"Prefiltered classes for review 5: {prefiltered_classes5}") # Output: ['Books']
print(f"Prefiltered classes for review 6: {prefiltered_classes6}") # Output: ['Food']
```

Here, instead of keywords, a `LogisticRegression` classifier is trained to map text to broad categories.  The `coarse_grained_prefilter` function then uses this model to predict the broad category, limiting the subsequent multi-class classifier's scope. The inclusion of a `TfidfVectorizer` transforms the text into numeric features.  In practice, a more complex model may be beneficial for greater pre-filtering accuracy.

**Example 3: Rule-Based Pre-Filtering with Multiple Conditions**

Combining multiple rules can further refine the pre-filtering.  This is useful when the relationship between the input data and classes is not entirely captured by keyword or simple classifiers. I've found that careful combinations of multiple conditions can greatly enhance precision.

```python
def rule_based_prefilter(text):
    """
    Filters classes based on combination of rules.

    Args:
    text (str): Input text to be classified.

    Returns:
    list: List of potentially relevant broad categories.
    """
    text = text.lower()
    relevant_categories = []

    if "battery" in text or "screen" in text or "processor" in text:
        relevant_categories.append("Electronics")

    if ("book" in text or "novel" in text or "author" in text) and ("page" in text or "chapter" in text):
         relevant_categories.append("Books")

    if ("restaurant" in text or "food" in text or "taste" in text) and ("delicious" in text or "recommend" in text or "order" in text) :
        relevant_categories.append("Food")

    return list(set(relevant_categories))

# Example Usage:
review7 = "The book has an interesting story and the writing is captivating"
review8 = "This phone's screen is really bright"
review9 = "This restaurant has delicious food, I recommend it."
review10 = "I liked the story of the novel and it has many pages"

prefiltered_classes7 = rule_based_prefilter(review7)
prefiltered_classes8 = rule_based_prefilter(review8)
prefiltered_classes9 = rule_based_prefilter(review9)
prefiltered_classes10 = rule_based_prefilter(review10)


print(f"Prefiltered classes for review 7: {prefiltered_classes7}")  # Output: []
print(f"Prefiltered classes for review 8: {prefiltered_classes8}") # Output: ['Electronics']
print(f"Prefiltered classes for review 9: {prefiltered_classes9}") # Output: ['Food']
print(f"Prefiltered classes for review 10: {prefiltered_classes10}")  #Output: ['Books']
```

Here, the `rule_based_prefilter` function checks for multiple conditions linked by logical operators. The "Books" category, for instance, requires both book-related keywords *and* page/chapter related keywords.  This combination allows for a more precise filter compared to using single keyword conditions. This is often my go-to approach when the data patterns are more intricate.

To further improve results, consider exploring techniques like ensemble pre-filtering, where multiple pre-filtering methods are combined, and probabilistic pre-filtering where the pre-filter provides probability scores for each class, indicating the likelihood of belonging to the candidate class which can be used to set a threshold. The trade-offs between accuracy, computational cost, and implementation complexity should be carefully evaluated when selecting an approach.

For further study on multi-class classification, I recommend exploring resources specializing in pattern recognition, machine learning, and natural language processing.  Textbooks covering topics such as statistical learning theory, information retrieval and classification, and advanced machine learning algorithms can be immensely beneficial. Research publications from prominent conferences in these areas also offer in-depth theoretical discussions and practical insights. These materials will provide a strong foundation to further explore and implement these concepts in real-world applications.
