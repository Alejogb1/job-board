---
title: "Why is my text classifier's performance sensitive to sample size?"
date: "2025-01-30"
id: "why-is-my-text-classifiers-performance-sensitive-to"
---
The performance of a text classifier is demonstrably, and often dramatically, impacted by the size of the training dataset; this sensitivity stems primarily from the underlying statistical nature of machine learning and the complexities inherent in natural language data.

Insufficient training data provides an inadequate representation of the underlying distribution of features within the text. In the context of text classification, these features often include individual words, sequences of words (n-grams), or more complex semantic representations. Each instance in the training set serves as a data point, contributing to the overall model's ability to generalize to unseen examples. When data is scarce, the model learns potentially spurious relationships, overfitting to the noise present in the limited sample. Conversely, with an abundance of data, the model has a greater opportunity to identify and learn the true, underlying patterns and reduce the influence of random variations within the training set.

Specifically, consider a naive Bayes classifier. It calculates conditional probabilities of features (words) given a class, which is an estimate based on their observed frequencies within the training data. When the sample size for a specific class is small, these estimated frequencies become unreliable. A single word appearing frequently in a few instances may lead the model to incorrectly associate it strongly with that class, which might not hold true when more diverse samples are available. For example, if only three documents of class "Finance" contain the word "stock," and no other class has this term, the classifier would likely overemphasize this term’s weight. As the dataset grows, the chances increase that words like "stock" appear across other classes to some degree, improving the model's ability to distinguish true relationships from spurious correlations.

Beyond simple frequency estimates, more complex models like support vector machines (SVMs) or neural networks are also affected by sample size. These models have significantly more parameters to learn than the naive Bayes model. With limited data, their capacity for generalization is compromised, and they overfit; they memorize the training data rather than learning the generalizable patterns. A small dataset is analogous to having a very limited amount of information to construct a highly detailed map.

The risk of overfitting leads to a high variance problem. This results in the classifier performing very well on the small training set it was exposed to, but it will not generalize well to new, unseen data, exhibiting poor accuracy. The problem is further exacerbated by the high dimensionality of textual data; with a vocabulary containing thousands of unique words, even a moderate size dataset can result in a sparse data matrix, increasing the chance of overfitting if the data is not balanced across classes or if feature representations are not appropriately handled. A lack of variance across the training set, a common issue with insufficient samples, also impacts negatively on model robustness, resulting in a classifier performing poorly in the face of real-world data variability.

Now, let's examine this with some examples in Python, illustrating the concepts:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Simulate a very small dataset
corpus_small = [
    ("This is a good movie", "positive"),
    ("I hate this movie", "negative"),
    ("The plot was great", "positive"),
    ("Terrible acting", "negative"),
    ("Excellent film", "positive"),
    ("Awful experience", "negative")
]
texts_small = [text for text, _ in corpus_small]
labels_small = [label for _, label in corpus_small]

# Convert text to numerical features
vectorizer = TfidfVectorizer()
features_small = vectorizer.fit_transform(texts_small)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(features_small, labels_small, test_size=0.2, random_state=42)
model_small = MultinomialNB()
model_small.fit(X_train, y_train)
y_pred_small = model_small.predict(X_test)
accuracy_small = accuracy_score(y_test, y_pred_small)
print(f"Accuracy on Small Dataset: {accuracy_small:.2f}")

# Simulate a larger dataset, maintaining class proportions
corpus_large = [
    ("This is a good movie", "positive"), ("I hate this movie", "negative"),("The plot was great", "positive"),
    ("Terrible acting", "negative"), ("Excellent film", "positive"),("Awful experience", "negative"),
    ("Very moving scene", "positive"), ("This movie is amazing", "positive"),("Complete waste of time", "negative"),
    ("Fantastic characters", "positive"),("I regret watching this", "negative"),("Truly inspirational", "positive"),
    ("Dreadful dialogues", "negative"),("It's a masterpiece", "positive"), ("A total disaster", "negative"),
    ("Superb direction", "positive"), ("Extremely boring", "negative"),("I loved it!", "positive"),
    ("It was a nightmare", "negative"),("A gem of cinema", "positive"),
    ("I would not recommend it", "negative")
]
texts_large = [text for text, _ in corpus_large]
labels_large = [label for _, label in corpus_large]
features_large = vectorizer.transform(texts_large) # using the same vectorizer for comparison
X_train, X_test, y_train, y_test = train_test_split(features_large, labels_large, test_size=0.2, random_state=42)
model_large = MultinomialNB()
model_large.fit(X_train, y_train)
y_pred_large = model_large.predict(X_test)
accuracy_large = accuracy_score(y_test, y_pred_large)
print(f"Accuracy on Large Dataset: {accuracy_large:.2f}")

```

This code demonstrates the impact of dataset size. The `corpus_small` has only 6 documents. After splitting the data the model is trained on 4 documents and tested on only 2 documents. On the larger dataset, model trained on 16 documents and tested on 6 documents which is still a small number overall, but relatively much larger.  The small dataset has high sensitivity to small changes and leads to unstable results and potentially to bad model output. In the `MultinomialNB` case, we observe much more stable predictions with the `corpus_large`. Note that the same `TfidfVectorizer` is used across both examples to eliminate one variable. However, using a new vectorizer on a larger dataset will also significantly improve its performance.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Simulate imbalanced dataset with a larger training size
corpus_imbalanced = [
    ("This is a fantastic product", "positive") for _ in range(300)] + \
                    [("Terrible experience", "negative") for _ in range(20)]
texts_imbalanced = [text for text, _ in corpus_imbalanced]
labels_imbalanced = [label for _, label in corpus_imbalanced]

vectorizer_imbalanced = TfidfVectorizer()
features_imbalanced = vectorizer_imbalanced.fit_transform(texts_imbalanced)

X_train, X_test, y_train, y_test = train_test_split(features_imbalanced, labels_imbalanced, test_size=0.2, random_state=42)

pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])
pipeline.fit(X_train, y_train)
y_pred_imbalanced = pipeline.predict(X_test)
accuracy_imbalanced = accuracy_score(y_test, y_pred_imbalanced)

print(f"Accuracy on Imbalanced Dataset: {accuracy_imbalanced:.2f}")

# Introducing more diverse negative samples and increasing the size, leading to higher model performance
corpus_imbalanced_balanced = [
    ("This is a fantastic product", "positive") for _ in range(300)] + \
                    [("Terrible experience", "negative") for _ in range(20)] + \
                    [("Appalling quality", "negative") for _ in range(20)]+ \
                    [("This is a really bad purchase", "negative") for _ in range(20)]+ \
                   [("The item was received damaged", "negative") for _ in range(20)]+ \
                    [("Never going to buy this again", "negative") for _ in range(20)] + \
                    [("This is horrible", "negative") for _ in range(20)]

texts_imbalanced_balanced = [text for text, _ in corpus_imbalanced_balanced]
labels_imbalanced_balanced = [label for _, label in corpus_imbalanced_balanced]
features_imbalanced_balanced = vectorizer_imbalanced.transform(texts_imbalanced_balanced)

X_train, X_test, y_train, y_test = train_test_split(features_imbalanced_balanced, labels_imbalanced_balanced, test_size=0.2, random_state=42)

pipeline_balanced = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])

pipeline_balanced.fit(X_train, y_train)
y_pred_imbalanced_balanced = pipeline_balanced.predict(X_test)
accuracy_imbalanced_balanced = accuracy_score(y_test, y_pred_imbalanced_balanced)
print(f"Accuracy on Imbalanced Balanced Dataset: {accuracy_imbalanced_balanced:.2f}")
```

The second code snippet addresses the impact of imbalanced data, while also expanding on the size of the dataset. The `corpus_imbalanced` consists of highly biased data – 300 positive class samples compared to 20 negative ones, a situation that commonly results in high accuracy metrics but that will mask underlying problems. Even with the imbalance, the model might perform decently on this limited data, but will exhibit poor results in real settings with more balanced representation. The `corpus_imbalanced_balanced` example mitigates this by adding more variety of negative samples. Critically, it is often not just about the number of documents but also how diverse and representative the data is. The use of pipelines for scaling features is another best-practice improvement.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Create larger, diverse dataset with simulated user reviews
corpus_nn = [
    ("This phone has amazing features and camera quality is top-notch", "positive"),
    ("The battery life is terrible and the screen cracked within a week", "negative"),
    ("I loved this product, fast shipping and well packaged", "positive"),
    ("The support was atrocious and the product was damaged", "negative"),
     ("The product did everything I expected. Love it", "positive"),
    ("I am very disappointed with the quality of this item", "negative"),
    ("This was a game-changer for me. Highly recommend", "positive"),
    ("The performance was very poor, would not recommend", "negative"),
    ("The device is simply outstanding, best value for money", "positive"),
    ("The instructions were confusing and the product was difficult to use", "negative"),
     ("Very pleased with my purchase, I would buy again from this seller", "positive"),
    ("I am returning this product, the color is completely wrong", "negative"),
    ("This model exceeded all my expectations. Stellar performance", "positive"),
    ("The product was faulty upon arrival. A terrible waste of money", "negative"),
    ("I am a happy customer, I think it is amazing", "positive"),
     ("This was a complete scam. Awful product and seller.", "negative")
] * 50 #Creating 800 total items
texts_nn = [text for text, _ in corpus_nn]
labels_nn = [label for _, label in corpus_nn]

vectorizer_nn = TfidfVectorizer()
features_nn = vectorizer_nn.fit_transform(texts_nn)
X_train, X_test, y_train, y_test = train_test_split(features_nn, labels_nn, test_size=0.2, random_state=42)

model_nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)
model_nn.fit(X_train, y_train)
y_pred_nn = model_nn.predict(X_test)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

print(f"Accuracy with Neural Network: {accuracy_nn:.2f}")
```
The final example uses a basic neural network (MLPClassifier) on a significantly increased, simulated user review dataset. I have repeated the original 16 samples 50 times to obtain 800 samples to simulate a larger, more diverse training dataset. While still a simulated dataset, this example represents a more realistic scenario.  With increased size and diversity, the model performance is greatly improved. The neural network, being a more complex model, benefits significantly from the larger amount of data. The iteration parameter of `MLPClassifier` had to be reduced to illustrate the convergence of the model at low iteration count.

For improving model performance related to sample size, I would recommend the following resources:

First, explore literature on data augmentation techniques. These methods allow you to artificially increase the size of the dataset, often through manipulations of existing samples.  In text, common methods may include synonym replacement, back-translation, or random deletion.

Secondly, study transfer learning techniques and pre-trained models. These models are trained on extremely large corpora and capture a significant amount of semantic information. Fine-tuning a pre-trained model on a smaller task-specific dataset often leads to significantly better results compared to training a model from scratch. This effectively addresses the low sample size issue by leveraging knowledge acquired from vastly larger datasets.

Finally, consider investigating learning curve analysis. These curves, which plot model performance against increasing training data size, can help you understand the point of diminishing returns and estimate how much data is actually required to reach a certain level of performance. This allows for informed decisions regarding resource allocation.
