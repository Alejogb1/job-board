---
title: "How to use LIME text explainer with a model with preprocessed input?"
date: "2024-12-16"
id: "how-to-use-lime-text-explainer-with-a-model-with-preprocessed-input"
---

, let’s tackle this. It's a scenario I’ve actually run into more than a few times, usually when dealing with complex NLP pipelines where the input to the model is far removed from the raw text itself. The challenge is making sense of what the model is focusing on, particularly when the input has undergone significant transformations such as tokenization, vectorization, or the application of embeddings. LIME, or Local Interpretable Model-agnostic Explanations, is powerful for this, but requires a bit of careful setup.

My team once worked on a system for sentiment analysis of customer reviews. We weren't feeding the raw reviews directly into our neural net. Instead, we had a sophisticated pre-processing pipeline: tokenization using subword units, followed by embedding lookup, a couple of lstm layers, and finally, a dense classification layer. Trying to apply LIME on the unprocessed text was essentially useless – the model wasn't "seeing" those words anymore. What the model consumed were sequences of numerical vectors. The key to applying LIME effectively in such a case lies in understanding how to bridge the gap between human-readable text and the model's input space, and in telling LIME how to generate perturbations that make sense within *that* space.

The core idea behind LIME is that it perturbs the input data and observes how the model's output changes. LIME then builds a local linear model to approximate the original model's behavior in the neighborhood of the input point and uses the weights in the linear model to determine feature importance. When the input is raw text, the perturbations involve changing words or phrases. But when the input is preprocessed, this concept needs a translation.

Here’s the process broken down, focusing on the critical steps and including a few real-world solutions I have applied before. We’ll cover defining the prediction function, how we perturb the input data, and finally, how we generate understandable output.

**1. Defining the Prediction Function**

LIME needs a prediction function, a function that receives an input and returns the model’s prediction. Crucially, this function must receive *the same input format* that the model itself uses. In our sentiment analysis example, this meant taking preprocessed input (specifically, a sequence of numerical vectors). You cannot feed a text string directly. This function should be carefully crafted and encapsulate all your preprocessing steps, or at least return the *identical* output as if these steps had been performed.

Let's illustrate with some python code, using an example with simple text and a basic model with a tf-idf vectorizer as the preprocessing step:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer

# Sample data (for demonstration purposes)
texts = ["This is a great movie", "This is a bad movie", "The acting is amazing", "The plot was terrible", "I enjoyed it"]
labels = [1, 0, 1, 0, 1] # 1 for positive, 0 for negative

# Preprocessing: Tfidf vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Model (logistic regression)
model = LogisticRegression()
model.fit(X, labels)

# Prediction function for LIME
def prediction_function(text_strings):
    processed_input = vectorizer.transform(text_strings)
    return model.predict_proba(processed_input)

# Now proceed with LIME
explainer = LimeTextExplainer(class_names=[0, 1]) # 0: negative, 1: positive

# Explain a sample prediction
test_text = ["This movie was ."]
explanation = explainer.explain_instance(
    test_text[0],
    prediction_function,
    num_features=5
)

print(explanation.as_list())
```

In this code, the `prediction_function` first transforms the input text using `vectorizer.transform()`, ensuring it matches what the `model` expects. This step is fundamental. The `explainer` then is initialized with the class names which are needed for interpretation. The `explain_instance` method creates the explanation.

**2. Handling Perturbations**

Here’s where things get trickier with preprocessed data. LIME assumes it can change individual words or tokens. We have to adapt it to understand our processed space. The `LimeTextExplainer` class has some inbuilt mechanisms to generate perturbations. When you give it raw text as input, it does the tokenization internally and generates perturbations like removing or adding a single word, or replacing with another. However, when your input is already preprocessed as numerical arrays, these perturbation techniques don't apply. Instead, we need to provide *our own* functions that generate perturbations that are meaningful in the preprocessed input space.

Let's show that with a different example. We are going to use a sentence transformer that transforms text into a fixed-size vector. We can perturb that vector and we can reconstruct the closest text back via an inverse transformation

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import LabelEncoder

# Sample data (for demonstration purposes)
texts = ["This is a great movie", "This is a bad movie", "The acting is amazing", "The plot was terrible", "I enjoyed it"]
labels = [1, 0, 1, 0, 1] # 1 for positive, 0 for negative

# Preprocessing: Sentence Transformer embedding
embedding_model = SentenceTransformer('all-mpnet-base-v2')
X = np.array(embedding_model.encode(texts))

# Model (logistic regression)
model = LogisticRegression()
model.fit(X, labels)

# Function to convert from embedding back to a list of strings (needed for explanations)
def embedding_to_text_func(embedding_matrix, original_texts, num_neighbors=3):
   # the most naive approach: use the closest input vector to the embedding as the most explanatory token
  distances = np.sqrt(np.sum((embedding_matrix[:, np.newaxis] - embedding_model.encode(original_texts)) ** 2, axis=2))
  closest_indices = np.argsort(distances, axis=1)[:, :num_neighbors]
  return [[original_texts[i] for i in neighbors] for neighbors in closest_indices]

# Prediction function for LIME
def prediction_function(text_strings):
    processed_input = np.array(embedding_model.encode(text_strings))
    return model.predict_proba(processed_input)

# LIME doesn't work out of the box with the output of the sentence transformer
# but we can create our own perturbation function
def custom_perturbation_function(data_point, num_samples):
  # data_point is the numpy array representing a sentence vector
  # we are simply creating gaussian noise vectors around the provided one.
  noise = np.random.normal(0, 0.1, size=(num_samples, data_point.shape[0]))
  return data_point + noise

explainer = LimeTextExplainer(class_names=[0, 1]) # 0: negative, 1: positive

# Explain a sample prediction
test_text = ["This movie was ."]
text_embedding = embedding_model.encode(test_text)
explanation = explainer.explain_instance(
    text_embedding[0],
    prediction_function,
    num_features=5,
    num_samples = 200,
    perturbation_fn=custom_perturbation_function,
    distance_metric='cosine', # Important: pick a meaningful metric in your vector space
    top_labels=2,
    text_to_labels_function = lambda x: embedding_to_text_func(x, texts)
)

print(explanation.as_list())
```

In this example, we directly perturb the numerical representation. The `custom_perturbation_function` applies Gaussian noise to the embedding vectors, which are what the `prediction_function` receives. The explanation is then shown using similar sentences to the perturbed vectors using `embedding_to_text_func`. This approach often works surprisingly well for embeddings, provided you understand the underlying structure of the vector space you're working with. The selected `distance_metric` is important since we are calculating distances between embedding vectors.

**3. Generating Understandable Output**

The last stage is connecting LIME's explanation with human-understandable text. In the first example, this is already taken care of as we are using `LimeTextExplainer`’s built-in token-based approach. In the second example, the interpretation is more difficult because we perturbed embeddings not text.

Let's do one more example and demonstrate how we can handle complex feature engineering before the actual deep learning model. In this case, we'll encode text into its character-level n-gram representation, and feed this representation to a simple cnn for classification:

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer

# Sample data (for demonstration purposes)
texts = ["This is a great movie", "This is a bad movie", "The acting is amazing", "The plot was terrible", "I enjoyed it"]
labels = [1, 0, 1, 0, 1] # 1 for positive, 0 for negative

# Preprocessing: character-level n-gram
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3)) # using 2- and 3-character n-grams
X = vectorizer.fit_transform(texts).toarray()
vocab_size = X.shape[1]

# Model (simple CNN)
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(vocab_size, 1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X[:, :, np.newaxis], labels, epochs=50, verbose=0)

# Prediction function for LIME
def prediction_function(text_strings):
  processed_input = vectorizer.transform(text_strings).toarray()
  return model.predict(processed_input[:, :, np.newaxis]) # adding dimension for keras CNN

# Create a text_to_labels function that explains in terms of substrings
def text_to_labels_function(matrix, original_texts, num_neighbors=3):
    distances = np.sqrt(np.sum((matrix[:, np.newaxis] - vectorizer.transform(original_texts).toarray()) ** 2, axis=2))
    closest_indices = np.argsort(distances, axis=1)[:, :num_neighbors]
    return [[original_texts[i] for i in neighbors] for neighbors in closest_indices]


# Define perturbation method
def custom_perturbation_function(data_point, num_samples):
  # data_point is the numpy array representing a text n-gram vector
  # we are simply creating gaussian noise vectors around the provided one.
  noise = np.random.normal(0, 0.05, size=(num_samples, data_point.shape[0]))
  return data_point + noise

explainer = LimeTextExplainer(class_names=[0, 1])

# Explain a sample prediction
test_text = ["This movie was terrible."]

text_embedding = vectorizer.transform(test_text).toarray()
explanation = explainer.explain_instance(
    text_embedding[0],
    prediction_function,
    num_features=5,
    num_samples=200,
    perturbation_fn=custom_perturbation_function,
    distance_metric='cosine',
    top_labels=2,
    text_to_labels_function = lambda x: text_to_labels_function(x, texts)
)

print(explanation.as_list())
```

In this example, we use character-level n-grams as features for our model. LIME perturbs the feature representation directly, and the explanation is generated via the `text_to_labels_function` which picks out the closest original training examples in the space of character n-gram vectors.

**Key Takeaways**

*   **Prediction Function:** The `prediction_function` must be precisely what the model consumes. It must encompass all of your input pipeline steps.
*   **Perturbations:** When direct text manipulation doesn’t work, you have to define custom perturbation functions working on the input format consumed by the model.
*   **Interpretability**: You need to find a way to map back the perturbed inputs to human-interpretable text space, or the output will not make any sense.
*  **Distance Metrics**: Choosing a proper distance metric in the perturbed space is important. For instance, using 'cosine' for dense embedding vectors is much more meaningful that 'euclidean'.

For deeper dives into LIME and other model interpretation techniques, I strongly recommend consulting "Interpretable Machine Learning" by Christoph Molnar; it's a comprehensive resource. For a stronger understanding of sentence embeddings, refer to research papers by Reimers and Gurevych on Sentence-BERT. These resources provide the necessary theoretical and practical background to handle complex scenarios, where the model input is anything but the original raw text. I've used these resources for a long time, and they have proven themselves incredibly useful.

In conclusion, using LIME with models that have preprocessed input is not fundamentally hard, but it does require understanding and controlling each step in your pipeline. By being intentional in how you define the prediction function and how you implement your perturbations, you can successfully apply LIME even in complex contexts.
