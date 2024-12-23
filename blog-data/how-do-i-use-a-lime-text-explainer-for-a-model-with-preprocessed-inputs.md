---
title: "How do I use a LIME text explainer for a model with preprocessed inputs?"
date: "2024-12-23"
id: "how-do-i-use-a-lime-text-explainer-for-a-model-with-preprocessed-inputs"
---

Okay, let's unpack this. I've definitely been down this road before – specifically, I remember a project a few years back involving sentiment analysis on user-generated reviews where we had a rather involved preprocessing pipeline. We needed explainability, and LIME seemed like a solid choice, but the preprocessing step certainly added a layer of complexity. It's not insurmountable, but it requires a careful approach.

The fundamental challenge, as I understand it, is that LIME, at its core, works by perturbing the input features and observing how the model's output changes. When your model is trained on preprocessed data, the perturbed inputs need to be similarly processed *before* being fed into the model. If not, the model will essentially be evaluating data that doesn’t resemble what it saw during training, rendering the explanations meaningless.

Therefore, the key to effectively using LIME with preprocessed inputs is to ensure that your LIME explainer operates in the *same space* as the model. This involves carefully crafting the `predict_fn` that LIME uses, to incorporate your preprocessing steps.

Let me illustrate this with a breakdown of how it typically works, using some example code snippets. Imagine we have a model that uses TF-IDF vectors as input, a common text representation.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer

# 1. Preprocessing (TF-IDF) and model training:
texts = ["this is a great movie", "terrible acting, i hated it", "it was okay"]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(tfidf_matrix, [1, 0, 0.5]) # Mock sentiment labels for training

# 2. Now, the problem - naive LIME usage will not work:
explainer = LimeTextExplainer(class_names=['negative', 'positive'])
try:
  explanation = explainer.explain_instance(texts[0], model.predict_proba, num_features=5)
  print("Explanation:", explanation.as_list())
except Exception as e:
  print(f"Error (as expected):\n{e}") # This will raise an error, because of data input mismatch
```

If you run this, you’ll likely see an error about the incompatibility between the text input expected by LIME and the numerical input expected by the model. `model.predict_proba` expects a TF-IDF vector, but LIME is providing raw text. So, this illustrates exactly why we need to create a custom `predict_fn`.

Here’s the corrected version of the previous example:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer

# 1. Preprocessing (TF-IDF) and model training:
texts = ["this is a great movie", "terrible acting, i hated it", "it was okay"]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(tfidf_matrix, [1, 0, 0.5]) # Mock sentiment labels for training

# 2.  Create a custom predict_fn:
def custom_predict(texts):
    tfidf_matrix = tfidf_vectorizer.transform(texts)
    return model.predict_proba(tfidf_matrix)


# 3. Use LIME with the custom predict_fn:
explainer = LimeTextExplainer(class_names=['negative', 'positive'])
explanation = explainer.explain_instance(texts[0], custom_predict, num_features=5)
print("Explanation:", explanation.as_list())

```

In this improved version, I've defined `custom_predict`. This function takes a list of raw text strings as its argument, applies the same TF-IDF transformation used during model training via `tfidf_vectorizer.transform()`, and *then* feeds that transformed matrix into the trained model for prediction. LIME now passes raw strings, they get preprocessed, and fed correctly.

Now, for something a little more complex, let's say you're dealing with a model that combines tokenization, padding, and embeddings before feeding into a neural network. The pattern remains the same: our `predict_fn` must mirror the preprocessing pipeline.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from lime.lime_text import LimeTextExplainer

# 1. Preprocessing and Model Training (using tensorflow):
texts = ["this is a great movie", "terrible acting, i hated it", "it was okay"]
max_vocab = 100
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = max([len(s) for s in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len)


model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=16, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(padded_sequences, np.array([1, 0, 0.5]), epochs=20, verbose=0)


# 2. Custom predict_fn:
def custom_predict_nn(texts):
  sequences = tokenizer.texts_to_sequences(texts)
  padded_sequences = pad_sequences(sequences, maxlen=max_len)
  return model.predict(padded_sequences)

# 3. LIME Explanation
explainer = LimeTextExplainer(class_names=['negative', 'positive'])
explanation = explainer.explain_instance(texts[0], custom_predict_nn, num_features=5)
print("Explanation:", explanation.as_list())
```

In this neural network example, the `custom_predict_nn` function now includes tokenization using the tokenizer fitted on the original training set, padding to match the expected input shape of our neural network and *then* calls model.predict.

What I've shown here is that the approach is consistently about encapsulating the entire data preparation logic that your model expects within the function LIME calls to generate predictions. This ensures that the perturbations LIME introduces are relevant to the model and result in interpretable explanations.

For further reading and solid theoretical understanding, I’d suggest investigating a few key resources. Start with the original LIME paper, “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. This will give you the foundational concepts. For a deeper dive into text processing techniques specifically TF-IDF and other related techniques, consult *Speech and Language Processing* by Daniel Jurafsky and James H. Martin. It's a comprehensive text that covers the underlying concepts. If you want to go deeper into the theory and application of neural networks, “Deep Learning” by Goodfellow, Bengio, and Courville is an exhaustive resource.

In summary, the key to effective LIME usage when dealing with preprocessed inputs is to always ensure that the `predict_fn` given to LIME faithfully reproduces the model's data preprocessing steps. By doing this, you ensure that LIME operates in the same "space" as your model, leading to understandable and reliable explanations. I hope that helps.
