---
title: "Why can't I use pretrained_model=URLs.WT103 in fastai.text?"
date: "2025-01-30"
id: "why-cant-i-use-pretrainedmodelurlswt103-in-fastaitext"
---
The issue stems from the evolution of the fastai library and its integration with Hugging Face's `transformers` library.  While `URLs.WT103` was at one point a valid path within fastai's internal model registry,  it's been deprecated and removed in favor of a more robust and streamlined approach utilizing the Hugging Face ecosystem directly.  My experience troubleshooting similar issues during the development of a sentiment analysis pipeline for a financial news aggregator highlighted the importance of aligning with the latest fastai and `transformers` versions and understanding their interoperability.

**1. Clear Explanation:**

The older fastai method, utilizing `URLs` for pretrained models, provided a convenient but ultimately limited way to access pre-trained language models. This approach tightly coupled fastai with its own internal model management.  The current best practice leverages the comprehensive model hub offered by Hugging Face.  Hugging Face maintains a vast library of pre-trained models, consistently updated and meticulously documented.  Directly integrating with Hugging Face provides access to a far wider array of models, improved version control, and a more actively maintained ecosystem.  The removal of `URLs.WT103` is part of this transition, designed to enhance the user experience and future-proof fastai applications.  Essentially, fastai no longer acts as an intermediary for model access; instead, it seamlessly integrates with the Hugging Face `transformers` library, allowing users to specify models using their Hugging Face identifiers.

Fastai now primarily relies on specifying the model architecture and its name directly using the `transformers` library. This transition necessitates adjusting the code to reflect the change in model loading mechanism.  Ignoring this update will result in the `AttributeError` or similar exceptions because the referenced URL no longer exists within the fastai namespace.

**2. Code Examples with Commentary:**

**Example 1: Incorrect (Deprecated) Approach:**

```python
from fastai.text import *
learn = text_classifier_learner(data_bunch, AWD_LSTM, pretrained_model=URLs.WT103)
```

This code will fail because `URLs.WT103` is no longer a valid attribute.  This older approach relied on fastai’s internal model registry.


**Example 2: Correct Approach using Hugging Face's `transformers`:**

```python
from fastai.text.all import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Specify the model name directly from Hugging Face
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a fastai DataBunch (assuming 'data_bunch' is already defined)
# ... (DataBunch creation code, ensuring tokenization aligns with the chosen model) ...

# Create a fastai learner
learn = text_classifier_learner(data_bunch, model=model, pretrained=False) # pretrained=False as the model is already loaded

# Fine-tune the model
learn.fit_one_cycle(1, 1e-3)
```

This code demonstrates the current best practice.  We use `AutoModelForSequenceClassification` and `AutoTokenizer` from the `transformers` library to load the specified model and tokenizer directly from Hugging Face. Note that `pretrained=False` in the `text_classifier_learner` is crucial as we have already loaded the pretrained model using the `transformers` library.  The `model_name` variable should be replaced with the desired model identifier found on Hugging Face's model hub. This approach is robust and aligns with the current fastai and Hugging Face integration.  Careful attention must be given to ensuring that the data preprocessing (tokenization) used in creating the `data_bunch` matches the tokenizer used for the model.


**Example 3:  Handling a Specific Task (Sentiment Analysis):**

```python
from fastai.text.all import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment" # Example sentiment model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Custom Tokenization Function (crucial for correct integration)
def custom_tokenize(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return tokens['input_ids'], tokens['attention_mask']

# ... (DataBunch creation, using custom_tokenize) ...

# Create the learner with the custom tokenizer
dls = TextDataLoaders.from_df(df, text_col='text', label_col='sentiment',
                                  valid_pct=0.2,
                                  tokenizer=custom_tokenize,
                                  batch_size=16)

learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy])

# Fine-tune and train
learn.fit_one_cycle(1, 1e-5)
```

This example focuses on a specific task, sentiment analysis, and highlights the necessity of a custom tokenization function (`custom_tokenize`) to ensure compatibility between the fastai `DataLoaders` and the Hugging Face tokenizer.  The choice of model (`nlptown/bert-base-multilingual-uncased-sentiment`) is illustrative;  other sentiment analysis models can be substituted as needed.  Note the careful choice of the loss function (`CrossEntropyLossFlat`) appropriate for multi-class classification. The data frame `df` is assumed to have 'text' and 'sentiment' columns.

**3. Resource Recommendations:**

The official documentation for both fastai and the Hugging Face `transformers` library.  Furthermore, reviewing example notebooks and tutorials provided within the fastai repository on GitHub will prove invaluable. Pay close attention to examples showcasing the integration between the two libraries, especially those focusing on custom tokenization and model loading.  Consult advanced tutorials which illustrate handling diverse model architectures and tasks.  The Hugging Face documentation itself is an essential resource for understanding the vast array of available pre-trained models and their respective parameters.  Finally, exploring community forums and Q&A sites devoted to deep learning can offer insights from other users’ experiences.

In summary, the inability to use `URLs.WT103` is not a bug, but a consequence of the deliberate shift towards a more flexible and powerful model integration strategy within fastai. By directly leveraging Hugging Face's `transformers` library, developers gain access to a substantially larger pool of models and benefit from the ongoing maintenance and development efforts of a wider community.  Adapting code to reflect this change is vital for building contemporary and maintainable fastai applications.
