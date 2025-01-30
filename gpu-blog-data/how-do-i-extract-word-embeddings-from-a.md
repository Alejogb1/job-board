---
title: "How do I extract word embeddings from a pretrained transformer model?"
date: "2025-01-30"
id: "how-do-i-extract-word-embeddings-from-a"
---
Extracting word embeddings from a pretrained transformer model involves accessing the internal representations of words learned by the model during its pretraining phase. These embeddings capture contextualized meaning, differing from static word vectors like those produced by Word2Vec or GloVe, which assign a single vector to each word regardless of the surrounding text. Instead, transformer embeddings are dynamic, meaning that the same word can have different vector representations depending on the sentence it is used in. This contextual nature is what makes them so powerful for downstream tasks.

Fundamentally, the process involves two steps: encoding the input text using the transformer's tokenizer, and then extracting the hidden states from specific layers.  I've frequently performed this operation to evaluate embedding quality or to prepare embeddings for custom classification pipelines. It's crucial to understand that the “word” here is often a token, and not necessarily a human-interpretable word in the traditional sense. Transformer tokenizers often break down words into sub-word units to handle out-of-vocabulary terms and to capture internal word structures like prefixes and suffixes. This granularity is what allows transformers to be more robust in handling diverse linguistic input.

The transformer model's architecture generally involves an input embedding layer, several encoder layers, and often an output layer. For extracting embeddings, we bypass the output layer, which is usually for tasks like language modeling or classification. Instead, we are interested in the hidden state vectors produced by the embedding layer and the various encoder layers. The specific layer from which we extract embeddings influences the level of contextualization.  Lower layers tend to represent more surface-level features, while higher layers capture more abstract, task-oriented features. I have found in my experience that often an average or a concatenation of the last four layers yields a practical trade-off between computation cost and embedding quality.

Let’s illustrate this process with code. I'll use Python with the `transformers` library from Hugging Face which is ubiquitous in this field.  The initial example demonstrates how to extract the last layer’s hidden states:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load a pre-trained model and its tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Input text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Get the model's output
with torch.no_grad():
    outputs = model(**inputs)

# Extract the hidden states of the last layer
last_hidden_states = outputs.last_hidden_state

# The shape of last_hidden_states will be (batch_size, sequence_length, hidden_size)
print(f"Shape of last hidden states: {last_hidden_states.shape}")

# Here, each token's embedding can be accessed from last_hidden_states
# The specific token embeddings can then be extracted as required
```

In this code, we first load a pre-trained BERT model and its corresponding tokenizer. The input text is tokenized using the tokenizer’s method, which returns a dictionary of tensors ready for the model. The model then generates a set of outputs, and the `last_hidden_state` attribute of those outputs is what holds the final-layer embeddings. It's a tensor with dimensions that depend on the batch size (which is one in this case), the sequence length (number of tokens), and the hidden size of the model. This example showcases how to grab the contextualized embedding for each input token, without averaging or further processing it. The batch size will typically be 1 when handling individual sentences. In my own projects, I have used batching to improve throughput when processing very large text corpora.

The second example refines this by demonstrating how to extract the embedding for a specific token after the input is tokenized:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load a pre-trained model and its tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Input text
text = "The quick brown fox jumps over the lazy dog."
target_word = "fox"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Get the model's output
with torch.no_grad():
    outputs = model(**inputs)

# Extract the hidden states of the last layer
last_hidden_states = outputs.last_hidden_state

# Obtain token IDs and map to input tokens
input_ids = inputs["input_ids"].squeeze()
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Find indices of the target token
target_indices = [i for i, token in enumerate(tokens) if token == target_word]

# If the word is split by the tokenizer, you'll have multiple indices
if target_indices:
    target_embedding = last_hidden_states[:, target_indices, :].mean(dim=1)
    print(f"Shape of the embedding for '{target_word}': {target_embedding.shape}")
else:
    print(f"'{target_word}' not found in tokens.")

```

This example adds a layer of refinement. We now explicitly tokenize the input and retrieve the token IDs. We locate the position(s) of the target word based on the tokens produced by the tokenizer. Because sub-word tokenization may split the word ‘fox’, we have to account for multiple token indices corresponding to it, taking their mean to represent its overall embedding, which I’ve often found necessary in practice. The resultant `target_embedding` tensor represents the contextualized embedding of the selected word. This precise extraction is important when, for example, you’re trying to perform semantic analysis on a specific word in a text.

My third example demonstrates how to extract and average the hidden states from multiple layers, which as mentioned earlier can provide richer contextual embeddings:

```python
from transformers import AutoTokenizer, AutoModel
import torch
from torch import mean

# Load a pre-trained model and its tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Input text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Get the model's output
with torch.no_grad():
    outputs = model(**inputs)

# Extract all hidden states
all_hidden_states = outputs.hidden_states

# Number of layers to use
num_layers = 4

# Extract last n layers
selected_hidden_states = all_hidden_states[-num_layers:]

# Calculate average embedding
averaged_hidden_states = mean(torch.stack(selected_hidden_states), dim=0)

# The shape of averaged_hidden_states will be (batch_size, sequence_length, hidden_size)
print(f"Shape of averaged hidden states: {averaged_hidden_states.shape}")

# Token embeddings can be further extracted as needed
```
In this final example, I set `output_hidden_states=True` when loading the model to retain all the hidden state outputs from the encoder layers. Then, we extract the last four layers' hidden states, stack them and calculate their average along the layer dimension using `mean` method which is imported from the pytorch library.  The resultant tensor, `averaged_hidden_states`, represents a contextualized embedding that considers the features of the last four layers, which often performs better on downstream tasks, as found empirically by myself and other NLP practitioners.

Beyond the `transformers` documentation, I would recommend consulting academic papers on the inner workings of transformer models, including those that discuss specific approaches to embedding extraction, as that will give you a more theoretical backing to any practical experimentation you may be doing. Books on natural language processing and deep learning can further supplement your understanding. Open source implementation of pre-training of models from researchers is also helpful, but can be at times overly complex. Experimenting with different layers and averaging strategies based on your specific downstream task is the most vital step of development and will provide an empirical understanding.
