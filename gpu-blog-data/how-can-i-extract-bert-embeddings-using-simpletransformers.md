---
title: "How can I extract BERT embeddings using simpletransformers?"
date: "2025-01-30"
id: "how-can-i-extract-bert-embeddings-using-simpletransformers"
---
BERT, at its core, transforms textual input into dense vector representations—embeddings—capturing semantic meaning. Simpletransformers, built on top of the Hugging Face Transformers library, significantly simplifies this process, abstracting away much of the underlying complexity. My experience training and fine-tuning numerous NLP models has shown that extracting these embeddings is a crucial step for various downstream tasks, including similarity analysis, clustering, and information retrieval. I've found that the process involves initializing a Simpletransformers model, feeding it text data, and then accessing the output layer, or specific hidden layers, that contain the embeddings. The following discussion will detail how one can effectively achieve this using simpletransformers.

The fundamental process involves several key steps: model initialization, data preparation, embedding extraction, and potentially further manipulation based on specific application needs. First, one must select the desired BERT model variant for which embeddings are needed. Simpletransformers supports a wide range of BERT architectures, allowing one to choose a model based on computational resources and required accuracy. The key here is selecting the appropriate pre-trained model for the language or task under consideration. Once the model has been chosen, its corresponding `TransformerModel` instance is constructed.

Following the selection and construction of the model, the textual input must be structured. Simpletransformers accepts a list of strings as input, which it processes into sequences of token IDs using the model’s specific tokenizer. This tokenizer plays a vital role in converting the text into numerical format suitable for the BERT model. Internally, this involves operations such as tokenization, padding, and the addition of special tokens like `[CLS]` and `[SEP]`, depending on the model’s specifics. After text preparation, the embedding extraction process is initiated. I've frequently noticed that the `model.encode_sentences()` method is particularly helpful for this. It produces a NumPy array representing embeddings of the input sequences, which can subsequently be utilized.

Importantly, the layer from which embeddings are extracted can be altered. By default, `encode_sentences()` often returns the embeddings from the last hidden layer, which I have typically found useful for general sentence representation. However, BERT architectures comprise several hidden layers. Depending on the specific application, it might be advantageous to utilize embeddings from other hidden layers. These other layers might capture differing levels of semantic abstraction. Accessing them requires a more direct interaction with the underlying `forward()` function of the model and careful examination of the return structure. This is more involved, but it allows for the full potential of BERT architectures to be utilized. Furthermore, one can extract not just sentence embeddings, but also token-level embeddings, which are valuable for word-level tasks like named entity recognition. This again requires interacting with the model’s outputs directly.

The output of embedding extraction is typically a multidimensional tensor, where each sequence or token has its own vector representation. These embeddings are usually dense and capture semantic information from the input text. The shape of the tensor depends on several factors, including the number of input sentences or tokens, the sequence length, and the model’s hidden layer size. One must be cognizant of this to effectively utilize the embeddings for downstream tasks.

Here are a few code examples with commentary to illustrate the process further:

**Example 1: Sentence Embedding Extraction with `encode_sentences`**

```python
from simpletransformers.language_representation import TransformerModel
import numpy as np

# Model initialization
model = TransformerModel(model_type="bert", model_name="bert-base-uncased")

# Input sentences
sentences = [
    "This is a sample sentence.",
    "Another sentence for embedding.",
    "The quick brown fox jumps over the lazy dog."
]

# Extract sentence embeddings
embeddings = model.encode_sentences(sentences, combine_strategy='mean')

# Output shape and sample
print(f"Shape of embeddings: {embeddings.shape}")
print(f"First embedding: {embeddings[0][:10]} ...")
```
This example illustrates the basic procedure using the `encode_sentences()` method for obtaining sentence-level embeddings. The `'mean'` strategy calculates the mean of all token embeddings in the sentence for a compact fixed-size representation. I chose to print only the first 10 elements of the first embedding for brevity. The output shows the shape of the resulting NumPy array, representing a 3 x 768 matrix in this case, where the first dimension corresponds to the number of sentences and the second the size of the embedding vectors for the selected `bert-base-uncased` model.

**Example 2: Custom Embeddings from Last Hidden Layer**

```python
from simpletransformers.language_representation import TransformerModel
import torch

# Model initialization
model = TransformerModel(model_type="bert", model_name="bert-base-uncased")

# Input sentences
sentences = [
    "This is a sample sentence.",
    "Another sentence for embedding.",
    "The quick brown fox jumps over the lazy dog."
]

# Tokenize input
inputs = model.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Obtain model output
with torch.no_grad():
    outputs = model.model(**inputs)

# Extract last hidden layer
embeddings = outputs.last_hidden_state

# Output shape and sample
print(f"Shape of embeddings: {embeddings.shape}")
print(f"First sentence first token embedding: {embeddings[0][0][:10]} ...")
```
This example accesses the model’s `forward()` method directly. The tokenized inputs are passed to the BERT model. The `last_hidden_state` from the outputs represents the embeddings from the final hidden layer. The shape of the embeddings shows the structure of the output tensor, where the first dimension corresponds to the batch size, the second to the sequence length, and the third to the hidden size. I've found that this approach allows one to manipulate embedding extraction more directly, and it is useful when more granular embeddings are required. The output shows the shape, followed by the first 10 elements of the first token from the first sequence.

**Example 3: Token-Level Embeddings from a Specific Layer**

```python
from simpletransformers.language_representation import TransformerModel
import torch

# Model initialization
model = TransformerModel(model_type="bert", model_name="bert-base-uncased", use_cuda=False)

# Input sentences
sentences = [
    "This is a sample sentence.",
    "Another sentence for embedding.",
    "The quick brown fox jumps over the lazy dog."
]

# Tokenize input
inputs = model.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Obtain model output for a specific layer
with torch.no_grad():
  outputs = model.model(**inputs, output_hidden_states=True)
  
# Extract embeddings from, say, layer 6
layer_index = 6
embeddings = outputs.hidden_states[layer_index]


# Output shape and sample
print(f"Shape of embeddings: {embeddings.shape}")
print(f"First sentence first token embedding: {embeddings[0][0][:10]} ...")
```
This example illustrates the extraction of token-level embeddings from an intermediate hidden layer. The `output_hidden_states=True` argument added to the model call will return all hidden layers of BERT. Then, specific layer embeddings can be extracted as shown by `embeddings = outputs.hidden_states[layer_index]`. This can be important when task-specific tuning is performed. The output will show the dimension and values, in a similar manner to the previous examples. It is important to note that the layer indices start from 0 for the embedding layer and increase to the top layers.

For further learning and exploration, I recommend the following resources, although I will refrain from providing specific links:

1.  **The official Hugging Face Transformers documentation:** This is an indispensable resource for understanding the underlying mechanisms of the BERT models. It provides detailed explanations of the models, tokenizers, and the overall architecture.

2. **The Simpletransformers GitHub repository:** This repository contains examples, tutorials, and detailed explanations of how the Simpletransformers library can be used for tasks beyond embedding extraction. Careful reading will help you navigate various functionalities of this tool.

3.  **General resources on NLP and Deep Learning:** Having a solid grasp of core concepts like tokenization, embedding, neural networks, and Transformer architecture is beneficial. Books and tutorials focusing on these concepts will help expand your knowledge and deepen your ability to extract insights from BERT embeddings and their applications.

In closing, I've found that Simpletransformers offers a convenient method for extracting BERT embeddings with minimal code. It allows both direct and more tailored access to the layers, thus adapting to diverse requirements. Understanding these nuances is essential for achieving optimal performance in downstream NLP tasks. Utilizing the `encode_sentences` function or directly interacting with the model will allow effective use of BERT embeddings.
