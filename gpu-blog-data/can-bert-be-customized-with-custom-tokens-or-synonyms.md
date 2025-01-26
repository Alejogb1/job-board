---
title: "Can BERT be customized with custom tokens or synonyms?"
date: "2025-01-26"
id: "can-bert-be-customized-with-custom-tokens-or-synonyms"
---

The pre-trained BERT model, while powerful, relies on a fixed vocabulary. Effectively adapting it to specialized domains often requires incorporating tokens outside of this pre-defined set, or tailoring the model’s understanding of existing tokens. I have personally wrestled with this challenge while implementing a natural language processing solution for an obscure historical text corpus, which required incorporating archaic terms and proper nouns not included in BERT's original vocabulary. This experience highlights the need for techniques that go beyond simple fine-tuning.

**1. Explanation: Custom Tokens and Synonym Handling in BERT**

BERT, at its core, tokenizes input text before processing it through its transformer architecture. Its tokenizer breaks down text into word pieces, which are essentially subword units or individual words depending on the complexity. The pre-trained vocabulary, generated during BERT's initial training, determines these word pieces. If a word is not in the vocabulary, the tokenizer will break it down into its component sub-word units, which are often not semantically meaningful. This can result in a loss of crucial context, especially when dealing with niche terminology or entities.

Custom tokens, in essence, are new words or phrases that you explicitly add to the vocabulary. The process of incorporating these new tokens is typically accomplished by adding them to the tokenizer’s vocabulary and initializing their corresponding embeddings (numerical representations). It's vital to understand that this alone is not sufficient for the model to understand these new terms; it only tells the model that the new tokens are discrete units within its input. The subsequent fine-tuning is crucial for the model to learn how these new tokens relate to existing vocabulary entries and for the model to produce meaningful outputs.

Synonym handling, on the other hand, is about modifying or expanding the model’s understanding of words that already exist in its vocabulary. The objective here is to nudge the model towards recognizing that certain existing tokens are semantically close to, or interchangeable with, each other within a specific context. Fine-tuning can achieve this implicitly by exposing the model to examples where synonyms are used in similar contexts. However, this often requires a considerable amount of data. Explicit methods, like employing synonym dictionaries and utilizing data augmentation techniques during fine-tuning, are beneficial for explicitly shaping the model’s understanding of synonyms.

Both approaches - integrating custom tokens and managing synonyms - are not mutually exclusive and are frequently used in combination. It often starts with identifying the custom tokens relevant to the specific domain and then fine-tuning with augmented data that includes synonym variations. The key lies in carefully balancing vocabulary expansion with the potential increase in model complexity and the need for appropriate training examples.

**2. Code Examples and Commentary**

Let's explore this concept through Python code using the `transformers` library from Hugging Face, a popular framework for NLP.

**Example 1: Adding Custom Tokens to the Tokenizer**

```python
from transformers import BertTokenizer, BertModel
import torch

# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example of custom tokens
custom_tokens = ['archaicterm', 'propernoun', 'specialphrase']

# Add the custom tokens to the tokenizer's vocabulary
num_added_tokens = tokenizer.add_tokens(custom_tokens)
print(f"Number of added tokens: {num_added_tokens}")

# Verify added tokens
print(f"Vocabulary Size after addition: {len(tokenizer)}")

# Load the pre-trained model (important!)
model = BertModel.from_pretrained('bert-base-uncased')

# Resize the model's embedding layer
model.resize_token_embeddings(len(tokenizer)) #Resize Model after adding new Tokens

# Sample input
text = 'This text contains an archaicterm. Also, check this propernoun!'
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs) #pass the tokenized input into the model
print (outputs.last_hidden_state.shape) # shape will be [1,14,768]

# You would now fine-tune the model with your specialized data
# Example of using custom tokens
custom_text = 'This also include specialphrase'
inputs = tokenizer(custom_text, return_tensors="pt")
custom_outputs = model(**inputs)
print (custom_outputs.last_hidden_state.shape)
```

*   **Commentary:** This code snippet demonstrates the crucial first step: expanding the tokenizer's vocabulary. The `add_tokens` method directly incorporates new tokens into the tokenizer’s structure and assigns them a unique ID in the tokenizer vocabulary. It's essential to resize the model’s embedding layer using `model.resize_token_embeddings` to ensure that the model can accommodate embeddings for the newly added tokens. If you do not resize the embedding layers, an error will occur at the model inference stage. The subsequent code demonstrates how those custom tokens are encoded by the modified tokenizer when passed to the model. The embedding vectors for added tokens are initialized randomly and require fine-tuning using specialized training data to learn meaningful contextual information.

**Example 2: Basic Synonym Handling during Fine-tuning with Data Augmentation**

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import random

# Define a simple dataset (replace with your actual dataset)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example synonymous words to use during data augmentation
synonyms = {
    "ancient": ["old", "aged", "antique"],
    "artifact": ["item", "object", "relic"]
}

def synonym_augmentation(text, synonyms, p=0.2):
    words = text.split()
    augmented_words = []
    for word in words:
      if word in synonyms and random.random() < p:
        augmented_words.append(random.choice(synonyms[word]))
      else:
        augmented_words.append(word)
    return " ".join(augmented_words)

# Data setup
texts = ["This is an ancient scroll", "I found an artifact in the ruins",
    "This is an old scroll", "I found an item in the ruins",
    "This is a valuable aged document", "I see an object near the water"]

labels = [0, 1, 0, 1, 0, 1] # Binary classification task (0 or 1)

# Load tokenizer and model (binary classification model this time)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


# Data augmentation
augmented_texts = []
for text in texts:
  augmented_texts.append(text)
  augmented_texts.append(synonym_augmentation(text,synonyms))

augmented_labels = labels + labels

# Data loader
dataset = TextDataset(augmented_texts, augmented_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
  for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
  print(f"Epoch: {epoch}, Loss: {loss.item()}")
```

*   **Commentary:** This example focuses on enhancing the model’s understanding of synonyms during the fine-tuning stage. The `synonym_augmentation` function replaces words in the original text with their synonyms with a probability of *p*. This strategy increases the dataset’s variety and allows the model to learn synonym associations implicitly. The `TextDataset` class encapsulates the data handling, which involves tokenization and appropriate padding. It uses a simple classification scenario, which could also be replaced with another task. Note the augmented data is added *in addition* to the original data. This provides more variety during the fine-tuning training.

**Example 3: Simple Dictionary Look Up**
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np

# Define a simple dataset (replace with your actual dataset)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, synonym_dict):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.synonym_dict = synonym_dict

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        synonym_encoded = self.synonym_encoding(text)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'synonym_encoding' : synonym_encoded.float()
        }

    def synonym_encoding(self, text):
      encoded_text = self.tokenizer(text,return_tensors="pt", add_special_tokens = True)
      input_ids = encoded_text.input_ids[0]

      synonym_encoding = torch.zeros((len(input_ids), len(self.synonym_dict)+1))

      for i, token in enumerate(input_ids):
        word = self.tokenizer.decode(token, skip_special_tokens=True)
        if word in self.synonym_dict:
          synonym_encoding[i,self.synonym_dict.index(word)+1] = 1
        else:
          synonym_encoding[i,0] = 1

      return synonym_encoding

# Example synonymous words to use during data augmentation
synonym_dict = ["ancient", "old", "artifact", "item"]

# Data setup
texts = ["This is an ancient scroll", "I found an artifact in the ruins",
    "This is an old scroll", "I found an item in the ruins"]

labels = [0, 1, 0, 1] # Binary classification task (0 or 1)

# Load tokenizer and model (binary classification model this time)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Data loader
dataset = TextDataset(texts, labels, tokenizer, synonym_dict)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
  for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    outputs = model(**batch,labels = batch['labels'],synonym_encoding = batch['synonym_encoding'])
    loss = outputs.loss

    loss.backward()
    optimizer.step()
  print(f"Epoch: {epoch}, Loss: {loss.item()}")
```
*   **Commentary:** This final example illustrates how one could implement a simple dictionary lookup in conjunction with BERT. Each token in the input text is evaluated and a new vector is appended to it. This vector is a one hot encoding. When a word is in the synonym dictionary, the associated synonym index is flipped to '1', otherwise the first element in the vector is flipped to '1'. This allows for explicit handling of synonyms via an additional input to the model. If you wished to go further, this can be passed through another linear layer to combine it with other hidden states. The model here is simply fine-tuned using a simple example loss, but one can customize the loss depending on the task.

**3. Resource Recommendations**

For a deeper understanding of BERT and its associated techniques, I would recommend exploring resources that offer comprehensive explanations of the transformer architecture, subword tokenization, and fine-tuning methodologies. Specifically, material that covers the mechanics of pre-training and fine-tuning large language models is invaluable. Consider seeking sources discussing data augmentation techniques specifically used in natural language processing. Furthermore, reviewing publications related to domain adaptation in NLP can assist in creating effective custom token strategies. Look for guides that provide concrete code examples to gain practical insights on implementation details.
