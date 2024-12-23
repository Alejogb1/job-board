---
title: "How can text be vectorized using tensors with multiple label columns?"
date: "2024-12-23"
id: "how-can-text-be-vectorized-using-tensors-with-multiple-label-columns"
---

, let’s tackle this. Thinking back, I recall working on a complex sentiment analysis project a few years ago. We weren't dealing with simple single-label classifications; we had a dataset with various nuanced sentiment dimensions – like ‘joy,’ ‘anger,’ ‘sadness,’ and ‘surprise,’ all present to varying degrees in a single text snippet. It quickly became apparent that treating these as independent binary labels was inadequate. We needed a more nuanced approach, and that involved effective text vectorization when paired with multi-label targets.

The challenge lies in representing text data numerically in a way that's amenable to machine learning models when dealing with multiple labels. When you have multiple columns of labels, the vectorization process must consider not only the text itself but also how it relates to *all* labels simultaneously. The conventional approach of converting each text into a fixed-length vector, like what TF-IDF or basic word embeddings do, is just the initial step. You need that vector to be suitable to train or infer multiple columns at the same time.

Let’s break this down. The initial stage of text vectorization is crucial and can be done using techniques like *tokenization* followed by embedding strategies. Tokenization involves dividing text into smaller units, such as words or sub-words. Then, these tokens can be converted into numerical representations (embeddings). These embeddings capture the semantic and syntactic relationships between words. Common choices here are techniques such as *TF-IDF* (term frequency-inverse document frequency), word embeddings like *Word2Vec* or *GloVe*, or contextualized embeddings such as *BERT*, *RoBERTa* or *Sentence-BERT*.

While these techniques produce feature vectors for text, they are independent of labels. The 'trick,' if you can call it that, comes in how we use these text vectors in training models capable of handling multiple target label columns. Essentially, we need a model architecture that can map from our fixed text vector to several independent predictions, one for each label. Here, we aren't changing how we vectorize, but how we use those vectors as an input.

Let's illustrate the concept with some simplified code examples. These snippets utilize *PyTorch* because it’s quite prevalent in this domain and offers clarity in terms of tensors and operations. Remember, in production settings, we'd use libraries specialized for efficiency.

First, imagine we have our pre-processed text data in a format where we have a column for text and several columns for our multiple labels.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Simplified dataset structure
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32) # Multiple labels.
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text,
                                 add_special_tokens=True,
                                 max_length=self.max_len,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt'
                                )
        return encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze(), self.labels[idx]


# Sample Data using pandas
data = {'text': ['This is great!', 'I am so sad today.', 'What a fantastic surprise!', 'Terrible news.'],
        'joy': [1, 0, 1, 0],
        'anger': [0, 0, 0, 1],
        'sadness': [0, 1, 0, 1],
        'surprise': [0, 0, 1, 0]}

df = pd.DataFrame(data)

texts = df['text'].tolist()
labels = df[['joy', 'anger', 'sadness', 'surprise']].values.tolist()

from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 128

dataset = TextDataset(texts, labels, tokenizer, max_len)

train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Model Definition
class MultiLabelClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
      super(MultiLabelClassifier, self).__init__()
      self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
      self.dropout = nn.Dropout(0.1)
      self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels) # Output = number of labels

    def forward(self, input_ids, attention_mask):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
      pooled_output = self.dropout(pooled_output)
      logits = self.classifier(pooled_output)
      return logits

model = MultiLabelClassifier('distilbert-base-uncased', 4) # 4 Labels

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training
num_epochs = 2
for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

print("Training Finished")

```

In this code, notice that `TextDataset` returns both tokenized input and a tensor containing multiple label values. `MultiLabelClassifier` outputs a prediction for each label as a single tensor. The loss function `nn.BCEWithLogitsLoss` is appropriate because we are treating each label as an independent binary classification problem (this was a design choice made based on the problem I mentioned in the beginning). The model architecture also assumes that all labels are independent.

Let’s change things around slightly with another example. Suppose we're dealing with structured data alongside text, perhaps some categorical features relevant to the text content.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn.utils.rnn import pad_sequence

# Sample data with structured data

data = {'text': ['This is great!', 'I am so sad today.', 'What a fantastic surprise!', 'Terrible news.'],
        'category': ['positive', 'negative', 'positive', 'negative'],
        'joy': [1, 0, 1, 0],
        'anger': [0, 0, 0, 1],
        'sadness': [0, 1, 0, 1],
        'surprise': [0, 0, 1, 0]}

df = pd.DataFrame(data)
texts = df['text'].tolist()
labels = df[['joy', 'anger', 'sadness', 'surprise']].values.tolist()
categories = df['category'].tolist()


# Label Encode Categories
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)

# Create Dataset with both text and categories
class MultiLabelStructuredTextDataset(Dataset):
    def __init__(self, texts, labels, categories, tokenizer, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.categories = torch.tensor(categories, dtype=torch.long) # Assuming categories are integers
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       text = self.texts[idx]
       encoded = self.tokenizer(text,
                                 add_special_tokens=True,
                                 max_length=self.max_len,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt'
                                )

       return encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze(), self.categories[idx], self.labels[idx]

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 128
dataset = MultiLabelStructuredTextDataset(texts, labels, encoded_categories, tokenizer, max_len)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Modified Model to combine text and structured data
class MultiLabelClassifierWithStructuredData(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, num_categories, embedding_dim=16):
      super(MultiLabelClassifierWithStructuredData, self).__init__()
      self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
      self.category_embedding = nn.Embedding(num_categories, embedding_dim)
      self.dropout = nn.Dropout(0.1)
      self.fc1 = nn.Linear(self.bert.config.hidden_size + embedding_dim, 64)
      self.relu = nn.ReLU()
      self.classifier = nn.Linear(64, num_labels)

    def forward(self, input_ids, attention_mask, category_ids):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      pooled_output = outputs.last_hidden_state[:, 0, :]
      category_embed = self.category_embedding(category_ids)
      combined_features = torch.cat((pooled_output, category_embed), dim=1)
      combined_features = self.dropout(combined_features)
      features = self.relu(self.fc1(combined_features))
      logits = self.classifier(features)
      return logits

num_categories = len(set(encoded_categories)) # Number of unique categories
model = MultiLabelClassifierWithStructuredData('distilbert-base-uncased', 4, num_categories)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)


# Training loop (simplified)
num_epochs = 2
for epoch in range(num_epochs):
    for batch in train_dataloader:
      input_ids, attention_mask, categories, labels = batch
      optimizer.zero_grad()
      outputs = model(input_ids, attention_mask, categories)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

print("Training Finished")
```

This code illustrates that the vectors derived from text can be combined with numerical or categorical data, giving a more sophisticated representation. The main change here is that the input to the model now includes the `category` alongside the text, encoded as integers, which are then embedded.

Finally, let's think about a more complex situation: We might want to use the outputs of the model as a set of embeddings (the second to last hidden layers). This is common in many applications where we want to represent each text, *with its associated labels*.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn.utils.rnn import pad_sequence

# Sample data with structured data

data = {'text': ['This is great!', 'I am so sad today.', 'What a fantastic surprise!', 'Terrible news.'],
        'category': ['positive', 'negative', 'positive', 'negative'],
        'joy': [1, 0, 1, 0],
        'anger': [0, 0, 0, 1],
        'sadness': [0, 1, 0, 1],
        'surprise': [0, 0, 1, 0]}

df = pd.DataFrame(data)
texts = df['text'].tolist()
labels = df[['joy', 'anger', 'sadness', 'surprise']].values.tolist()
categories = df['category'].tolist()


# Label Encode Categories
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)

# Create Dataset with both text and categories
class MultiLabelStructuredTextDataset(Dataset):
    def __init__(self, texts, labels, categories, tokenizer, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.categories = torch.tensor(categories, dtype=torch.long) # Assuming categories are integers
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       text = self.texts[idx]
       encoded = self.tokenizer(text,
                                 add_special_tokens=True,
                                 max_length=self.max_len,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt'
                                )

       return encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze(), self.categories[idx], self.labels[idx]

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 128
dataset = MultiLabelStructuredTextDataset(texts, labels, encoded_categories, tokenizer, max_len)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Modified Model to combine text and structured data
class MultiLabelClassifierWithStructuredData(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, num_categories, embedding_dim=16):
      super(MultiLabelClassifierWithStructuredData, self).__init__()
      self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
      self.category_embedding = nn.Embedding(num_categories, embedding_dim)
      self.dropout = nn.Dropout(0.1)
      self.fc1 = nn.Linear(self.bert.config.hidden_size + embedding_dim, 64)
      self.relu = nn.ReLU()
      self.classifier = nn.Linear(64, num_labels)

    def forward(self, input_ids, attention_mask, category_ids):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      pooled_output = outputs.last_hidden_state[:, 0, :]
      category_embed = self.category_embedding(category_ids)
      combined_features = torch.cat((pooled_output, category_embed), dim=1)
      combined_features = self.dropout(combined_features)
      features = self.relu(self.fc1(combined_features))
      logits = self.classifier(features)
      return features, logits #Returning embeddings as well

num_categories = len(set(encoded_categories)) # Number of unique categories
model = MultiLabelClassifierWithStructuredData('distilbert-base-uncased', 4, num_categories)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)


# Training loop (simplified)
num_epochs = 2
for epoch in range(num_epochs):
    for batch in train_dataloader:
      input_ids, attention_mask, categories, labels = batch
      optimizer.zero_grad()
      embeddings, outputs = model(input_ids, attention_mask, categories)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

print("Training Finished")

```

Here, we added a return for the final features, and we can then use these embeddings for things like clustering or visualization. Notice that the training loss is still calculated from the final logits, but that the outputs of the model also return a tensor with the learned embeddings.

For deeper dives into these concepts, I'd recommend the "Speech and Language Processing" by Jurafsky and Martin for general text processing; "Natural Language Processing with Transformers" by Tunstall, von Werra, and Wolf for more practical deep learning-based text processing; and "Deep Learning" by Goodfellow, Bengio, and Courville for a broader grounding in the fundamental machine learning. These resources should offer more insight into the theory and practical aspects of the techniques.

Essentially, text vectorization for multi-label problems is not just about the initial conversion of text to numbers; it’s about how that numerical representation is used in the context of your multi-label prediction model. The model architecture, loss function, and training strategy need to be carefully tailored to consider multiple targets simultaneously. The examples given should give you a solid starting point.
