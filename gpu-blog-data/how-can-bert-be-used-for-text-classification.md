---
title: "How can BERT be used for text classification?"
date: "2025-01-30"
id: "how-can-bert-be-used-for-text-classification"
---
BERT, a transformer-based language model, excels in text classification by leveraging its contextual understanding of word relationships. Unlike traditional methods reliant on word frequencies or simple co-occurrence, BERT's bidirectional training allows it to capture nuances within a sentence, making it highly effective for a variety of classification tasks, including sentiment analysis, topic categorization, and intent detection. I've personally observed this during my work on a large-scale customer feedback analysis project, where a BERT model consistently outperformed other approaches by a significant margin in accurately classifying nuanced feedback sentiments.

The core of BERT's operation for classification lies in its transformer architecture. This architecture, comprised of multiple self-attention layers, processes input tokens in parallel, allowing each token to attend to all other tokens in the sequence. This bidirectional attention mechanism enables the model to grasp not only the immediate context of a word but also its relationship to words both preceding and following it within the sentence. Furthermore, BERT is pre-trained on massive text corpora, allowing it to learn a deep understanding of language, which is then fine-tuned on specific classification datasets. During fine-tuning, a classification layer, typically a simple linear layer, is added to the pre-trained BERT model. This layer takes the pooled representation from the final transformer layer and predicts the class label.

The process involves several steps. Initially, the text is tokenized using BERT's custom WordPiece tokenizer, which breaks words into subword units. Special tokens, such as `[CLS]` at the beginning and `[SEP]` at the end of each sequence, are added. The tokenized input is then converted into numerical representations, or token IDs, which are input to the BERT model along with corresponding attention masks to indicate which tokens are padding. The output of the BERT model is a sequence of contextualized embedding vectors. For classification tasks, the embedding corresponding to the `[CLS]` token is generally extracted and passed through the classification layer. The output of this layer then represents the class probabilities.

The fine-tuning procedure is crucial. The pre-trained BERT model's weights are adjusted by backpropagation on the specific classification dataset. The classification layer weights are also trained during this stage. The chosen loss function will depend on the number of classes. Binary cross-entropy for binary classification or categorical cross-entropy for multi-class. Hyperparameter tuning, such as learning rate, batch size and number of epochs, is performed to optimize performance.

To illustrate this further, let's consider practical code examples using the Python `transformers` library, a widely used package for working with transformer models.

**Example 1: Binary Sentiment Classification**

This example demonstrates how BERT can be fine-tuned for binary sentiment classification. I use a dataset containing movie reviews labeled as either positive or negative. The `transformers` library provides a convenient interface for loading pre-trained models and tokenizers.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load a sample dataset (replace with your data loading logic)
data = pd.DataFrame({'review': ["This movie was great!", "I hated this film.", "It was okay."], 'sentiment':[1,0,1]})

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)


# Define a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer(text,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  padding='max_length',
                                  truncation=True,
                                  return_attention_mask=True,
                                  return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Create datasets and dataloaders
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)


# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)


# Fine-tuning loop
num_epochs = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader)}")


# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_dataloader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      preds = torch.argmax(logits, dim=-1).cpu().numpy()
      predictions.extend(preds)
      true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy: {accuracy}")
```
This code segment demonstrates the process, from data loading and preprocessing to fine-tuning and evaluation, using a binary classification case. The key points are loading the tokenizer and model, preparing the data for training in the form of a dataloader, defining the optimizer, performing forward passes during training and calculating the loss, using backpropagation to update the model's weights and finally performing evaluation to compute the accuracy of the model in a validation set. This example is intentionally kept concise for clarity.

**Example 2: Multi-Class Topic Classification**

Moving to a multi-class problem, imagine classifying news articles into categories like 'sports', 'politics', and 'technology'.  The code structure is similar, but `num_labels` and the loss function may need to be modified. Here, the output layer will have as many nodes as there are classes.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load a sample dataset (replace with your data loading logic)
data = pd.DataFrame({'article': ["The soccer match was intense.", "The election results are in.", "New software released today"],
                    'topic': [0, 1, 2]}) # 0=sports, 1=politics, 2=tech

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(data['article'], data['topic'], test_size=0.2, random_state=42)


# Define a custom dataset class
class TopicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer(text,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  padding='max_length',
                                  truncation=True,
                                  return_attention_mask=True,
                                  return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3) #3 classes

# Create datasets and dataloaders
train_dataset = TopicDataset(train_texts, train_labels, tokenizer)
val_dataset = TopicDataset(val_texts, val_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)


# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)


# Fine-tuning loop
num_epochs = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader)}")


# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_dataloader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      preds = torch.argmax(logits, dim=-1).cpu().numpy()
      predictions.extend(preds)
      true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy: {accuracy}")
```
In this code, the only notable change from the binary example is setting the num_labels parameter in `BertForSequenceClassification` to 3 to accommodate our multi-class scenario. This change is fundamental, as it tailors the output layer to the specific number of categories being predicted. This shows how the same general process can be adapted to a different type of classification problem, making use of the flexibility of BERT.

**Example 3: Handling Variable Sequence Lengths**

A key consideration with BERT is the handling of input text with varying lengths. The `transformers` library handles this by padding shorter sequences with a special padding token (`[PAD]`) up to a defined maximum length. Longer sequences can be truncated to fit. This is why in our custom Dataset classes, we set the parameters `padding='max_length'` and `truncation=True`. The attention mask then ensures that BERT disregards these padding tokens. This ensures that the model is able to work with the varying input text lengths, without having any issues with the calculations.

While I haven’t presented a standalone example for varying lengths here, since they are explicitly handled by the tokenizer in the above examples, I want to emphasize the crucial role padding and attention masks play when working with text classification. Without them, shorter sequences would contribute less to the final prediction. This ensures that all sequences, regardless of their length are processed uniformly.

For additional learning on this topic, I recommend the following resources: The *Hugging Face Transformers* documentation offers a comprehensive guide to using their library, and it includes practical examples covering various classification tasks. The *BERT paper* itself is a valuable source for in-depth understanding of the model's architecture and training methodology. Academic papers on NLP, often published in venues like *ACL* and *EMNLP*, provide the cutting-edge research and advancements in the field of transformer-based models. These will equip a practitioner with in-depth knowledge and skills to fully grasp and effectively employ BERT for text classification projects.

In summary, BERT, with its bidirectional training and contextual understanding, is a potent tool for text classification. The ability to easily fine-tune pre-trained models with frameworks like `transformers`, enables developers to leverage its power for a broad array of real-world applications. Proper understanding of tokenization, padding and attention mechanisms are critical for achieving optimal results.
