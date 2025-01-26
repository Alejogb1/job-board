---
title: "How can I fine-tune a BERT model with new data?"
date: "2025-01-26"
id: "how-can-i-fine-tune-a-bert-model-with-new-data"
---

Fine-tuning a BERT model for a specific downstream task hinges on understanding that the pre-trained weights encode a vast amount of general linguistic knowledge. Instead of training a model from scratch, a process that would be computationally prohibitive, we leverage this pre-existing understanding and adapt it to our particular dataset and objective. This adaptation is achieved through fine-tuning, which primarily involves updating the model's parameters using task-specific data. I’ve personally fine-tuned BERT on several NLP tasks over the past few years, including sentiment analysis and document classification, and the techniques I’ll outline here have proven reliable and efficient.

The core principle behind fine-tuning is to add a task-specific layer on top of the pre-trained BERT model. This might be a simple linear layer for classification, a sequence labeling layer for named entity recognition, or a more complex architecture depending on the problem. Crucially, we retain most of the BERT parameters, only allowing them to be slightly adjusted during training while focusing on learning the weights of the new task-specific layer. This significantly reduces the training time and data requirements compared to training a language model from the ground up. The process essentially transforms BERT’s generic understanding into a specialized one.

Initially, BERT takes an input sequence of tokens, processes it through a series of transformer layers, and outputs contextualized word embeddings. During fine-tuning, the output from the final BERT layer serves as the input for our task-specific layer. This new layer calculates an output that corresponds to our problem’s requirements, e.g., the probability of a sentence having a positive sentiment. The model is then trained using a loss function appropriate for the specific task (cross-entropy loss for classification, for example), and an optimizer (such as Adam) adjusts both the task-specific layer’s parameters and, to a lesser degree, BERT's own parameters to minimize this loss. The learning rate for fine-tuning is typically much smaller than the learning rate used during the pre-training phase, to avoid drastically altering the pre-trained knowledge.

It's critical to note that data preparation is paramount. BERT and its variants require input text to be tokenized according to the model's vocabulary and padded or truncated to a fixed length. Incorrect preprocessing can lead to substantially poorer model performance, and it’s not a step to be underestimated.

Below, I'll present three code examples using Python with the `transformers` library from Hugging Face, which I’ve found to be exceptionally helpful in handling BERT-related tasks. These examples demonstrate variations in fine-tuning setup for different use cases.

**Example 1: Binary Text Classification (Sentiment Analysis)**

In this first example, we’ll fine-tune BERT for a binary sentiment classification task. This involves labeling each input text as either positive or negative.

```python
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Assume you have lists of texts and labels: texts, labels
texts = ["This movie was fantastic!", "I absolutely hated the film.", "The acting was okay."]
labels = [1, 0, 0]

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

max_len = 128
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    total_accuracy = 0
    total_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).sum().item()
            total_accuracy += accuracy
            total_batches += len(labels)
    print(f"Epoch {epoch+1} Validation Accuracy: {total_accuracy / total_batches}")
```

Here, `BertForSequenceClassification` adds a classification head on top of the BERT base model. The `SentimentDataset` prepares the input data for BERT, handling tokenization, padding and truncation. The training loop iterates through the training data, computes loss, backpropagates, and updates model parameters. We evaluate using validation data to track progress and generalization.

**Example 2: Multi-Class Text Classification**

This scenario extends the previous one to multi-class classification, for example topic classification. We make minor modifications to reflect the change in output dimension.

```python
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TopicDataset(Dataset):
     def __init__(self, texts, labels, tokenizer, max_len):
          self.texts = texts
          self.labels = labels
          self.tokenizer = tokenizer
          self.max_len = max_len

     def __len__(self):
          return len(self.texts)

     def __getitem__(self, idx):
          text = str(self.texts[idx])
          label = self.labels[idx]
          encoding = self.tokenizer.encode_plus(
               text,
               add_special_tokens=True,
               max_length=self.max_len,
               padding='max_length',
               truncation=True,
               return_attention_mask=True,
               return_tensors='pt'
          )
          return {
               'input_ids': encoding['input_ids'].flatten(),
               'attention_mask': encoding['attention_mask'].flatten(),
               'labels': torch.tensor(label, dtype=torch.long)
          }

#Assume you have lists of texts and labels: texts, labels
texts = ["The stock market soared today.", "Quantum computing breakthroughs announced.", "The team won the championship."]
labels = [0, 1, 2] # 0 - Finance, 1- Science, 2- Sports

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

max_len = 128
train_dataset = TopicDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = TopicDataset(val_texts, val_labels, tokenizer, max_len)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)


optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    total_accuracy = 0
    total_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
             input_ids = batch['input_ids'].to(device)
             attention_mask = batch['attention_mask'].to(device)
             labels = batch['labels'].to(device)
             outputs = model(input_ids, attention_mask=attention_mask)
             logits = outputs.logits
             predictions = torch.argmax(logits, dim=1)
             accuracy = (predictions == labels).sum().item()
             total_accuracy += accuracy
             total_batches += len(labels)
    print(f"Epoch {epoch+1} Validation Accuracy: {total_accuracy / total_batches}")
```

The main modification is setting `num_labels=3`, which changes the size of the output layer. The underlying concepts and structure remain identical to the binary classification.

**Example 3: Named Entity Recognition (Sequence Labeling)**

This demonstrates a more complex fine-tuning application where we assign a label to each word in a sentence.

```python
from transformers import BertTokenizer, BertForTokenClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx] # list of labels corresponding to tokens
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        #Adjust labels to match tokenization and padding
        tokenized_labels = []
        offset_mapping = encoding['offset_mapping'].squeeze().tolist()
        word_ids = encoding.word_ids()
        for i in range(len(word_ids)):
          if word_ids[i] is None:
              tokenized_labels.append(-100)
          else:
              tokenized_labels.append(labels[word_ids[i]])

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(tokenized_labels, dtype=torch.long)
        }


# Assume you have lists of texts and labels: texts, labels. Each text has a corresponding list of labels
texts = ["John lives in New York.", "Apple is a great company."]
labels = [[0,0,0,2,3,1], [2,0,0,0,0,0]] # 0:O, 1:LOC, 2:ORG, 3:PER


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=4)

max_len = 128
train_dataset = NERDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = NERDataset(val_texts, val_labels, tokenizer, max_len)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    total_accuracy = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_dataloader:
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids, attention_mask=attention_mask)
          logits = outputs.logits
          predictions = torch.argmax(logits, dim=2)
          mask = labels != -100
          accuracy = (predictions[mask] == labels[mask]).sum().item()
          total_accuracy += accuracy
          total_tokens += mask.sum().item()
    print(f"Epoch {epoch+1} Validation Accuracy: {total_accuracy / total_tokens}")
```
For sequence labeling, `BertForTokenClassification` is used. The labels are token-level, and the dataset needs to adjust to the tokenization, often requiring `offset_mapping`. Labels for special tokens and padded tokens are usually ignored with a value like -100 during loss calculation to avoid influencing training.

For further exploration, I recommend consulting resources like the documentation of the `transformers` library, which provides comprehensive guides and examples.  Additionally, "Deep Learning with PyTorch" and “Natural Language Processing with Transformers" are excellent books covering the theoretical and practical aspects of these topics. Finally, research papers on fine-tuning strategies for BERT, such as those available through academic databases like ACM Digital Library and IEEE Xplore, can offer advanced insights and recent developments in this field. These resources offer depth that complements the practical examples given here, enabling a much more robust grasp of BERT fine-tuning techniques.
