---
title: "How can BERT be used for multi-class text classification?"
date: "2024-12-23"
id: "how-can-bert-be-used-for-multi-class-text-classification"
---

Alright, let’s tackle this. It's a common challenge, and frankly, one I've spent quite a bit of time navigating. When we talk about using BERT for multi-class text classification, we're essentially leveraging its powerful understanding of language to categorize text into one of several predefined categories. It’s not a trivial task, but the results can be impressively accurate, especially when compared to older techniques. I've personally deployed solutions like this for tasks ranging from categorizing customer support tickets to classifying news articles across different topics.

The fundamental idea revolves around fine-tuning a pre-trained BERT model. Now, ‘pre-trained’ is key here. BERT, being a transformer-based model, has already seen vast amounts of text during its initial training, learning contextual relationships between words and phrases. This is the foundation. We’re not starting from scratch; instead, we're adapting this generalized language understanding to our specific classification problem. This significantly reduces the amount of data we need for our own training and drastically increases our model's performance.

The core process involves a few crucial steps. First, we feed our text data, along with our labels, into the BERT model. The input goes through the transformer layers, generating a contextualized representation for each token (word or subword). For classification, we're not directly interested in every single token's representation. Instead, we typically focus on the representation of the special `[CLS]` token. This token, added at the beginning of the input sequence, is designed to aggregate the information from the entire input and, after fine-tuning, serves as a good representation of the whole text.

On top of this `[CLS]` token representation, we append a simple feed-forward neural network, also called a classification head. This layer takes the `[CLS]` vector as input and projects it onto a vector that matches the number of classes we have. A softmax function is then applied to these values, converting them into probabilities that sum to one, corresponding to the likelihood of a given piece of text belonging to each specific class. We then compute our loss against the true class labels, and backpropagate the loss to fine-tune both the classification head and, crucially, the layers of the BERT model itself. This is where the magic happens—BERT learns how to adapt its understanding of language to discriminate between our specific categories.

Let's break this down with some code examples using Python and the `transformers` library from Hugging Face, which has become almost standard in the field.

**Example 1: Basic Setup using PyTorch:**

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# 1. Initialize tokenizer and model.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5) # 5 classes

# 2. Preprocess data (example data)
texts = ["This is a very positive review.", "I hated this product.", "Neutral feedback here.", "Another positive one!", "This is bad."]
labels = [0, 4, 2, 0, 4]  # Example: 0 = positive, 4 = negative, 2 = neutral
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
labels = torch.tensor(labels)
dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=2)

# 3. Setup optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# 4. Training loop (basic)
epochs = 3
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete")
```

This is a very basic illustration. Here, we’ve taken a pre-trained `bert-base-uncased` model, configured it for 5 classes, and trained it on a small dataset. Crucially, note the use of `BertForSequenceClassification`, the class from the transformers library which contains the `[CLS]` token functionality I was discussing earlier. The dataloader handles batches, which is key for efficient learning. The AdamW optimizer is generally a good choice for transformer models.

**Example 2: Using a Custom Dataset:**

In a real scenario, you wouldn't have the data conveniently in a list. We'd typically load data from a file or database. This example demonstrates loading data, then creating a custom dataset class to manage it:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd

class CustomTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['label']
        encoding = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)}


# Load data (replace with your actual data loading)
data = {'text': ["First review", "Second one.", "Third."], 'label': [0,1,2]}
df = pd.DataFrame(data)

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Create dataset and dataloader
dataset = CustomTextDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 2
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
print("Training complete")
```

Here, a custom dataset class handles loading the data. We are taking advantage of the `**batch` syntax which allows us to pass the output of a batch directly into the model. This flexibility in data handling is vital for handling realistic data sources.

**Example 3: Prediction:**

Once you've fine-tuned your model, you'll want to use it for prediction. Here’s a simplified example of how to predict class labels for new data:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('fine_tuned_model') # load model you trained

# Prepare new data
new_texts = ["This is a new text to classify", "Another piece of text", "More text here."]

encodings = tokenizer(new_texts, truncation=True, padding=True, return_tensors='pt')

# Prediction
with torch.no_grad(): # Disable gradient calculation during inference
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=-1).tolist()

print("Predicted Classes:", predictions)

```

Note the usage of `torch.no_grad()` during prediction. This is crucial because we don't need to keep track of gradients during inference which saves computation resources. It's also important to save and reload your model when predicting after training. I've used a placeholder 'fine_tuned_model' here, assuming it was saved during training. The predicted classes are extracted with `torch.argmax` and converted to a standard list for readability.

For further reading on the theoretical foundations of transformers and BERT, I highly recommend “Attention is All You Need” (Vaswani et al., 2017), the seminal paper introducing the transformer architecture, and the original BERT paper “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” (Devlin et al., 2018). These resources offer a deep dive into the mechanisms that drive these models. For more applied knowledge, "Natural Language Processing with Transformers" by Lewis Tunstall, Leandro von Werra, and Thomas Wolf is an excellent practical guide. Additionally, the Hugging Face documentation for the `transformers` library is a valuable resource. Finally, the book, "Speech and Language Processing" by Jurafsky and Martin provides broader context into NLP techniques, although it may be a deeper dive than necessary for this specific task.

In practice, you'll likely need to spend time fine-tuning your hyperparameters, experimenting with different optimizers, learning rates and batch sizes. Remember that model evaluation metrics like accuracy, precision, recall and F1-score are all important to monitor and choose the optimal model. The code snippets above should give you a working starting point, but fine-tuning your approach based on your specific dataset and task is key to achieving superior results. This kind of problem requires iterative refinement, but once you have the base concept down, and some practical experience you'll be well on your way to getting good accuracy using BERT for multi-class classification.
