---
title: "How to use Bert for relation extraction?"
date: "2024-12-14"
id: "how-to-use-bert-for-relation-extraction"
---

alright, so you're looking to use bert for relation extraction. it's a pretty common task these days, and bert's actually a solid choice for it. i've spent some time in this area myself, and i can share what i've learned.

basically, relation extraction is about identifying relationships between entities in text. think about it like this: in the sentence "steve jobs co-founded apple," we want to extract that 'steve jobs' is related to 'apple' through the relation 'co-founded'. there are other relations of course, like 'is a' or 'located in' and so on, it all depends on your specific task.

now, bert by itself doesn't magically do this, we need to train it for this specific purpose. you see, bert is a language model trained on huge amounts of text to understand language. for relation extraction, we need to fine-tune it. the idea is to take the pre-trained bert model and add a classification layer on top. this classifier will predict the relation between the entities we've identified in the sentence.

so the first thing to get straight is data. you need training data in a format that bert and our classification model can consume. a popular data format is triples, where each triple consists of (subject entity, relation, object entity). for example, for the sentence about steve jobs we could format the data in such a way: ("steve jobs", "co-founded", "apple"). a bunch of those constitutes the training data, that is the main food source for your fine-tuning process.

let's look at a code example. we will use pytorch and the transformers library since that's what i've used the most.

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# assume you have a list of your training data in triples
# that looks like this [(sent1, ent1_start, ent1_end, ent2_start, ent2_end, relation), ....]
# where the start and end indexes are related to the tokenized sentence
# and relationship is an integer that represents that relation in your dataset.

class RelationExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent, ent1_start, ent1_end, ent2_start, ent2_end, relation = self.data[idx]
        
        encoding = self.tokenizer(
            sent,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # creating a 'mention mask' that is 1 on the entity tokens and 0 otherwise
        mention_mask = torch.zeros(self.max_len, dtype=torch.long)
        
        # mapping the character based indexes from the original string 
        # to the tokenized indexes
        token_offsets = encoding.offset_mapping[0]
        
        token_start_1 = -1
        token_end_1 = -1

        for token_index, (start, end) in enumerate(token_offsets):
            if start <= ent1_start < end:
                token_start_1 = token_index
            if start < ent1_end <= end:
                token_end_1 = token_index
        
        if token_start_1 != -1 and token_end_1 != -1:
          mention_mask[token_start_1:token_end_1+1] = 1

        token_start_2 = -1
        token_end_2 = -1

        for token_index, (start, end) in enumerate(token_offsets):
            if start <= ent2_start < end:
                token_start_2 = token_index
            if start < ent2_end <= end:
                token_end_2 = token_index

        if token_start_2 != -1 and token_end_2 != -1:
          mention_mask[token_start_2:token_end_2+1] = 1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'mention_mask': mention_mask,
            'labels': torch.tensor(relation, dtype=torch.long)
        }

# you would initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=num_relations
)

# some setup hyper params
max_len = 128
learning_rate = 2e-5
batch_size = 32
num_epochs = 3

# and you would create the dataset and dataloaders
train_dataset = RelationExtractionDataset(train_data, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = RelationExtractionDataset(val_data, tokenizer, max_len)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# and you setup your optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# example of how the training would look like
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        mention_mask = batch['mention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
      total_val_loss = 0
      for batch in val_loader:
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          mention_mask = batch['mention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
          total_val_loss += outputs.loss.item()

      avg_val_loss = total_val_loss / len(val_loader)
      print(f"validation loss at epoch {epoch}: {avg_val_loss}")

```

this code does a few things:

*   it sets up a custom `dataset` class for your relation data which loads your text, identifies the start and end positions of each entity and converts it into a format that is useful for bert's input. importantly, it uses `offset_mapping` from the tokenizer to map the original character-level positions of the entities into the positions of the tokens from the bert tokenization process. this is very important since the tokenization is not the same as a character or word based index.
*   it shows how to set up the bert model for sequence classification. you specify the number of classes in the `num_labels` argument. this will define how many distinct relations that you want to predict.
* it shows a standard training process that iterates through the training set in batches with the backward pass and the optimizer that will adapt the weights of the model
*   it also shows validation which is important to track the performance of your training and detect overfitting

but, that's only one way to do it. there are others, here's another approach.

instead of just using the bert model directly for classification, we can create entity markers, using them to signal to bert which tokens are of interest. this requires a bit of preprocessing but it might be more effective in certain cases. you basically add some special tokens into your original sentence before passing it to bert, something like "\[entity1] steve jobs \[/entity1] co-founded \[entity2] apple \[/entity2]".

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

class RelationExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_tokens = ["[entity1]", "[/entity1]", "[entity2]", "[/entity2]"]
        self.tokenizer.add_tokens(self.special_tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent, ent1_start, ent1_end, ent2_start, ent2_end, relation = self.data[idx]

        # insert entity markers into the original sentence
        marked_sent = sent[:ent2_end] + self.special_tokens[3] + sent[ent2_end:] 
        marked_sent = marked_sent[:ent2_start] + self.special_tokens[2] + marked_sent[ent2_start:] 
        marked_sent = marked_sent[:ent1_end] + self.special_tokens[1] + marked_sent[ent1_end:]
        marked_sent = marked_sent[:ent1_start] + self.special_tokens[0] + marked_sent[ent1_start:]
                
        encoding = self.tokenizer(
            marked_sent,
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
            'labels': torch.tensor(relation, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=num_relations
)

# resizing the model embeddings after adding the new special tokens
model.resize_token_embeddings(len(tokenizer))


max_len = 128
learning_rate = 2e-5
batch_size = 32
num_epochs = 3

# and you would create the dataset and dataloaders
train_dataset = RelationExtractionDataset(train_data, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = RelationExtractionDataset(val_data, tokenizer, max_len)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


optimizer = AdamW(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
      total_val_loss = 0
      for batch in val_loader:
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
          total_val_loss += outputs.loss.item()

      avg_val_loss = total_val_loss / len(val_loader)
      print(f"validation loss at epoch {epoch}: {avg_val_loss}")
```

this is how that previous code is different:

*   it uses special tokens to mark the entity mentions. these are added to the sentence strings before tokenization, which requires adding new tokens to the tokenizer vocabulary. also after adding the new tokens it's necessary to resize the embedding layer of the bert model.
* the rest of the training code follows the same logic from before.

now, there are other ways too, like using a pooling strategy over the entity tokens or modifying the attention mechanism, but those are more complex, and usually only give marginal improvements. but you could look into that after you have something basic working.

one more thing that can be helpful. sometimes in my experience, instead of using a single classification layer on top of bert, you can use bi-directional lstm before doing the classification. i had this one case, when i was fine-tuning bert for chemical relation extraction (weird stuff i know) and this improved the results by a couple of percentage points, it might not seem like much, but in the research world every percentage point matters. i spent a good couple of weeks on that trying various architectures. it turned out that adding the lstm layer made a big difference in that particular dataset. who would have thought.

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

class RelationExtractionModel(nn.Module):
    def __init__(self, bert_model_name, num_relations, lstm_hidden_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, 
                            lstm_hidden_size, 
                            batch_first=True, 
                            bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_relations)

    def forward(self, input_ids, attention_mask, mention_mask):
      
        bert_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state

        masked_output = bert_output * mention_mask.unsqueeze(-1)

        lstm_out, _ = self.lstm(masked_output)
        
        # mean pooling
        pooled_output = lstm_out.sum(dim=1) / (mention_mask.sum(dim=1, keepdim=True)+1e-10)

        logits = self.classifier(pooled_output)
        return logits


class RelationExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent, ent1_start, ent1_end, ent2_start, ent2_end, relation = self.data[idx]
        
        encoding = self.tokenizer(
            sent,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # creating a 'mention mask' that is 1 on the entity tokens and 0 otherwise
        mention_mask = torch.zeros(self.max_len, dtype=torch.long)
        
        # mapping the character based indexes from the original string 
        # to the tokenized indexes
        token_offsets = encoding.offset_mapping[0]
        
        token_start_1 = -1
        token_end_1 = -1

        for token_index, (start, end) in enumerate(token_offsets):
            if start <= ent1_start < end:
                token_start_1 = token_index
            if start < ent1_end <= end:
                token_end_1 = token_index
        
        if token_start_1 != -1 and token_end_1 != -1:
          mention_mask[token_start_1:token_end_1+1] = 1

        token_start_2 = -1
        token_end_2 = -1

        for token_index, (start, end) in enumerate(token_offsets):
            if start <= ent2_start < end:
                token_start_2 = token_index
            if start < ent2_end <= end:
                token_end_2 = token_index

        if token_start_2 != -1 and token_end_2 != -1:
          mention_mask[token_start_2:token_end_2+1] = 1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'mention_mask': mention_mask,
            'labels': torch.tensor(relation, dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = RelationExtractionModel('bert-base-uncased', num_relations=5, lstm_hidden_size=128)


max_len = 128
learning_rate = 2e-5
batch_size = 32
num_epochs = 3

train_dataset = RelationExtractionDataset(train_data, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = RelationExtractionDataset(val_data, tokenizer, max_len)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


optimizer = AdamW(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        mention_mask = batch['mention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, mention_mask=mention_mask)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
      total_val_loss = 0
      for batch in val_loader:
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          mention_mask = batch['mention_mask'].to(device)
          labels = batch['labels'].to(device)
          logits = model(input_ids=input_ids, attention_mask=attention_mask, mention_mask=mention_mask)
          loss_fn = CrossEntropyLoss()
          loss = loss_fn(logits, labels)
          total_val_loss += loss.item()

      avg_val_loss = total_val_loss / len(val_loader)
      print(f"validation loss at epoch {epoch}: {avg_val_loss}")
```

here is what is different:

*   it creates a custom `RelationExtractionModel` class that takes bert as a base and adds a bi-lstm layer and a final linear classifier on top.
*   it mask the output of bert with `mention_mask` and uses that as input to the lstm
*   instead of using the loss output of `BertForSequenceClassification` a manual loss function is used for training with the output logits of the model.
*  rest of the code remains the same.

for further reading, i would suggest looking at the original bert paper "bert: pre-training of deep bidirectional transformers for language understanding" which gives you the fundamentals. also, papers on relation extraction are worth looking into to understand the nuances like "joint entity and relation extraction using a hybrid neural network" or "relation extraction with multi-instance learning" that will give you different insights into the common techniques used.

also, textbooks like "speech and language processing" by dan jurafsky and james h. martin (third edition) is very good. while its not specific to bert it does provide a more broader overview of relation extraction and its historical context. sometimes it is good to look back into older methods to improve our current ones!

remember, always experiment with different hyper-parameters and model architectures. the "best" setup varies a lot depending on the data you have.
