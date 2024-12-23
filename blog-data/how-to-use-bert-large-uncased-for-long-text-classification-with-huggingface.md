---
title: "How to use bert-large-uncased for long text classification with HuggingFace?"
date: "2024-12-16"
id: "how-to-use-bert-large-uncased-for-long-text-classification-with-huggingface"
---

Alright,  I remember back in '19, working on a project involving legal document analysis – that’s when I first really grappled with the practicalities of applying `bert-large-uncased` to long text classification. It's a classic problem: the input sequence length limitation inherent in BERT clashes directly with real-world, lengthy documents. The straightforward approach of simply feeding in the entire text just doesn’t cut it; you’ll run into tokenization length limits and, consequently, errors. We need strategies, and fortunately, there are several viable ones.

First and foremost, let’s acknowledge the core constraint: BERT’s input token sequence limit, usually around 512 tokens for `bert-large-uncased`. Going beyond that directly leads to truncated inputs. We can't have that. This forces us into preprocessing the text. The common approaches revolve around chunking, sliding windows, or hierarchical methods. My experience leans heavily on chunking, especially when efficiency is a concern.

Chunking involves dividing the long text into smaller, manageable segments, each falling within BERT’s sequence length constraint. Each chunk is then passed through the model independently. This is conceptually simpler but requires a strategy for aggregating the results of these individual chunks to get a final prediction for the entire document. Common methods include averaging the logits, using max-pooling, or a more sophisticated approach like training a separate aggregation layer, which I'll describe shortly. I generally tend to favor averaging when it’s a fairly balanced dataset, but the task specifics heavily dictate the strategy.

Let's dive into some code. Assume we’re working with Hugging Face’s `transformers` library. Here's a basic implementation of chunking:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2) # binary classification


def chunk_text(text, max_length=512):
  tokens = tokenizer.tokenize(text)
  chunks = []
  for i in range(0, len(tokens), max_length - 2): # -2 for [CLS] and [SEP]
      chunk = ['[CLS]'] + tokens[i:i+max_length - 2] + ['[SEP]']
      chunks.append(chunk)
  return chunks

def predict_on_chunks(chunks):
  logits_list = []
  for chunk in chunks:
    input_ids = tokenizer.convert_tokens_to_ids(chunk)
    input_ids = torch.tensor([input_ids]) # batch size of 1
    with torch.no_grad():
      outputs = model(input_ids)
    logits = outputs.logits
    logits_list.append(logits.numpy())
  return np.array(logits_list)


def aggregate_predictions(logits_list):
  # simple averaging
  return np.mean(logits_list, axis=0)


text = "your long text here... " * 200 # long text to test
chunks = chunk_text(text)
logits_from_chunks = predict_on_chunks(chunks)
final_prediction = aggregate_predictions(logits_from_chunks)
print(f"Aggregated logits: {final_prediction}")

```

This first snippet illustrates the basic chunking and prediction process. The `chunk_text` function tokenizes and splits the text into suitable chunks, paying attention to the `[CLS]` and `[SEP]` tokens for BERT's structure.  `predict_on_chunks` runs each chunk through the model, and `aggregate_predictions` simply averages the resultant logits. This is a baseline and works reasonably well for some tasks.

Now, while simple averaging can be useful, it’s a fairly naive approach. Consider a scenario where some segments of the text are more crucial for the classification than others. In such situations, an averaging aggregation will not be optimal. This is where a trained aggregation layer comes into play. Here's a refined version:

```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ChunkDataset(Dataset):
    def __init__(self, chunks, labels, tokenizer):
        self.chunks = chunks
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
       chunk = self.chunks[idx]
       label = self.labels[idx]
       input_ids = self.tokenizer.convert_tokens_to_ids(chunk)
       return torch.tensor(input_ids), torch.tensor(label)

class AggregationLayer(nn.Module):
    def __init__(self, num_labels, bert_dim = 1024):
       super(AggregationLayer, self).__init__()
       self.linear = nn.Linear(bert_dim, num_labels)
       self.softmax = nn.Softmax(dim=1)

    def forward(self, pooled_outputs):
       return self.softmax(self.linear(pooled_outputs)) # softmax


def train_aggregation_layer(chunk_dataset, aggregation_layer, bert_model, num_epochs=10, learning_rate=0.001):
     dataloader = DataLoader(chunk_dataset, batch_size=4, shuffle = True)
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.Adam(aggregation_layer.parameters(), lr=learning_rate)

     for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()

            with torch.no_grad():
              bert_outputs = bert_model(inputs)
              pooled_outputs = bert_outputs.pooler_output
              #print(f"Pooled out: {pooled_outputs.shape}")
            aggregation_layer_output = aggregation_layer(pooled_outputs)
            #print(f"Agg out: {aggregation_layer_output.shape}")
            loss = criterion(aggregation_layer_output, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

def predict_with_agg_layer(chunks, aggregation_layer, bert_model):
  logits_list = []
  for chunk in chunks:
      input_ids = tokenizer.convert_tokens_to_ids(chunk)
      input_ids = torch.tensor([input_ids])
      with torch.no_grad():
        bert_outputs = bert_model(input_ids)
        pooled_output = bert_outputs.pooler_output
      aggregated_logits = aggregation_layer(pooled_output)
      logits_list.append(aggregated_logits.numpy())
  return np.mean(np.array(logits_list), axis=0)

# Example usage
long_text = "this is long text"*100
chunks_train = chunk_text(long_text)
labels_train = np.random.randint(0,2,len(chunks_train))
train_dataset = ChunkDataset(chunks_train, labels_train, tokenizer)

aggregation_layer = AggregationLayer(num_labels = 2) # binary again
train_aggregation_layer(train_dataset, aggregation_layer, model)
test_text = "this is a long test text"*100
test_chunks = chunk_text(test_text)
final_prediction_agg = predict_with_agg_layer(test_chunks, aggregation_layer, model)
print(f"Aggregated predictions with trainable layer: {final_prediction_agg}")

```

This snippet introduces a learnable `AggregationLayer` that takes the pooled output from each chunk and learns an optimal representation for final classification.  We utilize a `ChunkDataset` to feed the training data, and during training, we freeze the BERT weights and only train the `AggregationLayer`. `predict_with_agg_layer` performs inference using this learned layer. This approach, while more involved, often provides noticeable improvements over basic averaging, especially when certain text segments are more informative.

Finally, let’s briefly mention another common approach: sliding windows. This technique, while similar to chunking, instead of having non-overlapping chunks, it slides a window over the text with a defined stride. This captures more context, potentially mitigating information loss at segment boundaries. However, it can increase computational costs due to overlapping data points. Here’s a code snippet:

```python
def sliding_window_chunks(text, window_size = 512, stride = 256):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens) - window_size, stride):
        chunk = ['[CLS]'] + tokens[i:i + window_size - 2] + ['[SEP]']
        chunks.append(chunk)
    if len(tokens) % stride > 0: # Add last bit if not perfectly divisible
      last_chunk = ['[CLS]'] + tokens[len(tokens) - window_size + stride: ] + ['[SEP]']
      chunks.append(last_chunk)
    return chunks

# Example of usage:
sliding_chunks = sliding_window_chunks(long_text)
logits_sliding = predict_on_chunks(sliding_chunks)
final_prediction_sliding = aggregate_predictions(logits_sliding)
print(f"Aggregated logits (sliding): {final_prediction_sliding}")

```

This final code block demonstrates the `sliding_window_chunks` function, creating overlapping sequences, a technique often used for situations that require more contextual awareness.

When deciding which method to use, you must consider the context of your task. For instance, if your long documents are composed of somewhat independent paragraphs, chunking with averaging or a learned aggregation may perform well. When you need to capture continuity, sliding window can be preferable. Hierarchical methods like hierarchical attention networks (HAN) are alternatives, but add further complexity that may not always justify themselves for specific tasks. For further reading, I’d highly recommend consulting papers on transformer-based document classification, especially those focusing on handling long sequences. Also, Chapter 11 (Sequence-to-Sequence Models) in *Deep Learning with Python* by François Chollet and chapters on transformers in the *Speech and Language Processing* book by Daniel Jurafsky and James H. Martin are valuable. The original BERT paper provides foundational understanding of the underlying architecture.

In summary, dealing with long texts with BERT requires a strategy. Chunking, sliding windows, and learned aggregation layers are valuable techniques. Each has strengths and weaknesses, so evaluate them based on your task’s specific needs. This is, as many problems are, not a singular “one-size-fits-all” approach. You must adapt. Good luck!
