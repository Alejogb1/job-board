---
title: "How can LSTM be applied to BERT embeddings?"
date: "2025-01-30"
id: "how-can-lstm-be-applied-to-bert-embeddings"
---
The challenge with directly using BERT embeddings in an LSTM lies in the fundamental nature of BERT's output: fixed-size contextualized representations for each token in an input sequence. Unlike word embeddings designed for sequential processing, BERT's embeddings already encode information about the entire sentence. Consequently, feeding these directly into an LSTM might not leverage the LSTM's strength in modeling long-range dependencies in a nuanced way. Therefore, a successful approach requires careful consideration of how BERT and LSTM can collaborate, and where the strengths of each can be best utilized.

My experience in natural language processing projects, particularly in sequential tasks like text classification and named entity recognition, has shown that a common and effective strategy is to use BERT primarily for feature extraction and the LSTM for sequence modeling after the initial contextualization provided by BERT. We can view BERT as a powerful encoder that preprocesses the input, providing a rich representation. We can then use this representation, as opposed to the original token ids, as the input into a subsequent LSTM layer.

The key here isn't to feed the entire sequence of BERT embeddings directly into an LSTM for *all* subsequent computations. Instead, the LSTM is positioned to refine, aggregate, or extract specific features that are important for the target task. Specifically, it is useful to recognize that BERT's output is not necessarily a single summary of an input. Rather, we can access the hidden layer outputs at each token. These are the vectors that can serve as inputs to an LSTM, not just a representation of the entire sequence.

Here are three examples illustrating different ways to apply LSTMs to BERT embeddings, each with commentary explaining its function and purpose:

**Example 1: Sequence Classification with LSTM on Top of BERT**

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERT_LSTM_Classifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=128, num_classes=2):
        super(BERT_LSTM_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_states)
        # use the output at the [CLS] position
        cls_out = lstm_out[:, 0, :]
        logits = self.fc(cls_out)
        return logits

# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERT_LSTM_Classifier(num_classes=3) # example with 3 classes
text = ["This is a positive sentence.", "This is a negative sentence.", "This is neutral."]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
  outputs = model(inputs['input_ids'], inputs['attention_mask'])
  predictions = torch.argmax(outputs, dim=1)
  print(predictions) # show predictions of class id
```

This code defines a `BERT_LSTM_Classifier` class designed for text classification. The BERT model outputs the last hidden states for all tokens. These per-token embeddings are then passed to the LSTM layer. Note the use of `batch_first=True` in the LSTM initialization. This ensures the input tensor has the batch size in the first dimension, consistent with BERT’s output. We subsequently select the output corresponding to the `[CLS]` token in the sequence after being processed by the LSTM to get our final representation for classification. This `[CLS]` token representation, often described as a summary of the input sequence, is then sent through a fully connected layer to get a probability distribution across all classes.

**Example 2: Named Entity Recognition (NER) with LSTM on Top of BERT**

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERT_LSTM_NER(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=128, num_labels=5):
      super(BERT_LSTM_NER, self).__init__()
      self.bert = BertModel.from_pretrained(bert_model_name)
      self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first=True,
                         bidirectional=True)
      self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_states)
        # now get a prediction for each token outputted by lstm
        logits = self.fc(lstm_out)
        return logits

# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERT_LSTM_NER(num_labels=7) # example with 7 labels
text = ["Apple is planning to build a new store in London."]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    predictions = torch.argmax(outputs, dim=2)
    print(predictions) # output prediction at each token
```

In this example, we adapt the model for Named Entity Recognition (NER). Here, instead of using only the output associated with the `[CLS]` token, we need to classify *every* token. The architecture is nearly identical to the previous example except for the fully connected layer. We process the entire sequence of LSTM outputs through the linear layer to get per-token predictions. This is crucial for NER, where each token has to be assigned a class. The predictions tensor will be of shape `[batch_size, sequence_length, num_labels]`.

**Example 3: Custom Sequence Processing with LSTM on a Specific BERT Layer**

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERT_LSTM_Custom(nn.Module):
  def __init__(self, bert_model_name='bert-base-uncased', hidden_size=128, num_outputs=10):
    super(BERT_LSTM_Custom, self).__init__()
    self.bert = BertModel.from_pretrained(bert_model_name)
    self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                       hidden_size=hidden_size,
                       num_layers=1,
                       batch_first=True,
                       bidirectional=True)
    self.fc = nn.Linear(hidden_size * 2, num_outputs)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    # extract output of layer 7, as an example
    layer_output = outputs.hidden_states[7]
    lstm_out, _ = self.lstm(layer_output)
    # take the output associated with the [CLS] token
    cls_out = lstm_out[:, 0, :]
    logits = self.fc(cls_out)
    return logits

# Example Usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERT_LSTM_Custom(num_outputs = 5) # an example with 5 outputs
text = ["This is a test sentence."]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    print(outputs) # prints out the output of the linear layer.
```

This example demonstrates a customized approach where instead of relying only on the final hidden states, we access and process the hidden states of a *specific layer* within the BERT model. I have found that by accessing the intermediate hidden layers, one might better capture a representation that fits the needs of a downstream task. Note that, `output_hidden_states=True` needs to be added as a parameter when calling the BERT model to get the outputs of all layers. In this example, I have selected the hidden layer at position 7 of the bert layers as the input for the LSTM, although any layer could be used. The remainder of the processing, using an LSTM and subsequently a linear layer, is identical to that in example 1.

When implementing these models, remember several key points. Batch size has a major effect on performance; it will be important to set this parameter based on your available resources. Also, while the provided examples use a single LSTM layer, adding more layers or using a different architecture such as GRU might enhance performance in specific tasks, but would also increase complexity. Be sure to fine-tune these models on a task specific dataset to achieve the best performance.

For further learning, I recommend exploring the following resources. First, the official documentation for the `transformers` library provides detailed explanations of BERT and the various pre-trained models. Next, focus on documentation for `torch.nn.LSTM` to gain a deeper understanding of the LSTM mechanics and its tunable parameters. Finally, explore research papers on sequence modeling with BERT and LSTM, paying close attention to the specific experimental settings and dataset. This will provide a more detailed understanding and allow one to apply the same concepts to more complex use cases.
