---
title: "How can transformers (e.g., BERT, RoBERTa) be fine-tuned in PyTorch?"
date: "2025-01-30"
id: "how-can-transformers-eg-bert-roberta-be-fine-tuned"
---
Fine-tuning transformer models in PyTorch, while conceptually straightforward, involves several critical steps to achieve optimal performance on a target task. The process fundamentally relies on leveraging pre-trained knowledge encoded within these models and adapting it to a specific downstream application. I've navigated this process extensively during several NLP projects, ranging from sentiment analysis to text classification, and have consistently found that meticulous attention to detail regarding data preparation, hyperparameter selection, and model evaluation is crucial.

First, let's define the core idea. Pre-trained transformer models, like BERT, RoBERTa, and their variants, are trained on massive text corpora to learn general-purpose language representations. This pre-training phase yields models that can understand contextual nuances and relationships between words. Fine-tuning involves taking this pre-trained model and training it further on a dataset specific to the target task. This transfer learning approach significantly reduces the amount of labeled data required and the training time compared to training a model from scratch. The key is to modify the model's final classification layers for the target output, while keeping a majority of the pre-trained transformer layers frozen, and carefully unfreezing them over the training process.

The process generally follows these steps: Data preparation, loading the pre-trained model and tokenizer, modifying the model head, setting up the training loop and loss function, fine-tuning the model, and evaluation of the fine-tuned model.

**Data Preparation**

The initial step is to prepare the dataset for input to the model. This involves tokenizing the text data using the same tokenizer that was used during the pre-training phase, which typically involves converting text into a sequence of tokens understandable by the transformer model. Crucially, special tokens such as `[CLS]` and `[SEP]` are added to indicate the start and separation of sequences, respectively. Each token is then converted into a numerical identifier based on the tokenizer’s vocabulary. The processed data is further batched for efficient processing during training and then converted to PyTorch tensors. This includes creating input ids, attention masks, and labels. The input ids are the token IDs, the attention masks denote which tokens should be attended to (typically 1) and which are padding (typically 0) and the labels are the true labels of the dataset for supervision.

**Loading the Model and Tokenizer**

PyTorch facilitates easy access to pre-trained transformer models through the `transformers` library. Using a method like `AutoModelForSequenceClassification.from_pretrained()`, I can load the pre-trained weights and tokenizer of models such as BERT. It is paramount that the model loaded using this method includes a head capable of classifying sequences. For example, `AutoModelForSequenceClassification` will include an appropriate classifier head.

**Modifying the Model Head**

The final layer of the pre-trained model needs to be adjusted to match the requirements of the target task. For example, for a binary sentiment classification task, the last linear layer would be modified to have an output dimension of 2 (for positive and negative sentiments). Similarly, for multi-class classification the output dimension should correspond to the number of classes. Importantly, I avoid modifications to the encoder layers unless fine-tuning them.

**Training Loop and Loss Function**

Training involves iterating over the data batches, passing the input through the model to get predicted labels, calculating the loss between predicted labels and true labels using a suitable loss function (e.g., cross-entropy loss for classification), and then backpropagating the loss using an optimizer (e.g., AdamW). Gradient clipping is implemented to prevent gradient explosions during backpropagation. This whole process is wrapped into a training loop.

**Fine-tuning the Model**

Fine-tuning the model requires a carefully constructed training loop. One important aspect of this loop is the learning rate and learning rate scheduler. Smaller learning rates are typically used, compared to training from scratch, and I often find that using a learning rate scheduler, such as the linear warm-up scheduler, improves results. The most common practice is to initially freeze the encoder layers and only tune the modified classifier head. After a few epochs, all layers are unfrozen for the full fine-tuning process.

**Evaluation**

After fine-tuning, I evaluate the model performance on a validation or test dataset using appropriate metrics, such as accuracy, precision, recall, F1-score, or AUC depending on the task at hand.

Let’s illustrate these concepts with code examples.

**Code Example 1: Loading a Pre-trained Model**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) #binary classification

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Example input
text = "This is a test sentence."
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Perform inference and print output shape
with torch.no_grad():
  outputs = model(**inputs)
print(outputs.logits.shape)
```

This example demonstrates loading the pre-trained BERT model for binary classification. `AutoTokenizer` handles the tokenization of the text, while `AutoModelForSequenceClassification` provides the pre-trained model, also creating the required classification head. By setting the `num_labels` parameter, the output dimension of the final classification layer is configured for a binary task. After moving both the input and the model onto the appropriate device, an input is passed to the model. Finally, the `logits` output of the model, representing the unnormalized probabilities for each class, are output. The shape of the output is `[1,2]`, representing one input sample and two output classes.

**Code Example 2: Fine-tuning on a Simple Dataset**

```python
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Sample data
texts = ["I loved this movie", "This is terrible", "The food was great", "Avoid it at all costs"]
labels = [1, 0, 1, 0] # 1: positive, 0: negative

# Tokenize the data
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor(labels)

# Create Dataset and DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
model.train()
for epoch in range(num_epochs):
  for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
    optimizer.zero_grad()
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
```
Here, a very simple and small synthetic dataset is created and loaded into a `DataLoader`. This data is then used to fine-tune the model from the first example. An `AdamW` optimizer is used to optimize the loss. Additionally, a learning rate scheduler is utilized to control the learning rate over time. The data and model are moved to GPU, the loss is backpropagated and optimized, and the scheduler is updated.

**Code Example 3: Evaluation**
```python
from sklearn.metrics import accuracy_score, f1_score

model.eval() #Set to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad(): #Disable gradient calculations for evaluation
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 score: {f1:.4f}")
```

This code illustrates the evaluation stage after training, switching the model to eval mode and disables gradient calculation for efficiency. The labels and predictions are collected and moved back to the CPU, which are then fed into metric functions to calculate overall performance.

**Resource Recommendations**

For individuals seeking to deepen their understanding of transformer fine-tuning, several resources are available. Documentation for PyTorch and the `transformers` library, readily accessible through their respective official sites, should be considered essential. Additionally, numerous online courses covering deep learning and NLP provide practical insights into the techniques. Many textbooks on machine learning and natural language processing also offer detailed explanations of transformer architectures and fine-tuning processes. Finally, accessing and reviewing research papers published by the major NLP conferences and journals can give a deeper understanding of the advances and best practices in the field. These resources, used in conjunction with experimentation and practical implementation, are what I have used to gain a comprehensive understanding of fine-tuning transformer models in PyTorch.
