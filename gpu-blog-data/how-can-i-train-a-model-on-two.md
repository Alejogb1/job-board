---
title: "How can I train a model on two separate datasets?"
date: "2025-01-30"
id: "how-can-i-train-a-model-on-two"
---
Training a single model on two distinct datasets requires careful consideration of potential biases, data distributions, and the desired outcome of the combined training process. In my experience, addressing this often involves choosing an appropriate training strategy – typically, either a merged dataset approach or a multi-head architecture, depending on whether the datasets share label spaces or require separate outputs. I've encountered this situation several times while building models for heterogeneous data types across diverse sensor inputs.

Firstly, understanding the nature of your datasets is critical. Consider whether the two datasets represent different aspects of the *same* problem space, or whether they capture entirely *different* phenomena, even if superficially related. If the datasets fundamentally differ in their underlying distributions, naively combining them can lead to a model that performs poorly on both or biases towards the larger dataset.

The simplest, and sometimes most effective, method is to merge the two datasets into a single larger dataset if the datasets share the same feature space and label space. This approach is viable when the datasets are sufficiently similar and the joint distribution makes intuitive sense. Crucially, one must always examine distributions for discrepancies and apply balancing techniques if needed. This prevents the model from over-fitting towards one dataset's unique distribution. It's prudent to examine feature scaling as well. It is often necessary to standardize or normalize the features before the merging process to maintain feature relevance and stability during training. Feature scaling should be applied *after* datasets are combined and before dataset splitting. For datasets that capture different phenomena, this process is ill-advised, as the merged space may not be meaningful.

A more nuanced approach, particularly for datasets with distinct label spaces, involves using a multi-head architecture. This allows the model to learn shared features in a common embedding space while simultaneously predicting separate labels for each dataset through dedicated output heads. This method is especially valuable when the datasets are related but require independent analysis. For instance, in my previous work analyzing sensor data, I used a shared feature extraction module to interpret common sensor signals. This was followed by two distinct regression heads, each trained on different types of environmental measurements. This approach maximizes shared knowledge, allows separate label spaces, and reduces the need to retrain completely different models. However, it introduces complexity into the model architecture and loss function design.

Let's illustrate with code. The first example demonstrates merging two datasets, where we assume both datasets contain numerical features and corresponding target values, suitable for regression tasks, and the label spaces are the same:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Dummy Dataset Generation
def generate_dummy_data(n_samples, n_features, noise_scale=0.1):
    X = np.random.rand(n_samples, n_features)
    y = np.dot(X, np.random.rand(n_features, 1)) + np.random.randn(n_samples, 1) * noise_scale
    return X, y

# Generate two dummy datasets
X1, y1 = generate_dummy_data(100, 5, noise_scale=0.1)
X2, y2 = generate_dummy_data(150, 5, noise_scale=0.2)

# Merge datasets
X_merged = np.concatenate((X1, X2), axis=0)
y_merged = np.concatenate((y1, y2), axis=0)

# Data scaling
scaler = StandardScaler()
X_merged_scaled = scaler.fit_transform(X_merged)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_merged_scaled, y_merged, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
score = model.score(X_test, y_test)
print(f"R-squared score: {score}")
```

In this example, the datasets are directly concatenated, scaled to improve convergence and prevent any one dataset's features from dominating the training. The data is split into training and testing sets, and a regression model is then trained. In my experience, checking distributions of merged data is a crucial step after concatenation but not represented in the simplified code here, though the use of `StandardScaler` is important.

The following example demonstrates using a multi-head architecture for classification where we assume that input datasets contain the same number of features but different classification labels:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Dummy Data Generation
def generate_dummy_classification_data(n_samples, n_features, n_classes):
    X = torch.rand(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y

# Generate two dummy classification datasets with distinct labels
X1, y1 = generate_dummy_classification_data(100, 5, 3) # 3 Classes
X2, y2 = generate_dummy_classification_data(150, 5, 4) # 4 Classes

# Convert to datasets and dataloaders
train_dataset1 = TensorDataset(X1, y1)
train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
train_dataset2 = TensorDataset(X2, y2)
train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True)

# Define a Multi-Head Model
class MultiHeadModel(nn.Module):
    def __init__(self, n_features, n_classes1, n_classes2):
        super(MultiHeadModel, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.head1 = nn.Linear(32, n_classes1)
        self.head2 = nn.Linear(32, n_classes2)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        output1 = self.head1(shared_output)
        output2 = self.head2(shared_output)
        return output1, output2

# Model Instantiation
model = MultiHeadModel(n_features=5, n_classes1=3, n_classes2=4)

# Optimizer and Loss Functions
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

# Training Loop (simplified)
epochs = 10
for epoch in range(epochs):
    for (batch_x1, batch_y1), (batch_x2, batch_y2) in zip(train_loader1, train_loader2):
         optimizer.zero_grad()
         output1, output2 = model(torch.cat((batch_x1, batch_x2), dim=0))
         loss1 = criterion1(output1[:batch_x1.size(0)], batch_y1)
         loss2 = criterion2(output2[batch_x1.size(0):], batch_y2)
         total_loss = loss1 + loss2
         total_loss.backward()
         optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {total_loss.item()}")
```

This example uses a custom `MultiHeadModel` class. The model's shared layers extract features, then two independent output heads predict respective classification labels. Notice we train *jointly* by passing the concatenated batch data, but compute loss for each head and backpropagate in tandem. In my experience, careful choice of optimizers and tuning can drastically impact the performance of such a setup.

Finally, let’s consider another scenario where feature representations can be different and where a more advanced approach using transformers is applicable. In this case we use the Huggingface Transformers library to fine-tune a pre-trained model on two different datasets. Assume we are working with text data for sentiment classification, where each dataset is in a different language but we want a single model that performs classification on both languages.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import torch
import numpy as np

# Dummy Dataset Creation (mimicking text classification datasets)
def create_dummy_dataset(num_samples, num_classes):
    texts = ["This is a sample text." for _ in range(num_samples)]
    labels = np.random.randint(0, num_classes, num_samples)
    return Dataset.from_dict({"text":texts, "label":labels})

# Dummy datasets for different languages and classes
dataset1 = create_dummy_dataset(100, 2) #  2 classes for example, language 1
dataset2 = create_dummy_dataset(150, 3) #  3 classes for example, language 2

# Pre-trained model and tokenizer
checkpoint = "bert-base-uncased" # Using a multi-lingual base model would be preferable, e.g., bert-base-multilingual-cased, but this works for illustration purposes
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset1 = dataset1.map(tokenize_function, batched=True)
tokenized_dataset2 = dataset2.map(tokenize_function, batched=True)

# Model loading, explicitly handling the labels for the different data-sets
num_labels1 = len(set(dataset1['label']))
num_labels2 = len(set(dataset2['label']))
model1 = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels1)
model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels2)

# Combine datasets and models for fine-tuning
combined_dataset = Dataset.from_dict({'input_ids': tokenized_dataset1['input_ids'] + tokenized_dataset2['input_ids'], 'attention_mask': tokenized_dataset1['attention_mask'] + tokenized_dataset2['attention_mask'], 'label' : tokenized_dataset1['label'] + tokenized_dataset2['label'], 'language': [0 for _ in range(len(tokenized_dataset1))] + [1 for _ in range(len(tokenized_dataset2))]}) # language ID for each sample


# Trainer setup
def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01
)

# Joint training step
class JointModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super(JointModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, input_ids, attention_mask, language, labels):
        output = []
        for batch_idx, batch_language in enumerate(language):
           if batch_language == 0:
                out_batch = self.model1(input_ids=input_ids[batch_idx].unsqueeze(0), attention_mask=attention_mask[batch_idx].unsqueeze(0), labels=labels[batch_idx].unsqueeze(0))
           else:
                 out_batch = self.model2(input_ids=input_ids[batch_idx].unsqueeze(0), attention_mask=attention_mask[batch_idx].unsqueeze(0), labels=labels[batch_idx].unsqueeze(0))
           output.append(out_batch)

        loss = torch.stack([o.loss for o in output]).mean()
        logits = torch.cat([o.logits for o in output], axis=0)

        return {'loss' : loss, 'logits' : logits}

joint_model = JointModel(model1, model2)
trainer = Trainer(
    model=joint_model,
    args=training_args,
    train_dataset=combined_dataset,
    eval_dataset=combined_dataset, # for simplicity just use train
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

```
In this final example, two datasets with different labels are handled by using two distinct models (same architecture but different classification heads). Here, the data is combined by simply concatenating the data across the two datasets. The `JointModel` class facilitates the forward pass by selecting which model is to be used based on the `language` field in the dataset. This structure makes use of separate heads to correctly label data from both datasets, and enables training of a single, unified model. It is critical in practice to use metrics that allow comparing classification on different datasets (e.g. F1 score when datasets are unbalanced).

For further exploration, I would recommend reviewing literature on domain adaptation and transfer learning, which offer advanced strategies for handling datasets with varying distributions. Books on deep learning, particularly those covering multi-task learning, offer theoretical and practical insights into how to efficiently structure and train models on heterogeneous datasets. Exploring specialized libraries can facilitate specific types of joint training, such as those for handling graph-based or time-series data. Careful planning with regards to model selection, training strategy, and loss function, are key to achieving your desired model performance when using multiple datasets for training.
