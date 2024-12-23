---
title: "Is a validation set required for fine-tuning models with synthetic data?"
date: "2024-12-23"
id: "is-a-validation-set-required-for-fine-tuning-models-with-synthetic-data"
---

Alright, let’s tackle this topic head-on. It’s a question I’ve grappled with myself, especially back in the day when I was working on a somewhat unorthodox optical character recognition (ocr) project. We were heavily reliant on synthetic data to augment our rather limited real-world samples, and the question of how to properly validate model performance, particularly during fine-tuning, was paramount. The short answer, unequivocally, is yes: a validation set *is* essential, even—and perhaps especially—when using synthetic data for fine-tuning. However, the nature of that validation set and its purpose become even more critical in this context. Let’s break down why.

Firstly, the core issue stems from the inherent bias present in synthetic data. While synthetic datasets allow us to address the lack of labeled real-world examples, they are, by definition, artificially constructed. This construction, even with the most advanced generation techniques, often fails to fully capture the real-world variations and nuances present in the target domain. Consequently, a model fine-tuned solely on synthetic data risks overfitting to the specifics of the synthetic distribution, exhibiting poor generalization performance on actual unseen data. We found this out the hard way on that ocr project. We spent weeks generating fantastically diverse (on paper) synthetic fonts and text variations, only to find the model performing quite poorly when confronted with actual, scanned documents.

The role of a validation set in this scenario is, therefore, twofold. Firstly, it functions as a sanity check, helping to evaluate the degree of overfitting to the synthetic training data during the fine-tuning process. The key metric here is the performance gap between the training set (the synthetic data) and the validation set. A significant disparity between performance on the two is a strong indicator of overfitting and the need to adjust training parameters, data augmentation strategies, or even revisit the method by which synthetic data is generated. Secondly, and perhaps even more importantly, the validation set ideally should represent data from the actual domain the model is intended to be used on. This helps gauge how effectively the model trained on the synthetics is actually translating over to the real world. If we are dealing with rare edge cases that the synthetic data didn't model, the validation set is the critical tool for finding this deficiency.

Now, let’s look at some specific cases and code examples to concretize this point. Assume we are fine-tuning a pre-trained transformer model for sentiment analysis, but our available labelled real-world sentiment data is limited. We decide to generate synthetic text data labelled for sentiment.

**Example 1: Basic Fine-Tuning with Validation Set**

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume synthetic_data and real_data are preloaded DataSets objects
# synthetic_data has labelled text with the synthetic data.
# real_data is a smaller dataset of real examples for use as the validation set.

#Split the real data into validation and test sets
real_train, real_test = train_test_split(real_data, test_size=0.5, random_state=42)

# Create a combined data set of synthetic and real for the training set.
combined_training_set = torch.utils.data.ConcatDataset([synthetic_data, real_train])

train_dataloader = DataLoader(combined_training_set, batch_size=16, shuffle=True)
val_dataloader = DataLoader(real_test, batch_size=16, shuffle=False)


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) # Assume binary sentiment
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    val_acc = evaluate(model, val_dataloader)
    print(f"Epoch {epoch+1}, validation accuracy: {val_acc}")

```

In this example, we are intentionally combining the synthetic training data with a small amount of real data, alongside a validation set of real data. The `evaluate` function is key here because it uses data separate from the training data, giving a more accurate picture of model performance.

**Example 2: Focus on Synthetic Data Domain Bias:**

Sometimes, the synthetic data might have a specific bias that is unknown to us. Let's imagine a specific sentiment in the synthetic data is more common than it would be in the real world. This introduces a bias in our model. To try and account for this, we can monitor the accuracy of various categories in our validation dataset. This requires us to have granular labels in our validation data.

```python
from sklearn.metrics import classification_report

def detailed_evaluate(model, dataloader, label_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=label_names))


label_names=['negative', 'positive'] #Assuming the sentiment labels
print("detailed results on validation set:")
detailed_evaluate(model, val_dataloader, label_names)
```

Here, we utilize the `classification_report` from `sklearn` to provide a more detailed view of our model's performance in the validation set. This would highlight if there is an imbalance in the model and where it is struggling. In practice, I've seen situations where some minority classes within the real data are simply never accurately learned when trained with synthetic data that over-represents the dominant categories.

**Example 3: An Adaptive Training Approach:**

Lastly, consider a scenario where the synthetic data and real data come from distinctly different distributions. In this case, a more adaptive approach may be warranted, where validation performance informs us of when to stop training on the synthetic data. This involves iteratively assessing the performance and potentially only using the synthetic data to bootstrap.

```python

def adaptive_training(model, combined_dataloader, val_dataloader, optimizer, device):
  epochs = 3 # Initial epochs with synthetic data
  best_val_acc = 0
  patience_counter = 0
  max_patience = 3

  for epoch in range(epochs):
      model.train()
      for batch in combined_dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
      val_acc = evaluate(model, val_dataloader)
      print(f"Epoch {epoch+1}, Validation accuracy: {val_acc}")

      if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0 # Reset patience if improvement
      else:
        patience_counter += 1 # Patience if no improvement
        if patience_counter > max_patience:
            print("Early stopping with synthetic data.")
            break # Stop training if the validation accuracy stops improving

  # Now continue fine-tuning with the real training data only
  print("Fine-tuning only on real data:")

  real_train_dataloader = DataLoader(real_train, batch_size=16, shuffle=True)
  for epoch in range(epochs):
      model.train()
      for batch in real_train_dataloader:
          optimizer.zero_grad()
          inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
          labels = batch['labels'].to(device)
          outputs = model(**inputs, labels=labels)
          loss = outputs.loss
          loss.backward()
          optimizer.step()
      val_acc = evaluate(model, val_dataloader)
      print(f"Epoch {epoch+1}, Validation accuracy: {val_acc}")

```

In this approach, the training initially uses the combined synthetic and real data sets, but it will halt training on the synthetic data when the validation accuracy stops improving. Afterwards it will continue to fine-tune on the real training data only. The `evaluate` function continues to help us evaluate the real-world performance and to make training decisions.

In conclusion, a validation set is not optional, especially when using synthetic data for fine-tuning models. It is vital for catching model over-fitting, domain shift, and for assessing the real-world utility of a model trained on synthetics. The specific strategy we use might vary on the nature of the data, but the core principle remains constant.

For further reading, I recommend delving into the works by Ian Goodfellow, Yoshua Bengio and Aaron Courville, “Deep Learning”, MIT Press, which contains detailed discussions about model validation and generalization. Additionally, “Foundations of Data Science” by Avrim Blum, John Hopcroft and Ravindran Kannan offers a fantastic theoretical grounding on the topic of generalisation error which is helpful to understand what makes a validation set work. Finally, research papers focused on domain adaptation and transfer learning, particularly those that consider scenarios with limited real-world data, will also prove to be invaluable resources. These papers are easily accessible through academic search engines like Google Scholar by searching those topics. The key takeaway is that, in the realm of machine learning, rigorously assessing and validating model performance is absolutely crucial for deployment success.
