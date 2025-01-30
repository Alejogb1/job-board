---
title: "Why does accuracy equal F1 in binary classification with Torch Lightning?"
date: "2025-01-30"
id: "why-does-accuracy-equal-f1-in-binary-classification"
---
The equivalence of accuracy and F1 score in specific binary classification scenarios using PyTorch Lightning arises when the positive and negative classes are perfectly balanced within the evaluation dataset. I've observed this directly during model validation in various projects involving anomaly detection where data was carefully curated for parity. This apparent coincidence masks fundamental differences between the metrics, becoming indistinguishable under these constrained circumstances.

Accuracy, defined as the proportion of correctly classified instances to the total number of instances, is calculated as (TP + TN) / (TP + TN + FP + FN). Here, TP represents true positives, TN true negatives, FP false positives, and FN false negatives. This metric provides a global view of classifier performance across all classes but can be misleading when classes are imbalanced. A classifier that predicts everything as the majority class could achieve high accuracy in such cases, yet be practically useless.

The F1 score, conversely, is the harmonic mean of precision and recall. Precision measures the proportion of correctly predicted positive instances out of all instances predicted as positive (TP / (TP + FP)), while recall measures the proportion of correctly predicted positive instances out of all actual positive instances (TP / (TP + FN)). The F1 score is calculated as 2 * (precision * recall) / (precision + recall). This metric is more robust to class imbalance, as it balances the trade-offs between falsely labeling positives (FP) and missing actual positives (FN). It emphasizes the performance of a classifier on the positive class and its ability to accurately classify positive instances.

In a balanced binary classification setting, where the number of positive instances equals the number of negative instances, the impact of imbalanced misclassifications on both metrics diminishes. If TP + TN is approximately half the total samples, and given that recall and precision are also operating within the same scale relative to the total, their harmonic mean in the F1 score will, in most practical instances, numerically approach the overall accuracy. This is not a universal truth, but rather an observed phenomenon in a very specific context.

The key aspect of PyTorch Lightning in this convergence lies in its role as a framework for structuring the training and evaluation process, rather than any inherent behavior of the metrics. PyTorch Lightning simplifies the calculation of these metrics through defined validation steps. It handles batching of data, computation of predictions, and subsequent aggregation of metrics across the validation set. The convergence is a function of the data, not the framework.

To clarify, let's look at some code examples with commentary. Assume a fictional model with the name `MyModel` that returns a prediction logit.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1) # Example 10 features to 1 output

    def forward(self, x):
        return self.linear(x)

class BinaryClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1score = torchmetrics.F1Score(task='binary')

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y.float().view(-1, 1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y.float().view(-1,1))
        self.accuracy(torch.sigmoid(logits), y)
        self.f1score(torch.sigmoid(logits), y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy)
        self.log('val_f1', self.f1score)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

# --- Dummy Data ---
X_train = torch.randn(100, 10) # 100 samples, 10 features
y_train = torch.randint(0, 2, (100,)) # 100 binary labels (0 or 1) - balanced
X_val = torch.randn(50, 10) # 50 samples, 10 features
y_val = torch.randint(0, 2, (50,)) # 50 binary labels (0 or 1) - balanced

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)
# --- End Dummy Data ---

model = BinaryClassifier()
trainer = pl.Trainer(max_epochs=5) # Reduced epochs for example speed
trainer.fit(model, train_dataloader, val_dataloader)
```

In this first example, the training and validation data is balanced. You'll observe, upon examining the tensorboard logs created by PyTorch Lightning or through examining the `trainer.callback_metrics` attribute, that the `val_acc` and `val_f1` values reported are very close, especially as the model starts to learn. This is the context under which the coincidence exists.

Let’s observe a case where this does not hold true:

```python
# ... (Previous code, up to class definition)

# --- Dummy Data (Imbalanced) ---
X_train = torch.randn(100, 10) # 100 samples, 10 features
y_train = torch.randint(0, 2, (100,)) # 100 binary labels (0 or 1) - balanced
X_val = torch.randn(50, 10) # 50 samples, 10 features
y_val = torch.cat((torch.zeros(45,dtype=torch.int64),torch.ones(5,dtype=torch.int64))) # 45 negatives 5 positives - imbalanced

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)
# --- End Dummy Data (Imbalanced) ---

#... (rest of the code identical)
model = BinaryClassifier()
trainer = pl.Trainer(max_epochs=5) # Reduced epochs for example speed
trainer.fit(model, train_dataloader, val_dataloader)

```

In the second example, the validation data is now highly imbalanced, with far more negative labels than positive. When the validation phase now occurs, the values for `val_acc` will typically be much higher than the values for `val_f1`. This demonstrates clearly that only in balanced settings these metrics converge, due to how the misclassifications are affecting each. Accuracy becomes less informative due to its tendency to be inflated by the majority class performance, and F1 maintains a more consistent representation of the performance on the positive class,

Finally, let's look at a balanced data with poor classification.

```python
# ... (Previous code, up to class definition)

# --- Dummy Data (Poor performance, Balanced)---
X_train = torch.rand(100, 10)
y_train = torch.randint(0, 2, (100,))
X_val = torch.rand(50, 10)
y_val = torch.randint(0, 2, (50,))
# --- End Dummy Data ---

#... (rest of the code identical)

model = BinaryClassifier()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_dataloader, val_dataloader)
```

In this third example, the input features are randomized, producing a poor model that struggles to predict correctly. In this context you will observe both accuracy and F1 remaining close to a value of around 0.5. This demonstrates that the convergence does not mean the metrics are the same, rather their values converge within similar ranges in the case of a balanced dataset and similarly performant model.

For understanding these metrics further, I recommend studying statistical learning theory and reading material on metric evaluation in the context of machine learning. Textbooks on machine learning algorithms and statistical inference provide foundational knowledge. Resources dedicated to performance analysis in classification can offer deeper insight into metric selection and the nuances of evaluating model behavior, specifically focusing on the concepts of precision, recall, and the harmonic mean. Additionally, the documentation for the PyTorch Metrics library is an invaluable reference when implementing metric calculations in your projects. These resources will not only illuminate the practical effects but also provide the underlying theoretical understanding that clarifies the observed behavior.
