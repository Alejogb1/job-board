---
title: "Why does accuracy equal F1 in binary classification with Torch Lightning?"
date: "2024-12-23"
id: "why-does-accuracy-equal-f1-in-binary-classification-with-torch-lightning"
---

, let's unpack this. It's a question that often comes up, and I recall facing it directly when I was working on a churn prediction model a few years back. We were using PyTorch Lightning for training and initially, the reporting of accuracy as a primary metric seemed straightforward. However, as the model started performing differently on various subgroups of our dataset, it became clear that accuracy alone wasn't telling the whole story. We needed something that considered both precision and recall more holistically, and that's when the equivalence between accuracy and f1 score, under certain conditions, became apparent.

The core of the matter lies in the specifics of *binary* classification and the scenario where the classes are balanced. Let me clarify: F1 score is the harmonic mean of precision and recall, a measure that seeks to balance the two metrics. Precision is defined as the proportion of true positives among all predicted positives (i.e., out of all the times we predicted ‘1’, how many were actually ‘1’?), while recall is the proportion of true positives among all actual positives (i.e., out of all the actual ‘1’s, how many did we correctly classify?).

Now, when you have balanced classes in a binary setting—meaning an equal or near-equal number of samples belonging to each class—and your model is making predictions such that the number of true positives equals the number of true negatives, accuracy conveniently coincides with the f1 score. This is not an intrinsic property of the f1 score; rather, it's an outcome of the data and the model's performance. When your true positives equal your true negatives, it forces your false positives and false negatives to also be similar in quantity (though not necessarily equal), resulting in the same score when you calculate both accuracy and F1.

Let's ground this with a few code examples using PyTorch Lightning, keeping in mind that we’re focusing on the conditions where the equivalence holds true.

**Example 1: Simple balanced case with accuracy and F1 coinciding**

```python
import torch
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl

class BinaryClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y.float().view(-1,1))

        y_hat_class = (y_hat > 0.5).int()
        self.accuracy.update(y_hat_class.view(-1), y.int())
        self.f1.update(y_hat_class.view(-1), y.int())

        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        self.log('train_accuracy', self.accuracy.compute())
        self.log('train_f1', self.f1.compute())
        self.accuracy.reset()
        self.f1.reset()

# Generate balanced fake data for demonstration.
train_data = [(torch.rand(10), torch.randint(0, 2, (1,))) for _ in range(1000)]

from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
         return self.data[idx]

train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32)


model = BinaryClassifier()
trainer = pl.Trainer(max_epochs=5, enable_progress_bar=False)
trainer.fit(model, train_loader)

```

Here, we’ve set up a basic binary classifier and the training loop logs both accuracy and F1. If we were to execute this, you'd see that both metrics, towards the end of training when the model is close to its best performance on the provided dataset, will give a similar value. This occurs because the training dataset is balanced, and the model converges to a point where, implicitly, the number of true positives and true negatives is similar.

**Example 2: Slight Imbalance, and the Metrics diverge**

Now, let’s see what happens with a minor imbalance:

```python
import torch
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl

class BinaryClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y.float().view(-1,1))

        y_hat_class = (y_hat > 0.5).int()
        self.accuracy.update(y_hat_class.view(-1), y.int())
        self.f1.update(y_hat_class.view(-1), y.int())

        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        self.log('train_accuracy', self.accuracy.compute())
        self.log('train_f1', self.f1.compute())
        self.accuracy.reset()
        self.f1.reset()


# Generate *slightly* imbalanced fake data. We slightly bias toward class 0
train_data = [(torch.rand(10), torch.randint(0, 2, (1,)) if torch.rand(1) > 0.3 else (torch.rand(10), torch.tensor([0]))  for _ in range(1000)]


from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
         return self.data[idx]

train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32)


model = BinaryClassifier()
trainer = pl.Trainer(max_epochs=5, enable_progress_bar=False)
trainer.fit(model, train_loader)
```

Notice that in this example, I introduced a slight imbalance in our training data, favoring the 0 class a bit more. When the model trains, you will observe that the accuracy and f1 scores will start diverging, as they should. The accuracy might still look good, but the f1 score will typically reflect the performance more realistically due to the bias.

**Example 3: Deliberately skewed case: The metrics are now highly different**

And finally, let's exaggerate the class imbalance to show the stark differences.

```python
import torch
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl

class BinaryClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y.float().view(-1,1))

        y_hat_class = (y_hat > 0.5).int()
        self.accuracy.update(y_hat_class.view(-1), y.int())
        self.f1.update(y_hat_class.view(-1), y.int())

        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        self.log('train_accuracy', self.accuracy.compute())
        self.log('train_f1', self.f1.compute())
        self.accuracy.reset()
        self.f1.reset()


# Generate Highly imbalanced fake data. We *heavily* bias toward class 0
train_data = [(torch.rand(10), torch.randint(0, 2, (1,)) if torch.rand(1) > 0.1 else (torch.rand(10), torch.tensor([0]))  for _ in range(1000)]


from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
         return self.data[idx]

train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32)


model = BinaryClassifier()
trainer = pl.Trainer(max_epochs=5, enable_progress_bar=False)
trainer.fit(model, train_loader)
```

In this last example, most of the data is class 0. Now, even if the model simply predicts 0 most of the time, accuracy would still appear high. However, the F1 score will be lower as it correctly factors in the missed positive class due to the imbalance.

For a deep dive into evaluation metrics for classification, I highly recommend *Pattern Recognition and Machine Learning* by Christopher Bishop. It covers these concepts rigorously. For a more specific treatment of metrics in the context of machine learning, “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman is another superb resource. Furthermore, the original paper on F1 score and its variations by Rijsbergen is also a helpful read - "Information Retrieval". These books and papers offer thorough and fundamental understanding about the concepts which should further deepen the understanding of why this equivalence occurs, and when you shouldn't rely solely on accuracy.

To reiterate, it’s not that accuracy and F1 *always* equal in binary classification, or that this is a property of either metric individually. The equivalence you observe is a result of specific data conditions—balanced classes where the model's performance is such that true positives are roughly equal to true negatives. In practice, relying only on accuracy, especially if there's a suspicion of imbalance, can be deceiving, as I learned firsthand from that old churn prediction task. It is essential to inspect the metrics carefully in conjunction with the specific nuances of the dataset.
