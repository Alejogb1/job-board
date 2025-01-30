---
title: "How do torchmetrics represent uncertainty?"
date: "2025-01-30"
id: "how-do-torchmetrics-represent-uncertainty"
---
Torchmetrics, while not explicitly designed for direct uncertainty quantification in the manner of Bayesian neural networks or similar probabilistic models, often indirectly facilitates understanding uncertainty through the metrics it provides, alongside features related to their calculation. My work, particularly within computer vision projects analyzing noisy satellite imagery, has highlighted the nuanced ways in which torchmetrics can reveal, though not fundamentally model, uncertainty. Uncertainty manifests in these applications as both aleatoric uncertainty stemming from the data's inherent variability, and epistemic uncertainty arising from model limitations.

**Clear Explanation**

Torchmetrics, a library dedicated to evaluating machine learning model performance, focuses on providing concrete numerical summaries of predicted values compared to ground truth data. These metrics themselves do not encode uncertainty directly. A metric like accuracy, for example, reports a single number representing the proportion of correctly classified samples. That number does not convey how confident the model is for *individual* predictions or for its overall performance on specific data subsets. Instead, torchmetrics, through the types of metrics offered and their calculation methodologies, indirectly reveal aspects related to uncertainty.

Consider metrics like precision and recall in classification tasks. A low precision score signals many false positives. This suggests the model is uncertain about which instances genuinely belong to a particular class, potentially misclassifying many negatives as positives. High recall, on the other hand, may come at the cost of lower precision, signifying the model’s difficulty in confidently distinguishing different classes, which contributes to uncertainty. Similarly, in regression tasks, the Mean Squared Error (MSE) aggregates errors across all data points. While a high MSE indicates a poor overall fit, it doesn't directly point to uncertainty at specific locations of input space. However, variations in per-sample errors, which can be computed and analyzed outside of torchmetrics, can be highly indicative of uncertainty. Specifically, when we see groups of input instances consistently yielding higher errors, we can assume the model is less confident within that subspace.

Furthermore, torchmetrics often calculates metrics over entire batches or datasets and then reports aggregated values, making it harder to analyze instance-level uncertainty directly. This is not a flaw of the library, but it highlights its purpose: model *evaluation* rather than uncertainty *estimation*. The library's value emerges when we leverage its metrics alongside techniques that provide uncertainty estimates in our models. For example, we might use ensemble methods that produce a distribution of predictions or Bayesian neural networks that directly produce predictive variance. In such cases, metrics become invaluable for quantifying the impact of incorporating uncertainty in decision-making.

**Code Examples and Commentary**

The following examples illustrate how torchmetrics are applied and how subtle variations in metric results might point towards issues that suggest areas of uncertainty in models. I will use a fictitious classification project involving object detection in satellite imagery. The classes are "urban", "forest", and "water".

**Example 1: Classification Accuracy Across Different Subsets**

```python
import torch
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader

class SatelliteImageDataset(Dataset): # Fictitious satellite image dataset class
    def __init__(self, labels, predictions, subsets):
        self.labels = labels
        self.predictions = predictions
        self.subsets = subsets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.predictions[idx], self.labels[idx], self.subsets[idx]


def calculate_subset_accuracy(predictions, labels, subsets):
    metric = Accuracy(task="multiclass", num_classes=3)
    unique_subsets = torch.unique(subsets)
    subset_accuracies = {}
    for subset in unique_subsets:
        subset_indices = subsets == subset
        subset_predictions = predictions[subset_indices]
        subset_labels = labels[subset_indices]
        metric.update(subset_predictions, subset_labels)
        subset_accuracies[subset.item()] = metric.compute()
        metric.reset()
    return subset_accuracies

# Simulate Data for subset analysis
labels = torch.randint(0, 3, (100,))
predictions = torch.randint(0, 3, (100,))
subsets = torch.randint(0, 3, (100,))
dataset = SatelliteImageDataset(labels, predictions, subsets)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

all_predictions, all_labels, all_subsets = [], [], []
for batch_preds, batch_labels, batch_subsets in loader:
   all_predictions.append(batch_preds)
   all_labels.append(batch_labels)
   all_subsets.append(batch_subsets)

all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)
all_subsets = torch.cat(all_subsets)

subset_accuracy = calculate_subset_accuracy(all_predictions, all_labels, all_subsets)
print(subset_accuracy)
```
This example shows how to calculate accuracy for different data subsets. Suppose we observe lower accuracy for subset ‘2’, which consists of images taken under heavy cloud cover. This indicates the model struggles when presented with such data, and this could signal greater uncertainty in those particular scenarios. While the accuracy score itself is a simple number, its variation across subsets suggests variations in model confidence.

**Example 2: Classification Metrics with Probability Scores**

```python
import torch
from torchmetrics import Precision, Recall, F1Score
from torch.nn.functional import softmax
class SatelliteImageDataset(Dataset): # Fictitious satellite image dataset class
    def __init__(self, labels, logits, subsets):
      self.labels = labels
      self.logits = logits
      self.subsets = subsets

    def __len__(self):
       return len(self.labels)

    def __getitem__(self, idx):
       return self.logits[idx], self.labels[idx], self.subsets[idx]

def calculate_subset_metrics_with_logits(logits, labels, subsets):
   precision = Precision(task='multiclass', num_classes=3, average='macro')
   recall = Recall(task='multiclass', num_classes=3, average='macro')
   f1score = F1Score(task='multiclass', num_classes=3, average='macro')
   subset_metrics = {}
   unique_subsets = torch.unique(subsets)

   for subset in unique_subsets:
       subset_indices = subsets == subset
       subset_logits = logits[subset_indices]
       subset_labels = labels[subset_indices]
       probs = softmax(subset_logits, dim=1) # probability scores
       predictions = torch.argmax(probs, dim=1)
       precision.update(predictions, subset_labels)
       recall.update(predictions, subset_labels)
       f1score.update(predictions, subset_labels)
       subset_metrics[subset.item()] = {'precision': precision.compute().item(),
                                   'recall': recall.compute().item(),
                                   'f1score': f1score.compute().item()}
       precision.reset()
       recall.reset()
       f1score.reset()
   return subset_metrics

# Simulate Logits
labels = torch.randint(0, 3, (100,))
logits = torch.randn(100, 3) # Simulated Logits
subsets = torch.randint(0, 3, (100,))

dataset = SatelliteImageDataset(labels, logits, subsets)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

all_logits, all_labels, all_subsets = [], [], []
for batch_logits, batch_labels, batch_subsets in loader:
    all_logits.append(batch_logits)
    all_labels.append(batch_labels)
    all_subsets.append(batch_subsets)
all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)
all_subsets = torch.cat(all_subsets)


subset_metrics = calculate_subset_metrics_with_logits(all_logits, all_labels, all_subsets)
print(subset_metrics)
```

This example builds on the previous one by considering precision, recall, and F1-score. I've computed a probability score for each class using softmax on the model's logits. Suppose, for a given subset, we have a high recall but a low precision for the class “urban”. This indicates the model is frequently detecting urban areas, but is also mislabeling many non-urban features as urban, signaling potential confusion between classes, and, thus, uncertainty. The precision-recall balance provides a more fine-grained look at model behavior than accuracy alone and reveals the types of misclassification errors the model is prone to make.

**Example 3: Regression Metrics and Error Analysis**

```python
import torch
from torchmetrics import MeanSquaredError
import numpy as np

class SatelliteImageDataset(Dataset): # Fictitious satellite image dataset class
    def __init__(self, labels, predictions, subsets):
        self.labels = labels
        self.predictions = predictions
        self.subsets = subsets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.predictions[idx], self.labels[idx], self.subsets[idx]


def calculate_subset_mse(predictions, labels, subsets):
  metric = MeanSquaredError()
  subset_mses = {}
  unique_subsets = torch.unique(subsets)
  for subset in unique_subsets:
    subset_indices = subsets == subset
    subset_predictions = predictions[subset_indices]
    subset_labels = labels[subset_indices]
    metric.update(subset_predictions, subset_labels)
    subset_mses[subset.item()] = metric.compute()
    metric.reset()
  return subset_mses

# Simulate Regression Data
labels = torch.randn(100)
predictions = torch.randn(100)
subsets = torch.randint(0, 3, (100,))
dataset = SatelliteImageDataset(labels, predictions, subsets)
loader = DataLoader(dataset, batch_size=16, shuffle=False)


all_predictions, all_labels, all_subsets = [], [], []
for batch_preds, batch_labels, batch_subsets in loader:
    all_predictions.append(batch_preds)
    all_labels.append(batch_labels)
    all_subsets.append(batch_subsets)
all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)
all_subsets = torch.cat(all_subsets)


subset_mse = calculate_subset_mse(all_predictions, all_labels, all_subsets)
print(subset_mse)
```

Here, I showcase MSE for a fictitious regression task involving pixel-level temperature prediction. Suppose we observe higher MSE for subsets that include coastal regions. This suggests the model struggles with boundary areas, possibly because of complex temperature gradients or atmospheric interference unique to such areas. Again, the metric doesn't *model* uncertainty, but the variations in MSE reveal scenarios where the model has difficulty providing accurate predictions. When combined with per-sample error analysis we can gain deeper insight. For instance we can compute the standard deviation of absolute errors on subset '2' and interpret this as a proxy for model uncertainty on this type of data.

**Resource Recommendations**

For those interested in learning more about the topics involved, I would recommend exploring academic texts on machine learning evaluation metrics and also on Bayesian methods for uncertainty estimation. Furthermore, various online learning platforms offer courses on both the practical application of evaluation metrics in machine learning and advanced methods for quantifying uncertainty, such as variational inference. Reading documentation and tutorials related to PyTorch's ecosystem, particularly related to probabilistic programming (e.g., Pyro, Edward) can also be highly beneficial.
