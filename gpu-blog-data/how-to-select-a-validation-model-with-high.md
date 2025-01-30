---
title: "How to select a validation model with high MCC and sufficient sensitivity for an imbalanced dataset in PyTorch?"
date: "2025-01-30"
id: "how-to-select-a-validation-model-with-high"
---
The inherent class imbalance often encountered in real-world datasets significantly impacts model validation, particularly when relying solely on accuracy. Optimizing for a single metric like accuracy can be misleading, pushing the model toward the majority class while failing to adequately capture the minority class, which is often of greater interest. Selecting a suitable validation model requires a nuanced approach, involving metrics beyond accuracy, specifically Matthew’s Correlation Coefficient (MCC) and sensitivity (recall), within the PyTorch framework. My experience developing a rare disease diagnostic tool highlighted the crucial necessity of this approach. The dataset was severely imbalanced, with a 98:2 ratio between negative and positive cases. Standard evaluation yielded a high accuracy score, often over 95%, even with a poor underlying model performance.

Achieving both high MCC and sufficient sensitivity necessitates a multi-pronged strategy. MCC, ranging from -1 to +1, provides a robust measure of correlation between the predicted and actual classifications, even in the presence of class imbalances. A score of +1 indicates a perfect prediction, 0 suggests performance no better than random chance, and -1 indicates a total disagreement. Sensitivity, also known as recall, is crucial for identifying all positive cases. In imbalanced datasets, it becomes imperative to boost the sensitivity to ensure the model captures a sufficient proportion of the minority class. Blindly maximizing MCC alone could potentially sacrifice sensitivity, resulting in a model that identifies only a fraction of the positive instances. Therefore, I will demonstrate techniques to achieve a balance between these two essential validation metrics.

The validation process starts with a clear understanding of both metrics and how they influence model selection. I prefer to compute both metrics alongside others during training using callbacks or within the validation loop itself. Consider the following PyTorch code snippet for metric calculations:

```python
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef, recall_score

def calculate_metrics(outputs, labels):
    """
    Calculates MCC and sensitivity for binary classification predictions.
    Assumes outputs are probabilities from sigmoid activation.
    Args:
        outputs (torch.Tensor): Model output probabilities (shape [batch_size, 1]).
        labels (torch.Tensor): True binary labels (shape [batch_size,]).
    Returns:
        tuple: MCC and Sensitivity scores.
    """
    preds = (outputs > 0.5).long().squeeze() #Convert probabilities to binary predictions
    labels = labels.long().squeeze()  # Ensure labels are long type

    mcc = matthews_corrcoef(labels.cpu(), preds.cpu())
    sensitivity = recall_score(labels.cpu(), preds.cpu(), zero_division=0) # zero_division=0 handles the case with no predicted positives
    return mcc, sensitivity


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1) # Simplified model for example
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)


#Example usage within validation loop
def validate(model, dataloader, criterion):
    model.eval()
    total_mcc = 0.0
    total_sensitivity = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.float())
            mcc, sensitivity = calculate_metrics(outputs, labels)

            total_mcc += mcc
            total_sensitivity += sensitivity
            num_batches += 1

    avg_mcc = total_mcc / num_batches
    avg_sensitivity = total_sensitivity / num_batches

    return avg_mcc, avg_sensitivity
```

This code provides a function `calculate_metrics` that computes the MCC and sensitivity using `sklearn` functions, ensuring that they're computed on the CPU, avoiding common errors related to device mismatch. This was a major troubleshooting point during my research. The `validate` function calculates these metrics across all batches in the validation dataset and provides average results. This allows for a consolidated validation score. The `BinaryClassifier` class provides an example of a simple model for demonstration. The use of the sigmoid activation function is critical when treating this as a binary classification problem.

To address the challenge of imbalanced datasets, I found oversampling the minority class during training effective. I have also tested various undersampling techniques but prefer oversampling due to data loss with undersampling. This can be combined with weighted loss functions. The following code demonstrates the implementation of oversampling with PyTorch's `WeightedRandomSampler` class in conjunction with a loss function that incorporates class weights:

```python
import numpy as np
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
import torch.nn.functional as F

def get_weighted_sampler(labels):
    """
    Generates a weighted sampler for imbalanced datasets.
    Args:
        labels (torch.Tensor): True labels.
    Returns:
        WeightedRandomSampler: Sampler for the dataset.
    """
    class_counts = torch.bincount(labels)
    weights = 1.0 / class_counts.float()
    sample_weights = weights[labels].double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)
    return sampler


def get_class_weights(labels):
  """
    Calculates weights to compensate for class imbalances.
    Args:
      labels: True labels
    Returns:
      Weights in the form of a torch.Tensor.
    """
  class_counts = torch.bincount(labels)
  weights = 1 / class_counts.float()
  return weights

#Example Usage
num_samples = 1000
input_features = 10
imbalance_ratio = 0.95
num_positive = int(num_samples * (1-imbalance_ratio))
num_negative = num_samples - num_positive

data_inputs = torch.randn(num_samples, input_features)
data_labels = torch.cat([torch.ones(num_positive),torch.zeros(num_negative)]).long()

sampler = get_weighted_sampler(data_labels)
dataset = TensorDataset(data_inputs, data_labels)
dataloader = DataLoader(dataset, batch_size=32, sampler = sampler)


def weighted_loss_function(outputs, labels, class_weights):
  """Calculates weighted cross entropy loss"""
  return F.binary_cross_entropy(outputs.squeeze(), labels.float(), weight=class_weights[labels].float())


model = BinaryClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
class_weights = get_class_weights(data_labels) #compute weights based on labels

for epoch in range(100):
    model.train()
    for inputs, labels in dataloader:
      optimizer.zero_grad()
      outputs = model(inputs.float())
      loss = weighted_loss_function(outputs, labels, class_weights)
      loss.backward()
      optimizer.step()
    mcc, sensitivity = validate(model, dataloader, weighted_loss_function) #use validation function defined previously
    print(f"Epoch {epoch+1}: MCC {mcc:.3f}, Sensitivity {sensitivity:.3f}")
```
The `get_weighted_sampler` function computes weights inversely proportional to the class frequencies, thereby allowing a more balanced representation of the data during training. Using PyTorch's `DataLoader` with this sampler ensures each batch has an appropriate proportion of positive and negative samples. Additionally, the `weighted_loss_function` calculates the loss with weights based on the class labels. This combination ensures that the model is not predominantly trained on the majority class, further improving the model's sensitivity to the minority class. The model selection should be done on the epoch with the optimal balance between mcc and sensitivity.

Finally, model selection should not rely solely on a single validation split; cross-validation can provide a more realistic measure of model performance across different data subsets. The following code illustrates a basic implementation of k-fold cross-validation with the functions defined above:

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validate(model, data_inputs, data_labels, k=5, num_epochs = 100):
    """
    Performs k-fold cross-validation.
    Args:
        model: Model to be trained.
        data_inputs (torch.Tensor): Input data.
        data_labels (torch.Tensor): Target data.
        k (int): Number of folds.
        num_epochs (int): Number of epochs per training loop
    Returns:
      list of tuple: List of (average_mcc, average_sensitivity) for each fold
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for fold, (train_index, val_index) in enumerate(skf.split(data_inputs, data_labels)):
      print(f"Fold {fold+1}/{k}")
      train_inputs, train_labels = data_inputs[train_index], data_labels[train_index]
      val_inputs, val_labels = data_inputs[val_index], data_labels[val_index]

      train_dataset = TensorDataset(train_inputs, train_labels)
      sampler = get_weighted_sampler(train_labels)
      train_dataloader = DataLoader(train_dataset, batch_size=32, sampler = sampler)

      val_dataset = TensorDataset(val_inputs, val_labels)
      val_dataloader = DataLoader(val_dataset, batch_size=32)

      optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
      class_weights = get_class_weights(train_labels) #compute weights based on labels
      for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
          optimizer.zero_grad()
          outputs = model(inputs.float())
          loss = weighted_loss_function(outputs, labels, class_weights)
          loss.backward()
          optimizer.step()
        mcc, sensitivity = validate(model, val_dataloader, weighted_loss_function) #use validation function defined previously
      results.append((mcc, sensitivity))
    return results


# Example usage
k = 5
cross_val_results = cross_validate(model, data_inputs, data_labels, k)

for fold, (mcc, sensitivity) in enumerate(cross_val_results):
  print(f"Fold {fold+1} MCC: {mcc:.3f}, Sensitivity {sensitivity:.3f}")
```

Here, `StratifiedKFold` ensures that each fold retains the class distribution. The `cross_validate` function performs the training and validation on each fold using the functions previously defined. The result provides a more robust validation score since it's not based on a single split. I would emphasize that selecting the final model should be guided by the results of this cross-validation.

For further study, I recommend exploring the following resources. For deeper understanding of metric evaluation, specifically on imbalanced datasets, research the performance characteristics of Matthew's Correlation Coefficient and its relationship to other metrics. For strategies related to dealing with imbalanced datasets in general, I suggest learning more about sampling methods, especially oversampling techniques like SMOTE, as well as weighted loss functions. Finally, I recommend studying the theory and implementation of cross validation methods like k-fold, and repeated k-fold cross validation.
