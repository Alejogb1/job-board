---
title: "How can multiple metrics be used in PyTorch?"
date: "2025-01-30"
id: "how-can-multiple-metrics-be-used-in-pytorch"
---
The efficacy of a neural network model isn’t defined by a singular performance number. Real-world scenarios often require observing various facets of a model’s behavior concurrently to make informed decisions about training and deployment. In my experience developing anomaly detection systems, for instance, I’ve found that relying solely on accuracy can be misleading; a model might achieve high accuracy by simply labeling everything as the majority class. PyTorch provides flexible mechanisms to track, combine, and interpret multiple metrics, enabling a more nuanced evaluation.

The core idea involves calculating different metrics during the training loop and storing them for analysis. This is generally achieved by leveraging PyTorch's native tensor operations in conjunction with external libraries designed for metric computation. PyTorch itself doesn’t offer dedicated metric classes but works seamlessly with packages like scikit-learn or torchmetrics. I generally prefer torchmetrics due to its PyTorch integration and hardware acceleration capabilities.

The foundational approach involves these steps: first, initializing metric objects before the training loop. Then, within each training or validation step, you compute the metrics based on the model's predictions and ground truth labels. Crucially, you must update the metrics for each batch. Finally, after each epoch or after a period of evaluation, you can retrieve the aggregated metric values. This process requires careful handling of tensors and appropriate accumulation of values across batches.

Let’s delve into concrete examples demonstrating this principle. Consider a classification task where we want to monitor accuracy, precision, and recall simultaneously. Here’s how it can be implemented:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics

# Dummy data for demonstration
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Simple model
model = nn.Linear(10, 2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize metrics
accuracy_metric = torchmetrics.Accuracy(task='binary')
precision_metric = torchmetrics.Precision(task='binary')
recall_metric = torchmetrics.Recall(task='binary')

num_epochs = 5

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update metrics with batch data
        preds = torch.argmax(output, dim=1) # converts raw logits into class labels
        accuracy_metric.update(preds, target)
        precision_metric.update(preds, target)
        recall_metric.update(preds, target)

    # Compute and log the metrics after each epoch
    epoch_accuracy = accuracy_metric.compute()
    epoch_precision = precision_metric.compute()
    epoch_recall = recall_metric.compute()

    print(f'Epoch: {epoch+1}, Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}')

    # Reset the metrics for next epoch
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()

```
This first example showcases a fundamental binary classification scenario. I begin by defining a dummy dataset and model. I then instantiate the `Accuracy`, `Precision`, and `Recall` metrics from the `torchmetrics` library, specifying `'binary'` as the task. Crucially, within the training loop, I call `update()` with the model's predictions and true labels for each batch. After processing all batches of an epoch, `compute()` is invoked to aggregate metrics across the entire epoch. Finally, the metrics are reset before the next epoch. This pattern is consistent regardless of the specific metrics being used. This reset step is essential; if omitted the metrics accumulate across all epochs rendering the results meaningless.

Now let's examine a case where we deal with multi-class classification, modifying the previous code accordingly:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics

# Dummy multi-class data
X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,)) # Labels now 0,1,2
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Multi-class model
model = nn.Linear(10, 3) # 3 output classes

# Loss and optimizer remain
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Metrics for multi-class
accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=3)
precision_metric = torchmetrics.Precision(task='multiclass', num_classes=3, average='macro') # Macro average
recall_metric = torchmetrics.Recall(task='multiclass', num_classes=3, average='macro') # Macro average

num_epochs = 5

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update metrics
        preds = torch.argmax(output, dim=1)
        accuracy_metric.update(preds, target)
        precision_metric.update(preds, target)
        recall_metric.update(preds, target)

    # Compute and log metrics for each epoch
    epoch_accuracy = accuracy_metric.compute()
    epoch_precision = precision_metric.compute()
    epoch_recall = recall_metric.compute()

    print(f'Epoch: {epoch+1}, Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}')

    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
```
This example adapts the prior case to a multi-class setup. The primary changes involve generating multi-class labels (0, 1, 2), adjusting the model’s output to 3 classes, and most importantly, specifying `task='multiclass'` and the number of classes when initializing the metrics. I also specify `average='macro'` for precision and recall. Macro averaging calculates the metric for each class separately and then takes the unweighted average, providing a more balanced perspective when dealing with class imbalance. If I hadn't included this, or used the default of `average='micro'`, a weighted average would be returned by default. Micro averaging will compute the metric globally by counting the total true positives, false negatives and false positives, so this choice should depend on the specific problem and evaluation requirements.

Finally, let's explore a regression scenario, including a custom metric, mean absolute error, in addition to R-squared:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics

# Dummy regression data
X = torch.randn(100, 10)
y = torch.randn(100, 1)  # Regression targets
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Regression model
model = nn.Linear(10, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Regression metrics
r2_metric = torchmetrics.R2Score()

def mean_absolute_error(preds, target): # Custom metric
    return torch.mean(torch.abs(preds - target))

num_epochs = 5

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update R^2 metric
        r2_metric.update(output, target)

        # Compute and log custom metric (MAE)
        mae = mean_absolute_error(output, target)

    # Compute and log epoch R2 metric
    epoch_r2 = r2_metric.compute()
    print(f'Epoch: {epoch+1}, R2: {epoch_r2:.4f}, MAE: {mae:.4f}')

    r2_metric.reset()
```
This last example handles regression. The dataset now consists of continuous target values, and the model is a linear regressor. Here, I use the `R2Score` metric from torchmetrics. Furthermore, I define a simple function to calculate the Mean Absolute Error (MAE) which is not a standard `torchmetrics` metric. I compute MAE on the batch level, as there’s no accumulator for this single value. For more complex custom metrics, creating a dedicated class extending `torchmetrics.Metric` will provide similar flexibility as the built in metrics. As before, I reset the R2 metric each epoch, to prevent cross epoch contamination of the final results.

For further learning and implementation details I would highly suggest referring to the documentation of the torchmetrics library. You will also find useful examples in the official PyTorch documentation. Additionally, exploration of other specialized libraries focused on specific metrics might be advantageous depending on the nature of the problems being addressed. It's beneficial to familiarize yourself with the conceptual differences between micro, macro and weighted averaging, as these impact how aggregated metrics represent the overall model performance.
