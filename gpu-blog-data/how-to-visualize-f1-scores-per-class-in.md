---
title: "How to visualize F1 scores per class in TensorBoard during training?"
date: "2025-01-30"
id: "how-to-visualize-f1-scores-per-class-in"
---
The efficacy of a multi-class classification model is critically assessed not just by overall accuracy, but also by the per-class performance, often measured using F1 scores. Monitoring these during training within TensorBoard, especially as class imbalances or specific feature learning difficulties manifest, is essential for informed adjustments. I’ve regularly utilized this method when building computer vision systems where subtle differences between object classes are common, and this has proven invaluable.

To visualize F1 scores per class in TensorBoard, the core methodology involves calculating these scores within the training loop and then logging them using TensorBoard's `SummaryWriter` API. The challenge here is not the TensorBoard integration itself, but rather the correct calculation and formatting of the per-class F1 scores from the model's predictions and the ground truth labels.

Let’s break down the necessary steps. First, I define a function that takes the true labels and predicted labels and calculates the F1 score for each class. In a multi-class classification setting, we need to consider macro or weighted averaging, but for TensorBoard logging, per-class scores are directly more informative, allowing us to track which classes are being handled effectively by the model and which are lagging.

Here is the Python function to achieve this using libraries commonly found in deep learning projects:

```python
import numpy as np
from sklearn.metrics import f1_score

def calculate_f1_per_class(true_labels, predicted_labels, num_classes):
    """
    Calculates the F1 score for each class.

    Args:
        true_labels (np.ndarray): Ground truth labels as one-hot encoded or integer values.
        predicted_labels (np.ndarray): Predicted labels as probabilities or integer values.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: Array of F1 scores per class.
    """

    # Handle both predicted probabilities and class integer labels
    if predicted_labels.ndim > 1:  # Probabilities
        predicted_labels = np.argmax(predicted_labels, axis=1)

    if true_labels.ndim > 1: # One-hot encoded
        true_labels = np.argmax(true_labels, axis=1)

    f1_scores = []
    for class_id in range(num_classes):
       # Create binary labels for the specific class
        true_class = (true_labels == class_id).astype(int)
        predicted_class = (predicted_labels == class_id).astype(int)
        
        # Calculate F1 score, setting to 0 if no predictions of that class exist.
        try:
            f1_class = f1_score(true_class, predicted_class, zero_division=0)
        except ValueError:
             f1_class = 0
        f1_scores.append(f1_class)
    return np.array(f1_scores)
```

This function, `calculate_f1_per_class`, is critical. It accepts ground truth labels and predicted labels—both of which could be in the form of one-hot encoded vectors or integer labels—along with the number of classes. It then iterates through each class, creating binary labels where '1' indicates the presence of that specific class and ‘0’ otherwise. F1 scores are then calculated using `sklearn.metrics.f1_score` for each class. I’ve added a zero_division=0 parameter to avoid errors if the model doesn’t predict any instances of a specific class. Further, I handle the case where one-hot encoding is used, converting to class indices using argmax prior to calculation.

Next, during the training loop, I’ll be using the function with data batch, after the prediction stage, and prior to backpropagation. I typically set up a step counter for TensorBoard logging so that we get a time series of F1 score changes, instead of just one final number.

Here’s how I’d incorporate it into a PyTorch-like training loop:

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# Placeholder for your model and data loader
# Assuming a model named 'model' and dataloader named 'train_loader'
class DummyModel(nn.Module):
    def __init__(self, num_classes):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, num_classes) #Dummy input
    def forward(self, x):
        return self.fc(x)

num_classes = 3
model = DummyModel(num_classes)
optimizer = optim.Adam(model.parameters()) # Using Adam optimizer as an example
criterion = nn.CrossEntropyLoss()
train_loader = [(torch.randn(64,10), torch.randint(0,3,(64,))) for _ in range(20)] # Dummy training data

writer = SummaryWriter('runs/f1_per_class_example')
step = 0

for epoch in range(2):  # Dummy epochs for example
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted_labels = outputs.detach().cpu().numpy()
        true_labels = labels.cpu().numpy()
        f1_per_class = calculate_f1_per_class(true_labels, predicted_labels, num_classes)

        for class_id, f1_score in enumerate(f1_per_class):
            writer.add_scalar(f'F1/Class_{class_id}', f1_score, step)
        step += 1
writer.close()

```

In this example, we see the `calculate_f1_per_class` function is used after the forward pass and before the next iteration. `SummaryWriter` logs each class’s F1 score as a separate scalar using a formatted tag: “F1/Class_{class_id}.” This creates a time series graph of the F1 score for each class during training. Notice the `detach().cpu().numpy()` calls, these are important for detaching from the computation graph, converting to CPU memory and formatting as Numpy arrays for use in the F1 calculation.

The key insight here is that we iterate through the `f1_per_class` results, logging each with its own unique tag. This way TensorBoard displays separate curves for each class, not just an aggregated metric.

Finally, if the datasets are not available at the end of each batch, and if the processing of the F1 scores is done outside the training loop, then I need to collect all the model's predictions and corresponding true labels within the validation loop and calculate a final value for F1 per class:

```python
from collections import defaultdict
import torch
# Previous DummyModel definition and validation dataloader
class DummyModel(nn.Module):
    def __init__(self, num_classes):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, num_classes) #Dummy input
    def forward(self, x):
        return self.fc(x)

num_classes = 3
model = DummyModel(num_classes)

val_loader = [(torch.randn(64,10), torch.randint(0,3,(64,))) for _ in range(10)] # Dummy validation data
writer = SummaryWriter('runs/f1_per_class_validation_example')
step = 0

# Validation loop
all_predictions = defaultdict(list)
all_labels = defaultdict(list)
with torch.no_grad():
    model.eval() # Set the model to evaluation mode

    for inputs, labels in val_loader:
      outputs = model(inputs)
      predicted_labels = outputs.cpu().numpy()
      true_labels = labels.cpu().numpy()

      all_predictions["val"].append(predicted_labels)
      all_labels["val"].append(true_labels)

    # Consolidate data from validation loop
    all_predictions["val"] = np.concatenate(all_predictions["val"],axis=0)
    all_labels["val"] = np.concatenate(all_labels["val"], axis=0)

    # Calculate F1 per class for the validation set
    f1_per_class = calculate_f1_per_class(all_labels["val"], all_predictions["val"], num_classes)

    for class_id, f1_score in enumerate(f1_per_class):
       writer.add_scalar(f'Validation_F1/Class_{class_id}', f1_score, step)

writer.close()
```

Here, during the validation loop, I collect all the batches of predictions and true labels and then process all of them together. After all the batches of data are processed by the model, I then aggregate the true labels and predicted labels to be used to compute the final set of F1 scores. The calculated F1 scores per class are then logged to TensorBoard using scalar summaries. This approach ensures that the F1 score represents the overall validation set performance, rather than individual batch results.

For further reading and understanding of the topic, I recommend studying the official documentation of the scikit-learn library, particularly around metrics, and the TensorBoard documentation, particularly regarding scalar summaries and the use of `SummaryWriter` API. Understanding the underlying principles of how these metrics are calculated and how TensorBoard logs them will enable more effective debugging and iteration of models. Additionally, it is beneficial to review research papers or online tutorials focusing on model evaluation for multi-class classification, as there are various interpretations of performance, but the per-class F1 score metric tends to be most revealing.
