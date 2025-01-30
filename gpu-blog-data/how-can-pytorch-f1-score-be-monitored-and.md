---
title: "How can PyTorch F1 score be monitored and retraining triggered?"
date: "2025-01-30"
id: "how-can-pytorch-f1-score-be-monitored-and"
---
The challenge with directly using F1 score as a loss function in PyTorch stems from its non-differentiable nature. Gradient-based optimization, the core of backpropagation, necessitates differentiable loss functions. Monitoring F1 score, therefore, becomes a crucial task during model training and subsequent retraining triggers require an understanding of its performance. My experience developing image classification models for medical imaging has demonstrated the importance of not solely relying on accuracy; F1 score provides a balanced evaluation of precision and recall, particularly when dealing with imbalanced datasets, as was frequently encountered in my projects involving rare diseases.

Fundamentally, F1 score is calculated as the harmonic mean of precision and recall. Precision measures the proportion of true positives among all positive predictions, while recall quantifies the proportion of true positives that were correctly identified out of all actual positives. Mathematically, precision is TP/(TP + FP) and recall is TP/(TP + FN), where TP represents true positives, FP represents false positives, and FN represents false negatives. The F1 score then becomes 2 * (precision * recall) / (precision + recall). Because the computation involves comparisons of integer counts (TP, FP, FN), it introduces discontinuities and, thus, prevents directly deriving gradients to be backpropagated into the network's trainable parameters.

Monitoring F1 score during PyTorch training typically involves calculating the metric during the validation phase at the end of each epoch or a certain number of mini-batches. This allows us to observe the model's performance on unseen data, avoiding overfitting. To accomplish this, we maintain running totals of TP, FP, and FN for each class of the classification task, accumulate these statistics across the validation set, and then, based on these accumulated values, compute the F1 score. It is important to use a validation set separate from training to provide accurate and unbiased evaluation.

Let's examine several practical code examples, starting with a basic implementation of F1 score calculation:

```python
import torch

def calculate_f1(predictions, targets, num_classes, average='macro'):
    """Calculates the F1 score for multi-class classification.

    Args:
        predictions (torch.Tensor): Predicted class indices.
        targets (torch.Tensor): True class indices.
        num_classes (int): Number of classes.
        average (str): Averaging strategy ('macro' or 'weighted').

    Returns:
        float: F1 score.
    """

    tp = torch.zeros(num_classes, dtype=torch.int64)
    fp = torch.zeros(num_classes, dtype=torch.int64)
    fn = torch.zeros(num_classes, dtype=torch.int64)

    for i in range(num_classes):
        tp[i] = ((predictions == i) & (targets == i)).sum()
        fp[i] = ((predictions == i) & (targets != i)).sum()
        fn[i] = ((predictions != i) & (targets == i)).sum()

    precision = tp / (tp + fp + 1e-8) # Adding a small value to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    if average == 'macro':
       return f1.mean().item()
    elif average == 'weighted':
        class_counts = torch.bincount(targets, minlength=num_classes)
        weights = class_counts / class_counts.sum()
        return (f1 * weights).sum().item()
    else:
        raise ValueError("Invalid average method.")

# Example usage:
num_classes = 3
predictions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 2, 1])
targets = torch.tensor([0, 1, 2, 1, 2, 0, 0, 2, 1])
f1_macro = calculate_f1(predictions, targets, num_classes, average='macro')
f1_weighted = calculate_f1(predictions, targets, num_classes, average='weighted')
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")
```

This function provides both macro and weighted F1 calculations. Macro averaging computes the F1 score for each class and then averages those scores equally, regardless of the class imbalances. Weighted averaging, on the other hand, weights the F1 scores for each class by the class's support (number of occurrences) in the target tensor. Selecting the appropriate averaging type is important. In my experience, if classes are significantly imbalanced, a weighted average can provide a better indication of the overall performance. The next code snippet focuses on embedding this calculation into a typical PyTorch training loop.

```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assume the existence of a simple CNN model, replace with actual model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 6 * 6, num_classes) # Assuming input image of 10x10

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Example usage with simulated data:
num_classes = 3
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simulate training and validation data
train_data = torch.randn(100, 1, 10, 10)
train_labels = torch.randint(0, num_classes, (100,))
val_data = torch.randn(50, 1, 10, 10)
val_labels = torch.randint(0, num_classes, (50,))

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def train_and_validate(model, criterion, optimizer, train_loader, val_loader, num_epochs, num_classes):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
             for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)
                _, predicted = torch.max(val_outputs, 1)
                all_predictions.extend(predicted.tolist())
                all_targets.extend(val_labels.tolist())
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        f1_score = calculate_f1(all_predictions, all_targets, num_classes, average='macro')
        print(f"Epoch: {epoch+1}, Validation F1 Score: {f1_score:.4f}")
        # Add retraining trigger logic here if needed

num_epochs = 10
train_and_validate(model, criterion, optimizer, train_loader, val_loader, num_epochs, num_classes)
```

This example sets up a basic training and validation loop, integrating the F1 score calculation after each epoch. Crucially, the model is placed into evaluation mode (`model.eval()`) during validation, which disables dropout and batch normalization layers behaving differently than during training. A key aspect of this example is the accumulation of predictions and targets during the validation pass before F1 score computation, ensuring an accurate evaluation across the entire validation set. Building upon this code, we introduce a basic retraining trigger:

```python
# Continuation of the previous code block
def train_and_validate(model, criterion, optimizer, train_loader, val_loader, num_epochs, num_classes, min_f1_delta=0.01, patience = 3):
    best_f1 = 0.0
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)
                _, predicted = torch.max(val_outputs, 1)
                all_predictions.extend(predicted.tolist())
                all_targets.extend(val_labels.tolist())
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        f1_score = calculate_f1(all_predictions, all_targets, num_classes, average='macro')
        print(f"Epoch: {epoch+1}, Validation F1 Score: {f1_score:.4f}")

        if f1_score - best_f1 > min_f1_delta:
             best_f1 = f1_score
             epochs_without_improvement = 0
             print(f"F1 improved to {best_f1:.4f}, resetting patience.")
        else:
             epochs_without_improvement +=1

        if epochs_without_improvement >= patience:
            print(f"No significant improvement after {patience} epochs, triggering retraining.")
            # Add logic to load weights of the best model before retraining (as checkpoint)
            # In this example, we simply reset model's weights
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                  layer.reset_parameters()

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            best_f1 = 0.0
            epochs_without_improvement = 0
            print("Model weights reset and retraining initiated with new optimizer.")

num_epochs = 30 # Modified number of epochs to allow more chance of retraining
min_f1_delta = 0.01 # Minimum improvement
patience = 3  # Max number of epochs without significant improvement before trigger
train_and_validate(model, criterion, optimizer, train_loader, val_loader, num_epochs, num_classes, min_f1_delta, patience)
```

In this enhanced version, we store the best F1 score seen so far, and if the F1 score does not improve by `min_f1_delta` within `patience` epochs, the model's weights are reset (mimicking re-initialization in this simple example). Then the training process continues from the reinitialized model with the new optimizer. In real-world scenarios, we would typically save the model’s best weights and load them instead of re-initialization for a more effective retraining procedure, leveraging the model state that achieved the highest F1 score. The parameters `min_f1_delta` and `patience` can be tuned based on the dataset and application.

For further exploration, I would recommend consulting resources such as scikit-learn documentation for deeper insights into metrics calculation, particularly the handling of precision, recall and F1 in multi-class problems. Furthermore, publications within the fields of machine learning and medical imaging provide additional context on practical application of evaluation metrics.
