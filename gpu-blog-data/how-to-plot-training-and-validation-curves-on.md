---
title: "How to plot training and validation curves on the same PyTorch graph?"
date: "2025-01-30"
id: "how-to-plot-training-and-validation-curves-on"
---
In my experience building and optimizing deep learning models, visualizing training and validation performance concurrently on a single graph is fundamental for assessing model behavior and identifying common problems such as overfitting or underfitting. This process involves tracking loss and metric values over each training epoch, storing these values, and then using a plotting library to generate the combined visualization.

The core mechanism lies in collecting data during the training and validation phases separately and then rendering both datasets on the same axes of a plot. This allows for a direct comparison of how the model generalizes, or fails to generalize, to unseen data. I'll outline the steps necessary, including representative code snippets using standard PyTorch practices and matplotlib.

Firstly, the training loop must be modified to not only calculate loss and gradients during the training phase but also record relevant metrics, typically loss and accuracy, for both the training and validation sets. This is typically achieved through two distinct but largely parallel processes within the training loop, one for the training data and the other for the validation data. The validation set calculations must be performed without gradient computation, accomplished through the `torch.no_grad()` context manager.

The values obtained during training and validation are then stored in lists that will be used for plotting. Let me exemplify the training and evaluation loop, where the crucial metric is loss, using `torch` and `torch.nn` for modeling and `torch.optim` for training.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train() # Sets the model to training mode
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clears old gradients from last step
            outputs = model(inputs) # Performs forward propagation
            loss = criterion(outputs, targets)  # Loss calculation
            loss.backward()        # Backward propagation
            optimizer.step()       # Updates weights based on gradients
            epoch_train_loss += loss.item() # Accumulates loss for the epoch

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss) # Store training loss

        model.eval()  # Sets the model to evaluation mode
        epoch_val_loss = 0.0
        with torch.no_grad(): # Turn off gradient computation to accelerate evaluation and conserve memory
            for inputs, targets in val_loader:
                outputs = model(inputs)  # Forward propagation for validation
                loss = criterion(outputs, targets) # Calculates validation loss
                epoch_val_loss += loss.item() # Accumulates loss for the validation

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss) # Store validation loss

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    return train_losses, val_losses

# Example usage
if __name__ == '__main__':
    # Define a simple linear model
    class LinearModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(input_size, output_size)

        def forward(self, x):
            return self.linear(x)

    # Generate synthetic data
    input_size = 10
    output_size = 1
    num_samples = 1000
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randn(num_samples, output_size)

    # Split into training and validation sets
    train_size = int(0.8 * num_samples)
    train_inputs = inputs[:train_size]
    train_targets = targets[:train_size]
    val_inputs = inputs[train_size:]
    val_targets = targets[train_size:]

    # Create DataLoader
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, criterion, and optimizer
    model = LinearModel(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 10

    # Train and evaluate
    train_losses, val_losses = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs)

    # Plot the losses
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```
The code first defines a basic linear regression model, then generates some synthetic data, splits the data into training and validation sets, and encapsulates them within PyTorch DataLoaders. In the `train_and_evaluate` function, the model's training is performed inside an outer loop for each epoch. The loss is calculated and backpropagated in batches of the training data and model weights are updated through the optimizer. The same process is conducted for validation data while suppressing gradient computation to speed up evaluation and reduce memory usage. The accumulated losses for both sets are then returned as separate lists. Lastly, Matplotlib visualizes the losses across all epochs as a plot. The `model.train()` and `model.eval()` calls are crucial for toggling between the dropout and batchnorm behavior that is often used during training but suppressed during validation or inference.

The output plot clearly displays both the training and validation losses across each epoch. A considerable divergence between training loss and validation loss can indicate overfitting. A constant training and validation loss, where losses are not decreasing, is often a sign of underfitting.

This initial example only incorporates loss as the performance metric.  However, plotting accuracy metrics, or any metric relevant to the model performance, alongside loss can provide a more complete picture. Consider the following adaptation where a classification metric, specifically accuracy, is added:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def train_and_evaluate_classification(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)


        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    class SimpleClassifier(nn.Module):
       def __init__(self, input_size, hidden_size, num_classes):
           super(SimpleClassifier, self).__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(hidden_size, num_classes)

       def forward(self, x):
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x


    # Define data and hyperparameters
    input_size = 20
    hidden_size = 50
    num_classes = 5
    num_samples = 1000
    num_epochs = 10
    batch_size = 32

    # Generate synthetic data
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randint(0, num_classes, (num_samples,))
    train_size = int(0.8 * num_samples)
    train_inputs = inputs[:train_size]
    train_targets = targets[:train_size]
    val_inputs = inputs[train_size:]
    val_targets = targets[train_size:]

    # Create dataloaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = SimpleClassifier(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate_classification(model, train_loader, val_loader, optimizer, criterion, num_epochs)


    epochs = range(1, num_epochs + 1)

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
```
This extended code includes accuracy computation within both the training and validation loops. The training loop calculates the number of correctly classified samples, which is then used to derive the accuracy percentage, and is appended to a list for later plotting. In addition to generating the loss plots, this adaptation also renders a second plot displaying the accuracy metrics. This provides a more holistic view of model performance, as a loss reduction could potentially occur alongside a reduction in accuracy or vice versa.

Finally, I would like to demonstrate the tracking and plotting of F1 score, which is frequently utilized in situations with class imbalance. Here is the modified version of the code above, with F1 score calculation.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def train_and_evaluate_classification_f1(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_targets_all = []
        train_preds_all = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            train_targets_all.extend(targets.cpu().numpy())
            train_preds_all.extend(predicted.cpu().numpy())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_f1 = f1_score(train_targets_all, train_preds_all, average='weighted', zero_division=0)
        train_f1_scores.append(train_f1)


        model.eval()
        epoch_val_loss = 0.0
        val_targets_all = []
        val_preds_all = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_targets_all.extend(targets.cpu().numpy())
                val_preds_all.extend(predicted.cpu().numpy())


        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_f1 = f1_score(val_targets_all, val_preds_all, average='weighted', zero_division=0)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    return train_losses, val_losses, train_f1_scores, val_f1_scores


if __name__ == '__main__':
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Define data and hyperparameters
    input_size = 20
    hidden_size = 50
    num_classes = 5
    num_samples = 1000
    num_epochs = 10
    batch_size = 32

    # Generate synthetic data
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randint(0, num_classes, (num_samples,))
    train_size = int(0.8 * num_samples)
    train_inputs = inputs[:train_size]
    train_targets = targets[:train_size]
    val_inputs = inputs[train_size:]
    val_targets = targets[train_size:]

    # Create dataloaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = SimpleClassifier(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_losses, val_losses, train_f1_scores, val_f1_scores = train_and_evaluate_classification_f1(model, train_loader, val_loader, optimizer, criterion, num_epochs)

    epochs = range(1, num_epochs + 1)

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1_scores, label='Training F1 Score')
    plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

The F1 score calculation requires accumulating predictions and ground truth labels, then using `sklearn.metrics.f1_score`, and specifying the averaging method.  This is necessary due to F1 score not being an immediate output from typical PyTorch model predictions. This example shows that additional metrics beyond accuracy and loss can also be tracked and visualized over the epochs, enabling even more comprehensive performance evaluation.

In addition to what I have already exemplified, the user should examine other plotting libraries such as seaborn for generating more detailed visualizations. Also the tensorboard utility available in PyTorch can be used to monitor the model's progress, including the mentioned metrics.
Furthermore, the user should become comfortable with the use of metric computation functions available in libraries like Scikit-Learn, to efficiently obtain the required performance statistics for model validation, such as the precision, recall or F1 scores. The choice of metrics should be made based on the specific needs of the use case.
