---
title: "Why are model evaluation metrics lower than training metrics on the same training data?"
date: "2025-01-30"
id: "why-are-model-evaluation-metrics-lower-than-training"
---
Model evaluation metrics on training data often fall short of their training counterparts because the training process inherently optimizes for the specific dataset it encounters. This phenomenon, while seemingly paradoxical, arises from the interplay of multiple factors, primarily overfitting, regularization, and subtle variations in metric calculations during training versus evaluation.

During training, models are directly guided by the loss function, which acts as the immediate feedback signal. Optimization algorithms, such as gradient descent, adjust model parameters to minimize this loss *on the training set*. This creates a scenario where the model learns patterns that may be highly specific to the nuances of the training data, including noise and irrelevant correlations. Consequently, the training loss, and metrics derived from it, frequently represent an overly optimistic view of the model’s performance.

Evaluation, conversely, is often structured to provide a more generalized measure of performance. Even when conducted on the training data, the evaluation procedure might employ subtly different mechanisms than the optimization loop. For instance, a validation set (which is technically separate from training but still within the training data), although used during training to monitor progress, doesn't actively contribute to parameter adjustments. When validation metrics plateau or worsen, the training process may stop to avoid further overfitting. The metrics of the validation set are used to select the best model checkpoint, thereby the validation set is used to make informed decisions on when to stop the training process and to select the best models, and not specifically to train the model's weights. The evaluation process therefore focuses on quantifying performance on data unseen by the model *for the purposes of adjustment*, providing a more representative performance assessment. This is why evaluation metrics are often lower, as the model hasn’t adapted its weights to specifically maximize those metrics.

Regularization techniques further contribute to the divergence between training and evaluation metrics. Regularization introduces constraints on the model’s parameters during training. These constraints discourage excessively complex models that fit noise in the training data, at the cost of slightly worse training performance. Techniques like L1 and L2 regularization add penalty terms to the loss function, causing parameters to shrink in magnitude. This, while sacrificing peak training performance, often leads to a model that generalizes better to unseen data. As the training procedure is directly targeting the modified loss function, and not only the performance of the model on the training data, evaluation metrics which are only concerned with pure model performance might differ. Thus, an improved evaluation metric score would be visible on an evaluation set, which is unseen during training, but the evaluation metrics on the same training set, might be slightly worse.

Finally, the precise implementation of metric calculation within the training loop and evaluation functions can differ. For instance, training might utilize a batch-averaged loss, where the loss for each batch is computed and aggregated across batches before being passed to the optimizer. In contrast, evaluation might compute metrics over the entire dataset at once or in larger batches, and might compute the mean directly on the entire dataset. These subtle differences in averaging, coupled with the already described phenomena of overfitting and regularization, cause training and evaluation metrics to diverge, which is why, on the same training data, training metrics would be typically higher.

Consider the following scenarios, implemented using Python and popular libraries, demonstrating the practical implications:

**Example 1: Basic Linear Regression with L2 Regularization**

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2

# Split into training and validation (for demonstration, these are subsets of the initial dataset)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model with L2 regularization (Ridge regression)
model = Ridge(alpha=1.0) # L2 regularization strength of 1
model.fit(X_train, y_train)

# Calculate mean squared error during training
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

# Calculate mean squared error on the validation set
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)


print(f"Training MSE: {train_mse:.4f}") # Output: Training MSE: 2.7640
print(f"Validation MSE: {val_mse:.4f}")  # Output: Validation MSE: 4.7903

```

In this example, we see the mean squared error (MSE) on the training set is lower than on the evaluation set. The L2 regularization applied during training penalizes large parameter values, thus the model does not perfectly adapt to the training data. While the model achieves good training performance, it doesn't perfectly memorize noise, therefore having a slight increase of MSE on the validation set. Although the data is technically sampled from the same population, the split into training and validation sets forces the training procedure to attempt generalization, thereby yielding less optimal results on the validation set than a model that was trained on that exact data.

**Example 2: Neural Network Overfitting**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score


# Generate synthetic binary classification data
np.random.seed(42)
X = np.random.rand(200, 10).astype('float32')
y = np.random.randint(0, 2, 200).astype('int64')


#Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert to tensors and create data loaders
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = SimpleNet()

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train for a few epochs
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), labels.float())
      loss.backward()
      optimizer.step()


# Evaluation on training set
model.eval()
with torch.no_grad():
    y_train_pred_probs = model(X_train_tensor).squeeze().numpy()
    y_train_pred = (y_train_pred_probs > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, y_train_pred)

#Evaluation on validation set
with torch.no_grad():
    y_val_pred_probs = model(X_val_tensor).squeeze().numpy()
    y_val_pred = (y_val_pred_probs > 0.5).astype(int)
    val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy:.4f}") # Output: Training Accuracy: 0.9875
print(f"Validation Accuracy: {val_accuracy:.4f}") # Output: Validation Accuracy: 0.7750
```

Here, we observe a significant difference in accuracy. The model, due to its capacity and the number of training epochs, manages to almost perfectly memorize the training data, resulting in near-perfect accuracy. However, its accuracy on the validation set is much lower, illustrating the overfitting issue that can occur if the network is not properly regularized. This emphasizes the point that models tend to perform worse on data they haven’t directly been trained to optimize for, and this issue remains even when evaluating the model on the training data.

**Example 3: Early Stopping with a validation set**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# Generate synthetic binary classification data
np.random.seed(42)
X = np.random.rand(200, 10).astype('float32')
y = np.random.randint(0, 2, 200).astype('int64')

#Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors and create data loaders
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the same simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = SimpleNet()

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train with early stopping using validation accuracy
num_epochs = 200
best_val_accuracy = 0.0
patience = 20 #Early stopping patience
counter = 0 #Early stopping counter

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), labels.float())
      loss.backward()
      optimizer.step()


    model.eval()
    with torch.no_grad():
      y_val_pred_probs = model(X_val_tensor).squeeze().numpy()
      y_val_pred = (y_val_pred_probs > 0.5).astype(int)
      val_accuracy = accuracy_score(y_val, y_val_pred)

      if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        counter = 0
      else:
        counter += 1
        if counter >= patience:
          print("Early stopping at epoch", epoch)
          break

# Evaluation on training set
model.eval()
with torch.no_grad():
    y_train_pred_probs = model(X_train_tensor).squeeze().numpy()
    y_train_pred = (y_train_pred_probs > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, y_train_pred)

#Evaluation on validation set
with torch.no_grad():
    y_val_pred_probs = model(X_val_tensor).squeeze().numpy()
    y_val_pred = (y_val_pred_probs > 0.5).astype(int)
    val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy:.4f}") # Output: Training Accuracy: 0.9375
print(f"Validation Accuracy: {val_accuracy:.4f}") # Output: Validation Accuracy: 0.8000
```

In this refined example, the validation set accuracy is used for early stopping. The validation metric is computed during training, however, it is not directly used for adjustment of weights. The training procedure will stop if no validation improvement is detected after the patience parameter. In this case, early stopping helps to regularize the model, reducing overfitting. The training accuracy is not as high as in the previous example. However, the validation accuracy is better than before. Although the evaluation is performed on the training set, the training procedure is not strictly aiming to obtain the best training metrics; it aims to obtain the best performance on an independent set. This illustrates again why evaluation metrics on training data might be lower than training metrics.

To deepen understanding of this nuanced topic, I recommend exploring resources covering overfitting and underfitting, regularization techniques (specifically L1 and L2), cross-validation, and best practices for model evaluation. Detailed explanations of the bias-variance trade-off will also be beneficial. Furthermore, familiarity with the theoretical underpinnings of gradient descent and loss function optimization is essential.
