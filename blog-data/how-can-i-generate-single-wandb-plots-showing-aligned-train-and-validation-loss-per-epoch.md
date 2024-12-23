---
title: "How can I generate single WandB plots showing aligned train and validation loss per epoch?"
date: "2024-12-23"
id: "how-can-i-generate-single-wandb-plots-showing-aligned-train-and-validation-loss-per-epoch"
---

Okay, let's tackle this. I remember a project back in '21, working on a complex segmentation model where visualizing the training and validation losses in a clear, aligned manner was crucial. It wasn't just about seeing the numbers; it was about understanding *how* the model was learning and identifying any potential overfitting or underfitting trends. The default WandB setup, while useful, wasn't quite cutting it for the granular analysis I needed. So, here's the breakdown of how I approached this, along with some practical examples.

The core challenge here isn’t the logging of the loss values themselves, but rather, the *presentation* of those values in a single, aligned plot. What we want to avoid is the mess of separate plots or, even worse, having to manually align them after the fact. To accomplish this, we need to leverage WandB's custom plotting capabilities effectively, ensuring that both train and validation losses are plotted against the same epoch axis. We will achieve this by creating custom chart panels and logging data appropriately. We're going to be creating a ‘line plot’ in WandB, which by default, plots the data points logged as we are training.

The key is to log both the training and validation losses using the same `step` value, which in our case is going to be the epoch number. This might seem trivial, but it's where many stumble. Instead of logging the losses with different, potentially implicit steps, we explicitly log them with the corresponding epoch. Let’s walk through some code examples.

**Example 1: Basic Logging Within a Training Loop**

Let's say we have a basic training loop that iterates through epochs. Within this loop, we will calculate and log both the training and validation loss.

```python
import wandb
import numpy as np

# Initialize a wandb run
wandb.init(project="loss_visualization")

num_epochs = 10
for epoch in range(num_epochs):
    # Simulate training loss
    train_loss = np.random.rand() * (1 - epoch / num_epochs)  # Decreasing loss

    # Simulate validation loss
    validation_loss = np.random.rand() * (1 - epoch / num_epochs) + 0.1 # Decreasing loss but consistently higher

    wandb.log({
        "epoch": epoch, # Explicitly log the epoch
        "train_loss": train_loss,
        "validation_loss": validation_loss
    }, step=epoch) # Important to set this as the step
wandb.finish()

```

In this basic setup, the `step=epoch` parameter in `wandb.log()` is crucial. If you omit this, wandb would assume a sequential integer count by default, not the actual epoch number, and your plot alignment will not work as expected. The `epoch` value itself is logged as a custom data point, alongside the losses.

**Example 2: More Realistic Scenario using Training and Validation Data**

In more complex scenarios, you may have separate training and validation loaders. Here's how you might log the losses in such a setup, assuming the use of PyTorch or a similar framework for model training:

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Mock Data for Demonstration
train_data = torch.randn(100, 10)
train_labels = torch.randint(0, 2, (100,))
val_data = torch.randn(50, 10)
val_labels = torch.randint(0, 2, (50,))

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=10)
val_loader = DataLoader(val_dataset, batch_size=10)

# Simple Model (Example)
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Setup
model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

wandb.init(project="loss_visualization_realistic")
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss_sum = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()

    avg_train_loss = train_loss_sum / len(train_loader)

    model.eval()
    validation_loss_sum = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss = criterion(output, target)
            validation_loss_sum += val_loss.item()

    avg_validation_loss = validation_loss_sum / len(val_loader)

    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "validation_loss": avg_validation_loss
        }, step=epoch)
wandb.finish()
```

This version mirrors a realistic training setup more closely, with training and validation loops, and the logging follows the same pattern, but with the `average` loss values computed over the entire loader dataset, instead of a single iteration. Again, the `step=epoch` is critical for proper plot alignment.

**Example 3: Using Custom Charts for Enhanced Control**

While the previous examples suffice for basic plotting, you might want more control over your chart’s appearance. Here's how to create custom chart configurations using `wandb.define_metric` and then log your metrics for that chart.

```python
import wandb
import numpy as np

wandb.init(project="loss_visualization_custom")

wandb.define_metric("epoch") # Define 'epoch' as the step
wandb.define_metric("train_loss", step_metric="epoch")
wandb.define_metric("validation_loss", step_metric="epoch")

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = np.random.rand() * (1 - epoch / num_epochs)
    validation_loss = np.random.rand() * (1 - epoch / num_epochs) + 0.1

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss
    }, step=epoch)
wandb.finish()
```

By using `wandb.define_metric`, you are defining that you intend to use the `epoch` parameter as your x-axis or ‘step’. The other metrics are then linked to this step metric. Although this example looks similar to example 1, there are more advanced customisation options available through `wandb.define_metric` that are helpful for more complex chart designs. For instance, if you have various other metrics being calculated, this helps to group them correctly into separate charts, which also reduces any confusion.

**Key Takeaways and Additional Guidance**

These examples demonstrate the core principles. The takeaway is that aligning your training and validation losses in a single plot hinges on using the same `step` value for both during logging.

For further learning:

*   **WandB Documentation:** The official WandB documentation is the most comprehensive resource for all things WandB. You will want to especially focus on the sections related to `wandb.log()`, metric definition, and custom charts.
*   **"Deep Learning" by Goodfellow, Bengio, and Courville:** This is a fantastic resource that gives an understanding of how to create the training/validation process for deep learning systems. It is essential for understanding what the train and validation loss actually means.
*   **PyTorch Tutorials/Tensorflow Tutorials**: If you are using one of these frameworks, going through their guides on training loops and validation procedures will help provide you with more practical experience and a strong basis to work with, in addition to the examples provided.

Remember, the key is consistency in your logging and leveraging WandB's features to visualise your model's learning process effectively. My own experiences suggest that spending time getting this visualisation right is often as beneficial as refining the model architecture itself. It truly helps you understand the model's training dynamics and facilitates better decision-making for hyperparameter tuning and model architecture selection. Good luck, and let me know if you have any further questions.
