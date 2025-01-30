---
title: "How can I prevent Weights & Biases from saving the best model checkpoints?"
date: "2025-01-30"
id: "how-can-i-prevent-weights--biases-from"
---
Preventing Weights & Biases (W&B) from automatically saving the "best" model checkpoints requires understanding its default behavior and strategically controlling the checkpointing process. W&B, by default, tracks metrics and can automatically save models based on improvements in a monitored metric. This "best" model is usually determined by minimizing a loss or maximizing an accuracy. However, in several situations, this automatic saving is not desirable. I've personally encountered this during experimentation with continual learning where I wanted to save model versions at distinct points in training, irrespective of performance, or during pretraining where performance might fluctuate wildly but the model’s state at every epoch is relevant. Here’s how we can effectively override the default checkpointing.

W&B's model checkpointing is primarily managed through its `wandb.save()` function, which under the hood leverages the `wandb.Artifact` functionality to manage versions and store model files. The key to preventing the automatic "best" model saving is to disable or modify the automatic tracking of a monitored metric that triggers it. We must then take direct control of when, and *if*, we call `wandb.save()` explicitly, based on custom logic or requirements. The automatic selection of the best model is driven by what `wandb.watch()` observes and the default `wandb.config.save_best_model` and `wandb.config.monitor_metrics` settings. Simply put, if you don't want W&B to save the "best" model based on a particular metric, do not instruct it to monitor that metric for model checkpointing. We’ll achieve this via a combination of configuration changes and direct control of saving artifacts.

The common scenarios where you might want to prevent W&B from automatically saving the best model are often when:

*   You are using a complex training regime that requires the model to go through phases where performance drops temporarily, before increasing. In such cases, saving only the best model based on a single metric can result in missing valuable model checkpoints.
*   You're not focused on maximizing a particular metric; for example, you might be analyzing convergence patterns or exploring model behavior at various training stages.
*   You're debugging training issues and need to evaluate model performance at specific epochs, even if the performance is poor.
*   You’re performing a pretraining stage, where the metric is unstable and only the final pretrained model is of interest.
*   You want to decouple the metrics from the model saving behavior and handle it manually in order to apply custom logic.

Let’s examine concrete code examples to illustrate different methods to prevent the automatic saving, each with a specific application and commentary.

**Example 1: Disabling Automatic Best Model Tracking**

This example demonstrates the simplest method, turning off the `save_best_model` option and explicitely not using the monitoring feature when calling `wandb.watch()`. This disables W&B’s automated saving of the "best" model by metric performance. Instead we control the checkpointing process explicitly.

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize a new W&B run
wandb.init(project="custom-checkpointing", config={"save_best_model": False}) # Disable Automatic Best Model

# Instantiate the model, optimizer, and loss function
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example Training Loop
for epoch in range(5):
    # Training steps - Fake Data
    inputs = torch.randn(32, 10)
    labels = torch.randint(0, 2, (32,))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Log the training metrics - crucial for W&B
    wandb.log({"loss": loss.item(), "epoch": epoch})
    # Instead of automatically saving the best, we manually save the checkpoint in each epoch.
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, f"model_epoch_{epoch}.pth")
    
    # Explicitly Save as Artifact
    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(f"model_epoch_{epoch}.pth")
    wandb.log_artifact(artifact)


# Finish the W&B run.
wandb.finish()
```

In this example, we disable the default automatic model saving, configure the training loop, explicitly save the models in each epoch, and explicitly log it to W&B as an artifact. We disable the automatic saving by setting `save_best_model` to False when initializing the run. The model’s state is saved to a file each epoch and we use `wandb.log_artifact` to save the model as an artifact. This grants us direct control over what model versions are logged.

**Example 2: Manual Control with `wandb.save()` based on Custom Criteria**

Here, we retain metric tracking, but use a conditional save using `wandb.save()`, which only saves based on a custom condition, for example only saving the model at specific epochs.

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize a new W&B run
wandb.init(project="custom-checkpointing", config={"save_best_model": False})

# Instantiate the model, optimizer, and loss function
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# W&B watch to see metrics for debugging/logging, but not for saving best model
wandb.watch(model, criterion, log="all")

# Example Training Loop
for epoch in range(5):
    # Training steps - Fake Data
    inputs = torch.randn(32, 10)
    labels = torch.randint(0, 2, (32,))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Log the training metrics - crucial for W&B
    wandb.log({"loss": loss.item(), "epoch": epoch})
    # Only save models every other epoch
    if epoch % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"model_epoch_{epoch}.pth")
            
         # Explicitly Save as Artifact
        artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
        artifact.add_file(f"model_epoch_{epoch}.pth")
        wandb.log_artifact(artifact)


# Finish the W&B run.
wandb.finish()
```

In this iteration, we still use `wandb.watch()` for logging, but we circumvent the default best model selection behavior. The code now saves the model only at specific epochs.  `wandb.watch` allows us to see the behavior of training in the W&B UI while we manually control when to save the model as an artifact with `wandb.log_artifact`. The save condition can be any complex function.

**Example 3:  No Automatic Metric Monitoring**

This example doesn't leverage the `wandb.watch()` feature at all and avoids any dependence on metric tracking by manually saving the models. This gives complete control and might be useful for certain pretraining scenarios.

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize a new W&B run
wandb.init(project="custom-checkpointing", config={"save_best_model": False})

# Instantiate the model, optimizer, and loss function
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example Training Loop
for epoch in range(5):
    # Training steps - Fake Data
    inputs = torch.randn(32, 10)
    labels = torch.randint(0, 2, (32,))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Log the training metrics - crucial for W&B
    wandb.log({"loss": loss.item(), "epoch": epoch})

    # Directly save models at each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, f"model_epoch_{epoch}.pth")
    
     # Explicitly Save as Artifact
    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(f"model_epoch_{epoch}.pth")
    wandb.log_artifact(artifact)


# Finish the W&B run.
wandb.finish()
```

In this final version, we completely disconnect W&B’s automatic monitoring of metrics to guide model saving. We log the loss and epoch but we do not save the "best" one, instead we choose to save a checkpoint on every epoch. This gives the most granular control over checkpointing but still retains logging.

In summary, controlling W&B’s model saving requires actively preventing the default automatic checkpointing behavior which is driven by metric observation. By setting `save_best_model` to `False` and by leveraging `wandb.save` and `wandb.log_artifact` based on custom logic, users can ensure that model saving aligns precisely with their research or deployment needs.  These techniques allow for the saving of models irrespective of their performance on any given metric, enabling a more flexible experimentation process.

**Resource Recommendations:**

To understand W&B's functionalities better I recommend reviewing the documentation, specifically focusing on:

*   Artifact Management: This covers versioning, uploading, and downloading models and other files, which are crucial for implementing the direct control over saving models shown.
*   Configuration Tracking: Understand the use of `wandb.config`, where you can explicitly define whether to enable/disable the automatic saving of the best model.
*   Metric Tracking and logging: Become familiar with the different logging methods and the use of `wandb.log()`.
*   The `wandb.watch()` function and the relationship to `wandb.config.monitor_metrics` and how to manipulate this behavior.

These sections will provide the knowledge to customize W&B to work within a variety of experiments and allow for granular control of the model checkpointing process.
