---
title: "How can I disable Weights & Biases from saving the best model checkpoint?"
date: "2024-12-23"
id: "how-can-i-disable-weights--biases-from-saving-the-best-model-checkpoint"
---

Alright, let's unpack this. It's a nuanced request because, as anyone who's worked with Weights & Biases (wandb) extensively knows, its checkpointing behavior is designed to be helpful and somewhat "automatic." The good news is, you definitely have control, just not in a "one-click" sort of way. I've encountered this exact scenario in several projects, specifically when deploying models that had their own, more precise checkpointing logic baked in, or when I wanted to minimize storage usage for the wandb project. Let me illustrate the approach with a few practical examples and some reasoning behind why this works.

The core challenge is that wandb’s `wandb.log` function, when used in conjunction with training loops, and especially its `wandb.watch` feature, often implicitly saves models that perform the best according to a given metric. This happens because it is coupled with the `wandb.Artifact` mechanism which, while great for reproducibility, can cause friction if you have existing save conventions.

Essentially, there isn't a single global "disable all best checkpoint saves" setting. You will need to manage this more granularly by controlling what you log to wandb and how you define ‘best’.

My first piece of advice would be to critically examine how your training loop handles model saving in relation to wandb. The problem arises when you directly invoke methods like `wandb.log({"metric_name": metric_value})` and also use `wandb.watch()` . In those scenarios, wandb will try to persist any model, assuming it to be the best model based on the metric that it’s tracking. The key, then, is to separate your own model-saving logic from wandb's automatic tracking.

Here's an initial code snippet demonstrating how you can selectively avoid having wandb automatically save the best model. Assume we are doing some typical training with a hypothetical model, called `MyModel`:

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize wandb
wandb.init(project="my-project", name="no-auto-save")

# Dummy Model and optimizer setup
class MyModel(nn.Module):
    def __init__(self):
      super(MyModel, self).__init__()
      self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define your own logic for best model saving
best_val_loss = float('inf')
save_path = "my_custom_model.pth" # where you will be saving your models

# Dummy training loop
for epoch in range(10):
    # Dummy training step (replace with actual training)
    train_loss = torch.randn(1).item()
    # Dummy validation step (replace with actual validation)
    val_loss = torch.randn(1).item()

    # Log metrics to wandb but don't associate with model saving
    wandb.log({"train_loss": train_loss, "val_loss": val_loss})

    # Check if current validation loss is the best and save model as you see fit.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saving new best model at epoch {epoch}")
    
    print(f"Epoch: {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Finish wandb run
wandb.finish()

```
In this first example, we are manually saving our models based on our evaluation of best loss, and `wandb` only gets the metrics, no associated models are saved via wandb's system. This gives you full control over how you save your models and when, bypassing wandb's default behavior. Notice `wandb.watch()` was not called here, because we don't want it to automatically save. We have not provided any model information with `wandb.log`, so no automatic checkpointing will occur.

Moving further, a more complex, common scenario involves creating an artifact when a model has completed training. Artifacts are great for versioning and reproducibility, but they may require slight changes in our control of model saving. Here's an example showing how to explicitly save the model as an artifact when *we* decide to, and avoid implicit saving of the best model:

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Initialize wandb
run = wandb.init(project="my-project", name="artifact-controlled-save")

# Dummy Model and optimizer setup
class MyModel(nn.Module):
    def __init__(self):
      super(MyModel, self).__init__()
      self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop
for epoch in range(10):
    # Dummy training step (replace with actual training)
    train_loss = torch.randn(1).item()
    # Dummy validation step (replace with actual validation)
    val_loss = torch.randn(1).item()
    # Log metrics to wandb
    wandb.log({"train_loss": train_loss, "val_loss": val_loss})

# This is the key part – manually creating and saving an artifact
artifact = wandb.Artifact("my-model-artifact", type="model")
save_path = "my_final_model.pth"
torch.save(model.state_dict(), save_path)
artifact.add_file(save_path)
run.log_artifact(artifact)

# Clean up the saved model file
os.remove(save_path)

# Finish wandb run
run.finish()
```

In this example, we've explicitly created a wandb artifact and added our saved model file. We are, in this case, bypassing wandb's "best model" tracking completely. This provides a clear separation between logging metrics and saving models as artifacts and provides control over which model version is uploaded as an artifact.

Finally, a critical consideration is when you have your own existing checkpointing system. Here's an example to demonstrate how to integrate your existing system and keep using it without wandb interfering by trying to auto-save 'better' models that aren’t actually better for your application:

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

# Initialize wandb
wandb.init(project="my-project", name="existing-checkpoint")

# Dummy Model and optimizer setup
class MyModel(nn.Module):
    def __init__(self):
      super(MyModel, self).__init__()
      self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define your own logic for saving and managing checkpoints
checkpoint_dir = "my_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_my_checkpoint(model, optimizer, epoch, checkpoint_dir):
  checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      }, checkpoint_path)
  print(f"Saved custom checkpoint at epoch {epoch} to {checkpoint_path}")

# Dummy training loop with custom checkpointing
for epoch in range(10):
    # Dummy training step (replace with actual training)
    train_loss = torch.randn(1).item()
    # Dummy validation step (replace with actual validation)
    val_loss = torch.randn(1).item()
    # Log metrics to wandb
    wandb.log({"train_loss": train_loss, "val_loss": val_loss})

    # Save checkpoint on a fixed interval
    if epoch % 3 == 0:
        save_my_checkpoint(model, optimizer, epoch, checkpoint_dir)

# Finish wandb run
wandb.finish()
```

Here, the crucial aspect is that we are in charge of saving our checkpoints. We use a custom `save_my_checkpoint` function that adheres to our standards and handles storing models at specific intervals. This approach keeps your existing checkpointing system completely independent of wandb’s automatic saving.

In conclusion, disabling the automatic “best model” saving in wandb requires adopting a more explicit approach that gives you full control over how models are managed, whether through your custom save method or through the creation of artifacts at defined moments. Key takeaways from my experience are that you must control what gets associated with `wandb.log` and how artifacts are created to prevent wandb from implicitly assuming what is the "best" checkpoint and overriding your own model saving conventions. For more information on best practices with wandb, the documentation is a must, and I would also advise looking at papers that explore deep learning pipeline management which can shed light on managing such systems effectively; specific papers will vary based on your chosen deep learning ecosystem and application. Finally, familiarize yourself with the `wandb.Artifact` documentation since that's often the source of most of these "automatic" model checkpoint issues. The key is understanding the mechanisms at play so that they serve you rather than the other way around.
