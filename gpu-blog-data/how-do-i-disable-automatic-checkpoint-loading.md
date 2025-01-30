---
title: "How do I disable automatic checkpoint loading?"
date: "2025-01-30"
id: "how-do-i-disable-automatic-checkpoint-loading"
---
In my experience managing large-scale distributed training jobs, I've frequently encountered situations where automatic checkpoint loading, while generally convenient, becomes a hindrance. Specifically, scenarios involving model restarts from specific iteration points, rather than the latest saved state, necessitate a mechanism to prevent the framework from unilaterally restoring the most recent checkpoint. Disabling automatic checkpoint loading, therefore, is less about eliminating a feature, and more about gaining precise control over the model initialization process.

The core issue revolves around the deep integration of checkpoint management within most high-level training frameworks, such as TensorFlow, PyTorch, or deep learning libraries built on top of them. These frameworks often implement a default behavior of automatically searching for and loading the most recent checkpoint upon model creation or training initialization. This automatic process attempts to seamlessly restore the training process, including the model's weights, optimizer state, and sometimes other relevant variables. While beneficial for training resumption after unexpected interruptions, this feature becomes restrictive when specific restoration conditions are needed, or when fine-grained management of training sessions is required. Consequently, a mechanism to bypass this automatic loading procedure is essential for advanced users. The method to disable this behavior is, however, framework-specific.

For TensorFlow, the primary mechanism for managing checkpoints involves using the `tf.train.CheckpointManager` or, more recently, Keras’s `ModelCheckpoint` callback and their associated saving/loading functionalities. Automatic loading typically occurs when the `CheckpointManager` finds an existing checkpoint directory during initialization or when the Keras `ModelCheckpoint` callback restores the latest checkpoint when resuming a training process. To disable automatic loading in this context, it is important not to pass a directory that already contains saved checkpoints when you initialize the `CheckpointManager` or the `ModelCheckpoint`. Instead, the model should be built afresh, and then, if necessary, a user-defined method should be applied to load a specific checkpoint if and when it’s needed. This level of control allows for the creation of customized restore strategies, where loading is done selectively and deliberately rather than automatically. Furthermore, it is critical to avoid loading from the `latest_checkpoint` from the `tf.train.CheckpointManager` function if you intend to start training from a specific checkpoint.

Here's an example illustrating this:

```python
import tensorflow as tf

# Assume a simple model definition exists:
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        return self.dense(inputs)

# Scenario 1: Disabling automatic load when starting training from scratch
model = SimpleModel()
optimizer = tf.keras.optimizers.Adam()

# Define a CheckpointManager for saving checkpoints, but don't load anything
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
# Note: we are not restoring from `manager.latest_checkpoint`.

# Now, train the model from scratch:
# Training loop not included for brevity

# Scenario 2: Loading from a specific point if necessary
# ...Training continues or restarts later...

# Restore the model from a specific checkpoint rather than automatic loading.
specific_checkpoint_number = 2 # For instance, the second checkpoint created
checkpoint_to_restore = manager.get_checkpoint_path(specific_checkpoint_number)
if checkpoint_to_restore:
  checkpoint.restore(checkpoint_to_restore)
  print(f"Model restored from checkpoint number {specific_checkpoint_number}.")
else:
  print("Specific checkpoint not found, training from scratch.")


# Continue training with the restored model, or not, based on needs.
```

In this code, the `CheckpointManager` is initialized with the same checkpoint directory, but we do not attempt to load a prior `latest_checkpoint`. Instead, we handle the restore operation manually by examining the available checkpoints with the `get_checkpoint_path()` method and then loading the weights only when required using the `restore` method. This process ensures that automatic loading is avoided, providing full control over model loading and ensuring a training from scratch state initially. We can then choose to restore later, using an exact checkpoint if required.

In PyTorch, the situation is similar, though checkpoint loading is primarily handled using `torch.save` and `torch.load`, or using utilities like the `torch.nn.DataParallel` module. Automatic loading isn't directly a feature in the low-level saving functions. However, libraries on top of PyTorch, like PyTorch Lightning or similar training harnesses, might incorporate auto-loading features. The key with plain PyTorch or similar custom training loops is ensuring that the model and optimizer are created without immediately loading a checkpoint. This involves simply constructing the model and optimizer and not calling `torch.load` at the start of a training session. Checkpoint loading in PyTorch is explicitly called through `torch.load` and the assignment of the resulting state dictionary to the model's weight and the optimizer's state dictionary.

Here's an illustrative PyTorch example:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Assume a simple model definition exists:
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# Scenario 1: Disabling automatic load by not loading from a checkpoint on model init
model = SimpleModel()
optimizer = optim.Adam(model.parameters())
checkpoint_dir = './torch_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)


# Train the model from scratch. No loading on model/optimizer construction
# Training loop not included for brevity


# Scenario 2: Loading a specific checkpoint, with manual loading
# ...Later training continues or a restart occurs...

checkpoint_path = os.path.join(checkpoint_dir, "specific_checkpoint.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model and optimizer restored from the specific checkpoint.")
else:
    print("Specific checkpoint not found. Training will start from scratch.")

# Optional: save the specific checkpoint when desired:
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#    }, checkpoint_path)

# Continue training using the model if needed.
```

In the above PyTorch example, by not calling the `torch.load` function when the model and optimizer are initialized, we effectively disable any automatic checkpoint loading. We later explicitly load a specific checkpoint, using its name, and then proceed. It's worth noting that while plain PyTorch doesn't implicitly auto-load, training libraries build on top of it might do so, so care must be taken during initialization to check the loading parameters.

Finally, consider a case where we are using a library that is built on top of a base framework, such as a model training library that implements a custom checkpoint system. Many such libraries provide specific mechanisms to disable this behavior. Assuming a hypothetical library with a `Trainer` class and `checkpoint_dir`, the solution might look like this:

```python
class MyCustomTrainer:
  def __init__(self, model, optimizer, checkpoint_dir, auto_load_checkpoint=True):
    self.model = model
    self.optimizer = optimizer
    self.checkpoint_dir = checkpoint_dir
    self.auto_load_checkpoint = auto_load_checkpoint

    if self.auto_load_checkpoint:
      self._load_checkpoint() # Hypothetical checkpoint loader

  def _load_checkpoint(self):
    print("Attempting checkpoint load...")
    # Hypothetical logic to load latest checkpoint here.

  def train(self):
    # Training logic
    print("Training...")


model = SimpleModel()
optimizer = tf.keras.optimizers.Adam() # Assuming a TF model/optimizer here.
trainer = MyCustomTrainer(model, optimizer, "./custom_checkpoints", auto_load_checkpoint=False)
trainer.train() # Will skip automatic load, as indicated by the parameter.
```

In this conceptual example, the `MyCustomTrainer` class incorporates a boolean flag, `auto_load_checkpoint`. When set to `False`, it skips the automatic checkpoint loading behavior during initialization, giving the user the option to manage the loading process manually later. Frameworks often allow for the overriding of this feature, or give configuration parameters to control it.

In conclusion, disabling automatic checkpoint loading requires a framework-specific approach. The core methodology, however, revolves around avoiding the automatic restoration mechanisms provided by these tools. This is achieved by not passing directories that contain existing checkpoints when initializing checkpoint managers, not invoking load functions on model construction, or using flags and parameters that are available within a specific library that manages the saving/loading logic. By employing these techniques, it is possible to maintain full control over the model initialization process and enable fine-grained management of training sessions, allowing for specific restoration of checkpoints or from-scratch training scenarios as needed.

For further in-depth exploration of checkpointing, I would recommend studying the official documentation of TensorFlow and PyTorch, including sections on `tf.train.CheckpointManager`, Keras' `ModelCheckpoint` callback, `torch.save` and `torch.load`, and associated tutorials. Also, investigation of library specific documentation on trainers and save/load procedures is critical to mastering checkpoint management in a specific use case. In addition, a thorough understanding of state management in machine learning is required to fully master the complexities of checkpointing.
