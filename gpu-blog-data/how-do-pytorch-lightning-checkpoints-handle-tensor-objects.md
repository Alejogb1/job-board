---
title: "How do PyTorch Lightning checkpoints handle Tensor objects?"
date: "2025-01-30"
id: "how-do-pytorch-lightning-checkpoints-handle-tensor-objects"
---
PyTorch Lightning’s checkpointing mechanism does not directly store Tensor objects as they exist in active memory. Instead, it captures the state dict of the model, optimizers, and optionally, other user-defined components, and serializes these state dicts to disk. This approach is crucial for portability, allowing trained models to be loaded and resumed on different hardware or across different Python processes, independent of the specific memory locations of Tensor objects during training.

The fundamental unit of a PyTorch model’s parameters are, of course, Tensors. However, these Tensors are managed within PyTorch as part of the model’s internal structure – typically, they are contained within `torch.nn.Parameter` objects, which themselves are held within `torch.nn.Module` instances. These modules, when part of a larger architecture, are often nested. Consequently, simply storing Tensor objects directly would be insufficient for preserving the state of the network; the architecture and the relationships between Tensors also need to be preserved.

The `state_dict()` method is central to how checkpointing is achieved. This method, implemented for both `torch.nn.Module` objects and `torch.optim.Optimizer` objects, produces an ordered dictionary that maps parameter names to their corresponding Tensors, or their associated values in the case of optimizers (like learning rates and momentum buffers). When PyTorch Lightning saves a checkpoint, it calls the `state_dict()` of the model, optimizers, and any other necessary component passed via the `checkpoint_callback`, and then serializes the dictionary using a library such as pickle or torch.save. The serialization is what makes the state transportable across environments, as the individual Tensors are effectively converted into serialized data that can be recreated into the identical parameter and optimizer states at a later time.

The process is reversed when loading a checkpoint: the serialized data is deserialized, typically back into a dictionary, and then loaded into the relevant objects via the `load_state_dict()` method. This method iterates through the items in the deserialized state dictionary, assigning Tensor data to the corresponding parameters of the model and also updating optimizer parameters as needed.

It's important to recognize that while the *data* of the Tensor is preserved through the `state_dict` and serialization process, the *memory location* of the original Tensor object is never stored nor recovered. The Tensor object, once loaded, is a new object, albeit holding the same numerical data and metadata that existed before serialization. This behavior is desirable as it allows for model loading on different hardware or with different batch sizes without complications arising from incompatible memory configurations.

Consider this concrete example. I often build models that employ a transformer architecture encapsulated in a custom module.

```python
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(100, d_model) # Simulating input embeddings
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.output_layer = nn.Linear(d_model, 10) # For a 10-class classification problem


    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.output_layer(x.mean(dim=1)) # Simple mean pooling
        return x

class LightningModel(pl.LightningModule):
    def __init__(self, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        self.transformer = SimpleTransformer(d_model, nhead, num_layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.transformer(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
       return Adam(self.parameters(), lr=1e-3)
```

This code defines a simple transformer model and wraps it in a LightningModule. A key takeaway here is that the `state_dict` method, inherited from `nn.Module`, will capture all of the necessary information about the parameters of the `SimpleTransformer` instance nested within `LightningModel`. For instance, it includes the weights and biases of the `nn.Embedding`, `nn.TransformerEncoder`, and `nn.Linear` layers. It also handles the optimizer configuration in `configure_optimizers()`.

Here is how PyTorch Lightning manages checkpointing, demonstrating the `state_dict` and how it is handled on the model object level.

```python
    model = LightningModel()
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=0.1, default_root_dir='checkpoints')

    # Dummy Data for the training.
    dummy_x = torch.randint(0,100, (16, 20))
    dummy_y = torch.randint(0, 10, (16,))

    trainer.fit(model, [[dummy_x, dummy_y]])

    # Checkpoint has been saved during training. We can now load the model.
    checkpoint_path = trainer.checkpoint_callback.best_model_path
    loaded_model = LightningModel.load_from_checkpoint(checkpoint_path)

    # We can confirm that the parameters are the same, using the equality check
    # of parameter values directly.
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), loaded_model.named_parameters()):
         print(f"Checking parameters for {name1}, are parameters equal: {torch.all(param1 == param2)}")

```

In the above snippet, after training, the `Trainer` saves a checkpoint automatically. Using `LightningModel.load_from_checkpoint()` loads the model state from the saved file into the new instance of `LightningModel`. The comparison loop verifies that the parameter values within both models are equivalent. Crucially, the parameters in `loaded_model` are not the same Tensor objects that were in `model` in the original training session – they are newly instantiated tensors populated with the data recovered from the checkpoint.

Finally, let’s look at a slightly modified example that demonstrates how user-defined objects can be included in the checkpoint, using `on_save_checkpoint` and `on_load_checkpoint` hooks.

```python
class ExtendedLightningModel(LightningModel):
     def __init__(self, d_model=512, nhead=8, num_layers=3):
        super().__init__(d_model, nhead, num_layers)
        self.custom_state = {"epoch_at_checkpoint": 0, "my_counter":0}

     def on_save_checkpoint(self, checkpoint):
        checkpoint["custom_state"] = self.custom_state
        print(f"Saving custom state at epoch {self.current_epoch} with counter {self.custom_state['my_counter']}")

     def on_load_checkpoint(self, checkpoint):
         self.custom_state = checkpoint["custom_state"]
         print(f"Loading custom state at epoch {self.current_epoch} with counter {self.custom_state['my_counter']}")

     def training_step(self, batch, batch_idx):
         loss = super().training_step(batch, batch_idx)
         self.custom_state["my_counter"] += 1 # Simulate changes between training steps
         return loss
```
In this extended version, a custom state dictionary is incorporated.  `on_save_checkpoint` adds this dictionary to the checkpoint before it is saved, and the corresponding `on_load_checkpoint` restores it after the checkpoint is loaded, maintaining state between training runs.  This illustrates that, while raw Tensor objects are managed via their parameter data within the model, arbitrary other data can also be integrated using these methods.

For further study, I would suggest consulting the official PyTorch documentation for thorough explanation of the `torch.nn.Module` and `torch.optim.Optimizer` classes. The PyTorch Lightning documentation is invaluable for understanding how the framework handles checkpointing. I have also found academic publications on deep learning best practices to give context for how this is used in a more involved model training situation. Moreover, practicing by implementing checkpointing in several real projects and exploring the behavior of `state_dict()` is one of the more effective ways to gain familiarity.
