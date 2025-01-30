---
title: "Can PyTorch Lightning's LightningModule handle `ModuleList` or `ModuleDict`?"
date: "2025-01-30"
id: "can-pytorch-lightnings-lightningmodule-handle-modulelist-or-moduledict"
---
Yes, PyTorch Lightning's `LightningModule` can effectively manage `torch.nn.ModuleList` and `torch.nn.ModuleDict`, provided the structures are correctly incorporated within the module's lifecycle methods. I’ve utilized both extensively in building complex sequence-to-sequence models and multi-task learning architectures, and a common misunderstanding is treating them as direct replacements for lists or dictionaries without considering their implications on parameter registration and automatic optimization.

The core principle is that `ModuleList` and `ModuleDict` are specifically designed to register their contained modules within the parent module's parameter list. This allows PyTorch's autograd engine to track the gradients correctly during backpropagation, which is paramount for training. If you simply use Python lists or dictionaries, PyTorch will not be aware of those sub-modules and their parameters, resulting in them not being updated during training. When integrated inside a `LightningModule`, this behavior extends smoothly, ensuring that the automatic optimization capabilities are applied correctly to modules defined within `ModuleList` or `ModuleDict` instances.

Let's delve into practical examples. Assume we're constructing a multi-head attention layer. Instead of hand-writing and explicitly managing each attention head, we can leverage `ModuleList`:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_heads = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_heads)
        ])
        self.projection = nn.Linear(embedding_dim * num_heads, embedding_dim)

    def forward(self, x):
        head_outputs = [head(x) for head in self.attention_heads]
        concatenated_output = torch.cat(head_outputs, dim=-1)
        return self.projection(concatenated_output)


class ExampleLightningModule(pl.LightningModule):
    def __init__(self, num_heads, embedding_dim):
      super().__init__()
      self.attention_layer = MultiHeadAttentionLayer(num_heads, embedding_dim)
      self.linear = nn.Linear(embedding_dim, 10)


    def forward(self, x):
        x = self.attention_layer(x)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters())

if __name__ == '__main__':
    model = ExampleLightningModule(num_heads = 4, embedding_dim=64)
    x = torch.randn(32, 10, 64)
    y = torch.randint(0, 10, (32, 10))
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=[(x, y)])
```

In this example, `MultiHeadAttentionLayer` uses a `ModuleList` to store each attention head. Inside the `ExampleLightningModule`, this entire structure, including the parameters of each linear layer within the `ModuleList`, is recognized by Lightning's training loop. Specifically, when calling `self.parameters()` within `configure_optimizers`, the parameters of all modules nested under `attention_layer`, are returned. Consequently, Adam optimizer will adjust weights during training, updating the module weights as expected.

Contrast this with using a plain Python list:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class BrokenMultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_heads = [
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_heads)
        ] # NOTE: This is a Python list not a ModuleList
        self.projection = nn.Linear(embedding_dim * num_heads, embedding_dim)

    def forward(self, x):
        head_outputs = [head(x) for head in self.attention_heads]
        concatenated_output = torch.cat(head_outputs, dim=-1)
        return self.projection(concatenated_output)


class BrokenExampleLightningModule(pl.LightningModule):
    def __init__(self, num_heads, embedding_dim):
      super().__init__()
      self.attention_layer = BrokenMultiHeadAttentionLayer(num_heads, embedding_dim)
      self.linear = nn.Linear(embedding_dim, 10)


    def forward(self, x):
        x = self.attention_layer(x)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

if __name__ == '__main__':
    model = BrokenExampleLightningModule(num_heads = 4, embedding_dim=64)
    x = torch.randn(32, 10, 64)
    y = torch.randint(0, 10, (32, 10))
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=[(x, y)])
```

In this revised example, while the code runs, the parameters of linear layers inside `self.attention_heads` are never updated. During backpropagation, the optimizer only sees parameters defined directly under `BrokenExampleLightningModule` or those within other proper `nn.Module` instances. Consequently the `attention_heads` layer behaves essentially like a static component. This highlights the crucial difference between ordinary Python lists and `ModuleList`.

Now let’s examine `ModuleDict`. This is incredibly useful for scenarios like implementing different heads in a multi-task learning scenario, where task-specific modules may be required:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiTaskModel(nn.Module):
    def __init__(self, embedding_dim, task_dims):
      super().__init__()
      self.encoder = nn.Linear(embedding_dim, embedding_dim)
      self.task_heads = nn.ModuleDict({
        task_name: nn.Linear(embedding_dim, task_dim) 
        for task_name, task_dim in task_dims.items()
      })

    def forward(self, x, task_name):
        encoded_x = self.encoder(x)
        return self.task_heads[task_name](encoded_x)

class MultiTaskLightningModule(pl.LightningModule):
    def __init__(self, embedding_dim, task_dims):
      super().__init__()
      self.model = MultiTaskModel(embedding_dim, task_dims)

    def training_step(self, batch, batch_idx):
        x, task_name, y = batch
        y_hat = self.model(x, task_name)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log(f'{task_name}_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

if __name__ == '__main__':
    task_dims = {"task_a": 5, "task_b": 10}
    model = MultiTaskLightningModule(embedding_dim=64, task_dims=task_dims)
    x = torch.randn(32, 64)
    y_a = torch.randint(0, 5, (32,))
    y_b = torch.randint(0, 10, (32,))
    trainer = pl.Trainer(max_epochs=1)
    train_data = [
        (x, "task_a", y_a),
        (x, "task_b", y_b)
    ]
    trainer.fit(model, train_dataloaders=[train_data])
```

Here, `MultiTaskModel` stores task-specific linear layers in a `ModuleDict`. The `MultiTaskLightningModule` then uses this structured model to carry out distinct loss computations, allowing for a multi-task learning objective. The key is again, the `ModuleDict` enables the optimizer in the LightningModule to see and update all the parameters within task heads. The optimizer will correctly update the parameters of the linear layer associated with each task, because the underlying `nn.Module` classes are used with `ModuleDict` and not a regular dictionary.

In summary, both `ModuleList` and `ModuleDict` are perfectly compatible with PyTorch Lightning's `LightningModule`. They are in fact encouraged for encapsulating modular components and they greatly assist in parameter management. The essential caveat is to use these PyTorch specific structures, rather than Python native lists or dictionaries, to ensure the trainable modules are properly recognized and their parameters can be updated during the optimization process within the `LightningModule`.

For those interested in exploring these concepts further, I recommend focusing on the official PyTorch documentation, specifically the sections detailing how `nn.Module`, `nn.ModuleList`, and `nn.ModuleDict` operate. Research on the internal mechanics of PyTorch's autograd will also provide valuable insight. Deep dives into optimization algorithms and how they interact with the graph formed by your model’s structure also offer useful context. Finally, studying existing code in public repositories implementing multi-head attention and multi-task learning scenarios utilizing these tools is always beneficial. These resources should collectively provide a firm grasp of effectively utilizing these tools within PyTorch Lightning.
