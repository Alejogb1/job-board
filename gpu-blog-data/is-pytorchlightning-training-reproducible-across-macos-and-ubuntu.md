---
title: "Is PyTorch/Lightning training reproducible across macOS and Ubuntu CPU environments?"
date: "2025-01-30"
id: "is-pytorchlightning-training-reproducible-across-macos-and-ubuntu"
---
Deterministic training outcomes across diverse hardware and operating systems represent a significant challenge in deep learning, particularly when relying on CPU-based computations. My experience across numerous projects, encompassing both research and applied machine learning, has shown that achieving bit-for-bit reproducibility between macOS and Ubuntu CPU environments using PyTorch Lightning, while theoretically possible, requires careful configuration and an understanding of the underlying factors that can introduce variability.

The primary source of non-determinism stems from the inherent parallelism employed by PyTorch and its interaction with underlying system libraries. While both macOS and Ubuntu operating systems share common architectural elements, their implementations of threading, numerical operations, and random number generation can differ subtly, resulting in variations during training. Specifically, thread scheduling and the order of floating-point operations are not always guaranteed to be identical, and these seemingly minor divergences can accumulate during iterative processes like gradient descent, leading to different model parameters and ultimately different performance. PyTorch Lightning, while simplifying the training process, does not inherently circumvent these underlying issues.

The first critical element in establishing reproducibility involves setting seeds for all relevant random number generators. Both the Python random module, which PyTorch utilizes in its DataLoader, and the PyTorch random number generator itself must be initialized with identical seeds. This establishes a consistent sequence of pseudo-random numbers which, in theory, should eliminate stochasticity originating from random initialization and data loading shuffles. However, this measure alone is frequently insufficient. Additionally, one must ensure that NumPy, often used for data preprocessing or augmentation, also uses a fixed seed.

The second, and more challenging, aspect lies in controlling non-deterministic algorithms. This often encompasses the numerical computations on the CPU and any libraries that PyTorch utilizes under the hood. When operating on a CPU, operations are not always executed in a strict and pre-defined sequence, especially if multiple threads are used. PyTorch's default behavior leverages available multi-core architectures to accelerate computations by operating on tensors in a parallel fashion. However, because the precise scheduling of operations across these threads is managed by the operating system’s scheduler, the order in which calculations are carried out can be slightly different across machines. This will impact operations sensitive to the order of addition and multiplication, like sums of gradients or accumulation of losses. These operations may have minor variations in the floating-point error.

PyTorch provides the `torch.use_deterministic_algorithms(True)` context manager to minimize this specific risk by forcing deterministic execution. However, it has limitations, since not all PyTorch operations have a deterministic implementation, and it may cause certain performance degradation. It’s also pertinent to note that this does not control system-level non-determinism.

In addition to the numerical and multi-threading variability, the versions of all components within the software stack must be identical. The version of Python, PyTorch, PyTorch Lightning, CUDA (even if using CPU), any associated libraries, and even the installed operating system version can introduce inconsistencies. In several of my prior attempts, I have observed variation introduced purely by upgrading a library dependency or the operating system. Containerization with tools like Docker addresses many of these version-related issues by encapsulating the entire environment.

Below, I will illustrate several code examples that demonstrate the strategies described above for pursuing reproducibility.

**Example 1: Basic Seed Setting and Deterministic Algorithms**

This example illustrates how to set seeds for PyTorch, NumPy, and the Python random module, as well as enforce deterministic operations via `torch.use_deterministic_algorithms(True)`. Note that despite these precautions, perfect reproducibility is not always guaranteed due to the operating system or hardware specifics discussed above.

```python
import torch
import numpy as np
import random
import pytorch_lightning as pl

# Seed the random number generators
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Ensure deterministic algorithms
torch.use_deterministic_algorithms(True)


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)
    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = (y_hat - torch.ones_like(y_hat)).pow(2).mean()
        return loss
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


# Generate Random Input Data
input_tensor = torch.rand(10, 10)


# Instantiate and train the model
model = SimpleModel()
trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, deterministic=True)  # Explicitly specify CPU
trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader(torch.rand(10, 10)))
print(f"Model Weight: {model.linear.weight}")

```

This example performs rudimentary training using randomly generated data and prints the trained weights of a linear layer. With the seeds and deterministic mode set, the initial weights and training process should ideally yield identical results across different runs on the same machine.

**Example 2: Data Loading and DataLoader Reproducibility**

Reproducibility extends to the data loading process. Data loaders can introduce variations if not properly seeded, particularly when shuffling is involved. This example illustrates how to explicitly control data loader randomness.

```python
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Seeds are applied as above.
SEED = 42
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


class RandomDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# Create Data Loader
train_dataset = RandomDataset(100)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True, # Shuffling can lead to variations if not handled carefully.
    generator=torch.Generator().manual_seed(SEED) #  Seed the dataloader directly.
)

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)
    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = (y_hat - torch.ones_like(y_hat)).pow(2).mean()
        return loss
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

model = SimpleModel()
trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, deterministic=True)  # Explicitly specify CPU
trainer.fit(model, train_dataloaders=train_dataloader)
print(f"Model Weight after dataset: {model.linear.weight}")

```

Here the `DataLoader` has a seed applied through `generator=torch.Generator().manual_seed(SEED)` ensuring consistent order of data being fetched by the dataloader during each epoch.

**Example 3: Addressing External Library Variability**

This example illustrates that external numerical libraries may impact reproducibility. Even when all PyTorch elements are controlled, inconsistencies might appear if a specific transformation (e.g., custom augmentation) uses external libraries like `scipy`. The general principle is that if one wants to make an external call more reproducible, they need to seed the random number generators of their external dependencies.

```python
import torch
import numpy as np
import random
import scipy
import pytorch_lightning as pl

# Apply global seeds.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True)

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)
    def training_step(self, batch, batch_idx):
        transformed_batch = self.apply_scipy_transformation(batch)
        y_hat = self(transformed_batch)
        loss = (y_hat - torch.ones_like(y_hat)).pow(2).mean()
        return loss
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
    def apply_scipy_transformation(self, batch):
        # This function demonstrates the need to seed dependencies. This simple example performs random permutation.
        transformed = []
        for data in batch:
            new_data = scipy.random.permutation(data.numpy())
            transformed.append(torch.tensor(new_data))
        return torch.stack(transformed)


model = SimpleModel()
trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, deterministic=True)  # Explicitly specify CPU
trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader(torch.rand(10, 10)))
print(f"Model Weight after scipy: {model.linear.weight}")
```

Here we use the scipy library, and we see that despite setting the global seeds, it is necessary to set scipy's random generator's seed as well. In this case, we perform a permutation, that should produce the same outcome when the seeds are consistent.

**Resource Recommendations**

To delve deeper into this topic, I advise studying the documentation for PyTorch concerning deterministic behavior (`torch.use_deterministic_algorithms`) and PyTorch Lightning documentation related to training reproducibility. Consulting research papers or books specializing in numerical computing and the effect of floating-point arithmetic can further illuminate the inherent complexities involved. Finally, the official documentation for Python's `random` module and NumPy’s random number generation is pertinent to understand seeding implications. It is vital to emphasize that absolute byte-for-byte reproducibility is challenging, and one must accept a certain degree of variance, even with best practices implemented. Focus should therefore shift to achieving functional reproducibility – that is, ensuring similar performance and trained model behavior rather than identical intermediate results.
