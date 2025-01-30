---
title: "Why do PyTorch Lightning tests yield different results?"
date: "2025-01-30"
id: "why-do-pytorch-lightning-tests-yield-different-results"
---
PyTorch Lightning, while streamlining PyTorch model training, introduces several abstractions that can lead to variations in test results if not handled meticulously. My experience across numerous projects utilizing it, from image segmentation to sequence modeling, has revealed that these discrepancies often stem from non-deterministic behaviors inadvertently introduced within the training and testing pipelines, rather than fundamental flaws in the library itself. Understanding these nuances is crucial for maintaining reproducible experiments.

Specifically, the inherent randomness in deep learning, when coupled with PyTorch Lightning's distributed training and data loading mechanisms, contributes significantly to the issue. Without strict control over seed values and data shuffling, slight variations can accumulate across multiple runs, leading to divergent test scores, even when using the same model architecture and hyperparameters. This issue isn't unique to PyTorch Lightning; it's a characteristic of stochastic gradient descent and related algorithms. However, the layered complexity that Lightning introduces can obscure these factors, making them harder to pinpoint.

Let's break down the common causes. PyTorch Lightning employs both `torch.random` and `numpy.random` internally, and managing seed settings is crucial for deterministic behavior. The `Trainer` in Lightning can operate on different devices (CPU, GPU, multi-GPU setups), each with its own potential source of randomness, especially with CUDA. During data loading, the `DataLoader` can incorporate randomness via its `sampler`, and if this is not controlled, it will lead to variations in the batches presented during training. If the `sampler` performs shuffling and it is not fixed, results will vary.

Additionally, consider any augmentations used in the datasets. If the augmentation functions are not deterministic, they can also contribute to different test results. This is especially relevant in image-related tasks, where transforms involving flipping or rotations can affect input data and model behavior in unpredictable ways unless controlled with seed values. If an augmentation is random and seeds are not set during test data loading, results will vary. This means data preparation and training have to adhere to the same deterministic settings.

Finally, a less obvious source of variability can stem from the data itself. If data generation involves randomness, any changes in how training and test data are obtained or preprocessed can create disparities. Similarly, the behavior of external libraries used for preprocessing, while generally stable, could have minor fluctuations. While they are not related to PyTorch Lightning itself, they can be exposed by the framework, as it often hides complex interactions.

To address these issues and achieve reproducible test results, several steps should be consistently followed. First, the global seed must be set for all random number generators before starting training. Second, deterministic behavior within PyTorch needs to be enabled by setting `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` if using CUDA. Third, for any data loading operations, shuffling should be disabled unless there's an explicit need for it, or seeds must be set in the `DataLoader`. Fourth, augmentation functions must be written to be deterministic, or deterministic versions of the library must be used, or controlled with a fixed seed.

Below are some examples and their implications:

**Example 1: Setting Random Seeds**

```python
import random
import numpy as np
import torch
import pytorch_lightning as pl

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self(x)
         loss = torch.nn.functional.mse_loss(y_hat, y)
         self.log('test_loss', loss)
         return loss
    
    def configure_optimizers(self):
         return torch.optim.Adam(self.parameters(), lr=1e-3)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def setup(self, stage = None):
         set_seed(self.seed)
         self.train_data = [(torch.randn(10), torch.randn(1)) for _ in range(100)]
         self.test_data = [(torch.randn(10), torch.randn(1)) for _ in range(20)]

    def train_dataloader(self):
         return torch.utils.data.DataLoader(self.train_data, batch_size = 10, shuffle = False)
    
    def test_dataloader(self):
         return torch.utils.data.DataLoader(self.test_data, batch_size = 10, shuffle = False)


# Set seed globally at beginning of execution.
seed = 42
set_seed(seed)
# This will still yield the same results because setup will set the seed
datamodule = MyDataModule(seed)
model = MyLightningModule()
trainer = pl.Trainer(max_epochs=2, accelerator="cpu")
trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule)

```
This example demonstrates the crucial practice of setting all random seeds at the very beginning of the program, covering `random`, `numpy`, and `torch` (including CUDA seeds). The custom `set_seed` function is called in the `DataModule` `setup` method, ensuring the data loading is always initialized with the same randomness, and at the top level of execution. The DataLoader shuffle option is set to `False`. Failure to include all these seed setters will lead to non-deterministic behavior in both training and testing, and the failure to make the dataloader deterministic during testing will lead to test result variations.

**Example 2: Disabling CUDNN Benchmarking**

```python
import random
import numpy as np
import torch
import pytorch_lightning as pl

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self(x)
         loss = torch.nn.functional.mse_loss(y_hat, y)
         self.log('test_loss', loss)
         return loss
    
    def configure_optimizers(self):
         return torch.optim.Adam(self.parameters(), lr=1e-3)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def setup(self, stage = None):
         set_seed(self.seed)
         self.train_data = [(torch.randn(10), torch.randn(1)) for _ in range(100)]
         self.test_data = [(torch.randn(10), torch.randn(1)) for _ in range(20)]

    def train_dataloader(self):
         return torch.utils.data.DataLoader(self.train_data, batch_size = 10, shuffle = False)
    
    def test_dataloader(self):
         return torch.utils.data.DataLoader(self.test_data, batch_size = 10, shuffle = False)


# Set seed globally at beginning of execution.
seed = 42
set_seed(seed)
# This will still yield the same results because setup will set the seed
datamodule = MyDataModule(seed)
model = MyLightningModule()
trainer = pl.Trainer(max_epochs=2, accelerator="gpu" if torch.cuda.is_available() else "cpu")
trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule)
```

Building upon the previous example, this code demonstrates how `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` force deterministic behavior in CUDA operations when using GPUs. While benchmarking can improve performance, it can introduce non-determinism. These flags will slow down your training, but they will ensure consistent and reproducible results. If your results vary even with a fixed seed, this is often the culprit when working with GPUs. The trainer is modified to use the GPU if one is available, otherwise the CPU is selected.

**Example 3: Deterministic Augmentations**

```python
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self(x)
         loss = torch.nn.functional.mse_loss(y_hat, y)
         self.log('test_loss', loss)
         return loss
    
    def configure_optimizers(self):
         return torch.optim.Adam(self.parameters(), lr=1e-3)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def setup(self, stage = None):
         set_seed(self.seed)
         self.train_data = [(torch.randn(10), torch.randn(1)) for _ in range(100)]
         self.test_data = [(torch.randn(10), torch.randn(1)) for _ in range(20)]
         self.transform = transforms.Compose([
              transforms.RandomHorizontalFlip(p = 1.0),
              transforms.ToTensor(),
         ])


    def train_dataloader(self):
         return torch.utils.data.DataLoader(self.train_data, batch_size = 10, shuffle = False)
    
    def test_dataloader(self):
         def transform(x, transform):
             img = Image.fromarray((x.cpu().numpy() * 255).astype(np.uint8), mode = "L")
             transformed_img = transform(img)
             return transformed_img.squeeze()
        
         test_data_transformed = [
            (transform(x, self.transform), y) for x, y in self.test_data
         ]
         return torch.utils.data.DataLoader(test_data_transformed, batch_size = 10, shuffle = False)

# Set seed globally at beginning of execution.
seed = 42
set_seed(seed)
# This will still yield the same results because setup will set the seed
datamodule = MyDataModule(seed)
model = MyLightningModule()
trainer = pl.Trainer(max_epochs=2, accelerator="gpu" if torch.cuda.is_available() else "cpu")
trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule)
```

This example incorporates a torchvision transform, namely, `RandomHorizontalFlip`, which is initially set to randomly flip the image, introducing non-determinism. To counteract this, we now use the same seed to initialize the data each time. Since this is not strictly deterministic, we modify how test data is loaded. We manually apply the transform, which is now initialized deterministically. The key idea is to control any transformations so that they always occur in the exact same way between runs. Note that we are passing the seed into the datamodule, which sets the seed during setup. This ensures the data is initialized consistently between training and testing.

For further investigation, I recommend reviewing the documentation for PyTorch Lightning's `Trainer` class, particularly the sections related to reproducibility and distributed training. The official PyTorch documentation on setting random seeds and ensuring deterministic behavior with CUDA is essential. Furthermore, understanding the specific behavior of any augmentation libraries used is crucial. I also advise paying close attention to data preprocessing steps. They can be a hidden source of variability.

In summary, achieving reproducibility in PyTorch Lightning requires meticulous attention to detail. Setting random seeds globally, disabling CUDNN benchmarking, and controlling data loading are foundational. Moreover, scrutinizing any data transformations or preprocessing steps for hidden sources of randomness is key to obtaining consistent test results. While initially challenging to fully master, these principles will ensure that your experimental results are dependable.
