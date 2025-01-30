---
title: "When should I use `prepare_data` vs `setup` in PyTorch Lightning?"
date: "2025-01-30"
id: "when-should-i-use-preparedata-vs-setup-in"
---
The core distinction between `prepare_data` and `setup` in PyTorch Lightning hinges on the lifecycle of data preparation and its relation to the training process's reproducibility.  `prepare_data` is designed for operations that should only run once, regardless of the number of trainers or devices involved, while `setup` is for per-process initialization that can be repeated across multiple processes.  This seemingly subtle difference significantly impacts data handling, especially in distributed training scenarios.  My experience debugging distributed training failures across diverse hardware configurations highlights this crucial aspect.

**1. Clear Explanation**

`prepare_data` executes only once across all processes involved in training. Its primary purpose is to perform potentially time-consuming and non-reproducible operations, such as downloading datasets from remote sources, pre-processing massive files which are too large to easily replicate across nodes, or creating highly computationally intensive features from raw data. This ensures that this work isn't redundantly performed across multiple GPUs or nodes. The method receives only the `trainer` object as an argument.  Therefore, it cannot access information that is specific to a particular process, like its GPU ID or rank within a distributed setup. This constraint ensures idempotency; repeated calls must produce the same result.

In contrast, `setup` executes once per process.  This means it will run multiple times in a distributed training setting – once per GPU or node.  This function receives the `stage` argument (`"fit"`, `"validate"`, `"test"`, `"predict"`) which enables stage-specific data manipulations. Unlike `prepare_data`, it has access to the `trainer` object, allowing interaction with aspects specific to the current process, such as setting up dataloaders with specific batch sizes tailored for each device or loading only a subset of data pertinent to the current process.  This offers greater flexibility in managing data specific to a single training process but necessitates ensuring that its operations are consistent and repeatable for each process.

Failure to appreciate this difference often leads to unexpected behavior, especially when dealing with distributed training or model checkpointing. For instance, if you inadvertently use `setup` for downloading a dataset, each process will download the dataset independently, leading to increased training time and potential storage issues. Conversely, placing code that requires process-specific configuration (e.g., setting up dataloaders with unique indices) within `prepare_data` would lead to incorrect behavior.


**2. Code Examples with Commentary**

**Example 1: Using `prepare_data` for dataset download and preprocessing:**

```python
import pytorch_lightning as pl
import torch
import os
from urllib.request import urlretrieve

class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_url):
        super().__init__()
        self.data_url = data_url
        self.data_dir = "data"

    def prepare_data(self):
        # Download dataset if it doesn't exist. This is done only once.
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print("Downloading data...")
            urlretrieve(self.data_url, os.path.join(self.data_dir, "dataset.txt"))
            print("Data downloaded successfully.")

        # Perform computationally intensive preprocessing steps.  This is done only once.
        print("Preprocessing data...")
        # ... (complex preprocessing of dataset.txt)...

    def setup(self, stage=None):
        # Load preprocessed data into PyTorch Dataset objects
        # ... (load dataset into training, validation, and test datasets)...

    # ... (other methods: train_dataloader, val_dataloader, test_dataloader, etc.) ...
```

Here, the dataset is downloaded and preprocessed in `prepare_data`, leveraging the single execution guarantee. The actual PyTorch Datasets are created in `setup`, allowing for process-specific configuration if needed.


**Example 2: Using `setup` for DataLoader configuration:**

```python
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

class MyDataModule(pl.LightningDataModule):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TensorDataset(torch.tensor(self.data['train'][0]), torch.tensor(self.data['train'][1]))
        if stage == 'validate' or stage is None:
            self.val_dataset = TensorDataset(torch.tensor(self.data['val'][0]), torch.tensor(self.data['val'][1]))
        if stage == 'test' or stage is None:
            self.test_dataset = TensorDataset(torch.tensor(self.data['test'][0]), torch.tensor(self.data['test'][1]))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)
```

In this example, data loading and DataLoader creation are handled within `setup`, ensuring the creation of separate dataloaders tailored to each process and stage.



**Example 3:  Illustrating an incorrect approach:**

```python
import pytorch_lightning as pl
import torch
import os

class IncorrectDataModule(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        # INCORRECT: This should be in setup
        if os.path.exists(self.data_dir):
            self.train_data = torch.load(os.path.join(self.data_dir, "train_data.pt"))
            self.val_data = torch.load(os.path.join(self.data_dir, "val_data.pt"))

    def setup(self, stage=None):
        # This is redundant and potentially incorrect due to prepare_data
        #  having already loaded the data
        pass #Should load data here if prepare_data was removed

    # ... dataloaders ...
```

This example incorrectly attempts to load data within `prepare_data`.  In a multi-process scenario, this would lead to incorrect data splits or even errors because data loading must happen per process within `setup`.



**3. Resource Recommendations**

The official PyTorch Lightning documentation.  Thorough understanding of distributed training concepts within the context of deep learning frameworks.  A solid grasp of Python's multiprocessing and concurrency models.  Familiarity with the intricacies of PyTorch's DataLoader and Dataset classes. Consulting relevant chapters in advanced deep learning textbooks which discuss efficient data handling strategies for large-scale model training.
