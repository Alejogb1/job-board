---
title: "When should I use prepare_data vs setup in PyTorch Lightning?"
date: "2024-12-16"
id: "when-should-i-use-preparedata-vs-setup-in-pytorch-lightning"
---

Alright,  I’ve seen this confusion pop up more times than I can count, and it’s understandable. Both `prepare_data` and `setup` in PyTorch Lightning seem to deal with data, but they have distinctly different responsibilities. The key lies in understanding *when* each method is called and what resources they’re meant to handle. Think of `prepare_data` as your data *download and preprocessing stage*, while `setup` is where you *assign and instantiate* the processed data ready for training/evaluation. They're sequential, not interchangeable. I recall a particularly thorny project early in my career involving large image datasets where not understanding this distinction led to some… let’s call them *interesting* debugging sessions.

Let’s break it down step-by-step, focusing on the practicalities and avoiding overly abstract explanations.

**`prepare_data()`: The Download and Preprocessing Phase**

The core function of `prepare_data()` is, as the name suggests, to get your data *ready*. Crucially, this method is executed *only on a single process* across your entire distributed training setup. This is incredibly important for avoiding redundant operations when using multiple GPUs or machines. Consider it a global preprocessing step.

Here's what typically goes inside `prepare_data()`:

*   **Downloading Data:** If your data isn't already present, this is where you'd programmatically download it (e.g., using `torchvision.datasets` with a download flag). This helps in reproducible experiments.
*   **Data Unzipping/Extraction:** If your data comes in an archive, this is the place to unpack it.
*   **Data Preparation (but not *instantiation*):** This might include things like converting to a specific format, generating necessary files, or downloading embeddings. Note that you're *not* yet creating your actual `torch.utils.data.Dataset` instance. You’re just prepping the data for later usage. Think of it as preparing your ingredients in the kitchen, not cooking the meal yet.
*   **Saving Prepared Data to Disk:** Often, you'll want to cache the results of these potentially time-consuming operations to avoid re-running them every time you train.

It's crucial that this function is deterministic based on the provided arguments, ensuring consistent results across runs. Avoid any operations involving random number generation in this method. If you need non-deterministic processes, these are better placed elsewhere, specifically within the `setup` method, or within the `Dataset` class itself. Also, keep data loading (creating `torch.utils.data.Dataset` instances) *out* of `prepare_data()`. This is a common mistake.

**`setup()`: The Assignment and Instantiation Phase**

`setup()`, on the other hand, is called on *every process* and is executed *after* `prepare_data()` has finished. This is where you now use the data prepared by `prepare_data()` and create your actual `torch.utils.data.Dataset` instances (and dataloaders if required). It's where data processing and assignment to specific training, validation, and test datasets occur.

Here’s what you'd expect to find inside `setup()`:

*   **Dataset Instantiation:** You create your `torch.utils.data.Dataset` objects here, passing any necessary arguments including the preprocessed data paths. You will likely be using the paths you prepared in `prepare_data()`.
*   **Splitting Data:** If you are using a pre-split dataset, or need to implement your own splits (train/validation/test), this is the place to assign the respective datasets.
*   **Data Transforms:** Apply transformations specific to the training, validation, or testing data. This is where you can use augmentations for training, and apply the necessary normalization to all datasets.
*   **Dataloader Creation (Optional):** In many cases, you'll also create your `torch.utils.data.DataLoader` instances inside `setup()`. While this isn't mandatory, it's a very standard and practical setup.

Now, let’s look at some code examples to concretize these concepts:

**Example 1: Image Dataset Preparation**

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

class ImageClassifier(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = (64, 64)

    def prepare_data(self):
        # download dataset if not already available
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

        # (optionally) resize data here, before loading into datasets
        for split in ["train", "test"]:
          data_path = os.path.join(self.data_dir, f"cifar10_data_{split}")
          os.makedirs(data_path, exist_ok=True)
          # imagine a resizing process is implemented here, saving the processed data to "data_path"

    def setup(self, stage=None):
        transform = transforms.Compose([
          transforms.Resize(self.image_size),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # standard
        ])

        if stage == "fit" or stage is None: # note: if you train without validation set, stage may be none
          self.train_dataset = datasets.CIFAR10(root=os.path.join(self.data_dir), train=True, transform=transform)
          self.val_dataset = datasets.CIFAR10(root=os.path.join(self.data_dir), train=False, transform=transform)

        if stage == "test" or stage is None:
          self.test_dataset = datasets.CIFAR10(root=os.path.join(self.data_dir), train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
```

In this example, `prepare_data()` handles the download, while `setup()` instantiates and transforms the datasets.

**Example 2: Text Dataset Preparation**

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class TextClassifier(pl.LightningDataModule):
    def __init__(self, data_dir='./text_data', vocab_size=10000, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.tokenizer = None

    def prepare_data(self):
        # (Imagine) this is where you download or create your raw text files
        raw_text = ["This is the first document.", "This is the second document.", "Another short sentence"]

        # (Imagine) here is where you pre-process the text to create tokens and save it into text file,
        # where each line is a tokenized document. In our example it will only be stored in memory.
        tokens = [[word for word in doc.lower().split(" ")] for doc in raw_text]
        self.tokenizer = Tokenizer(tokens, self.vocab_size) # create tokenizer

        # (Imagine) the following code would save tokenized data to the disk
        self.tokenized_data =  [self.tokenizer.encode(doc) for doc in tokens]
        # save to self.data_dir

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TextDataset(self.tokenized_data[:2], tokenizer=self.tokenizer) # first two samples
            self.val_dataset = TextDataset(self.tokenized_data[2:], tokenizer=self.tokenizer) # third sample

        if stage == "test" or stage is None:
            self.test_dataset = TextDataset(self.tokenized_data, tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size)


class Tokenizer:
  def __init__(self, tokens, vocab_size):
    self.vocab = sorted(list(set([item for sublist in tokens for item in sublist]))) # unnested list of tokens
    self.vocab = ["<pad>", "<unk>"] + self.vocab # pad first
    self.vocab = self.vocab[:vocab_size]
    self.vocab_size = len(self.vocab)
    self.stoi = {char:i for i, char in enumerate(self.vocab)}
    self.itos = {i:char for i, char in enumerate(self.vocab)}

  def encode(self, text):
    return [self.stoi.get(c, self.stoi["<unk>"]) for c in text]
  def decode(self, ids):
    return [self.itos[id] for id in ids]

class TextDataset(Dataset):
  def __init__(self, tokenized_data, tokenizer):
      self.tokenized_data = tokenized_data
      self.tokenizer = tokenizer

  def __len__(self):
    return len(self.tokenized_data)

  def __getitem__(self, index):
    input_ids = torch.tensor(self.tokenized_data[index])
    return input_ids
```

Here, `prepare_data()` handles the raw text processing (even though our tokenizer is basic) and `setup()` constructs the actual datasets from tokenized data. We use a custom `Tokenizer` class to perform simple tokenization.

**Example 3: A Dataset without a Download Step**

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

class DummyDataModule(pl.LightningDataModule):
  def __init__(self, data_size=100, batch_size=32):
    super().__init__()
    self.data_size = data_size
    self.batch_size = batch_size
    self.dataset = None

  def prepare_data(self):
      # No data download is necessary here, this could be for datasets already generated.
      pass

  def setup(self, stage=None):
      data = np.random.rand(self.data_size, 10)
      labels = np.random.randint(0, 2, self.data_size)
      self.dataset = CustomDataset(data, labels)

  def train_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
      return DataLoader(self.dataset, batch_size=self.batch_size)

  def test_dataloader(self):
      return DataLoader(self.dataset, batch_size=self.batch_size)

class CustomDataset(Dataset):
  def __init__(self, data, labels):
    self.data = torch.tensor(data, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.int64)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index], self.labels[index]
```

In this streamlined example, `prepare_data()` does nothing, as the data is already present (generated on the fly). All data handling and dataset creation happens within `setup()`.

**Key Takeaways and Resources**

In essence, always aim to separate your *data acquisition/preparation* from your *dataset instantiations*. `prepare_data()` should be your single point of contact for making your data ready to use and should be run only once, while `setup()` is responsible for creating datasets and dataloaders.

For further information, I recommend:

*   **The PyTorch Lightning Documentation**: The official documentation is your best resource for up-to-date information, especially the sections covering DataModules.
*   **“Deep Learning with PyTorch” by Eli Stevens, Luca Antiga, and Thomas Viehmann**: This book offers a thorough and practical overview of PyTorch concepts, and its data section provides excellent guidance.
*   **“Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron**: While not specifically PyTorch, the general machine learning data handling principles explained in this book are applicable across deep learning frameworks. Understanding how to preprocess data is a critical piece of understanding how to build robust ML systems.

By adhering to this separation of concerns, your code will be more maintainable, less prone to errors, and will scale properly when using distributed training. It will also help keep your research reproducible. I hope this clarification helps, and feel free to ask further questions if any other aspect remains unclear!
