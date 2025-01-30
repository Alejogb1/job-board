---
title: "How can I efficiently load batches of data to a GPU using a BERT model in PyTorch?"
date: "2025-01-30"
id: "how-can-i-efficiently-load-batches-of-data"
---
Transferring large datasets to a GPU for BERT model training or inference is often a bottleneck. The common practice of loading an entire dataset into memory and then moving it to the GPU can lead to substantial delays, especially with datasets exceeding available GPU RAM. Efficient batch loading, therefore, becomes essential for optimal throughput.

My experience stems from several large-scale NLP projects involving BERT and its variants. One particularly memorable project involved processing a multi-terabyte text corpus for a document classification task. Initial attempts at loading the whole dataset into memory predictably crashed, highlighting the critical need for an optimized batch loading strategy. We significantly improved performance through meticulous pipeline construction based on PyTorch’s `DataLoader`, utilizing a combination of custom dataset classes and memory-mapping techniques.

The core concept relies on constructing a pipeline where data is loaded, preprocessed, and transferred to the GPU in small, manageable chunks. This pipeline avoids loading the entire dataset into RAM at once. Instead, the `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` classes are used to abstract the data source and perform batch iteration.

The `Dataset` class must inherit from `torch.utils.data.Dataset` and implement two crucial methods: `__len__` which returns the size of the dataset, and `__getitem__` which takes an index and returns the corresponding data item. A basic implementation would handle data loading and preprocessing at this stage. However, this can still load the data sequentially and be too slow if you use something like a text file.

To address this, memory-mapping libraries like `memmap` in NumPy can be used with the `Dataset` class. With memory mapping, files remain stored on disk, but are treated as if they are in RAM, avoiding actual bulk loads to memory until required. The `__getitem__` method then reads relevant slices of the mapped file on demand, making the process much faster.

The `DataLoader`, in turn, iterates over the `Dataset`, generating batches of data. Multiple worker processes can be used in the `DataLoader` through the `num_workers` argument to achieve parallel loading and preprocessing. Crucially, the `DataLoader` can pin memory to page-locked RAM using the `pin_memory=True` argument before transferring data to the GPU. This often improves transfer times by avoiding extra overhead and is recommended when copying tensors to a CUDA device. Finally, data tensors are moved to the GPU using the `.to(device)` method inside the training loop.

**Code Examples**

**Example 1: Basic Dataset and DataLoader**

This demonstrates a simple dataset that loads text from a list of strings and then tokenizes it using a basic tokenizer. It is not optimal but serves as a baseline.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
      text = self.texts[idx]
      encoding = self.tokenizer(text, 
                                 add_special_tokens=True,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
      return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts = ["This is a first sentence.", "Here is another one.", "And yet another example."]
    max_length = 128
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_input_ids, batch_attention_mask in dataloader:
      batch_input_ids = batch_input_ids.to(device)
      batch_attention_mask = batch_attention_mask.to(device)
      print("Input IDs batch shape:", batch_input_ids.shape)
      print("Attention mask batch shape:", batch_attention_mask.shape)

```
*Commentary:* This example creates a basic `Dataset` class that initializes with a list of texts, a tokenizer, and a maximum length. Inside `__getitem__`, the text is tokenized and returned with its attention mask. The main section initializes this dataset, creates a `DataLoader`, and iterates through it to show batch processing and basic GPU transfer. `squeeze()` is used to remove the batch dimension before transfer since it is created by the tokenizer.
    
**Example 2: Memory-Mapped Dataset with Custom Tokenization**
 This provides a more efficient implementation using memory-mapped files for loading text.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer
import os

class MemmapTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_path = file_path
        # Assumes each line in file represents one text example
        with open(self.file_path, 'r') as f:
            self.num_samples = len(f.readlines())  
        self.memmap = np.memmap(self.file_path, dtype='S', mode='r')
        self.line_starts = self._get_line_starts()


    def _get_line_starts(self):
         starts = [0]
         with open(self.file_path, 'r') as file:
           offset = 0
           for line in file:
            offset += len(line.encode('utf-8'))
            starts.append(offset)
         return starts
    

    def __len__(self):
        return self.num_samples
    

    def __getitem__(self, idx):
        start_idx = self.line_starts[idx]
        end_idx = self.line_starts[idx+1]
        raw_text = self.memmap[start_idx:end_idx].tobytes().decode('utf-8').strip()

        encoding = self.tokenizer(raw_text,
                                  add_special_tokens=True,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

if __name__ == '__main__':
    # Create dummy data file for demo
    dummy_file_path = 'dummy_data.txt'
    with open(dummy_file_path, 'w') as f:
        f.write("This is text one.\n")
        f.write("This is text two, a bit longer.\n")
        f.write("Short text.\n")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128
    dataset = MemmapTextDataset(dummy_file_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_input_ids, batch_attention_mask in dataloader:
         batch_input_ids = batch_input_ids.to(device)
         batch_attention_mask = batch_attention_mask.to(device)
         print("Input IDs batch shape:", batch_input_ids.shape)
         print("Attention mask batch shape:", batch_attention_mask.shape)
    os.remove(dummy_file_path)

```

*Commentary:* This enhanced example employs `np.memmap` to read from a file and store a view on the file without loading it entirely into RAM. Instead, `_get_line_starts` gets the byte offset for each line of the file to index the memmap object. In `__getitem__`, byte ranges are loaded as necessary. The `DataLoader` now includes `num_workers` to increase loading speed and `pin_memory=True` to optimize transfers to the GPU. A dummy text file is created and deleted, as well.

**Example 3: Incorporating a Custom Collate Function**

This example illustrates the use of a custom `collate_fn` to handle variable-length sequences when the maximum length is not suitable for every sequence.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os


class VariableTextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
      self.tokenizer = tokenizer
      self.file_path = file_path
      with open(self.file_path, 'r') as f:
          self.num_samples = len(f.readlines())
      self.memmap = np.memmap(self.file_path, dtype='S', mode='r')
      self.line_starts = self._get_line_starts()

    def _get_line_starts(self):
         starts = [0]
         with open(self.file_path, 'r') as file:
           offset = 0
           for line in file:
            offset += len(line.encode('utf-8'))
            starts.append(offset)
         return starts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
      start_idx = self.line_starts[idx]
      end_idx = self.line_starts[idx+1]
      raw_text = self.memmap[start_idx:end_idx].tobytes().decode('utf-8').strip()
      encoding = self.tokenizer(raw_text, add_special_tokens=True, return_tensors='pt')
      return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()


def variable_length_collate_fn(batch):
  input_ids = [item[0] for item in batch]
  attention_masks = [item[1] for item in batch]

  input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
  attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

  return input_ids_padded, attention_masks_padded

if __name__ == '__main__':
    dummy_file_path = 'dummy_data_variable.txt'
    with open(dummy_file_path, 'w') as f:
        f.write("This is a short one.\n")
        f.write("This is a much longer text sentence.\n")
        f.write("A very short sentence.\n")
        f.write("This sentence is medium.\n")
        f.write("Another long text sentence, for testing.\n")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = VariableTextDataset(dummy_file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=variable_length_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_input_ids, batch_attention_mask in dataloader:
      batch_input_ids = batch_input_ids.to(device)
      batch_attention_mask = batch_attention_mask.to(device)
      print("Input IDs batch shape:", batch_input_ids.shape)
      print("Attention mask batch shape:", batch_attention_mask.shape)
    os.remove(dummy_file_path)
```

*Commentary:* This example uses `pad_sequence` within a custom `collate_fn` function to pad input sequences to the maximum length in each batch. This is beneficial when handling datasets with varying sequence lengths. This avoids unnecessary padding and speed up operations.  A dummy text file is created and deleted here, as well.

**Resource Recommendations**

For deepening understanding of these techniques, I recommend exploring the PyTorch documentation regarding `torch.utils.data`, specifically the sections covering `Dataset`, `DataLoader`, and custom collate functions.  For memory mapping, the NumPy documentation’s information on `memmap` is useful.  Additionally, several NLP frameworks' documentation, like Hugging Face’s Transformers, provide extensive examples and tutorials on effectively using `DataLoader` with text data.   Finally, some community-driven resources may be valuable for finding different code examples and discussions on these topics.
