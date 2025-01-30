---
title: "How can PyTorch DataLoader be used to create batches where all examples share an attribute value?"
date: "2025-01-30"
id: "how-can-pytorch-dataloader-be-used-to-create"
---
Data loading in machine learning often requires specific structural constraints beyond simple batching. I’ve frequently encountered situations where processing efficiency and model performance rely on batches containing examples sharing a common categorical attribute. Directly utilizing the default PyTorch `DataLoader` functionality proves inadequate for this task; it indiscriminately shuffles and batches data, disregarding any inherent similarities. This requires a custom approach integrating data indexing and careful sample selection.

The core challenge stems from the `DataLoader`'s reliance on sequential or random sample selection. To create batches grouped by a shared attribute, we must first establish an index that maps attribute values to their corresponding dataset indices. Then, instead of directly passing the entire dataset to the `DataLoader`, we feed it a custom `Sampler` that intelligently selects indices based on the attribute's value.

Here's the breakdown of the solution, starting with index construction:

1.  **Index Creation:** We first iterate over the dataset and create a Python dictionary where keys are the unique attribute values and values are lists of indices pointing to data samples exhibiting that attribute value. This index allows us to efficiently locate all examples within the dataset that share a specific attribute.

2.  **Custom Sampler:** A custom `torch.utils.data.Sampler` class is then implemented. This sampler receives the index dictionary and, for each batch, randomly selects an attribute value. From that attribute's list of indices, it randomly samples a desired number of examples, which forms the batch’s indices. This ensures each batch is homogenous in terms of the shared attribute.

3. **DataLoader Integration:**  Finally, a standard `DataLoader` is created, but it is passed the custom `Sampler` instance instead of relying on default batching. This allows it to utilize the sampler’s intelligent index selection logic.

This approach provides a flexible framework suitable for various scenarios. It's also worth highlighting that data loading speed is impacted due to the index sampling operation, and I've found profiling the custom sampler performance to be crucial when using very large datasets.

Below are three example implementations illustrating progressively complex attribute-based batching:

**Example 1: Basic Attribute Grouping**

This simplest example groups data samples according to one categorical attribute, such as the class label in image classification.  I will demonstrate this with a fictional data structure.

```python
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random


class DummyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class AttributeSampler(Sampler):
    def __init__(self, dataset, attribute_idx, batch_size):
        self.dataset = dataset
        self.attribute_idx = attribute_idx
        self.batch_size = batch_size
        self.attribute_index = self._build_attribute_index()
        self.num_batches = len(self.attribute_index)


    def _build_attribute_index(self):
        index = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in index:
                index[label] = []
            index[label].append(idx)
        return index

    def __iter__(self):
        attribute_values = list(self.attribute_index.keys())
        for _ in range(self.num_batches):
             selected_attribute = random.choice(attribute_values)
             indices = random.sample(self.attribute_index[selected_attribute], min(self.batch_size, len(self.attribute_index[selected_attribute])))
             yield indices



    def __len__(self):
        return self.num_batches

# Fictional Data
data = [torch.rand(10) for _ in range(100)]
labels = [random.randint(0, 4) for _ in range(100)] # 5 classes
dummy_dataset = DummyDataset(data, labels)
attribute_sampler = AttributeSampler(dummy_dataset,1, 10)
dataloader = DataLoader(dummy_dataset, batch_sampler = attribute_sampler)

for batch in dataloader:
    batch_data, batch_labels = batch
    print(f"Batch labels: {batch_labels}, Unique Labels in Batch: {torch.unique(torch.tensor(batch_labels))}")

```
In this code, `DummyDataset` simulates a dataset with data and labels. The `AttributeSampler` creates an index based on the `labels`, randomly selects a label for each batch, then retrieves indices of the corresponding data examples. Notice that the DataLoader is initialized with a `batch_sampler` argument rather than `batch_size`. The output demonstrates that each batch contains examples exclusively from one class.

**Example 2: Batches Based on Multiple Attributes**

Situations may demand grouping by multiple attributes, such as object type and lighting condition. This is achieved by creating a composite index key from these attributes.

```python
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random

class DummyDataset(Dataset):
    def __init__(self, data, attributes):
        self.data = data
        self.attributes = attributes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.attributes[idx]

class MultiAttributeSampler(Sampler):
    def __init__(self, dataset, attribute_idx, batch_size):
        self.dataset = dataset
        self.attribute_idx = attribute_idx
        self.batch_size = batch_size
        self.attribute_index = self._build_attribute_index()
        self.num_batches = len(self.attribute_index)


    def _build_attribute_index(self):
        index = {}
        for idx, (_, attributes) in enumerate(self.dataset):
            key = tuple(attributes)  # Create a tuple from attributes for multi-attribute key
            if key not in index:
                index[key] = []
            index[key].append(idx)
        return index

    def __iter__(self):
        attribute_values = list(self.attribute_index.keys())
        for _ in range(self.num_batches):
             selected_attribute = random.choice(attribute_values)
             indices = random.sample(self.attribute_index[selected_attribute], min(self.batch_size, len(self.attribute_index[selected_attribute])))
             yield indices

    def __len__(self):
        return self.num_batches

# Fictional data with multiple attributes
data = [torch.rand(10) for _ in range(100)]
attributes = [ (random.randint(0, 2), random.choice(['day', 'night'])) for _ in range(100) ]
dummy_dataset = DummyDataset(data, attributes)

attribute_sampler = MultiAttributeSampler(dummy_dataset,1,10)
dataloader = DataLoader(dummy_dataset, batch_sampler=attribute_sampler)


for batch in dataloader:
    batch_data, batch_attributes = batch
    unique_attributes = set(batch_attributes)
    print(f"Unique attributes in batch: {unique_attributes}")

```
In this variant,  `attributes` is a list of tuples with each tuple containing multiple attributes of the data item. The index now keys on the tuple of attributes, ensuring that all data in a batch share all the listed attribute values.

**Example 3: Variable Batch Sizes Per Attribute**

Sometimes, data availability per attribute varies. It's important to handle cases where there might be very few examples for a category, potentially resulting in smaller batches. This final example demonstrates how batch size can be dynamic, limited by the smallest number of samples present for a given category within that batch.

```python
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random


class DummyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class VariableBatchSampler(Sampler):
    def __init__(self, dataset, attribute_idx, max_batch_size):
        self.dataset = dataset
        self.attribute_idx = attribute_idx
        self.max_batch_size = max_batch_size
        self.attribute_index = self._build_attribute_index()
        self.num_batches = len(self.attribute_index)

    def _build_attribute_index(self):
        index = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in index:
                index[label] = []
            index[label].append(idx)
        return index

    def __iter__(self):
        attribute_values = list(self.attribute_index.keys())
        for _ in range(self.num_batches):
            selected_attribute = random.choice(attribute_values)
            available_count = len(self.attribute_index[selected_attribute])
            batch_size = min(self.max_batch_size, available_count) # Dynamic batch size
            indices = random.sample(self.attribute_index[selected_attribute], batch_size)
            yield indices

    def __len__(self):
        return self.num_batches


# Fictional data with imbalanced attribute distribution
data = [torch.rand(10) for _ in range(100)]
labels = [0] * 10 + [1] * 30 + [2] * 60  # Imbalanced classes
dummy_dataset = DummyDataset(data, labels)

attribute_sampler = VariableBatchSampler(dummy_dataset, 1, 20)
dataloader = DataLoader(dummy_dataset, batch_sampler=attribute_sampler)

for batch in dataloader:
    batch_data, batch_labels = batch
    batch_size = len(batch_data)
    print(f"Batch labels: {batch_labels}, Batch Size: {batch_size}")
```
Here, `VariableBatchSampler` dynamically determines the actual batch size, which may differ from the `max_batch_size`, adapting to the number of samples available for the chosen attribute, handling imbalanced datasets effectively.

In summary, constructing custom `Sampler` classes offers fine-grained control over how data is batched. Resource recommendations should include careful review of the PyTorch documentation for `torch.utils.data.Dataset`, `torch.utils.data.Sampler`, and `torch.utils.data.DataLoader`. Furthermore, understanding Python data structures and random number generation is key to implementing these techniques. The techniques discussed provide a solid framework for controlling batch formation and data loading with the `DataLoader`, beyond the standard defaults, and have proven essential in my previous work for specific training requirements.
