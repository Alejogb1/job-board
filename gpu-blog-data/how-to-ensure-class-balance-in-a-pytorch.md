---
title: "How to ensure class balance in a PyTorch DataLoader?"
date: "2025-01-30"
id: "how-to-ensure-class-balance-in-a-pytorch"
---
Data imbalance is a pervasive issue in machine learning, frequently leading to biased models that perform poorly on underrepresented classes.  In my experience working on image classification projects involving highly skewed datasets – such as satellite imagery analysis where certain land cover types are significantly rarer than others – ensuring class balance within the PyTorch `DataLoader` is paramount for robust model training.  This necessitates a nuanced approach beyond simply shuffling the dataset.  True class balance requires stratified sampling techniques to guarantee equitable representation of each class in every batch.


**1.  Understanding the Problem and the Solution:**

A naive approach using the standard `DataLoader` with `shuffle=True` only randomizes the entire dataset. While this mitigates some biases, it doesn't guarantee proportionate class representation in each batch.  Consider a binary classification problem with 90% positive samples and 10% negative samples.  Even with shuffling, a batch might disproportionately contain positive samples, hindering effective gradient updates for the underrepresented negative class.

The solution involves employing stratified sampling. This approach divides the data into strata (subsets) based on class labels and then samples proportionally from each stratum to form batches.  This ensures each batch maintains a representative distribution of the class labels, mirroring the overall class distribution in the dataset.


**2.  Implementation Strategies:**

There are several ways to achieve stratified sampling in PyTorch.  I've found the following three approaches particularly effective, each with its own strengths and weaknesses.

**Code Example 1:  Using `torch.utils.data.SubsetRandomSampler`**

This is a straightforward method that provides good control and flexibility.  We first stratify the dataset, creating indices for each class, then use `SubsetRandomSampler` to sample from these indices proportionally.

```python
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

# Sample Data (replace with your actual data loading)
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randint(0, 2, (100,)) # Binary classification: 0 or 1

# Stratification
class_indices = [[] for _ in range(2)] # For binary classification
for i, label in enumerate(y):
    class_indices[label.item()].append(i)

# Calculate sample counts for balanced batches
batch_size = 16
samples_per_class = batch_size // len(class_indices)
remainder = batch_size % len(class_indices)

# Create SubsetRandomSamplers
samplers = []
for i in range(len(class_indices)):
    num_samples = samples_per_class + (1 if i < remainder else 0)
    sampler = SubsetRandomSampler(class_indices[i][:num_samples])
    samplers.append(sampler)

# Combine samplers (converts to a single sampler across epochs)
combined_sampler = torch.utils.data.BatchSampler(
    samplers, batch_size=batch_size, drop_last=True
)


dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_sampler=combined_sampler)

#Training Loop (Example)
for epoch in range(10):
    for batch_X, batch_y in dataloader:
        # Your training code here
        pass
```

This example first stratifies the data based on the labels and then creates a `SubsetRandomSampler` for each class. Finally, it leverages `torch.utils.data.BatchSampler` to draw a balanced number of samples from each class's sampler within each batch.  The `drop_last=True` argument ensures that all batches are of the same size.  Note: The sample data generation is for illustration and should be replaced with a proper data loading pipeline relevant to your specific dataset.


**Code Example 2:  Using a Custom Dataset Class**

A more elegant approach involves creating a custom dataset class that handles stratification during data retrieval.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class StratifiedDataset(Dataset):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.class_counts = {}
        for label in y:
            self.class_counts[label.item()] = self.class_counts.get(label.item(), 0) + 1
        self.class_indices = {label: [] for label in self.class_counts}
        for i, label in enumerate(y):
            self.class_indices[label.item()].append(i)

        # for ease of calculations, we'll limit samples per class here; a more advanced implementation could handle uneven class sizes more robustly
        self.samples_per_class = batch_size // len(self.class_counts)
        self.num_batches = min(len(i) for i in self.class_indices.values()) // self.samples_per_class

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        batch = []
        for label, indices in self.class_indices.items():
            start = (idx // self.batch_size) * self.samples_per_class + (idx % self.samples_per_class)
            end = start + self.samples_per_class
            sample_indices = indices[start:end]
            batch.extend(sample_indices)

        batch_X = self.X[batch]
        batch_y = self.y[batch]
        return batch_X, batch_y

#Sample data (same as Example 1)
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
batch_size = 16
dataset = StratifiedDataset(X, y, batch_size)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Training Loop (Example)
for epoch in range(10):
  for batch_X, batch_y in dataloader:
    # Your training code here
    pass

```

This method integrates stratification directly into the dataset class, providing a cleaner and more encapsulated solution. The `__getitem__` method retrieves balanced batches directly.  This approach is particularly advantageous for larger datasets where generating a full list of indices beforehand is less efficient.


**Code Example 3:  Leveraging Imbalanced-learn Library (external library)**

For more advanced scenarios with complex class distributions and a need for more sophisticated sampling techniques, the `imbalanced-learn` library offers powerful tools.  While not strictly part of PyTorch, it seamlessly integrates.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
#or other techniques like SMOTE

# Sample Data (same as Example 1)
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Undersampling or Oversampling
sampler = RandomUnderSampler(random_state=42) #or RandomOverSampler()
X_resampled, y_resampled = sampler.fit_resample(X.numpy(), y.numpy()) #Requires numpy conversion

# Convert back to PyTorch tensors
X_resampled = torch.tensor(X_resampled, dtype=torch.float32)
y_resampled = torch.tensor(y_resampled, dtype=torch.int64)

dataset = TensorDataset(X_resampled, y_resampled)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training Loop (Example)
for epoch in range(10):
  for batch_X, batch_y in dataloader:
    # Your training code here
    pass
```

This approach uses `imbalanced-learn` for either undersampling the majority class or oversampling the minority class before creating the `DataLoader`.  This is suitable for situations where severe imbalance needs correction before batching.  Note that undersampling might lose valuable information, while oversampling can lead to overfitting if not carefully managed.  The library provides more advanced techniques like SMOTE (Synthetic Minority Over-sampling Technique) for more nuanced handling of class imbalances.


**3. Resource Recommendations:**

The PyTorch documentation, specifically sections on `DataLoader` and its customizability, is essential.  Consult textbooks on machine learning and its practical applications which will typically cover resampling techniques in detail. Explore documentation on the `imbalanced-learn` library for more sophisticated sampling strategies.  Finally, examining research papers on handling imbalanced datasets within the context of your specific application domain will provide further insights.
