---
title: "How can I split a dataset in PyTorch while preserving class ratios?"
date: "2025-01-30"
id: "how-can-i-split-a-dataset-in-pytorch"
---
Maintaining class distribution parity during dataset splitting is crucial for avoiding biased model training in machine learning.  I've encountered this challenge numerous times while working on image classification projects involving imbalanced datasets – particularly those dealing with rare pathologies in medical imaging.  A naive split risks over-representing common classes in the training set and under-representing less frequent ones, leading to poor generalization performance on unseen data, especially concerning the minority classes.  The optimal approach involves stratified sampling, ensuring each subset reflects the original dataset's class proportions.


**1.  Clear Explanation of Stratified Dataset Splitting**

Stratified splitting guarantees proportional representation of each class across all subsets (training, validation, testing).  This is achieved by first grouping data points by class label, then randomly sampling a specified percentage from each class group. This ensures that if, for example, class A represents 10% of your dataset, then approximately 10% of your training, validation, and testing sets will also consist of class A samples.

The standard `random.sample` function won't directly achieve this; it samples randomly from the entire dataset without considering class labels.  Therefore, we need a more sophisticated approach that explicitly handles class-based partitioning.  Efficient implementations often leverage the `scikit-learn` library's `train_test_split` function with the `stratify` parameter.  This function internally performs stratified sampling based on the provided labels.  However, for enhanced control and understanding, a custom implementation can be more informative.

**2. Code Examples with Commentary**

**Example 1: Using `scikit-learn`'s `train_test_split`**

This is the most straightforward approach, leveraging the power of a well-tested library function:

```python
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# Assume 'data' is your PyTorch tensor of features and 'labels' is a NumPy array of class labels
data = torch.randn(1000, 10)  # 1000 samples, 10 features
labels = np.random.randint(0, 3, 1000)  # 1000 labels, 3 classes

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Convert back to PyTorch tensors if needed
train_data = torch.tensor(train_data)
test_data = torch.tensor(test_data)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")
# Verify class distribution – proportions should be similar across train and test sets.
print(f"Training label distribution: {np.bincount(train_labels.numpy())}")
print(f"Testing label distribution: {np.bincount(test_labels.numpy())}")

```

This code demonstrates a simple 80/20 split.  The `random_state` ensures reproducibility.  Crucially, `stratify=labels` instructs `train_test_split` to perform stratified sampling based on the `labels` array.


**Example 2:  Manual Stratified Splitting**

This example illustrates a manual implementation, providing greater control over the process and a deeper understanding of its mechanics:


```python
import torch
import numpy as np
from collections import defaultdict

def stratified_split(data, labels, test_size=0.2, random_state=42):
    np.random.seed(random_state) #Ensure reproducibility
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[label].append(i)

    train_indices = []
    test_indices = []
    for indices in class_indices.values():
        n_samples = len(indices)
        n_test = int(n_samples * test_size)
        test_indices.extend(np.random.choice(indices, n_test, replace=False))
        train_indices.extend(list(set(indices) - set(test_indices)))

    train_data = data[train_indices]
    test_data = data[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    return train_data, test_data, train_labels, test_labels

# Usage:
data = torch.randn(1000, 10)
labels = np.random.randint(0, 3, 1000)

train_data, test_data, train_labels, test_labels = stratified_split(data, labels)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")
print(f"Training label distribution: {np.bincount(train_labels)}")
print(f"Testing label distribution: {np.bincount(test_labels)}")

```

This function iterates through each class, randomly selecting a portion for the testing set while ensuring the remaining samples form the training set. This approach maintains the class proportions.


**Example 3: Handling Imbalanced Datasets with Weighted Random Sampling**

In cases of extreme class imbalance, a simple stratified split might not be sufficient. We can incorporate weighted random sampling to further balance the representation of minority classes. This is particularly useful when one class significantly outnumbers others.

```python
import torch
import numpy as np

def weighted_stratified_split(data, labels, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights)  # Normalize weights


    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    #Weighting (optional, but improves minority class representation in heavily imbalanced datasets)
    #This is a simplified example; more sophisticated weighting strategies exist.
    train_weights = np.array([class_weights[label] for label in train_labels])
    test_weights = np.array([class_weights[label] for label in test_labels])

    return train_data, test_data, train_labels, test_weights, train_weights

# Usage (Illustrative):
data = torch.randn(1000, 10)
labels = np.array([0] * 900 + [1] * 50 + [2] * 50) # Highly imbalanced example

train_data, test_data, train_labels, test_weights, train_weights = weighted_stratified_split(data, labels)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")
print(f"Training label distribution: {np.bincount(train_labels)}")
print(f"Testing label distribution: {np.bincount(test_labels)}")

```

This example adds a weighting scheme, which can further improve the balance if there are significant class imbalances.  These weights can then be used during training with optimizers that support sample weighting.


**3. Resource Recommendations**

For a comprehensive understanding of stratified sampling, I recommend consulting introductory statistics textbooks and machine learning literature.  The documentation for `scikit-learn` is also an invaluable resource.  Furthermore, exploring research papers focusing on handling imbalanced datasets will provide additional insights and advanced techniques.  Finally, revisiting fundamental concepts of probability and sampling distributions enhances the grasp of this crucial aspect of data preparation.
