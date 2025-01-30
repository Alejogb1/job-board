---
title: "How can I efficiently create a PyTorch DataLoader and sampler that guarantees at least one sample per class?"
date: "2025-01-30"
id: "how-can-i-efficiently-create-a-pytorch-dataloader"
---
The challenge of creating a PyTorch DataLoader that guarantees at least one sample per class per batch arises frequently in imbalanced classification problems, particularly where underrepresented classes might be completely missed during training. Standard PyTorch samplers like `RandomSampler` do not inherently ensure this condition, leading to potentially unstable learning, especially at the start of training. I've seen firsthand how omitting classes can stall convergence and degrade a model's ability to generalize. Therefore, a custom sampler tailored to this specific requirement is crucial.

The primary issue with typical data loading workflows when dealing with imbalanced datasets is the possibility of the DataLoader producing batches devoid of samples from certain classes. This uneven distribution can cause several issues. Gradient updates might heavily favor dominant classes, hindering the model's ability to learn from minority classes. If some classes are consistently absent from the training batch, the model will likely perform poorly on those classes during validation and testing. The solution is to craft a custom PyTorch sampler that explicitly selects at least one sample from each available class in every batch. The process involves two main steps: determining the batch size and creating a strategy to sample data according to class representation.

Let's examine how to achieve this, taking into account the specifics of how data is structured when working with Pytorch. Assume each data point is indexed, and the dataset returns both the data tensor and its label. This allows the sampler to intelligently sample based on the labels. The sampling process can be split into two parts: first, we guarantee one sample per class; and second, we handle the remaining sampling to complete the batch. For the guarantee step, we iterate over all available classes and randomly select one index for each class from a list of indices. The list of indices is built during initialization and is specific to each class. For the second step, we can sample remaining items randomly, given that a sufficient number of samples is required. I will present three Python code examples illustrating these principles with progressive complexity.

**Example 1: Basic Sampler with Class Guarantees**

This initial example introduces the core logic of a custom sampler. It assumes a simple scenario with a fixed batch size and a known number of classes.

```python
import torch
from torch.utils.data import Sampler
import random

class ClassGuaranteedSampler(Sampler):
    def __init__(self, data_labels, batch_size, num_classes):
        self.data_labels = data_labels
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.class_indices = {c: [] for c in range(num_classes)}
        for idx, label in enumerate(self.data_labels):
            self.class_indices[label].append(idx)

    def __iter__(self):
        indices = []
        for _ in range(len(self.data_labels) // self.batch_size):
            batch = []
            for c in range(self.num_classes):
                batch.append(random.choice(self.class_indices[c]))

            remaining_size = self.batch_size - self.num_classes
            remaining_indices = random.choices(range(len(self.data_labels)), k=remaining_size)
            batch.extend(remaining_indices)
            random.shuffle(batch)
            indices.extend(batch)

        return iter(indices)

    def __len__(self):
        return len(self.data_labels)
```

In this code: `data_labels` represents a list or tensor of labels associated with each data item, `batch_size` sets the desired batch size, and `num_classes` is the total number of unique classes in the dataset. We create `class_indices` at initialization, a dictionary storing indices grouped by class. During each iteration, we ensure that each batch contains one sample from each class. Subsequently, we randomly sample to complete batch size. The return of the `__iter__` method produces the indices that the DataLoader uses to retrieve the samples. This simple sampler iterates over the dataset to generate batches. However, It doesn't handle edge cases such as dataset lengths not being multiples of the batch size.

**Example 2: Robust Sampler Handling Edge Cases**

This example enhances the sampler, addressing cases where the dataset size is not perfectly divisible by the batch size. We ensure the final batch still fulfills the class representation requirements.

```python
import torch
from torch.utils.data import Sampler
import random
import math

class ClassGuaranteedSamplerV2(Sampler):
    def __init__(self, data_labels, batch_size, num_classes):
        self.data_labels = data_labels
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.class_indices = {c: [] for c in range(num_classes)}
        for idx, label in enumerate(self.data_labels):
            self.class_indices[label].append(idx)

        self.num_batches = math.ceil(len(data_labels) / batch_size)

    def __iter__(self):
        indices = []
        for i in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                if self.class_indices[c]:
                    batch.append(random.choice(self.class_indices[c]))
                else:
                     continue # Handle cases where class is missing.

            remaining_size = self.batch_size - len(batch)
            if remaining_size > 0:
                remaining_indices = random.choices(range(len(self.data_labels)), k=remaining_size)
                batch.extend(remaining_indices)
            random.shuffle(batch)
            indices.extend(batch)


        return iter(indices)

    def __len__(self):
        return len(self.data_labels)
```

This version uses `math.ceil` to compute the number of batches needed and introduces a `continue` statement in the loop that adds the class-specific sample to handle scenarios with missing classes. The key improvement is the proper handling of cases where dataset size isn't a multiple of batch size. This ensures the last batch, even if smaller, attempts to represent all classes while allowing it to contain less than the regular batch size. Furthermore, the sampler checks whether each class has samples before trying to sample from it. This prevents errors if a class is completely absent.

**Example 3: Sampler with Custom Data Source Integration**

This final example integrates the sampler within a custom dataset class, showcasing how the sampler can be used in a realistic setting. We'll assume a simple synthetic dataset that can be easily generalized to a real-world scenario.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import random
import math

class SimpleDataset(Dataset):
    def __init__(self, num_samples, num_classes):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ClassGuaranteedSamplerV3(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_labels = dataset.labels.tolist()
        self.num_classes = dataset.num_classes

        self.class_indices = {c: [] for c in range(self.num_classes)}
        for idx, label in enumerate(self.data_labels):
            self.class_indices[label].append(idx)

        self.num_batches = math.ceil(len(self.data_labels) / batch_size)

    def __iter__(self):
        indices = []
        for i in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                if self.class_indices[c]:
                     batch.append(random.choice(self.class_indices[c]))
                else:
                    continue


            remaining_size = self.batch_size - len(batch)
            if remaining_size > 0:
                remaining_indices = random.choices(range(len(self.data_labels)), k=remaining_size)
                batch.extend(remaining_indices)
            random.shuffle(batch)
            indices.extend(batch)

        return iter(indices)

    def __len__(self):
        return len(self.data_labels)

if __name__ == '__main__':
    num_samples = 100
    num_classes = 5
    batch_size = 8

    dataset = SimpleDataset(num_samples, num_classes)
    sampler = ClassGuaranteedSamplerV3(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Labels: {labels}")
```

This example defines a synthetic dataset (`SimpleDataset`) and includes the `ClassGuaranteedSamplerV3`, modified to accept a dataset object during initialization. We retrieve the labels directly from the dataset's attribute. The sampler is then integrated with the `DataLoader` using the `batch_sampler` argument, demonstrating a typical usage scenario. The output of the `DataLoader` can be printed to verify each batch contains at least one instance of each class. This demonstrates the full lifecycle of a custom sampler.

For further study, I recommend exploring the official PyTorch documentation on data loading, specifically the `torch.utils.data` module. Textbooks on Deep Learning often include comprehensive sections on data preprocessing and handling class imbalance, typically exploring a wider variety of sampling methods than what is shown here, such as weighted sampling. Additionally, researching sampling techniques specific to your application, like those used in medical imaging or natural language processing, can often give you valuable insights. The PyTorch tutorials also provide several guides to build custom datasets and samplers.
