---
title: "Can DeepSpeed maintain the same training time with larger batch sizes?"
date: "2025-01-30"
id: "can-deepspeed-maintain-the-same-training-time-with"
---
DeepSpeed's ability to maintain training time while increasing batch size hinges primarily on its ZeRO optimization family, specifically ZeRO-Offload and ZeRO-3, which address limitations in GPU memory capacity and communication bandwidth. My experience scaling models, particularly during my time working on large language models, has demonstrated that naively increasing batch size often leads to GPU memory exhaustion, resulting in slower training or outright failure. DeepSpeed mitigates this by partitioning model states and optimizing data movement.

A fundamental constraint when training deep learning models arises from the limited memory available on GPUs. The memory footprint of a model during training includes parameters, gradients, and optimizer states (e.g., momentum, variance). As models grow in size and complexity, they quickly exceed the capacity of a single GPU. Furthermore, simply increasing batch size, even for smaller models, amplifies this memory demand as gradients are accumulated for each example in the batch. Without techniques to manage memory, larger batches become untenable.

DeepSpeed’s ZeRO (Zero Redundancy Optimizer) offers several strategies to overcome this limitation. ZeRO-Offload, the initial variant, focuses on offloading optimizer states to the CPU. During the backward pass, the gradients are computed on the GPU, and then the optimizer updates are performed on the CPU using the offloaded states. While this allows for a larger batch size compared to training without offloading, the CPU-GPU data transfer becomes a bottleneck. ZeRO-3, the most sophisticated variant, addresses this bottleneck by partitioning not only optimizer states but also model parameters and gradients across multiple GPUs.

In ZeRO-3, each GPU holds only a fraction of the model, optimizer states, and gradients. During the forward pass, each GPU computes its portion of the output. During the backward pass, each GPU computes the gradients of its portion. Before the optimizer step, the necessary gradients and parameters are gathered across GPUs using a communication collective, such as all-gather. Then, each GPU performs parameter updates on its portion of the model. The significant benefit is that each GPU stores only a fraction of the entire model, dramatically reducing memory footprint per GPU and enabling the use of a larger global batch size.

The efficiency gains stem from distributing the workload and the model data itself. When the total batch size increases but is spread across more GPUs, the amount of data being processed by each GPU remains roughly constant. As a result, the forward and backward passes are not necessarily slower with the larger batch, as it’s distributed over several cards. The key here is that efficient communication and data management are critical to offset any added overhead from distributed operations. If the communication network is slow or becomes a bottleneck, then even a well-partitioned model with large batch size across many cards can still slow down training.

Here are three code examples that demonstrate the conceptual process and illustrate how ZeRO optimization impacts model training with larger batch sizes.

**Example 1: Training without DeepSpeed**

This example shows a training loop without DeepSpeed, using a simple PyTorch model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Model Definition
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Dummy data
input_size = 100
hidden_size = 500
output_size = 10
batch_size = 64
num_samples = 1000
learning_rate = 0.001

data = torch.randn(num_samples, input_size)
labels = torch.randint(0, output_size, (num_samples,))

dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = SimpleModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(5):
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```
*Commentary:* In this example, we use a basic model, loading data via `DataLoader`, and performing standard forward, backward, and optimization steps. Without DeepSpeed, increasing `batch_size` excessively will likely cause an out-of-memory (OOM) error on a typical single GPU. The entire model, optimizer states, and gradients reside in a single GPU's memory.

**Example 2: Training with DeepSpeed and ZeRO-Offload**

This example shows a training loop using DeepSpeed with ZeRO-Offload, demonstrating partial model parameter and state offloading.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import deepspeed

# Model Definition
class SimpleModel(nn.Module):
   # Same model definition as in example 1...
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Dummy data (same as example 1)
input_size = 100
hidden_size = 500
output_size = 10
batch_size = 64
num_samples = 1000
learning_rate = 0.001

data = torch.randn(num_samples, input_size)
labels = torch.randint(0, output_size, (num_samples,))

dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# DeepSpeed Configuration
config = {
    "train_batch_size": batch_size,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": learning_rate
        }
    },
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 1, # ZeRO stage 1, which is equivalent to ZeRO-Offload
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
     }
}

# Model, Loss, Optimizer (wrapped with DeepSpeed)
model = SimpleModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=config
)

# Training loop
for epoch in range(5):
    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(model_engine.device)
        batch_labels = batch_labels.to(model_engine.device)
        outputs = model_engine(batch_data)
        loss = criterion(outputs, batch_labels)
        model_engine.backward(loss)
        model_engine.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

*Commentary:* In this version, the key is the configuration passed to `deepspeed.initialize`. We explicitly configure `zero_optimization` to use `stage: 1`. This configures ZeRO-Offload which offloads the optimizer states to the CPU, which allows us to use a moderately larger batch size without running out of GPU memory (compared to Example 1). However, the limitation is the transfer between the CPU and the GPU can become a bottleneck.

**Example 3: Training with DeepSpeed and ZeRO-3**

This example demonstrates the use of ZeRO-3 with a distributed setup, showcasing more substantial scaling and data parallelism.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import os

# Model Definition (same as example 1 & 2)
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Dummy data (same as example 1 & 2)
input_size = 100
hidden_size = 500
output_size = 10
batch_size = 64 # Note that this is the per-GPU batch size
num_samples = 1000
learning_rate = 0.001

data = torch.randn(num_samples, input_size)
labels = torch.randint(0, output_size, (num_samples,))

dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# DeepSpeed Configuration
config = {
    "train_batch_size": batch_size,
    "train_micro_batch_size_per_gpu": batch_size,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": learning_rate
        }
    },
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO stage 3
    }
}

# Initialize DeepSpeed
model = SimpleModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=config
)

# Training loop
for epoch in range(5):
    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(model_engine.device)
        batch_labels = batch_labels.to(model_engine.device)
        outputs = model_engine(batch_data)
        loss = criterion(outputs, batch_labels)
        model_engine.backward(loss)
        model_engine.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

*Commentary:* Here, `zero_optimization` is set to `stage: 3`. This means model parameters, gradients, and optimizer states are all partitioned across devices. The `train_batch_size` is actually the global batch size, implicitly derived from the per-GPU batch size and the number of GPUs. This enables significantly larger total batch sizes compared to the previous examples because the overall model is distributed, allowing us to potentially maintain training time even with a larger global batch size if sufficient resources are available. Launching this script would typically involve `deepspeed --num_gpus {num_gpus} your_script.py`.

While these examples utilize simplified models, they represent the underlying principles of DeepSpeed and its ability to handle larger batch sizes through memory optimization and distributed training.

For further understanding, I would recommend consulting the DeepSpeed documentation, paying particular attention to the "ZeRO Optimizer" section. Explore resources detailing distributed training concepts as well. Understanding the nuances of communication collectives (e.g., all-gather, all-reduce) will provide a deeper understanding of how data is exchanged across devices. Books on parallel computing and high-performance computing can also be valuable for building a comprehensive understanding of the underlying concepts. Finally, study practical examples of large language model training to observe these techniques used in real-world scenarios.
