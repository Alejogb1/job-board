---
title: "How can multiple GPUs be used to train GPT-Neox?"
date: "2025-01-30"
id: "how-can-multiple-gpus-be-used-to-train"
---
The computational demands of training large language models like GPT-NeoX necessitate the use of multiple GPUs to accelerate the process. Effectively distributing the training workload across several GPUs requires careful consideration of data and model parallelism strategies, along with appropriate software tools to orchestrate the distributed computation. I've personally navigated these complexities on multiple occasions while training models for NLP research, observing that a poorly implemented distribution strategy can easily lead to inefficient resource usage, diminishing the benefits of multiple GPUs.

**Understanding Distributed Training Paradigms**

Fundamentally, two dominant parallelization techniques are employed: data parallelism and model parallelism. In data parallelism, the training dataset is divided into subsets, with each GPU processing a different batch of data through a replica of the full model. The gradient updates from each GPU are then aggregated to update the model's weights. This is a relatively straightforward approach and works well when the model itself fits within the memory constraints of a single GPU. However, for larger models like GPT-NeoX, the memory footprint often exceeds these limits, rendering data parallelism alone insufficient.

Model parallelism, on the other hand, involves partitioning the model itself across multiple GPUs. Each GPU is responsible for computing a specific portion of the model's layers or operations. This approach is essential when dealing with models too large to fit on a single device but adds complexities concerning communication between GPUs to maintain a forward and backward pass through the model. There are several flavors of model parallelism, including tensor parallelism where single layers are split across devices, and pipeline parallelism where successive layers are placed on successive GPUs. A common practice is to combine these approaches. For instance, a model may be tensor parallelized within a stage and the stages can be arranged in a pipeline.

**Practical Implementation Choices**

The framework utilized for training plays a significant role in how these parallelization strategies are employed. While bespoke implementations are occasionally necessary, using frameworks like DeepSpeed, PyTorch Distributed, or Megatron-LM greatly simplifies the implementation. For GPT-NeoX, DeepSpeed has proven to be a reliable option due to its performance-focused approach. It provides optimized kernels, memory management features like ZeRO optimization, and support for a variety of distributed strategies. I've found its integration to be relatively seamless once the configuration is well understood.

Here are three code examples demonstrating increasingly complex approaches using a conceptualized training script:

**Example 1: Basic Data Parallelism with PyTorch Distributed**

This example shows the most fundamental approach with data parallelism. It assumes that the model fits on each device.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_step(model, data, optimizer, criterion):
  optimizer.zero_grad()
  outputs = model(data)
  loss = criterion(outputs, data)
  loss.backward()
  optimizer.step()
  return loss.item()


def main(rank, world_size):
    setup(rank, world_size)

    # Dummy model, loss function and optimizer for illustration purposes
    model = nn.Linear(100, 100).to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Dummy training data for illustration
    data = torch.randn(32, 100).to(rank)

    for epoch in range(5):
        loss = train_step(model, data, optimizer, criterion)
        print(f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}")

    cleanup()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

```

**Commentary:**

This example showcases a basic data-parallel implementation using PyTorch's `DistributedDataParallel`. It initializes a distributed process group via `dist.init_process_group` using NCCL for GPU communication.  A simple linear model is created, then wrapped with `DDP` to handle gradient synchronization across GPUs. The training loop is basic.  Each rank (GPU) computes a loss using its assigned portion of the data, and DDP handles the necessary gradient gathering from all ranks, enabling updates across the model replica on each device. It assumes that all GPUs are accessible, and no specific model parallelism strategy is implemented. This will not work for very large models which exceed GPU memory.

**Example 2: Combining Data Parallelism with Basic Tensor Parallelism**

This example introduces a rudimentary tensor parallelism alongside the data parallelism used earlier.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, world_size):
        super().__init__()
        self.out_features_per_rank = out_features // world_size
        self.weight = nn.Parameter(torch.randn(self.out_features_per_rank, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_rank))
        self.rank = rank
        self.world_size = world_size

    def forward(self, x):
        output = torch.matmul(self.weight, x.T) + self.bias.unsqueeze(1)
        output = output.T
        all_outputs = [torch.empty_like(output) for _ in range(self.world_size)]
        dist.all_gather(all_outputs, output)
        return torch.cat(all_outputs, dim=1)


def train_step(model, data, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, data)
    loss.backward()
    optimizer.step()
    return loss.item()

def main(rank, world_size):
    setup(rank, world_size)

    model = ParallelLinear(100, 200, rank, world_size).to(rank)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    data = torch.randn(32, 100).to(rank)

    for epoch in range(5):
        loss = train_step(model, data, optimizer, criterion)
        print(f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}")

    cleanup()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
```

**Commentary:**

This example shows a rudimentary form of tensor parallelism by dividing the output features of the linear layer across GPUs using the `ParallelLinear` class. Instead of the whole linear layer existing on each device, each device owns a slice of the output features. The `all_gather` operation collects the outputs from all devices, allowing the complete output to be available to each device.  The rest of the script remains similar to the first example using data parallelism. Note that `DDP` is no longer used, as our `ParallelLinear` class takes care of the inter-GPU communication. In practice, `torch.distributed.fsdp` should be used for tensor parallelism instead of what is implemented here.

**Example 3:  Conceptual DeepSpeed Implementation (Simplified)**

This is a demonstration of what a more realistic setup would resemble by abstracting some of the DeepSpeed functionality.

```python
import torch
import torch.nn as nn
import torch.optim as optim
# Assume DeepSpeed is abstracted with DeepSpeedManager
class DeepSpeedManager:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        # DeepSpeed engine initialization would happen here
        # For this example, we just return the model
    def get_model(self):
      return self.model

    def train_step(self, data, criterion):
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = criterion(outputs, data)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def main():
    # Configuration for DeepSpeed
    config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 0.01}
        },
       "zero_optimization": {
            "stage": 2
        }

    }

    # Dummy model, loss function and optimizer for illustration purposes
    model = nn.Linear(100, 100)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Initialize DeepSpeed
    deepspeed_manager = DeepSpeedManager(model, optimizer, config)

    # Dummy training data for illustration
    data = torch.randn(32, 100)
    model = deepspeed_manager.get_model()
    for epoch in range(5):
        loss = deepspeed_manager.train_step(data, criterion)
        print(f"Epoch: {epoch}, Loss: {loss}")


if __name__ == '__main__':
  main()

```

**Commentary:**

This example uses a conceptual `DeepSpeedManager` to simulate the initialization and training process with DeepSpeed. In reality, DeepSpeed would handle the distributed initialization, model wrapping, optimizer management, and the training loop. Here, it acts as an abstraction for the sake of brevity. The configuration dictionary outlines basic parameters. In a real implementation, this dictionary would include ZeRO optimization, tensor parallel parameters, pipeline configurations, and other configurations relevant to GPT-NeoX. The important takeaway is that DeepSpeed handles a lot of the complexity of the distributed training under the hood, making it easier to scale training.

**Resource Recommendations:**

To further develop expertise in this area, I recommend studying the documentation of the following projects directly: DeepSpeed, Megatron-LM, PyTorch Distributed, and specifically the PyTorch FSDP module. Theoretical understanding of distributed training methodologies is equally beneficial. Research papers on data and model parallelism in deep learning, specifically focusing on those that present strategies relevant to transformer-based models, will provide a solid foundation. Furthermore, examining open-source implementations of GPT-NeoX training, including those that utilize one of the mentioned distributed training frameworks, will provide hands-on insight and best practices. These resources are continuously being developed, and regular engagement with their development will provide a greater knowledge of current best practices.
