---
title: "How does training ResNet50 on CIFAR-10 improve with multiple GPUs?"
date: "2025-01-30"
id: "how-does-training-resnet50-on-cifar-10-improve-with"
---
The primary challenge in scaling ResNet50 training to multiple GPUs lies not in the inherent parallelizability of convolutional operations, but in managing data distribution, gradient aggregation, and ensuring minimal communication overhead. My experience optimizing large-scale models for image recognition indicates a significant performance boost is achievable with careful consideration of these factors, particularly when working with a relatively smaller dataset like CIFAR-10.

The training process benefits from multi-GPU utilization primarily through data parallelism. In essence, we replicate the ResNet50 model across each available GPU. During each training step, we divide the training batch into smaller sub-batches, each assigned to a different GPU. Each GPU computes its forward and backward passes using its sub-batch. The resulting gradients, representing the local updates to the model's weights, must be aggregated across all GPUs before being used to update the shared model parameters. Without an optimized strategy for this communication, the overhead can easily negate the computational advantages of using multiple GPUs. Crucially, we must ensure the weights are synchronized, or the training will not converge effectively.

There are two predominant methods for handling gradient aggregation: synchronous and asynchronous. Synchronous gradient aggregation requires all GPUs to finish their forward and backward passes before the gradients are averaged. This approach offers precise convergence and predictable behavior, but it also results in idle GPU time if one GPU encounters an unusual delay. Asynchronous aggregation, on the other hand, allows GPUs to update the shared model as soon as their gradients are computed. This is less likely to result in idle GPUs but might lead to unstable training and less predictable convergence behavior. Given the smaller dataset of CIFAR-10 and the relatively simple, well-understood nature of ResNet50, a synchronous method, often implemented using libraries like TensorFlow or PyTorch's distributed modules, is typically the more straightforward and effective starting point. It prioritizes stability and a reliable training trajectory over the potential performance gains of an asynchronous update.

Furthermore, the performance improvements depend significantly on the underlying communication infrastructure and its optimization. The most common method for intra-node communication (communication within the same physical machine) involves using shared memory, provided the GPUs and CPU are connected via a fast interconnect like NVLink. For multi-node configurations (where the training is distributed across multiple machines), high-speed networking like Infiniband is crucial. Regardless, the reduction of communication latency is key; the larger the dataset (though CIFAR-10 is comparatively small) and the larger the model, the more dominant communication overhead becomes.

Let's examine how multi-GPU training for ResNet50 with CIFAR-10 can be approached in practice with code examples using PyTorch, which I have used extensively:

**Example 1: Basic Data Parallelism (using `torch.nn.DataParallel`)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define device and check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")

# Data transformation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# Initialize ResNet50
model = torchvision.models.resnet50(pretrained=False, num_classes=10)
model.to(device)

# Use DataParallel for multi-GPU training if available.
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(2):  # Loop over the dataset multiple times
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 200 == 199:    # print every 200 mini-batches
      print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 200))
      running_loss = 0.0

print('Finished Training')
```
This basic example utilizes `nn.DataParallel`, which is the most straightforward approach in PyTorch for multi-GPU training. It duplicates the model onto each GPU and automatically handles data distribution.  The disadvantage is that it relies on a single process and does not scale well beyond a few GPUs or when more control over distributed behavior is required. It can be limited because the data loading and the final weight update is all done on a primary GPU which creates bottlenecks, especially with more GPUs and complex models.

**Example 2: Distributed Data Parallelism (using `torch.distributed`)**

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Setup distributed environment (example for local execution)
def init_distributed(backend='nccl'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
       rank = int(os.environ["RANK"])
       world_size = int(os.environ['WORLD_SIZE'])
       gpu = int(os.environ['LOCAL_RANK'])
    else:
       rank, world_size, gpu = 0, 1, 0
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu)
    return rank, world_size, gpu


rank, world_size, gpu = init_distributed()

# Data transformation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)

sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                        num_replicas=world_size,
                                                        rank=rank,
                                                        shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                         sampler=sampler, num_workers=2)



# Initialize ResNet50
model = torchvision.models.resnet50(pretrained=False, num_classes=10)
model.to(gpu)

# Wrap the model with DistributedDataParallel
model = DistributedDataParallel(model, device_ids=[gpu])

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(2):  # Loop over the dataset multiple times
  sampler.set_epoch(epoch) #Important for shuffle across multiple epochs with DDP
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data[0].to(gpu), data[1].to(gpu)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 200 == 199 and rank == 0:    # print every 200 mini-batches (rank 0)
      print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 200))
      running_loss = 0.0

if rank == 0:
  print('Finished Training')
```
This example showcases `DistributedDataParallel` (DDP) which leverages multiple processes. Each process has its own copy of the model, leading to a higher degree of independence and less overhead compared to DataParallel, thereby allowing better scaling as we increase the number of GPUs. The distributed sampler ensures that each process receives a different subset of the training data. The key part in setting up the training is the use of `torch.distributed` and `DistributedDataParallel` to properly synchronize the processes. The environment variables `RANK`, `WORLD_SIZE` and `LOCAL_RANK` need to be set for launching the training.

**Example 3: Gradient Accumulation with DDP (Hypothetical - not included in full)**

This strategy builds on the previous example by accumulating gradients over several iterations before updating the model's weights. It allows the simulation of larger batch sizes than would otherwise fit in GPU memory. While the code for this would require further modification to the DDP training loop of Example 2 to increment the optimizer every `n` steps, the fundamental idea is to call `loss.backward()` multiple times and then update the optimizer's parameters every 'n' steps. This allows us to train with a large effective batch size while still fitting within GPU memory bounds, which can be useful in situations where the batch size can't be increased directly because of GPU memory limitations.  The key is to call `.zero_grad()` less frequently than calling `.step()`.

In summary, training ResNet50 on CIFAR-10 benefits substantially from multi-GPU utilization. Data parallelism, especially using `torch.distributed`, combined with synchronous gradient aggregation provides effective scaling. The benefits are increased computational capacity, allowing for faster training times and enabling the use of larger effective batch sizes through gradient accumulation. To further optimize performance, one would need to fine tune batch size, learning rate and choose appropriate optimizer parameters.

For further study, I recommend the following resources: the PyTorch documentation on distributed training, the Horovod library documentation for another perspective on multi-GPU training strategies, and research papers on large-scale deep learning, specifically those detailing optimizations for distributed training. These resources provide a deeper understanding of the nuances of distributed training and offer advanced optimization techniques that may be necessary for more demanding scenarios.
