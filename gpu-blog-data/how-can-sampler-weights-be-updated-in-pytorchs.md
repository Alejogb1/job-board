---
title: "How can sampler weights be updated in PyTorch's DDP mode?"
date: "2025-01-30"
id: "how-can-sampler-weights-be-updated-in-pytorchs"
---
Batch sampling in distributed data parallel (DDP) training is not as straightforward as in single-GPU scenarios, particularly when employing techniques like weighted sampling to address class imbalance or prioritize certain training examples. Directly manipulating sampler weights within a DDP context requires careful consideration of how data loaders and the DDP process itself interact. Improper handling can lead to inconsistent data distribution across ranks, suboptimal training convergence, and even silent errors. I've encountered these challenges firsthand while developing a large-scale object detection model with highly skewed class distributions using a multi-node cluster.

My initial approach involved attempting to directly update the `weights` attribute of a `torch.utils.data.WeightedRandomSampler` instance at each training epoch. This naive method failed spectacularly because each DDP process initializes its own data loader and sampler. Modifying the weights in one process does not propagate to other processes, resulting in each worker drawing different samples, undermining the parallel training process. Furthermore, because the DDP wrapper shuffles and distributes batches based on the world size, it may appear like batches are similar initially but the actual underlying data distribution is completely non-deterministic.

The correct procedure hinges on two core principles: 1) ensuring that sampler state is synchronized *before* the data loader is instantiated on each worker, and 2) recalculating and re-assigning new sampling weights to each process. Let me elaborate.

Instead of directly modifying the `sampler.weights` object during training, the strategy I settled on is to store the sample weights globally, outside of any individual sampler instance. This is usually best achieved using a simple Python `list`. Before constructing a `DataLoader`, each process first accesses or regenerates these global weights. The process then creates its own `WeightedRandomSampler` with these synchronized weights. The key here is that the weights are identical *across all processes* at the point when the DataLoader is instantiated.

The following examples should illustrate the method I've found successful.

**Example 1: Synchronizing Initial Weights**

This code shows the initial setup of synchronized sampler weights.

```python
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

# Simplified custom dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 10)
        self.targets = torch.randint(0, 2, (num_samples,))  # Binary classification
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_initial_weights(dataset):
    """Compute initial weights based on class distribution."""
    class_counts = torch.bincount(dataset.targets)
    weights = 1.0 / class_counts[dataset.targets]
    return weights.tolist()


def create_dataloader(dataset, rank, world_size, sampler_weights=None, batch_size=4):
    if sampler_weights is None:
      sampler = None # Use the default sampler instead
    else:
      sampler = WeightedRandomSampler(
          weights=sampler_weights, num_samples=len(dataset), replacement=True
          )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
      dataset, num_replicas = world_size, rank = rank, shuffle=False # Shuffling controlled by weighted sampler
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler if sampler_weights is None else sampler, #Use either dist sampler or custom weighted one
        shuffle=False,
        drop_last=True
    )
    return dataloader

if __name__ == '__main__':
    world_size = 2  # Example world size
    rank = int(os.environ["LOCAL_RANK"]) # Set using torchrun or equivalent
    setup(rank, world_size)

    dataset = SimpleDataset(num_samples=100)
    global_sampler_weights = get_initial_weights(dataset) # Get initial weights
    dataloader = create_dataloader(dataset, rank, world_size, global_sampler_weights)

    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Rank: {rank}, Batch {batch_idx}, Batch size {len(data)}")

    cleanup()

```

In this example, `get_initial_weights` calculates and stores initial weights. In the `create_dataloader` function, I first check if there are `sampler_weights`. If so, I instantiate a `WeightedRandomSampler` for use by the DataLoader. Note, if sampler weights are `None` then the DistributedSampler handles shuffling and distribution. This initial setup ensures that at the start of training, all ranks utilize identical sampling probabilities.

**Example 2: Updating Weights at Each Epoch**

This shows the procedure for updating weights at the end of each training epoch.

```python
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import os
import random


# Simplified custom dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 10)
        self.targets = torch.randint(0, 2, (num_samples,))  # Binary classification
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_initial_weights(dataset):
    """Compute initial weights based on class distribution."""
    class_counts = torch.bincount(dataset.targets)
    weights = 1.0 / class_counts[dataset.targets]
    return weights.tolist()


def create_dataloader(dataset, rank, world_size, sampler_weights=None, batch_size=4):
    if sampler_weights is None:
      sampler = None # Use the default sampler instead
    else:
      sampler = WeightedRandomSampler(
          weights=sampler_weights, num_samples=len(dataset), replacement=True
          )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
      dataset, num_replicas = world_size, rank = rank, shuffle=False # Shuffling controlled by weighted sampler
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler if sampler_weights is None else sampler, #Use either dist sampler or custom weighted one
        shuffle=False,
        drop_last=True
    )
    return dataloader


def update_sampler_weights(dataset, global_sampler_weights, rank):
    # Simulate updates based on training results
    new_weights = [
      weight * (1 + (random.random()/5) if target==1 else 1-random.random()/5) #Randomize weights (more for class 1)
       for weight, target in zip(global_sampler_weights, dataset.targets.tolist())
    ]
    if rank==0:
        print(f"Rank {rank}: sample weights updated.")
    return new_weights


if __name__ == '__main__':
    world_size = 2  # Example world size
    rank = int(os.environ["LOCAL_RANK"]) # Set using torchrun or equivalent
    setup(rank, world_size)

    dataset = SimpleDataset(num_samples=100)
    global_sampler_weights = get_initial_weights(dataset)

    num_epochs = 3
    for epoch in range(num_epochs):
        dataloader = create_dataloader(dataset, rank, world_size, global_sampler_weights)
        for batch_idx, (data, target) in enumerate(dataloader):
            print(f"Rank: {rank}, Epoch: {epoch}, Batch: {batch_idx}, Batch size: {len(data)}")

        # Synchronized update of weights before next epoch
        global_sampler_weights = update_sampler_weights(dataset, global_sampler_weights, rank)
        if dist.is_initialized():
           global_sampler_weights =  dist.broadcast_object_list([global_sampler_weights], src=0)[0] # Sync all ranks
        else:
            pass
        # All processes now have identical sampler weights for the new epoch


    cleanup()
```

Here, I’ve introduced the `update_sampler_weights` function to simulate updating the global weights after an epoch. The weights for each sample are slightly adjusted based on the target. The updated weights are then broadcast to all ranks to ensure that each process uses identical weights for the next epoch. This is done with `dist.broadcast_object_list`. Without this step, each rank will have inconsistent weights.

**Example 3: Using Custom Logic**

This illustrates a more complex weight update based on sample difficulty. Note, this requires access to the training loss for each sample, something not normally provided in the vanilla PyTorch Dataloader.

```python
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import os
import random

# Simplified custom dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 10)
        self.targets = torch.randint(0, 2, (num_samples,))  # Binary classification
        self.loss_history = torch.zeros(num_samples)  # Track sample loss

    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def set_loss(self, idx, loss):
        self.loss_history[idx] = loss

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_initial_weights(dataset):
    """Compute initial weights based on class distribution."""
    class_counts = torch.bincount(dataset.targets)
    weights = 1.0 / class_counts[dataset.targets]
    return weights.tolist()

def create_dataloader(dataset, rank, world_size, sampler_weights=None, batch_size=4):
    if sampler_weights is None:
      sampler = None # Use the default sampler instead
    else:
      sampler = WeightedRandomSampler(
          weights=sampler_weights, num_samples=len(dataset), replacement=True
          )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
      dataset, num_replicas = world_size, rank = rank, shuffle=False # Shuffling controlled by weighted sampler
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler if sampler_weights is None else sampler, #Use either dist sampler or custom weighted one
        shuffle=False,
        drop_last=True
    )
    return dataloader

def update_sampler_weights(dataset, global_sampler_weights, rank, learning_rate=0.5):
  """Update weights based on sample difficulty."""
  new_weights = []
  for index, weight in enumerate(global_sampler_weights):
        loss = dataset.loss_history[index]  # Loss for that sample
        updated_weight = weight * (1 + learning_rate * loss.item()) # Boost hard samples
        new_weights.append(updated_weight)


  if rank==0:
      print(f"Rank {rank}: sample weights updated.")
  return new_weights

if __name__ == '__main__':
    world_size = 2  # Example world size
    rank = int(os.environ["LOCAL_RANK"]) # Set using torchrun or equivalent
    setup(rank, world_size)

    dataset = SimpleDataset(num_samples=100)
    global_sampler_weights = get_initial_weights(dataset)


    num_epochs = 3
    model = torch.nn.Linear(10, 2)
    criterion = torch.nn.CrossEntropyLoss(reduction='none') # Return losses per sample
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    for epoch in range(num_epochs):
        dataloader = create_dataloader(dataset, rank, world_size, global_sampler_weights)
        for batch_idx, (data, target) in enumerate(dataloader):

            optimizer.zero_grad()
            outputs = model(data)
            losses = criterion(outputs, target) # Get sample losses
            losses.mean().backward() # Back prop using mean
            optimizer.step()

            for index_in_batch, (loss, sample_data, sample_target) in enumerate(zip(losses, data, target)):
                dataset_index =  batch_idx * len(data) + index_in_batch  # Correct index in the whole dataset
                dataset.set_loss(dataset_index, loss)
            print(f"Rank: {rank}, Epoch: {epoch}, Batch: {batch_idx}, Batch size: {len(data)}")

        # Synchronized update of weights before next epoch
        global_sampler_weights = update_sampler_weights(dataset, global_sampler_weights, rank)
        if dist.is_initialized():
           global_sampler_weights = dist.broadcast_object_list([global_sampler_weights], src=0)[0] # Sync all ranks
        else:
           pass
    cleanup()

```
In this final example, I've modified the dataset to track the loss of each sample and updated the weight update method accordingly to consider these losses, giving higher weights to harder examples. Also, each worker also stores loss history, and updates only local sample weights. If the datasets are different on each worker (which might be more practical in a real-world setting), you could pass dataset index to the global sample weight tracker. This more complex method requires more careful management of the data, but it can lead to much faster training in some scenarios.

In terms of additional resources, I suggest examining PyTorch’s official documentation on distributed training and the `torch.utils.data` modules. I also recommend consulting research papers and tutorials related to active learning and sampling strategies for imbalanced datasets. Specifically, investigate publications discussing curriculum learning and hard example mining as these methods often employ similar weighted sampling techniques. Understanding the nuances of the `DistributedSampler`, as well as the underlying mechanisms of distributed data parallel processing is key. Finally, reviewing best practices for distributed training with PyTorch will help to avoid common pitfalls.
