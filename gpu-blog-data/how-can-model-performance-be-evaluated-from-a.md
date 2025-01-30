---
title: "How can model performance be evaluated from a checkpoint using multiple GPUs?"
date: "2025-01-30"
id: "how-can-model-performance-be-evaluated-from-a"
---
Checkpoint evaluation across multiple GPUs requires a nuanced approach beyond naive single-GPU loading, primarily due to the distributed nature of the model’s weights and the necessity for synchronized, representative performance metrics. Over years spent optimizing large-scale deep learning deployments, I've encountered numerous challenges when transitioning from single-GPU training to multi-GPU evaluation, leading to the development of robust patterns detailed below.

The central challenge is that model checkpoints, when generated from distributed training (e.g., using `torch.nn.DataParallel` or `torch.distributed`), often do not store weights that are readily consumable by a single process. Each GPU during training holds only a portion of the model’s parameter set, and the final checkpoint reflects this distributed state. Consequently, directly loading such a checkpoint onto a single GPU will often lead to errors or misrepresented performance, as the model lacks crucial portions of its parameter set. The evaluation process must therefore reassemble the distributed weights into a cohesive model representation.

To evaluate a distributed checkpoint correctly, the process can be broadly broken into these steps: 1) initialize a distributed environment mirroring the training setup, 2) load the checkpoint weights, ensuring they are correctly distributed or consolidated based on the framework’s saving conventions, 3) setup the evaluation dataset and dataloader, making sure the data is correctly distributed across processes (if doing distributed evaluation), and 4) execute the evaluation loop, gathering and aggregating the performance metrics across all the participating GPUs. Different deep learning frameworks and training paradigms necessitate slightly varied implementations of these steps, but the underlying principles remain consistent.

Let's examine a typical scenario using PyTorch with `torch.distributed`, using a `DistributedDataParallel` training pattern:

**Code Example 1: Distributed Checkpoint Loading and Model Setup**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def load_checkpoint_and_model(rank, world_size, checkpoint_path):
    setup(rank, world_size)

    model = SimpleModel()
    model = model.to(rank) # Move model to GPU based on rank
    model = DDP(model, device_ids=[rank]) # Encapsulate with DDP

    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0': f'cuda:{rank}'}) # Load to corresponding device
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

if __name__ == '__main__':
    world_size = torch.cuda.device_count() # Number of GPUs available
    if world_size > 1:
        torch.multiprocessing.spawn(run,
                                    args=(world_size,),
                                    nprocs=world_size)
    else:
        # single GPU run
        rank = 0
        checkpoint_path = 'checkpoint.pt'
        model = load_checkpoint_and_model(rank, world_size, checkpoint_path)
        # Run eval loop with loaded single GPU model
    
    #Example main function to spawn processes
def run(rank, world_size):

        checkpoint_path = 'checkpoint.pt' # Replace with actual checkpoint path
        model = load_checkpoint_and_model(rank, world_size, checkpoint_path)

        # Create dummy data for example evaluation
        batch_size = 32
        input_data = torch.randn(batch_size, 10).to(rank)
        labels = torch.randint(0, 2, (batch_size,)).to(rank)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        with torch.no_grad():
          outputs = model(input_data)
          loss = criterion(outputs, labels)

        gathered_loss = [torch.zeros_like(loss) for _ in range(world_size)]
        dist.all_gather(gathered_loss, loss)

        # Aggregate losses from all devices
        total_loss = sum(gathered_loss) / len(gathered_loss)

        if rank == 0:
            print(f"Loss: {total_loss}")
        
        cleanup()
```

**Commentary:**

1.  The `setup` function initializes the distributed environment using `torch.distributed`. It sets environment variables and initializes the process group, establishing inter-process communication necessary for distributed evaluation.
2.  `SimpleModel` is a basic model to represent the architecture being evaluated. This would be replaced with actual model architecture.
3.  In `load_checkpoint_and_model`, the model is instantiated, moved to the correct GPU based on the current process’s rank, and wrapped in `DDP`. Critically, the `map_location` argument during checkpoint loading ensures that model weights previously on `cuda:0` are redirected to the current process’s allocated GPU (`cuda:{rank}`), rather than just the first GPU available.
4. The evaluation is performed with a dummy input, loss calculation and loss aggregation across all GPUs.
5. The main function spawns multiple processes based on available CUDA GPUs and runs the training or inference functions respectively.

This example shows how to properly load a checkpoint from a `DDP` training environment for evaluation in a distributed fashion. This script requires a valid `checkpoint.pt` file created during `DDP` training for demonstration purposes.

**Code Example 2: Data Loading for Distributed Evaluation**

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import os

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = np.random.rand(size, 10).astype(np.float32)
        self.labels = np.random.randint(0, 2, size).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_distributed_dataloader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(run,
                                    args=(world_size,),
                                    nprocs=world_size)
    else:
        # Single GPU Evaluation
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=32)
        for inputs, labels in dataloader:
          print("Evaluation with single GPU data loader.")
          break

def run(rank, world_size):
    
    setup(rank, world_size)

    dataset = DummyDataset(size=1000) # Increased dataset size
    batch_size = 32
    dataloader = create_distributed_dataloader(dataset, batch_size, rank, world_size)

    for inputs, labels in dataloader:
        print(f"Rank {rank}, batch size: {inputs.shape[0]}")
        # Move input and label to correct GPU
        inputs = inputs.to(rank)
        labels = labels.to(rank)
        break

    cleanup()
```

**Commentary:**

1.  `DummyDataset` is a placeholder for a real dataset, creating random input data and labels.
2.  `create_distributed_dataloader` makes use of `DistributedSampler`. This sampler ensures that each process receives a non-overlapping partition of the dataset during evaluation. It is initialized with the number of devices involved and the current rank, so every GPU gets a different portion of the data. Shuffling should generally be disabled for evaluation to maintain consistent metric calculations.
3. The main function demonstrates both single-GPU and multi-GPU scenario. The multi-GPU scenario spawns the specified number of processes. The data loading and model evaluation are separated into the run function which is executed in each process.

This demonstrates how to set up a distributed data loader to be used in the evaluation step to make sure that the data used for evaluation is distributed properly across multiple GPUs.

**Code Example 3: All-Gather for Consolidated Metrics**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
      super(SimpleModel, self).__init__()
      self.fc = nn.Linear(10, 2)

    def forward(self, x):
      return self.fc(x)

def create_dummy_data(batch_size, rank):
    input_data = torch.randn(batch_size, 10).to(rank)
    labels = torch.randint(0, 2, (batch_size,)).to(rank)
    return input_data, labels

def evaluate_and_gather_metrics(model, data, labels, rank, world_size):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
       outputs = model(data)
       loss = criterion(outputs, labels)

    gathered_losses = [torch.zeros_like(loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss)
    # Aggregate losses
    total_loss = sum(gathered_losses) / len(gathered_losses)
    return total_loss

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(run,
                                    args=(world_size,),
                                    nprocs=world_size)
    else:
        #Single GPU Scenario
        rank = 0
        model = SimpleModel()
        model = model.to(rank)
        data, labels = create_dummy_data(batch_size=32, rank=rank)
        total_loss = evaluate_and_gather_metrics(model, data, labels, rank=rank, world_size=1)
        print(f"Loss: {total_loss}")
    
def run(rank, world_size):

    setup(rank, world_size)
    model = SimpleModel()
    model = model.to(rank)
    data, labels = create_dummy_data(batch_size=32, rank=rank)
    total_loss = evaluate_and_gather_metrics(model, data, labels, rank=rank, world_size=world_size)
    
    if rank == 0:
        print(f"Loss: {total_loss}")
    
    cleanup()

```

**Commentary:**

1.  This example demonstrates the use of `dist.all_gather`. After calculating metrics locally on each GPU, `all_gather` copies loss tensors from all devices to each individual device. This is needed so the total average loss across all devices can be calculated in each device.
2. The evaluation and gathering of metrics are separated into a function called `evaluate_and_gather_metrics`.
3. The main function sets up the multiple process setup if multiple GPUs are present. Single GPU is also handled in the main function. The logic for evaluation and metrics gathering are the same in single and multi GPU scenarios.

This example clearly shows the `all_gather` function call and calculation of global metric.

In summary, evaluating a distributed model checkpoint accurately requires careful consideration of how weights are distributed and how metrics should be aggregated across GPUs. Implementing `torch.distributed`, `DistributedDataParallel`, and `DistributedSampler` correctly are pivotal for a seamless and effective multi-GPU evaluation workflow.

For further investigation into distributed training and evaluation techniques I suggest referring to the official PyTorch documentation on distributed data parallelism, and the associated tutorials for building robust multi-GPU workflows. Additionally, research papers and blog posts by seasoned practitioners often detail best practices based on extensive practical experience, providing practical insights not necessarily covered in standard API documentation. Finally, scrutinizing open-source implementations of established model repositories can also provide valuable guidance and understanding on efficient distributed model evaluation techniques.
