---
title: "How to gather and concatenate tensor arrays of varying lengths using PyTorch distributed multiprocessing?"
date: "2025-01-30"
id: "how-to-gather-and-concatenate-tensor-arrays-of"
---
Asynchronous data collection in distributed PyTorch training, specifically when handling tensor arrays of varying lengths, presents a unique synchronization challenge. Simple concatenation after collection across processes can lead to significant errors, particularly if not every process contributes the same number of elements. The core problem arises from the variable nature of the data being produced within each distributed worker, making traditional collective operations like `torch.cat` unsuitable without careful preprocessing and alignment.

The principal issue stems from the fact that in distributed training, each process operates in relative isolation, generating tensors based on its assigned partition of the dataset. These tensors often vary in size, reflecting differences in data augmentation, batch processing, or sampling strategies applied locally. A naive attempt to directly concatenate these varying-length tensors across all processes would typically result in a tensor with inconsistent dimensions and ultimately, calculation errors.

My experience working on a large-scale sequence-to-sequence model trained with distributed data parallelism revealed this challenge acutely. Specifically, we employed dynamic sequence lengths to reduce padding overhead, leading to widely varying tensor sizes across processes after each model forward pass. To reliably collect these tensors, we need a multi-stage process involving local collection, size communication, global size agreement, and then finally, a coordinated data exchange.

To facilitate this, a robust procedure often involves the following steps. First, every process aggregates its locally generated tensors into a list. Second, each process communicates the length of their local list to all other processes. This communication is performed through a collective operation using PyTorch’s distributed package. Third, using the exchanged lengths, all processes determine the final size of the consolidated tensor across all processes. This step allows each process to construct an appropriately sized tensor to receive the combined data. Finally, after agreeing on the final output dimensions, each process sends their local tensor data to all other processes. This step often necessitates careful memory management and precise communication indices.

Here are three example implementations that reflect this methodology with a focus on varying complexity and scalability.

**Example 1: Basic Gathering with Explicit Communication:**

This example illustrates the most basic approach. Each process generates a tensor array of a random length, communicates that length, and then receives data from all processes into a preallocated receiving tensor.

```python
import torch
import torch.distributed as dist
import random

def gather_tensors_basic(local_tensors):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    local_lengths = torch.tensor([len(local_tensors)], dtype=torch.int32).to(rank)
    all_lengths = [torch.zeros_like(local_lengths) for _ in range(world_size)]
    dist.all_gather(all_lengths, local_lengths)
    all_lengths = torch.cat(all_lengths)
    
    global_total_length = int(torch.sum(all_lengths))
    
    output_tensor = torch.zeros(global_total_length, dtype=torch.float32)
    
    send_tensor = torch.cat(local_tensors) if local_tensors else torch.tensor([], dtype=torch.float32) 
    
    offset = int(torch.sum(all_lengths[:rank]))
    
    if send_tensor.numel() > 0:
        output_tensor[offset:offset+send_tensor.numel()] = send_tensor.cpu()
        
    for source_rank in range(world_size):
      if source_rank == rank:
        continue
      
      source_offset = int(torch.sum(all_lengths[:source_rank]))
      source_length = int(all_lengths[source_rank])

      if source_length > 0:
        source_tensor = torch.zeros(source_length, dtype=torch.float32)
        dist.recv(tensor=source_tensor, src=source_rank)
        output_tensor[source_offset:source_offset+source_tensor.numel()] = source_tensor

    if rank != 0:
        dist.send(tensor=send_tensor.cpu(), dst=0)
    
    if rank == 0:
        return output_tensor
    else:
        return None
    
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_tensors = [torch.randn(random.randint(1, 5)) for _ in range(random.randint(0,3))]
    gathered_tensor = gather_tensors_basic(local_tensors)

    if rank == 0:
      print(f"Process {rank}: Collected tensor of size {gathered_tensor.shape}")
    
    dist.destroy_process_group()
```

This basic example directly showcases the exchange of local lengths and data through explicit sends and receives. The root process, usually rank 0, accumulates all the tensor data. Note that while functional, the communication overhead of sending to/from every rank directly to rank 0 could be limiting for a large number of processes.

**Example 2: Utilizing `all_gather_into_tensor` for Efficiency:**

This revised example aims for more efficient memory management and communication by utilizing the `all_gather_into_tensor` for the exchange of both lengths and data. This operation is optimized to write directly into a preallocated global tensor.

```python
import torch
import torch.distributed as dist
import random

def gather_tensors_efficient(local_tensors):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    local_length = torch.tensor([len(local_tensors)], dtype=torch.int32).to(rank)
    all_lengths = torch.zeros(world_size, dtype=torch.int32).to(rank)
    dist.all_gather_into_tensor(all_lengths, local_length)
    
    global_total_length = int(torch.sum(all_lengths))
    output_tensor = torch.zeros(global_total_length, dtype=torch.float32).to(rank)

    send_tensor = torch.cat(local_tensors) if local_tensors else torch.tensor([], dtype=torch.float32).to(rank)
    send_tensor_length = send_tensor.numel()

    all_send_tensor_lengths = torch.zeros(world_size, dtype=torch.int32).to(rank)
    local_send_length = torch.tensor([send_tensor_length],dtype=torch.int32).to(rank)
    dist.all_gather_into_tensor(all_send_tensor_lengths, local_send_length)
    
    
    all_tensor_parts = torch.zeros(global_total_length, dtype=torch.float32).to(rank)

    offsets = [int(sum(all_send_tensor_lengths[:i])) for i in range(world_size)]
    
    if send_tensor_length > 0:
        all_tensor_parts[offsets[rank]:offsets[rank]+send_tensor_length] = send_tensor
    dist.all_reduce(all_tensor_parts, op=dist.ReduceOp.SUM)
        
    return all_tensor_parts
    
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_tensors = [torch.randn(random.randint(1, 5)) for _ in range(random.randint(0,3))]

    gathered_tensor = gather_tensors_efficient(local_tensors)
    print(f"Process {rank}: Collected tensor of size {gathered_tensor.shape}")
    
    dist.destroy_process_group()
```

Here, the use of `all_gather_into_tensor` for both the lengths of tensor arrays as well as the flattened send tensors allows for a more optimized data transfer, reducing multiple steps into one communication operation. A crucial optimization is the use of `all_reduce` to sum all the contributions into the full result tensor, which will have the result for each rank.

**Example 3: Employing a Sparse Representation and Collective Communications:**

This third example focuses on a scenario with highly varying lengths where we might want to leverage a sparse representation and further utilize collective operations for maximal efficiency.  This is particularly relevant when the total length across all processes is significantly larger, leading to memory concerns if a dense tensor is utilized during communication.

```python
import torch
import torch.distributed as dist
import random
from torch.nn.utils.rnn import pad_sequence

def gather_tensors_sparse(local_tensors):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    local_lengths = torch.tensor([len(local_tensors)], dtype=torch.int32).to(rank)
    all_lengths = [torch.zeros_like(local_lengths) for _ in range(world_size)]
    dist.all_gather(all_lengths, local_lengths)
    all_lengths = torch.cat(all_lengths).to(rank)

    max_length = max(all_lengths).item()
    
    padded_local_tensors = pad_sequence([torch.cat(local_tensors)] if local_tensors else [torch.tensor([], dtype=torch.float32).to(rank)], batch_first=True, padding_value=0.0).to(rank)
    
    all_tensors_padded = torch.zeros(world_size, max_length, dtype=torch.float32).to(rank)

    all_tensors_padded[rank] = padded_local_tensors
    dist.all_reduce(all_tensors_padded, op=dist.ReduceOp.SUM)
    
    
    mask = torch.zeros(world_size, max_length, dtype=torch.bool).to(rank)
    
    for i, length in enumerate(all_lengths):
      mask[i, :length] = True
    
    final_tensor = all_tensors_padded[mask]

    return final_tensor

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_tensors = [torch.randn(random.randint(1, 5)) for _ in range(random.randint(0,3))]

    gathered_tensor = gather_tensors_sparse(local_tensors)
    print(f"Process {rank}: Collected tensor of size {gathered_tensor.shape}")
    dist.destroy_process_group()
```

This example employs padding via `pad_sequence` to create tensors of equal sizes per process prior to global reduction using `all_reduce`.  This approach avoids explicit point-to-point communication, and the dense tensor formed by `all_reduce` is then masked to obtain the concatenated output.  This provides performance and usability advantages for high variability in input lengths.

These examples highlight key variations in strategies for gathering variable-length tensor arrays in a distributed PyTorch environment. Selection of the optimal approach depends on the specific characteristics of the data distribution, the system's communication capabilities, and the computational resources available.

For further exploration, I would recommend focusing on:

1.  **PyTorch Distributed Documentation:** The official PyTorch documentation provides comprehensive details on all distributed operations.
2.  **DeepSpeed and Horovod Tutorials:** Examine how these libraries handle similar challenges with optimized and scalable implementations.
3.  **Research Papers:** Review studies that benchmark different distributed data parallel approaches and communication algorithms.
