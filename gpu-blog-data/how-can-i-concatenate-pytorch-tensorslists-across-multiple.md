---
title: "How can I concatenate PyTorch tensors/lists across multiple nodes in a distributed setup?"
date: "2025-01-30"
id: "how-can-i-concatenate-pytorch-tensorslists-across-multiple"
---
Direct insight: Effective concatenation of PyTorch tensors or lists across multiple nodes in a distributed environment necessitates careful consideration of data locality and communication overhead. The naive approach of gathering all data to a single node before concatenating is often detrimental to performance, particularly with large tensors. I've encountered this bottleneck firsthand when training large language models distributed across multiple GPUs on a cluster.

The core challenge stems from the fact that in distributed training, data is typically partitioned across different nodes. These nodes operate mostly independently, and any interaction requires explicit communication. Simple Python concatenation, `torch.cat()` or list concatenation (`+`), operates solely within a single process and hence is inappropriate. To achieve cross-node concatenation, we must leverage PyTorch's distributed communication primitives.

Here's a breakdown of the most common strategies:

**1. Utilizing `torch.distributed.all_gather()`:**

The `all_gather()` operation is the foundation for cross-node concatenation of *tensors*. It works by collecting tensors of the same shape from all processes and placing them into a list of tensors at *every* process. Imagine each node possessing a slice of a tensor. `all_gather` gathers *all* the slices and redistributes the entire resulting tensor to each individual node. This ensures that after `all_gather`, every node has the full concatenated tensor. This is a strong choice when each process has a meaningful subset of the final tensor and you need the entire assembled result on *every* node.

Here's an example:

```python
import torch
import torch.distributed as dist

def distributed_tensor_concatenate_all_gather(local_tensor):
  """Concatenates a local tensor across all processes using all_gather."""

  world_size = dist.get_world_size()
  rank = dist.get_rank()

  # Ensure all tensors are of the same dimension except the dim we want to concatenate on
  all_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]
  dist.all_gather(all_tensors, local_tensor)

  # Concatenate along the first dimension (dim=0)
  concatenated_tensor = torch.cat(all_tensors, dim=0)
  return concatenated_tensor

if __name__ == "__main__":
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=2, rank=0)
    local_tensor_1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    concatenated_tensor_1 = distributed_tensor_concatenate_all_gather(local_tensor_1)
    print(f"Rank {dist.get_rank()} concatenated tensor: {concatenated_tensor_1}")
    dist.destroy_process_group()
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=2, rank=1)
    local_tensor_2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    concatenated_tensor_2 = distributed_tensor_concatenate_all_gather(local_tensor_2)
    print(f"Rank {dist.get_rank()} concatenated tensor: {concatenated_tensor_2}")
    dist.destroy_process_group()
```

**Commentary:**
   - I initialize the distributed environment using `dist.init_process_group()` which would happen earlier in a real distributed script. We demonstrate the behavior of `all_gather` on two separate runs.
   - In each run the individual process’s tensor is passed to `all_gather` which is going to populate a list of tensors of the same size and shape as the input. Note this is done on every process.
   - Each process then concatenates the list of tensors along `dim=0`, so the input tensors will stack on top of each other, which results in every process having the full concatenated tensor. This is not always desirable, and is a source of increased memory usage, but is sometimes the fastest method of concatenation.

**2. Utilizing `torch.distributed.gather()` and manual concatenation on a single node:**

  The `gather()` operation is a variant of `all_gather`. It collects tensors from all processes, but aggregates them only on a designated *rank*. The rest of the processes do *not* receive the full assembled tensor. This approach is useful when only one node requires the concatenated tensor. In my experience, I’ve found `gather()` particularly beneficial when processing evaluation metrics in distributed training, where the final aggregation is only needed on the primary process.

  Here's an example:

```python
import torch
import torch.distributed as dist

def distributed_tensor_concatenate_gather(local_tensor, gather_rank=0):
    """Concatenates a local tensor across all processes using gather, only on a designated rank."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    all_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)] if rank == gather_rank else None
    dist.gather(local_tensor, all_tensors, dst=gather_rank)

    if rank == gather_rank:
      concatenated_tensor = torch.cat(all_tensors, dim=0)
      return concatenated_tensor
    else:
      return None


if __name__ == "__main__":
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=2, rank=0)
    local_tensor_1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    concatenated_tensor_1 = distributed_tensor_concatenate_gather(local_tensor_1)
    if dist.get_rank() == 0:
        print(f"Rank {dist.get_rank()} concatenated tensor: {concatenated_tensor_1}")
    dist.destroy_process_group()
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=2, rank=1)
    local_tensor_2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    concatenated_tensor_2 = distributed_tensor_concatenate_gather(local_tensor_2)
    if dist.get_rank() == 0:
        print(f"Rank {dist.get_rank()} concatenated tensor: {concatenated_tensor_2}")
    dist.destroy_process_group()
```

**Commentary:**

   - The `gather` operation in the function `distributed_tensor_concatenate_gather` only populates the `all_tensors` list on the process specified by `gather_rank`, which in this case, is rank 0.
   - The `if rank == gather_rank` block is the critical step. It ensures that the concatenation operation, `torch.cat()`, is performed *only* on the designated rank. Other processes will return None and their `all_tensors` will be a `NoneType`
   - This approach is crucial for reducing memory overhead when only a specific process requires the result.

**3. Concatenating lists of tensors using a combination of `gather` and list concatenation:**

   The above approaches focus on tensors which can be concatenated using `torch.cat()`. When we have a list of tensors or a list of other data structures that need to be assembled from different ranks, the process requires a combination of the distributed communication primitives along with some manual list manipulation. Often the underlying data structure can be transformed into a tensor for cross-node transfer. However, sometimes the local data is heterogeneous or more suited to list manipulation after gathering. I've found this pattern useful when dealing with complex data augmentations and need to assemble the results from each node before batching them together for a final processing step.

   Here's an example:
```python
import torch
import torch.distributed as dist
import random

def distributed_list_concatenate(local_list, gather_rank=0):
    """Concatenates a local list across all processes using gather, only on a designated rank."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    all_lists = [None for _ in range(world_size)] if rank == gather_rank else None
    dist.gather(local_list, all_lists, dst=gather_rank)

    if rank == gather_rank:
      concatenated_list = []
      for lst in all_lists:
          if lst is not None: # Ensure no None values are appended
            concatenated_list.extend(lst)
      return concatenated_list
    else:
      return None


if __name__ == "__main__":
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=2, rank=0)
    local_list_1 = [torch.tensor([random.random()],dtype=torch.float32) for _ in range(3)]
    concatenated_list_1 = distributed_list_concatenate(local_list_1)
    if dist.get_rank() == 0:
        print(f"Rank {dist.get_rank()} concatenated list: {concatenated_list_1}")
    dist.destroy_process_group()
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=2, rank=1)
    local_list_2 = [torch.tensor([random.random()],dtype=torch.float32) for _ in range(4)]
    concatenated_list_2 = distributed_list_concatenate(local_list_2)
    if dist.get_rank() == 0:
        print(f"Rank {dist.get_rank()} concatenated list: {concatenated_list_2}")
    dist.destroy_process_group()
```

**Commentary:**
  - In this example, we have a list of tensors to be concatenated across ranks.
   - Like with the `gather` example above, the `dist.gather()` call collects the local lists, `local_list`, from each process on the designated `gather_rank`. Other ranks will have `all_lists` equal to None.
   - Once the list of lists is received on the root process, they are concatenated with standard list operations (`extend`), ensuring only non-`None` values are appended to the `concatenated_list`.

**Resource Recommendations:**

For a deeper understanding of distributed PyTorch, I recommend exploring the official PyTorch documentation for `torch.distributed`. Further resources can be found in articles and blog posts that cover strategies for distributed training with PyTorch. Look for articles focused on "Data Parallel" training.
