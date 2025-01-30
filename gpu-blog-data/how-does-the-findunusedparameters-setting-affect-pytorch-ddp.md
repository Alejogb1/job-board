---
title: "How does the `find_unused_parameters` setting affect PyTorch DDP training with multiple forward/backward passes?"
date: "2025-01-30"
id: "how-does-the-findunusedparameters-setting-affect-pytorch-ddp"
---
The `find_unused_parameters` argument in PyTorch's Distributed Data Parallel (DDP) module is often a source of confusion, particularly when dealing with models that have multiple forward passes or dynamically altered computation graphs. It directly impacts the efficiency of gradient synchronization during distributed training. Misunderstanding its implications can lead to performance bottlenecks, incorrect gradient calculations, or even training failures.

By default, DDP assumes that all parameters of a model participate in every forward and backward pass. This allows for an efficient all-reduce operation on gradients, where each process contributes its local gradient to compute the global average for weight updates. However, some models employ conditional execution paths; not all parameters will be used in every iteration. If parameters are unused, their gradients will be zero. In such cases, DDP's all-reduce, without accounting for this, will still attempt to gather and reduce gradients from these unused parameters, which is computationally wasteful.

The `find_unused_parameters=True` setting tells DDP to dynamically identify, before each backward pass, which parameters actually participated in the forward pass. It accomplishes this by traversing the computational graph generated during the forward propagation to determine which parameters were used in the current iteration. Only gradients from parameters identified as “used” are then communicated across the processes. This approach reduces unnecessary data transfer and computation during gradient aggregation, particularly when a significant portion of the model’s parameters may be unused in a given iteration.

However, the trade-off is increased overhead. The dynamic graph traversal adds to the computation time of each backward pass. This overhead is typically negligible for large models and complex conditional logic but might be noticeable for smaller models with less variability in their computational graphs. The decision to use this setting hinges on whether the gains from reduced communication outweigh the added computational cost of finding unused parameters.

My experience implementing a dynamic segmentation network, where only a subset of the network branch was active based on input conditions, demonstrated this trade-off clearly. Initially, I ran with the default `find_unused_parameters=False`. This resulted in slow performance as all gradients were being communicated even though significant portions of the model's parameters had gradients of zero in a given iteration. Enabling `find_unused_parameters=True` significantly improved training speed, outweighing the computational overhead by a large margin. The number of processes did not particularly change the outcome, the performance gain with `find_unused_parameters=True` scaled as the dataset grew in size and the conditional logic was more dynamic.

Let's illustrate with code examples. The first example shows a simple model with conditional branching where `find_unused_parameters` is *not* enabled:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)
        self.linear3 = nn.Linear(20, 30)

    def forward(self, x, condition):
        x = self.linear1(x)
        if condition:
          x = self.linear2(x)
        else:
           x = self.linear3(x)
        return x

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    model = ConditionalModel().to(dist.get_rank())
    ddp_model = DDP(model, device_ids=[dist.get_rank()]) # find_unused_parameters is False by default
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    input_data = torch.randn(4,10).to(dist.get_rank())
    condition = bool(dist.get_rank() % 2) # Different condition per process for demonstration

    output = ddp_model(input_data,condition)
    loss = torch.sum(output)
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()
```

In this setup, every parameter, `linear1`, `linear2`, and `linear3` gets its gradients gathered and synchronized in the all-reduce operation of DDP. Even if only either `linear2` or `linear3` were used in the forward pass on a particular device.

The next example shows the same model with `find_unused_parameters` set to `True`:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)
        self.linear3 = nn.Linear(20, 30)

    def forward(self, x, condition):
        x = self.linear1(x)
        if condition:
          x = self.linear2(x)
        else:
           x = self.linear3(x)
        return x

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    model = ConditionalModel().to(dist.get_rank())
    ddp_model = DDP(model, device_ids=[dist.get_rank()], find_unused_parameters=True)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    input_data = torch.randn(4,10).to(dist.get_rank())
    condition = bool(dist.get_rank() % 2) # Different condition per process for demonstration

    output = ddp_model(input_data,condition)
    loss = torch.sum(output)
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()
```

In this case, DDP will correctly identify that either `linear2` or `linear3` wasn't used in the current pass on each of the processes. Only gradients for parameters used on each rank are communicated during the all-reduce operation. Note that even if it is the same layer, if the rank doesn't execute it, it is skipped.

The final example explores a situation with multiple forward passes, where the first forward pass determines parameters for the second. A variant of a recurrent network, without explicit loops:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class MultiForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_init = nn.Linear(10,20)
        self.linear_first = nn.Linear(20, 30)
        self.linear_second = nn.Linear(30, 40)


    def forward(self,x):
        init_embed = self.linear_init(x)
        first_pass = self.linear_first(init_embed)

        second_pass = self.linear_second(first_pass)
        return second_pass

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    model = MultiForwardModel().to(dist.get_rank())
    ddp_model = DDP(model, device_ids=[dist.get_rank()], find_unused_parameters=True)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    input_data = torch.randn(4,10).to(dist.get_rank())

    output = ddp_model(input_data)
    loss = torch.sum(output)
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()
```

In this scenario, all layers (`linear_init`, `linear_first`, and `linear_second`) are *always* used in both passes. However, DDP correctly tracks all the parameters used during the first and second forward passes before performing the all-reduce. Setting `find_unused_parameters=False` would also work in this scenario, since all parameters are always used. However, should the second pass be made conditional, `find_unused_parameters=True` would be required for correct execution without synchronizing gradients for the unused layers during the second pass.

When considering whether to use `find_unused_parameters`, I recommend profiling your model with and without the setting to determine its effect on performance. It's not always a strict win; the overhead might not be worth it.

For further study, I would suggest reading the PyTorch Distributed documentation for details on the DDP module, and the PyTorch internals documentation on Autograd. Also, consulting any book covering advanced PyTorch usage, specifically the chapters concerning distributed training will be a great help.
