---
title: "Is this possible to train a Neural Network using DDP with a different number of GPUs in two nodes?"
date: "2024-12-14"
id: "is-this-possible-to-train-a-neural-network-using-ddp-with-a-different-number-of-gpus-in-two-nodes"
---

no problem, let's break this down. dealing with distributed data parallel (ddp) and varied gpu counts across nodes, it's definitely a situation i've bumped into more than a few times. it can seem tricky initially, but with a little understanding of how ddp works, and how we handle uneven hardware setups it's very doable. i've spent more nights than i'd like to recall staring at console outputs, so i have a few insights on the matter.

first, the core issue: ddp, in its typical usage, expects that each process will have an equal share of the data and be running on a similar hardware setup with the same number of gpus. that's the 'data parallel' part—each device gets a chunk, everyone computes in parallel and gradients get synced. now, when we throw in different gpu counts, like node one with four gpus and node two with two, we throw a wrench into the works. the default ddp setup tends to expect equal amounts of devices across all nodes.

when ddp sees unequal devices across the two nodes, the default behavior is usually it will try to create the same process count on each node, which isn't possible if the gpus are different, this leads to process crashes, or hangs, or some other nasty behavior that leaves you staring at your terminal wondering where your evening went.

but this isn't a dead end at all, here are a few strategies that have worked for me.

the first key point to remember is that even when you have unequal numbers of gpus, ddp is based on the 'rank'. each process running on each gpu will have its rank. the crucial thing to do is to manage and make this rank aware at the beginning of our training.

my personal favorite method is to think about ddp as 'processes per node' instead of gpus, this way we have to programmatically decide how many processes we are gonna have per node and how we are going to distribute the data.

the basic idea is to explicitly control the process launching on each node and to make sure that the data is sliced accordingly for each process, regardless of how many gpus that node has.

let's say node one has four gpus and node two has two. instead of running one process per gpu and letting ddp attempt to reconcile this uneven distribution, we treat each node as the fundamental unit. for example, we can still launch our application, but we only create one process per node, the process will then internally handle all of the gpus it has.

here's a snippet on how you could structure the launch part for a distributed application using something like pytorch's `torch.distributed.launch`:

```python
# this should be the shell script you execute in your nodes.
# this is an example for 2 nodes.
# you would execute this script two times:
# one on the first node and another on the second node.
# node one:
# python -m torch.distributed.launch --nproc_per_node=1  --nnodes=2 --node_rank=0 --master_addr="<node_one_ip>" --master_port=12355 train.py --num_gpus_per_node=4
# node two:
# python -m torch.distributed.launch --nproc_per_node=1  --nnodes=2 --node_rank=1 --master_addr="<node_one_ip>" --master_port=12355 train.py --num_gpus_per_node=2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_loop(rank, world_size, num_gpus_per_node):

    setup(rank, world_size)
    model = nn.Linear(10, 1).cuda(rank*num_gpus_per_node) # basic example, remember to adjust it
    ddp_model = DDP(model, device_ids=[rank*num_gpus_per_node], find_unused_parameters=True)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    # here the data loading logic will be different and will take into account the number of gpus
    # and also each device id in each node.
    # usually using a sampler to only give specific batches to each process.
    for i in range(100): # just for example
        inputs = torch.randn(100, 10).cuda(rank*num_gpus_per_node) # again, adjust your batch size.
        target = torch.randn(100, 1).cuda(rank*num_gpus_per_node)
        outputs = ddp_model(inputs)
        loss = nn.MSELoss()(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0 and i % 10 == 0:
            print(f"epoch: {i}, loss: {loss.item()}")
    cleanup()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus_per_node", type=int, default=1, help="num of gpus")
    args = parser.parse_args()

    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    train_loop(rank, world_size, args.num_gpus_per_node)

```

in the code above, we use `--nproc_per_node=1`, to force the launching to have only one process per node. `--nnodes=2` tells the distributed launching that we will use two nodes. the `--node_rank` tells each node its rank, starting in 0, and the `--master_addr` and `--master_port` tells each node where to communicate, the node 0 will work as the master.

then, the `train.py` script gets these parameters and we can know which node is running and how many gpus are available, and it's in charge to launch distributed processes within the node, to utilize the multiple gpus.

here's what's happening:

*   **process per node**: we launch a single process on each node, this process is responsible for handling all gpus of that node.
*   **distributed initialization**: the key idea is we are going to set the world size as the amount of nodes (two in this example). ddp doesn't care what you do within the node.
*   **data handling**: the most important part, now we handle data with samplers or use a custom implementation of the dataset to only give each process, its part of the data.

a second approach, if you're not fully comfortable changing the launch parameters is to keep the regular launch but, use a custom sampler for your dataset. the idea is to provide to each process, only the part of the dataset corresponding to its rank, this ensures that even with different numbers of gpus per node, the training is balanced and the gradient updates are consistent across the network.

let's suppose you have a dataset called `my_dataset`, you can do something like this to distribute the data based on the rank:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class MyDataset(Dataset):
    def __init__(self, total_size):
        self.total_size = total_size
    def __len__(self):
        return self.total_size
    def __getitem__(self, idx):
        return torch.randn(10), torch.randn(1) # random inputs and targets.

def train_with_sampler(rank, world_size):

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dataset = MyDataset(total_size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    model = torch.nn.Linear(10, 1).cuda(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(10):
        sampler.set_epoch(epoch) # very important, to have random batches in each epoch
        for inputs, target in dataloader:
            inputs = inputs.cuda(rank)
            target = target.cuda(rank)
            outputs = ddp_model(inputs)
            loss = torch.nn.MSELoss()(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.distributed.init_process_group("nccl", init_method='env://')
    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    train_with_sampler(rank, world_size)
```

in this second example, we are using the `DistributedSampler` to give each process, the portion of the data corresponding to its `rank`. this sampler, has the `set_epoch` function that is used to make sure that the batches are different in each training epoch.

the first example we change the launch configuration to have only one process per node, so the data parallel is handled within each node and the second example, uses a sampler, to distribute data within the processes, these are the most common two ways.

finally, another important detail is the usage of `find_unused_parameters`, in most cases, when we have models with some complex layers it's beneficial to have this enabled, if not, you might have unexpected crashes or weird bugs. i've learned that one by painful experience.

in summary, training with uneven gpu counts across nodes requires a slightly different perspective on how ddp is used and how the processes are organized and launched, but it is perfectly achievable. it's less about 'hacking' ddp and more about being explicit about your process management and your data handling.

for further reading i would recommend the pytorch documentation, of course, specifically the section on distributed training. in addition to that the paper 'efficient large-scale distributed training with pytorch' might be helpful. and for a more general background on distributed computing, i found the book 'distributed algorithms' by nancy lynch to be quite illuminating.

finally, remember that this is distributed computing, sometimes problems arise unexpectedly, i've spent hours looking at network configurations only to realize that i was missing a comma somewhere in my code, so debugging is an important skill that you'll develop during this journey. in fact, i had a similar issue once, spent 5 hours debugging, just to find out that i had the ethernet cable a little bit loose. it's funny now, but not that much at the time.
