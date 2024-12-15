---
title: "Why is PyTorch's data parallel not using all my GPUS (and it seems to not speed up my training either)?"
date: "2024-12-15"
id: "why-is-pytorchs-data-parallel-not-using-all-my-gpus-and-it-seems-to-not-speed-up-my-training-either"
---

alright, so you're seeing a classic pitfall with pytorch's data parallel, where it's seemingly ignoring some of your gpus and not giving you the performance boost you'd expect. i've been there, spent more nights than i care to recall staring at tensorboard and wondering what went wrong. it's never a pleasant feeling when you think your code should be flying but it's just kinda...puttering along.

let's break down why this happens, and some ways to troubleshoot. the first and most common issue i’ve seen is that the `dataparallel` implementation, while convenient, isn't always the most efficient way to parallelize training, particularly when you have a large number of gpus or when your model has some architectural specifics. it basically replicates the model across all available gpus and then shuffles the batches of your input data to those gpu. this is a simple solution that works most of the times but its drawbacks are many.

the initial clue, and probably the one i missed myself when i started with pytorch and parallel training, is to understand how `dataparallel` actually works under the hood. it creates copies of your model on each gpu. it's important that it means each copy is a full copy. so if your model is big, you’re using a huge chunk of memory per gpu just storing it. that's already a warning signal. then, during each training step it copies the data input into the corresponding gpus and each gpu computes the forward pass and backward pass locally. after that, it copies the gradients to the main gpu, averages them and then updates the model. that is a lot of copying between gpus and the main cpu.

that data movement is often the culprit. if the communication between the gpus and the main cpu is not fast enough, the computation units stay idle waiting for the data to arrive and become available. the gpu utilization becomes low, not to mention that this also can affect the performance on your main cpu. the more gpus you use the more you might notice it. that means that in some cases even with more gpus the training time might even be higher than with just one gpu, due to all the overhead.

 another very common situation is when your model itself has some parts that are only running on the main gpu. for instance, let’s say you have a custom layer that performs some post-processing or a very particular loss function only running on cpu or the primary gpu. these operations are sequential and often it’s not trivial to parallelize it which means you are still suffering from a bottleneck and not scaling properly. data parallel relies on having all or a good portion of the computations happening within the gpus and not on the cpu and main gpu.

here’s a first step i usually take when seeing this issue. i'll often check which gpus pytorch is even seeing using a simple command:

```python
import torch

if torch.cuda.is_available():
    print(f"number of gpus available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i} name: {torch.cuda.get_device_name(i)}")
else:
    print("cuda is not available on this machine.")
```

this will output the number of gpus and their names which is a good thing to have to be sure that pytorch is seeing everything and there is no hardware or driver issue. if that is fine, then i start checking the gpu utilization during training with `nvidia-smi`. if all gpus are there but only one shows a reasonable percentage of usage, and the other gpus are pretty much idle, it's clear that the problem is not that pytorch is not aware of the available devices. it’s a problem of utilization. in this situation there are a few things you can do and that i have applied many times to get good results.

first, and this is the approach i prefer, is to ditch `dataparallel` and move to `distributeddataparallel`. it is more complex to setup but it’s very good at avoiding many of the issues that we have already mentioned. it uses a different approach, spawning separate processes for each gpu instead of replicating the model on the primary gpu. this approach reduces a significant amount of the overhead related to the primary cpu, the transfer of data between cpu/gpu as well as the reduction of gradients. the communication between gpus is greatly reduced, thus increasing the utilization and achieving faster training. the downside is that is a bit more involved to setup and use. here’s a basic example. it’s not a full ready-to-run code but should give you a general idea:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    model = nn.Linear(10, 2).to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(100):
      #simplified training loop (without dataloader)
      input_tensor=torch.randn(20, 10).to(rank)
      target=torch.randn(20,2).to(rank)
      optimizer.zero_grad()
      output = ddp_model(input_tensor)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

      if rank == 0:
        print(f'epoch: {epoch}, loss:{loss.item()}')

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

this code is doing a minimal training loop using distributed data parallel. we are creating a process for each gpu and all the training is happening in each process using each gpu. notice that now the model is loaded directly to the gpu (`.to(rank)`), and then it is wrapped using `DistributedDataParallel`. to be able to run this example you need to have installed the necessary libraries and you need to use the multi-process launcher of pytorch. the syntax is `python -m torch.distributed.launch --nproc_per_node <num gpus> <your_file>.py`, where `<num gpus>` is the number of gpus that you have. this is just a minimal working example but it is important to understand how the training loop has to be changed to use `distributeddataparallel`. the important things to note are the usage of the rank variable, the setup and cleanup, and the `device_ids` when you initialize the distributed model.

another aspect to consider is data loading. if you are using a single data loader feeding data to all gpus then you might have a serious bottleneck on your primary cpu. you need to make sure that each process has its own dataloader for its own gpu. we want each gpu to do the data loading locally. pytorch provides samplers that you can use in order to achieve this. here is a simple snippet of code:

```python
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size=size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(10), torch.randn(2) #random inputs and targets

def get_dataloader(rank, world_size, batch_size):
    dataset=DummyDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False) # we pass the sampler
```

here we are creating a custom dataset and then when we initialize the dataloader we pass a `DistributedSampler` that will make sure that only data from the fraction corresponding to the process rank is loaded. a sampler is an object used in the dataloader to tell how to sample the data from the dataset, and in our case the sampler distributes the data over the gpus. we set shuffle to false, since the sampler is already shuffling the data. that means that in the training loop example that i've shown before you need to replace your data loader with something like this. for instance, instead of this:

```python
      input_tensor=torch.randn(20, 10).to(rank)
      target=torch.randn(20,2).to(rank)
```

you would have something like:

```python
    dataloader = get_dataloader(rank, world_size, batch_size=20)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
      inputs=inputs.to(rank)
      targets=targets.to(rank)
      optimizer.zero_grad()
      output = ddp_model(inputs)
      loss = criterion(output, targets)
      loss.backward()
      optimizer.step()
```
and i promise that you will probably notice a good performance boost on your gpus utilization. you should use tensorboard or `nvidia-smi` to check your results.

finally, if you're dealing with a particularly large model, you might find that even distributed data parallel is hitting memory limits. in such cases, exploring model parallelism in conjunction with data parallelism can be necessary. this is more advanced but i have used it in several occasions. the basic idea is to split the model itself over several gpus as well as the data, making it a more complex but sometimes necessary approach.

there are good resources available for a deeper dive. for data parallelism, i highly recommend reading pytorch’s official documentation. it is quite good, specially the section about distributed data parallel. the deep learning book by goodfellow, bengio, and courville covers parallelization techniques with a strong mathematical foundation. another good source to complement that reading is the book "distributed training of deep learning models" by mohamed elgendy. although this book is more on a higher level it gives you a good overview of the techniques and tools available.

i hope these points help. i've lost my fair share of hair over this stuff myself (i'm thinking about getting a wig, just kidding) so feel free to ask for more specifics if needed. i'll try to help.
