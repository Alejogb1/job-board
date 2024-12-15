---
title: "How to Split a Dataset over Multiple GPUs?"
date: "2024-12-15"
id: "how-to-split-a-dataset-over-multiple-gpus"
---

alright, so you're looking at splitting a dataset across multiple gpus, huh? it's a common problem, and honestly, i've spent a good chunk of my career knee-deep in this sort of thing. it’s not always straightforward, but once you get the hang of it, it becomes pretty routine. i remember back in '09, working on this image recognition project. we had a mountain of data and a single, pathetic gpu. training times were measured in geologic eras. that's when i first started really diving into multi-gpu training, and boy, did i learn some lessons the hard way.

the core idea is that instead of having one gpu process the entire dataset, you divide the data and the computational workload across several gpus. it's like having a team of chefs instead of one guy trying to cook a banquet single-handedly. different frameworks and libraries offer different approaches, but the underlying principles are the same.

there are generally two main strategies: data parallelism and model parallelism. let’s focus on data parallelism since that's usually the first hurdle people encounter. model parallelism, while useful, gets complex real quick, and generally is for very, very large models which I guess is not your case here.

data parallelism basically means we duplicate the model across all gpus, and each gpu gets a different batch of data to chew on. after each batch, gradients are synchronized (usually averaged), and the model weights are updated together. this way, effectively, the model is trained using the entire dataset, distributed across the available hardware.

now, for splitting data across gpus, there are a few important things to consider. first, you need a data loader that understands the multi-gpu setup. most deep learning frameworks, like pytorch and tensorflow, have these built in, or have ways to easily implement them. second, you need to make sure your data loader is not a bottleneck. if it spends too much time loading data, all the gpus just sit around twiddling their thumbs, which is obviously not ideal. the data loading needs to be asynchronous, and if the data loading is slow then you can prefetch the data on your cpu then send it to the gpu.

here’s how i would do it in pytorch. i’m assuming you’ve already installed pytorch and have a data set and a model defined. you probably have something like this to start with (but probably not as simple):

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# dummy dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# dataset and dataloader, single gpu version
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model and optimizer, single gpu version
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# simple train loop, single gpu
for epoch in range(2):
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'epoch:{epoch} batch:{i} loss:{loss.item()}')
```

this is pretty basic, running on a single gpu, but let's say you have multiple gpus on the machine and you want to use them. here's how you'd change it for multi-gpu use with `torch.nn.DataParallel`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# dummy dataset (same as before)
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# dummy model (same as before)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# dataset and dataloader, multi-gpu version
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model and optimizer, multi-gpu version
model = SimpleModel()

# check if multiple gpus are available and use them
if torch.cuda.device_count() > 1:
    print(f'using {torch.cuda.device_count()} gpus!')
    model = nn.DataParallel(model)

# moves the model to the gpu
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# simple train loop, multi gpu
for epoch in range(2):
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'epoch:{epoch} batch:{i} loss:{loss.item()}')
```

a few key changes here: the most important one is that the model is wrapped in `nn.DataParallel`. this automatically handles the data distribution and gradient averaging across the gpus, assuming your input tensors are on cuda device, which you need to set before passing to the forward pass. this is the standard way to do multi-gpu data parallelization in pytorch and works out of the box.

now this solution has its quirks. in particular `torch.nn.DataParallel` has some scalability limitations that can result in suboptimal gpu utilization, especially if using multiple machines. but for most simple cases its more than good.

if you are willing to go further you should probably consider the following alternative which is a bit more flexible and gives you full control, and this is what i use now in almost all my projects:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import os

# dummy dataset (same as before)
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# dummy model (same as before)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# setting up the distributed training parameters (make sure to run with torchrun!)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    # dataset and dataloader
    dataset = SimpleDataset()

    # setting the sampler so each process can train on different portion of the data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)

    # model and optimizer
    model = SimpleModel().cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # simple train loop
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda(rank)
            labels = labels.cuda(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'rank:{rank} epoch:{epoch} batch:{i} loss:{loss.item()}')
    
    cleanup()

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    setup(rank, world_size)
    train(rank, world_size)

```

this snippet uses `torch.distributed` and `DistributedDataParallel`, giving you much more control over multi-gpu training and more flexibility. the most important change is that now we use `torch.utils.data.distributed.DistributedSampler` to generate different portion of the dataset to each process, and then `DistributedDataParallel` to sync the gradients of each process. in this way we make sure that each gpu is training on different batches in every iteration.

for this setup you need to launch each script with `torchrun --nproc_per_node {number_of_gpus_available} {script_name.py}` to initialize the process group. this is a bit more involved but scales better and handles multi-machine training well. and as the added bonus it makes it easier for you to set the correct data sampling procedure for each gpu.

and one more thing, remember that your data needs to get to your gpus in a timely manner. i've seen projects where the cpu was the bottleneck. in that case you can use techniques such as prefetching data or using a separate process dedicated for loading data, or even saving to memory the data, though depending on the size that might not be possible. so keep an eye on your cpu usage during training. also ensure you're using a reasonably sized batch size. too small and the gpus are underutilized, too large and you might run out of memory. there's a sweet spot, but that depends a lot on your specific problem.

i also recommend reading the pytorch documentation on distributed training, it's a gold mine of information. you can probably find also a few good papers on multi-gpu training techniques, especially ones about gradient averaging and distributed data loaders. i think there is one by sergei ioffe from google about batch normalization that covers some parts of this subject, check it out.

oh, and one more thing… i once spent a whole day debugging a multi-gpu setup, only to discover that the gpus weren't even running at the same frequency. i felt like i had traveled back in time to the first age of the internet. so double-check your hardware, it can save you a lot of headaches.

that’s about it. this should get you started. remember, the devil is in the details, and multi-gpu training can be tricky, but with some practice, it will become second nature. just remember to take the proper steps and don’t assume anything. and always double-check your configurations, those small things tend to become big problems very quickly. and dont use a mac.
