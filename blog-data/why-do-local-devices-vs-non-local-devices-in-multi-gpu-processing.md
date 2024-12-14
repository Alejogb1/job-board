---
title: "Why do Local devices VS non local devices in multi GPU processing?"
date: "2024-12-14"
id: "why-do-local-devices-vs-non-local-devices-in-multi-gpu-processing"
---

alright, let's talk about local vs non-local devices in multi-gpu setups. it's a topic i've spent more hours than i care to count debugging, so hopefully, i can shed some light on why this distinction matters.

when we're dealing with multiple gpus, we're inherently talking about parallel processing. instead of one gpu doing all the work, we're splitting it up. the way data is shuffled around between these gpus can have a huge impact on performance. the key difference lies in where the data sits in memory: either it's directly accessible by a gpu (local), or it has to be moved across the system (non-local).

imagine you have a server with two gpus: gpu0 and gpu1. each gpu has its own dedicated high-bandwidth memory. if you're running a workload where all the data that gpu0 needs is already in its memory, it's accessing that data *locally*. this is the ideal scenario. the access is fast, usually at bus speeds, and there's minimal overhead.

now, let's say gpu1 needs a chunk of data that's currently sitting in gpu0's memory, or even worse in the main system's ram (the cpu's memory). that's when we start talking about non-local access. accessing the memory of another gpu requires moving the data across the pcie bus, and even worse if it is in system memory. this is much slower than local access and introduces overhead.

the core of the performance hit lies in data transfer time. a local memory access is like grabbing something off a desk, whereas non-local access is like retrieving it from another room across a hallway or maybe even another floor of the building. the latency for these transfers and the limited bandwidth in pcie can quickly turn an efficient parallelized workload into a bottleneck.

i encountered this firsthand when i was working on this image processing project years ago. i had a system with four gpus. initially, i naïvely spread the data across all four and had some seriously miserable results. the processing itself was pretty quick, but the time spent moving data between the gpus dwarfed the actual computation time. i thought, hey, this is a quad-gpu system, it should be blazing fast, right? wrong. it turns out that the code was moving data back and forth between gpus every few milliseconds which really killed the performance of the project.

to address it, i had to get intimate with my data placement strategy. instead of scattering data everywhere, i started to think about data locality which is a fancy way to say: "keep the data as close to where it's used as possible." i reorganized my code to keep a good amount of data local to each gpu before i start the compute operation. that’s a way better approach.

here is a simple example in pytorch how you can move data to a local gpu:

```python
import torch

# checking for available gpus
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"number of available gpus: {device_count}")
    # creating data
    data = torch.randn(1000, 1000)
    # moving data to the first available gpu
    device = torch.device("cuda:0")
    data_gpu0 = data.to(device)
    # creating a matrix on the second gpu
    if device_count > 1:
      device = torch.device("cuda:1")
      data_gpu1 = torch.randn(1000,1000).to(device)
      print(f"data tensor moved to first gpu device: {data_gpu0.device}")
      print(f"data tensor moved to second gpu device: {data_gpu1.device}")
    else:
        print("only one gpu, skipping the second gpu allocation.")
else:
    print("no gpu available, the code will run in cpu.")
```
this example shows how to allocate data in a gpu. if your machine has two gpus, the output will show two different devices assigned to each tensor.

in that image processing project mentioned before, the bottleneck was the need to combine some features extracted from images that were being processed in different gpus. after a lot of experimentation, i discovered i could do some preliminary local processing on each gpu, combine results locally as much as possible, and only then transfer the minimum required data to one dedicated gpu to do the combination of features and final calculation. this made a huge difference.

the way we move data around between gpus is important, it is a trade-off, it might be faster to compute some redundant work than move a lot of data around. for example, let's say i have this numpy operation:
```python
import numpy as np
import time

def do_some_numpy_work(data):
  # some work here
  start_time = time.time()
  for _ in range(100):
      data = np.dot(data, data.T)
  end_time = time.time()
  return data, end_time - start_time
  
def copy_numpy_data(data, n_copies):
  start_time = time.time()
  copies = [data.copy() for _ in range(n_copies)]
  end_time = time.time()
  return copies, end_time - start_time

if __name__ == '__main__':
    data_size = 2000
    data = np.random.rand(data_size, data_size)
    numpy_work_results, numpy_work_time = do_some_numpy_work(data)
    numpy_copies_results, numpy_copies_time = copy_numpy_data(data, 2)

    print(f"time for numpy work: {numpy_work_time} seconds")
    print(f"time for copy of numpy array twice: {numpy_copies_time} seconds")
    
```

this code shows that copying the numpy array to a different location might take more time than doing the actual work on a single numpy array.
this is a simplified version of the problem, but the idea is that we have to think if it is worth to move data around or it is better to calculate some values in different gpus before doing the combination. this depends on the size of the data, the size of the computational work, and the connection speed of your pcie bus.

to get a better grasp of all this you need to understand the concept of memory hierarchy and data placement strategies. a classic book in this area is "computer organization and design: the hardware/software interface" by david a. patterson and john l. hennessy. it might seem like an older text, but understanding the fundamentals of memory management and data placement still very relevant for modern multi-gpu programming.

there are other books and papers specifically about multi-gpu programming, but if you are just starting i think it's fundamental to know the basics before advancing to the more complex stuff. a good place to look for papers would be the ieee xplore digital library. specifically search for papers related to "multi-gpu memory management" and "data locality". there's a ton of material out there, and it helps to have these concepts well established. it’s quite common to spend more time optimizing data movement than doing the actual computation. that's the joke, if you are wondering about the joke i told you earlier.

another important concept to keep in mind is *asynchronous data transfer*. it means that the cpu doesn't have to sit idle waiting for the gpu to move data before starting the processing. if you can overlap the transfer of one batch with the processing of another batch, you can hide some of the latency overhead of the non-local access. here is a small example in pytorch to illustrate the idea:

```python
import torch
import time

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 1:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        data_size = 1000000
        # create dummy data on cpu
        data_cpu = torch.randn(data_size)
        # copy data to device 0 asynchronously
        data_gpu0 = data_cpu.to(device0, non_blocking=True)
        # do some work on cpu meanwhile, simulate some computation
        time.sleep(0.01)
        # copy data to device 1 asynchronously
        data_gpu1 = data_cpu.to(device1, non_blocking=True)
        # do some other work on cpu meanwhile
        time.sleep(0.01)
        
        # sync gpu to check if transfers are done
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)

        print("data transfer to gpu 0 and gpu 1 completed.")

    else:
       print("only one gpu available, skipping the demo")

else:
    print("no gpus are available.")
```
this last example shows how to move data to different gpus in an asynchronous way. this is how you can overlap data transfers and computations and hide transfer latencies.

in summary, the difference between local and non-local gpu access comes down to the cost of transferring data. local access is fast because the data is already available in the gpu's memory. non-local access implies moving data over the pcie bus which is slower. therefore, if possible, you should keep the data as local as possible to the gpu which will use it. thinking about your data placement is essential to get the most out of a multi-gpu system.
