---
title: "Why am I having a Custom Dataset and DataLoader very slow?"
date: "2024-12-15"
id: "why-am-i-having-a-custom-dataset-and-dataloader-very-slow"
---

alright, so you're hitting the classic slow custom dataset and dataloader wall, it's a pain i've definitely felt before. it’s almost a rite of passage, like debugging a pointer error in c for three days straight in your early uni days— yeah, i've been there, more times than i care to count. it's never just 'one thing', is it? there's usually a bunch of little performance bottlenecks working together to slow things to a crawl. lets go through some common suspects based on what i've seen, and hopefully, one of these will spark an 'aha!' moment for you.

first off, the most frequent culprit, and i've been burned by this myself, is inefficient data loading inside your dataset’s `__getitem__` method. specifically, are you doing any disk i/o, image resizing or some other heavy processing in that method each time a data item is requested? if so, that's likely where a lot of your time is going. your dataloader is probably creating a new process or thread to get data and that call is blocking the main training loop. it will become so slow that if you put breakpoints you will not notice much time difference between breakpoint lines. this is because most of the time is spent inside the actual loading part. i remember debugging some of my own code and thinking there was no problem with the code just to find out later i was doing heavy image processing per item request inside the `__getitem__` method and not doing proper batch processing and caching.

here's a simple example to illustrate the problem. let's say you have a directory full of image files, and each time you request an item, your `__getitem__` method reloads and resizes the images:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class SlowImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')  # this is slow
        if self.transform:
          image = self.transform(image)
        return image

# Example usage (this will be slow)
image_dir = "path/to/your/images"
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
dataset = SlowImageDataset(image_dir,transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for batch in dataloader:
  pass
```

notice the `Image.open` and the transforms inside `__getitem__`? those are the usual culprits. it's loading and transforming each image every single time a data point is needed. this is where it gets painfully slow if you are doing any kind of augmentations or complex image transforms inside that method, or any type of heavy calculation or processing.

a fix would be to pre-process your data as much as possible *outside* of the `__getitem__` method. ideally, pre-resize, pre-transform, and serialize your dataset, if it fits your memory and time constraints, or pre-process chunks using multiprocessing and storing in memory. if your dataset is massive this may be impractical. you can also use libraries like `opencv` or `pillow-simd` which use faster cpu libraries when working with images.

another approach is to do batch processing. `torchvision` already does batch transformations in an efficient manner, but maybe you are doing something custom and then it might be a place you are losing time. this example shows the difference:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class FasterImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, cache = False):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.cache = cache
        self.cached_images = {}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        if self.cache and idx in self.cached_images:
           return self.cached_images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
           image = self.transform(image)
        if self.cache:
          self.cached_images[idx] = image
        return image

# Example usage (this will be faster using batch processing)
image_dir = "path/to/your/images"
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
dataset = FasterImageDataset(image_dir,transform, cache=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for batch in dataloader:
  pass
```

this example uses a simple cache and it will be way faster. but also keep in mind that the dataloader can use multiprocessing to prepare the batches. if you are on a windows machine, you will most likely need to wrap your dataloader creation inside a `if __name__ == '__main__':` to avoid issues with multiprocessing on windows.

other bottlenecks can be related to the `num_workers` parameter in your `dataloader`. a zero `num_workers` means that the main process will be responsible for getting the data which will be slow. but using more than the amount of cores your processor has may not improve performance and actually degrade performance, as the operating system will have to do constant context switches and that is an overhead. using a value that is equal to the amount of your cores is a good place to start, and from there you can try to tweak it a bit for your specific case. usually `num_workers = 4` works fine most of the time.

another thing to be mindful of is disk access. if you have many small files, and your disk is not fast enough, the operating system will spend time moving the read head from one file to another. this can be mitigated by using a solid state drive instead of a conventional mechanical hard drive. another, more advanced approach, is to use sharding and pre-processing large chunks of data into large tensors which will reside inside the operating system cache as much as possible, reducing disk reads and processing times. this has a caveat, that you need to know how many elements you will need for each shard. it will need some code adaptation.

also, make sure that if you are doing augmentations, these are done in batch as `torchvision` does. a mistake i've seen is doing individual image transforms and then stacking them together into a batch using `torch.stack` or some other custom method which will be slower. if you are creating masks or some other target, do it in batch fashion. some libraries like `albumentations` are very fast, but their usage might not be straightforward. this is why most people use `torchvision` for simple tasks.

sometimes, it is just that you are doing some kind of heavy computation inside the dataloader that can be optimized. that’s where code profiling comes in handy. tools like `cProfile` in python will pinpoint exactly where your code is spending its time. you may find out some obscure part of your code is the actual culprit. i've spent hours trying to figure out why my code was slow only to find out that some random initialization was doing some heavy lifting every item request. i've also been there where the `num_workers` were set to zero, and that's because, in a moment of pure genius, i copy pasted the code from an example and i didn't pay attention to that small detail. yeah, we've all been there, i guess.

if you are working with large text datasets, you should definitely consider using libraries like `datasets` from huggingface. if your data fits into memory, you may not need any dataloader at all, and load everything as a large tensor which will be the fastest. but this is often not the case.

also if your network is doing something slow that is not directly related to data loading, then it could be the computation inside the neural network itself.

here's a snippet of a possible way to handle this in a more efficient way for large datasets where you are dealing with some custom operation in the targets:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, targets, transforms = None):
        self.data = data
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
      sample = self.data[idx]
      target = self.targets[idx]
      if self.transforms:
         sample = self.transforms(sample)
      # some custom operation in batch with the target instead of item by item
      # target = some_custom_operation(sample,target) # this is a bad idea
      return sample, target

def some_custom_operation_batch(batch_samples, batch_targets):
  # do calculations in a vectorized way with batch_samples and batch_targets
  return batch_targets  # replace with the real logic

# Generate some dummy data
data = torch.randn(1000, 3, 224, 224)
targets = torch.randint(0, 10, (1000,))

# Create custom transform
transform = transforms.RandomRotation(degrees = 45)

dataset = CustomDataset(data, targets, transform)

def collate_fn(batch):
  samples = torch.stack([item[0] for item in batch], dim=0)
  targets = torch.stack([item[1] for item in batch], dim=0)
  targets = some_custom_operation_batch(samples,targets)
  return samples, targets

dataloader = DataLoader(dataset, batch_size=32, collate_fn = collate_fn, num_workers = 4)

for batch in dataloader:
   pass
```

notice in this example that instead of processing the target inside `__getitem__` function, we do it inside a `collate_fn` that processes the target in batch, or you can also use the dataloader `map` method, if you have simple transforms, but for custom tasks, a custom `collate_fn` is often the best way. the `collate_fn` is called only once per batch.

regarding resources, if you are serious about optimizing your data pipelines, i'd suggest looking into the paper "prefetching in deep learning training pipelines" and also look for other papers or books about operating system caches and memory management, so you can get a deeper understanding of how the operating system handles memory and disk reads. this way, you can optimize more intelligently. and it might be useful to use specialized tools like tensorboard to monitor your data loading time. it will pinpoint exactly where your training pipeline is spending time. also, knowing the difference between sequential and random reads will also help a lot.

finally, try to profile your code, and test multiple combinations of parameters. the best combination will be the one that gives the best loading time without hogging all your resources or creating bottlenecks somewhere else. and one last tip, if you ever decide to optimize your data loading pipeline for the gpu, remember that the data needs to be stored in pinned memory, or it will be very slow moving data from regular ram to the gpu ram each time. also the type of data needs to be the same (float32 or float16) or you will pay a conversion overhead.

hope this helps, happy coding!
