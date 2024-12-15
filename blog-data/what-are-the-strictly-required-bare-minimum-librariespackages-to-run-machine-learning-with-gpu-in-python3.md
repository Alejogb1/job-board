---
title: "What are the strictly required (bare minimum) libraries/packages to run machine learning with GPU in python3?"
date: "2024-12-15"
id: "what-are-the-strictly-required-bare-minimum-librariespackages-to-run-machine-learning-with-gpu-in-python3"
---

alright, so you're asking about the absolute minimum python libraries needed to get machine learning with gpu acceleration going, eh? i've been down this rabbit hole more times than i care to count, so let me lay it out.

from personal experience, i remember my early days trying to get this working, it was a mess. think of trying to assemble a complex lego set, but the instructions are in hieroglyphics. i spent weeks fighting with driver versions, cuda toolkit installations, and what felt like an endless loop of dependency hell.  it was during a project back in '17, i was trying to build a convolutional neural network for image classification and i was absolutely struggling with just making it run on the gpu instead of my poor cpu that was screaming for mercy. that’s why i feel ya.

let’s cut to the chase.  the bare minimum you need isn't as extensive as some might make it out to be, but the installation and compatibility can be a beast if you don't get it perfectly aligned. let’s go library by library.

first and foremost, you're going to need **numpy**.  this is the fundamental package for scientific computing in python. it provides the array object, which is how you deal with data in most machine learning. if you don’t have it, you're dead in the water. trust me on this. i've seen projects trying to reinvent the array wheel without it and it always ends in tears.

next up is **tensorflow** or **pytorch**. you only *need* one, but they represent the two big players in the deep learning world. i'll detail both because they approach the gpu acceleration in different manners.  they are the frameworks that abstract away much of the low level calculations for deep learning and have gpu capabilities baked in, so, that’s a good thing.

for **tensorflow**, you need the tensorflow version that is compiled with gpu support. typically it’s `tensorflow-gpu`. however, with newer tensorflow versions, the distinction is less and less important. so, simply installing tensorflow with `pip install tensorflow` might just suffice if you have the corresponding cuda driver and cudnn (we'll get to those system level bits later) installed. you might end with a version with or without gpu capability, depending on your system's gpu drivers etc. tensorflow will attempt to utilise your gpu if it finds it available. remember, you dont *need* tensorflow-gpu package, since it’s pretty much deprecated.

for **pytorch**, you need `torch` and `torchvision`. the `torch` package provides the core tensor operations and gpu acceleration, while `torchvision` is an extra library for common datasets and transforms specific for vision tasks.  to enable gpu acceleration during install, you'll also use pip.

so to be clear that’s it for python libraries.  but, this is where people get tripped up, and believe me, i’ve been there, the real magic happens at the system level because these python libraries are just interfaces to the low level libraries that make the gpu compute possible.

let’s talk about the system level requirements. this is the less friendly part.

for **nvidia** gpus (the most common for machine learning), you absolutely need the correct **cuda toolkit** installed. this includes the drivers, and development libraries required by tensorflow and pytorch. you'll need to download the version specific for your system and your graphics card and your framework version.  it can be a real pain, i spent almost a full weekend back in '18 when i was upgrading my system struggling to get the correct versions that would work together without crashing. also crucial is the corresponding **cudnn** library, which is used for accelerating neural network operations on nvidia gpus. this needs to match your cuda version as well. there are other ways to use gpus such as with rocm for amd cards but that is a whole different can of worms.

for other gpus like intel or amd there are their own equivalents, but to be honest, i dont have too much experience with them, so i am not comfortable going to deep into those territories.

ok, let's get to some code, and some gotchas:

here is the simplest code that i can produce, one for tensorflow and one for pytorch, it won't do anything useful, but shows how to query if the system is using the gpu (as a first sanity check):

```python
# tensorflow sanity check
import tensorflow as tf
print("tensorflow version: " ,tf.__version__)
if tf.config.list_physical_devices('GPU'):
    print("tensorflow is using gpu")
    print("gpu devices:",tf.config.list_physical_devices('GPU'))
else:
  print("tensorflow is using cpu")
```

here, we're importing tensorflow, printing the version, then checking if it sees the gpu and prints it. you'll likely see your gpu listed by name. if it just prints "cpu", you messed up something in the system level installation. and again, i’ve been there, many times.

here’s the equivalent for pytorch:

```python
# pytorch sanity check
import torch
print("pytorch version: ", torch.__version__)
if torch.cuda.is_available():
  print("pytorch is using gpu")
  print("gpu device:", torch.cuda.get_device_name(0))
else:
  print("pytorch is using cpu")
```

in this one we’re importing torch and checking if cuda is available, and if it is, then we print the name of the gpu. again, if it says cpu is used it means there is an issue with the system setup.

and finally, here is an example of a simple numerical operation being done on the gpu, where we generate some random tensors and perform a matrix multiplication, again, just as a quick check:

```python
import torch
if torch.cuda.is_available():
  device = torch.device("cuda")
  a = torch.rand(3000, 3000).to(device)
  b = torch.rand(3000, 3000).to(device)
  c = torch.matmul(a, b)
  print("matrix multiplication done on gpu", c.sum())
else:
    print("gpu is not available, sorry")
```

for tensorflow, the code would be quite similar just using tensorflow tensor and operations instead of pytorch ones. i am leaving it out as an exercise for the reader.

now, let's go through the crucial bit: *system level setup*. nvidia's website has detailed instructions for cuda and cudnn installations. you need to select the right combination of cuda driver, cuda toolkit and cudnn.

if i could recommend some reading material other than official websites and forum threads which i strongly advise that you consult before proceeding, i would go for "deep learning with python" by francois chollet (for tensorflow) and "programming pytorch for deep learning" by ian mcloughlin for pytorch. these books will give you much more context and a better sense of the inner workings than any quick start guide. also check the official documentations, specifically for cuda and cudnn compatibility, that’s an important step.

getting this right is 90% of the battle. you might also need to manage python environments (conda or virtualenv are your best friends) and make sure the libraries are installed in a clean and separate environment.  it prevents dependency hell with version conflicts.  i’ve learned this the hard way.

oh, and here's a tech joke i stole from somewhere: why did the programmer quit his job? because he didn't get arrays. yeah, i know, not my best work, but i couldn't help myself.

anyway, to summarize. the bare minimum python libraries for gpu machine learning are numpy and either tensorflow or pytorch (with torchvision if you go for pytorch). and the system level, the drivers, the cuda toolkit and cudnn. remember, that compatibility between all these, the python libs, cuda and cudnn is crucial.

i'm not going to lie, setting up a gpu environment is always a bit of a pain, but once you get through it, the speed improvements are absolutely amazing and make it all worth it. i hope this helps and avoids some of the headaches i’ve experienced during the early days. good luck and let me know if you get stuck and need to re-check some installation configurations.
