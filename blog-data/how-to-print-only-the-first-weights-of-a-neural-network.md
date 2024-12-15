---
title: "How to print only the first weights of a neural network?"
date: "2024-12-15"
id: "how-to-print-only-the-first-weights-of-a-neural-network"
---

alright, so you're after printing just the initial weights of your neural network, huh? i’ve been there. it sounds simple but depending on the framework you're using things can get a little… layered, shall we say? i’ve spent my fair share of evenings staring at tensor shapes and trying to figure out if my matrix multiplication was actually doing what i thought it was doing. let me walk you through it, drawing on some past experiences.

first off, the reason you'd want to do this, and i suspect you already have an idea, is to inspect the initial random weights. it's a common practice, especially when you’re debugging your architecture, or trying to understand how your network's initialization scheme is working. sometimes those seemingly random numbers aren't so random, and it's good to have a look. other reasons may include wanting to check if the initialization was done correctly, perhaps you have a specific algorithm that dictates how your weights are initialized, or maybe you're trying to reproduce a research paper and need to verify their initial weights. been there. i once spent an entire day debugging a poorly implemented kaiming initialization that kept spitting out nans, i had copy and pasted from some github gist. my fault really.

the process will differ a bit based on the deep learning framework, but the core idea is the same: access the weight tensors of your model’s layers and print their first values, or some subset of the values. let’s take a look at how this could be done in a couple of common frameworks, and i'll throw in a numpy example too.

**pytorch**

pytorch is pretty explicit about how it stores weights. each layer in a `torch.nn.module` has its parameters, and weights are a specific type of those parameters, typically the `weight` attribute in many layer classes like `nn.linear`, `nn.conv2d` and so on. to get the initial weights, all you need is access to the layer parameters before any training occurs.

here’s how you can do it:

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = simpleNet()

# iterate over the layers and print the first 5 elements of the weight
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"layer: {name}, first weights: {param.data.flatten()[:5]}")
```

in this example, we’re looping through each parameter of the model, filtering those that have the word 'weight' in their name, and then printing the first 5 flattened elements from the tensor. the `data` attribute is used to access the actual tensor underlying the parameter. this is important as the `param` itself is a `torch.nn.parameter` object which has some added metadata but the actual numerical data is inside that attribute. i used to mess this up always and it was a head scratcher.

**tensorflow/keras**

tensorflow with keras also makes it straightforward to get initial weights. the weights are properties of the layers themselves, accessible before training begins.

here's a snippet:

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])


for layer in model.layers:
    if hasattr(layer, 'kernel'):
       weights = layer.kernel.numpy() # changed to kernel for the weight matrix
       print(f"layer: {layer.name}, first weights: {weights.flatten()[:5]}")

```

here, we are using the `kernel` attribute for dense layers, which holds the actual weight matrix and not the bias or other tensors. the `.numpy()` call transforms the tensor into a numpy array that we can further slice into the first few values, just like in the pytorch example. in tensorflow, all parameters are internally kept as tensors, but `keras` makes it easy to deal with them, and the `.numpy()` method is very convenient.

**numpy**

if you want to just inspect matrices initialized with some numpy function, this is equally easy to achieve:

```python
import numpy as np

weights = np.random.rand(784, 128)

print(f" first weights: {weights.flatten()[:5]}")

```

this example shows you how to create an array with numpy and print out the first 5 values. it should be noted this numpy example has zero relation with neural networks but illustrates how easy it is to access weights or elements from tensors in numpy arrays.

**some more things to consider**

*   **initialization schemes**: pay attention to your initialization. the weights are not always random. things like xavier/glorot, he/kaiming initialization are very common and lead to very specific initial weight distributions. the `torch.nn.init` module and the `tensorflow.keras.initializers` module are good places to check how these are implemented. this is often overlooked but makes a huge difference. i had one bad experience when i had some custom initialization i thought was correct and it turned out that was generating a very strange distribution.
*   **bias parameters:** don't forget about the biases. layers such as `nn.linear` in pytorch or `dense` in tensorflow will have a bias term. if you want to print these you'd use a similar logic, looking for the `'bias'` parameter names, or the `bias` attribute in tensorflow.
*   **shape inspection:** if you just want the shape of your layers parameters without the values, you can use the `.shape` attribute of the weight or bias tensors. that can be helpful to confirm your architecture has been constructed as you thought it was.
*   **reproducibility:** if you are getting random weights, it's important to set your random seeds with `torch.manual_seed(some_number)` for pytorch, `tf.random.set_seed(some_number)` for tensorflow and `np.random.seed(some_number)` for numpy before initializing your model. this way the process is deterministic.
*   **large models:** printing the entire weight matrix is not always practical or even possible for large models. use slices or other selection methods. printing the first few values is a good start. some people think that printing `param.data.mean()` is more useful, if you do want to print only one number. also, tensorboard is a good way to visualize weights if you want to inspect the whole distribution of values.

if you need to deepen your knowledge about the math behind neural networks, i really recommend the deep learning book by goodfellow, bengio and courville. for more practical implementations and pytorch specifically, i like the official pytorch tutorials a lot, they are well organized and well written.

i hope this helps. i think at one point all of us have had a "what are my weights, really?" moment. it's a common step in the process of understanding what our models are doing under the hood. and well, if your initialization was a bit off, at least you didn't get a nan. i once had to manually trace gradient propagation with pen and paper because i had done a simple multiplication wrong. don't ask, it was painful. well, i actually have to go i am debugging something right now, seems i forgot to add a dropout layer in my cnn. my excuse is that it’s 4 am and i am not in my best shape right now.
