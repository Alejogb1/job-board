---
title: "Why does a PyTorch: Fully Connected Layer have no Parameters?"
date: "2024-12-15"
id: "why-does-a-pytorch-fully-connected-layer-have-no-parameters"
---

hey there,

so, you're seeing a fully connected layer in pytorch and it's showing zero parameters and thinking, 'what gives?'. yeah, i've been there, staring blankly at the output of a `print(model)` and feeling like i'm losing my mind a little. it's one of those things that makes you question your sanity at 3am. let me walk you through it. it’s a bit of a head scratcher but hopefully, we can iron it out.

first, let's get the basics straight. a fully connected layer, often called a dense layer, is all about connections. every single neuron in the layer is connected to every single neuron in the layer before it. each of those connections has a weight, and the neurons have biases, those are your parameters. those weights and biases are what get tweaked during training. pytorch uses them to learn.

now, here’s the kicker, a fully connected layer doesn’t magically create these weights and biases just because you define it as fully connected in code. it doesn't mean it has zero parameters. it simply means that those parameters haven’t been *initialized* yet. the layer is there, ready and waiting, but it doesn't have any actual numbers to work with, initially.

what happens usually is that these parameters get created once you, in a sense, "push" some data through the layer for the first time, or when you explicitly declare its shape. that’s because pytorch has to figure out the size of the weight matrix and bias vector based on the size of the input to that layer. if you don't give the layer any input then, well, it doesn't know what size to make those weight and bias parameters. it's a bit like trying to make a suit without knowing the wearer’s size.

i remember once i spent a day and a half chasing this down in a complex model i was working on way back in the day. i had defined my fully connected layers but hadn't run any data through it, and the parameter count was zero. kept thinking my gpu was fried or something. i went down a rabbit hole of checking cuda driver versions, you name it i probably checked it. turns out it was just that my data input shape was being passed incorrectly. oh man, i felt so dumb, but hey, we learn, or at least we try to.

let me show some examples.

example 1: no input, no parameters

```python
import torch
import torch.nn as nn

# define a fully connected layer
fc_layer = nn.Linear(in_features=5, out_features=10)

# print the number of parameters
print(f'number of parameters: {sum(p.numel() for p in fc_layer.parameters())}')
```

if you run this, you'll see that the number of parameters is zero. why? because, as i said, we haven't given it an input yet. this is very common. the layer is just waiting. it knows the `in_features` and the `out_features` which define *how* parameters will be formed, but they are not formed until you use the layer.

example 2: input, parameters appear

```python
import torch
import torch.nn as nn

# define a fully connected layer
fc_layer = nn.Linear(in_features=5, out_features=10)

# create a dummy input
dummy_input = torch.randn(1, 5) # batch size 1, 5 features

# pass the input through the layer (this initializes the parameters)
_ = fc_layer(dummy_input)

# print the number of parameters
print(f'number of parameters: {sum(p.numel() for p in fc_layer.parameters())}')

```

here’s what happens: when you pass a `dummy_input` through the `fc_layer`, pytorch goes "ah, okay, the input size is 5 and output size is 10, that means my weight matrix should be 5x10 and biases should be size of 10". it creates those weights and biases and bam, we now have parameters, and this time, we don't get zero.

example 3: explicitly defining the shapes at the start.

```python
import torch
import torch.nn as nn

# define a fully connected layer and initialize weights manually
fc_layer = nn.Linear(in_features=5, out_features=10)

# lets initialize weights
with torch.no_grad():
  fc_layer.weight.data.normal_(0, 1)
  fc_layer.bias.data.zero_()


# print the number of parameters
print(f'number of parameters: {sum(p.numel() for p in fc_layer.parameters())}')

```

here we don't pass data immediately. instead, we initialize manually. the weights and biases are there. so the layer has now parameters at the start and doesn't wait for the input data to get them. the weights start with random values and biases start at zero in this example. you could use other initialization strategies. this is a common way to ensure the parameters of a network are not too large at start that might lead to instability.

so, in short, a fully connected layer in pytorch isn't born with parameters; it makes them as it grows, once it gets a sense of the kind of input it’s going to be handling. the layer itself doesn't really exist as a "thing" unless it has the parameters, and you don't have them unless you either pass data through or explicitly define the shapes at the start of the layer. if you don’t give it that nudge, it just hangs out with zero parameters like some lazy slacker of a neuron. we have all been there, right? i mean the part of not knowing what's going on, not the slacking, hehe.

if you want to understand this kind of stuff more deeply, i suggest hitting the books a little. "deep learning" by goodfellow et al. is the go-to for a solid understanding of the math behind these things. another fantastic option is "neural networks and deep learning" by michael nielsen, it’s great for more practical examples. they are worth their weight in gold, especially when you are in a "parameter zero" situation like this.

hope this helps clear things up. always, feel free to ask more questions. this is how we learn.
