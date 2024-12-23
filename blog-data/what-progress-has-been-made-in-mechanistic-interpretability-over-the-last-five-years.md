---
title: "What progress has been made in mechanistic interpretability over the last five years?"
date: "2024-12-11"
id: "what-progress-has-been-made-in-mechanistic-interpretability-over-the-last-five-years"
---

 so mechanistic interpretability right super cool area  massive progress last five years  like whoa  feels like things exploded  before it was mostly hand wavy stuff  lots of "we think this part does this" now we're actually getting into the nitty gritty  tracing information flow  seeing how individual neurons or groups of neurons interact  it's nuts

Remember like five years ago  most interpretability was just looking at activation patterns  heatmaps  gradcam that kind of thing  useful for getting a general idea but not really explaining *why* a model behaved a certain way  it was like looking at the outside of a car and guessing how the engine worked

Now we got way more sophisticated tools  we can actually dissect the internal workings  probe individual units  even modify their behavior  and see how that ripples through the rest of the network  think of it like having an x-ray for neural networks but way better  it's not just a picture  we can interact with it  poke it and see what happens

One major advance is in understanding the role of individual neurons  we're moving beyond just looking at neuron activation to analyzing their functional roles  like identifying neurons specialized for specific features  or even neurons that act like gates controlling information flow   this helps in decomposing the model's reasoning into smaller manageable units instead of a black box

Another huge area is tracing information flow   we're not just looking at what neurons are active but how information propagates through the network  techniques like integrated gradients help  but there's so much more  researchers are developing methods to visualize information pathways  identifying bottlenecks  and pinpointing where the model makes crucial decisions  think of it like mapping the roads and highways of a city  except the city is a neural network

And then theres the work on circuit analysis  literally trying to understand the network as a circuit  with neurons acting as components  and connections as wires   this is where things get really powerful  because we can apply tools from circuit theory  and even use those to predict how the network will react  its still early days but its showing lots of promise

Code example 1  a super simplified illustrative example of tracing information flow using something like saliency maps  totally not realistic for real big models but gets the concept across

```python
import numpy as np
# mock data  replace with actual model activations
activations = np.random.rand(100, 10) # 100 neurons 10 datapoints

# simple saliency map calculation  replace with a real method
saliency = np.mean(activations, axis=1)

# find top k neurons
k = 5
top_neurons = np.argsort(saliency)[-k:]

print(f"Top {k} neurons: {top_neurons}")
```


This kind of stuff  super basic I know  but the idea is similar in more advanced methods   its about tracking the influence  the path  of a particular input  through the network

Then theres the work on probing  actively interacting with the model to understand its internal state  this is where we shine a light into different parts of the model and see how the model reacts   imagine tweaking individual neuron weights or even replacing neurons entirely  seeing how that changes the model's output  its a bit like reverse engineering   

Code Example 2  a super dumbed down example of probing  again way simplified but shows the idea  we are probing by changing activations directly


```python
# Mock data again
activations = np.random.rand(10,10)
original_output = np.sum(activations)

#Probe by adding a small perturbation to a single neuron
perturbation = 0.1
activations[0,0] += perturbation

perturbed_output = np.sum(activations)

print(f"Original output:{original_output}, perturbed output: {perturbed_output}")
```


This isn’t exactly how probing is done in real life but conceptually  its about carefully manipulating the network to see cause and effect  

And finally theres the  growing emphasis on using formal methods  we're starting to use  mathematical tools to prove properties of neural networks  to verify their behavior and even synthesize networks with specific properties   its like adding a layer of mathematical rigor  which is really great because it moves beyond just empirical observation

Code Example 3 showing a super basic example  this is more about the concept than any real practical application


```python
# A super simplified example of verifying a property
def is_monotonic(function):
  # Check if the function is monotonic increasing
  x = np.linspace(0,1,100)
  y = function(x)
  return np.all(np.diff(y) >= 0)

# Assume some simple function representing a part of the network
def my_simple_function(x):
    return x**2 # example

if is_monotonic(my_simple_function):
    print("Function is monotonic increasing")
else:
    print("Function is not monotonic increasing")

```

This stuff is mostly in the realm of theoretical computer science  but its showing promise in verifying parts of networks  showing that specific parts of the network meet certain criteria

So yeah  mechanistic interpretability  huge progress  lots of exciting new tools and methods  but its still early days  a lot more to do  check out some papers on  "circuit analysis in neural networks" and "information flow in deep learning" and books on  "formal methods for machine learning"  that would be a good start    it's a rapidly evolving field  so keep your eyes peeled  lots of cool stuff coming
