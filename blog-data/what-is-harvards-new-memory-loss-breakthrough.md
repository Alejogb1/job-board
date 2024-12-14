---
title: "What is Harvard’s new memory loss breakthrough?"
date: "2024-12-14"
id: "what-is-harvards-new-memory-loss-breakthrough"
---

alright, let's break down this memory loss thing harvard's been up to, or seems like they've been up to. it's not really a simple on/off switch, more like a very complicated circuit board with a bunch of connections getting fuzzy. i've been playing around with similar concepts for years, not in the medical sense, but in my own data management projects, where memory loss can be a real pain. so, what's likely going on is not some magic pill, but a deeper understanding of how memories are encoded and retrieved at a biological level and translating this to some practical use.

the core issue here is synaptic plasticity, the ability of the connections between neurons to strengthen or weaken over time. think of it like the wires in a computer, the more a wire is used, the better the connection. but when a wire isn’t used, that connection gets less efficient, maybe even disappears. this isn't just about forgetting what you had for breakfast, this is about the whole neural network that holds onto the data of your life getting its connections slightly degraded, that’s where the ‘memory loss’ is.

from what i've gathered from various academic papers, and some open access research journals i've been reading, harvard (and other labs) is probably not focusing on ‘curing memory loss’, but more on enhancing the process of memory stabilization. that is, ensuring those neural connections that represent a memory remain strong for longer periods. one technique might involve neurotrophic factors, proteins that encourage neuron growth and survival. that is probably the biological part of what is happening, and from where harvard got the data from. i've done something similar in some personal projects, using evolutionary algorithms to refine certain computational neural networks to get better at recognizing patterns. it's similar, but with bits and bytes instead of neurons.

the practical side of things probably involves some kind of targeted stimulation of the brain. there's transcranial magnetic stimulation (tms) which is non-invasive, and probably is what harvard or any other similar lab is testing, but there are other more invasive methods, but i seriously doubt those are the ones harvard is using.

think of it as using targeted pulses of electromagnetic energy to activate certain brain regions involved in memory encoding and retrieval. it’s like giving a gentle nudge to those neuronal connections that are fading. the trick here, is not just turning something on, but turning it on *at the right moment*, and at the *right intensity* so that the memory gets a boost of activity and, consequently, a better chance of getting consolidated into long-term memory. that’s where the complexity comes in, as you need a lot of data about an individual before doing any kind of intervention like that.

on a personal note, i’ve messed up my own fair share of neural networks while trying to get my machine learning models to remember specific data. one time, i created a model for classifying different types of vintage cameras and i kept feeding it the same dataset over and over and forgot to use ‘dropout’ on some layers. let's just say, it became *really good* at recognizing the same ten cameras, and *utterly terrible* at anything else. it had almost no generalization capacity, it was literally a memorization machine, it was just remembering details about the training data instead of learning to generalize. i had to re-architect the network and start the training again from scratch, this is very similar to what i'm guessing they are doing with the neurons, but instead of re-architecture the network they do stimulation.

here's a quick example of how i would use dropout in a pytorch model i made for a simpler task, to avoid what happened with the camera project:

```python
import torch
import torch.nn as nn
import torch.nn.functional as f

class simple_nn(nn.module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(simple_nn, self).__init__()
        self.fc1 = nn.linear(input_size, hidden_size)
        self.dropout = nn.dropout(dropout_rate)
        self.fc2 = nn.linear(hidden_size, output_size)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_size = 10
hidden_size = 50
output_size = 2
dropout_rate = 0.3
model = simple_nn(input_size, hidden_size, output_size, dropout_rate)

```
this example shows a basic neural network, with a dropout layer, and that’s a good way to reduce memorization and promote generalization. a similar principle might be at play in the harvard research, they are probably trying to enhance the generalization and long term retention capabilities of the neurons.

another area of research is around the consolidation of memories during sleep. it’s known that during sleep there’s a replay of neural patterns related to previously acquired memories, this re-activation during sleep helps consolidate the information. harvard researchers might be looking into ways of enhancing this consolidation process, probably by combining neurostimulation techniques with some form of sleep monitoring. this is similar to the concept of data backups and redundancy i used in one of my projects. i've worked in distributed database systems where we implemented data replication and error-checking mechanisms that would rebuild the database if one node got corrupted. that is, if one of the memory parts of the system degraded or failed, the system would reconstruct the lost information. the brain seems to do something similar during sleep, that's the beauty of it.

the exact details of harvard's work are probably under wraps for now, as is usually the case with cutting-edge stuff like this. but it's probably not some miracle solution that will cure all kinds of memory loss overnight, but probably a collection of techniques that enhance the biological processes of memory consolidation.

here’s a simplified example using python of how i would model some kind of memory decay and consolidation using a linear model (this is obviously very simplistic but helps illustrate the process):

```python
import numpy as np
import matplotlib.pyplot as plt

def memory_decay(initial_strength, time, decay_rate):
    strength = initial_strength * np.exp(-decay_rate * time)
    return strength

def memory_consolidation(strength, consolidation_boost, sleep_time):
    boosted_strength = strength + (consolidation_boost * sleep_time)
    return boosted_strength

time_points = np.arange(0, 10, 0.1)
initial_strength = 1.0
decay_rate = 0.2
consolidation_boost = 0.1
sleep_time = 2.0
memory_strengths = memory_decay(initial_strength, time_points, decay_rate)
consolidated_strengths = memory_consolidation(memory_strengths, consolidation_boost, sleep_time)

plt.figure(figsize=(10, 6))
plt.plot(time_points, memory_strengths, label='memory strength without consolidation')
plt.plot(time_points, consolidated_strengths, label='memory strength with consolidation')
plt.xlabel('time (arbitrary units)')
plt.ylabel('memory strength (arbitrary units)')
plt.title('memory decay and consolidation simulation')
plt.legend()
plt.grid(true)
plt.show()
```
this script simulates how memory decays and is recovered during sleep by boosting it, this helps illustrate the concepts of consolidation.

it is very important to highlight here that these models are way too simplistic and should not be seen as how things actually work, but they give a good intuition of some of the principles involved in the process.

another key aspect they are probably looking into is the role of neuroinflammation in memory loss. when the brain has some kind of inflammation it impacts neuron function and connectivity, making it harder for memories to form and consolidate. harvard researchers probably are exploring ways to reduce this inflammation using therapies. this is somewhat similar to debugging my code. sometimes a bug is not just a problem with one specific line but it's a systemic issue that's causing a cascade of errors. i need to go back, identify the real issue and correct it and sometimes the issues are not so obvious and need a systematic approach. memory issues are probably similar, the issue is not with one neuron failing, but with a system getting less efficient. it’s about the overall health of the neuronal network.

i had a very interesting debugging case once, it was a production issue on a very large e-commerce system, the system was underperforming and it took me a while to realize it was not an issue with the current version but a problem with a caching system that was implemented very badly a few versions before that. i had to reconstruct the state of the system to understand the root cause and then go back and correct it.

anyway, this is all educated speculation on my part, but given what i know from my work and the field, it is most likely the direction they are heading. it’s not going to be just one breakthrough that fixes everything, but a series of small advancements in the understanding of neural mechanisms that will, hopefully, lead to a real improvement in how we treat memory-related issues. the real deal will likely not be about finding a cure but about learning to enhance the natural biological processes of the brain, that way is possible that we will be capable of improving our memory.

if you want to get deep into this i would recommend reading ‘principles of neural science’ by kandel, schwartz and jessell, it's a great overview of how the brain works at the biological level. and if you want to get more technical in the modelling area check “deep learning” by goodfellow, bengio, and courville.

and, a joke because i think this is a very serious subject that needs a bit of comic relief: why did the programmer quit his job? because he didn't get arrays. ha.

here's a final python example that shows how i would visualize a more complex decay process with different types of memory strength decaying at different speeds:

```python
import numpy as np
import matplotlib.pyplot as plt

def memory_decay_complex(initial_strengths, time, decay_rates):
    strengths = initial_strengths * np.exp(-decay_rates * time[:, np.newaxis])
    return strengths

time_points = np.arange(0, 10, 0.1)
initial_strengths = np.array([1.0, 0.8, 0.6])
decay_rates = np.array([0.1, 0.2, 0.3])
memory_strengths = memory_decay_complex(initial_strengths, time_points, decay_rates)

plt.figure(figsize=(10, 6))
for i in range(memory_strengths.shape[1]):
    plt.plot(time_points, memory_strengths[:, i], label=f'memory {i+1}')

plt.xlabel('time (arbitrary units)')
plt.ylabel('memory strength (arbitrary units)')
plt.title('memory decay complex simulation')
plt.legend()
plt.grid(true)
plt.show()
```

in this script, you see that different types of memories can have different decay rates, making the simulation closer to what happens in reality, as some of our memories fade away quicker than others.
