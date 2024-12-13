---
title: "non von neumann architecture computer science?"
date: "2024-12-13"
id: "non-von-neumann-architecture-computer-science"
---

Okay so non-von neumann architecture huh Been there done that wrestled with that beast more times than I care to remember Let's unpack this thing because it’s not as straightforward as some textbooks make it out to be I'm not gonna give you a history lesson you can google that but I will share some scars from my own encounters with this stuff

First off when we talk about non-von neumann we're talking about stepping away from the classic sequential execution model you know the fetch-decode-execute cycle that's been the backbone of computing for decades Now that's not inherently bad it's been incredibly useful but it has limitations especially when we're dealing with the kind of massively parallel stuff we need today think AI or large scale simulations And the limitations are not just "speed" but also the whole power efficiency and memory access bottleneck issues we can encounter

I remember this project I did back in the day We were building a custom image processing pipeline this was way before the fancy gpus became commonplace This involved a bunch of complex transformations on raw image data So I tried using a traditional multicore system to do this like a normal person you would think but turns out doing each task step in a different core was not the ideal solution because even if the core was free it had to wait for the data in memory which the other core was currently using it felt like some cores where idle while the others where working to death You might say you can implement all sort of caching and memory access strategies but those come with the whole bunch of synchronization and race conditions issues and we where trying to avoid those.

The whole point was that the von Neumann architecture became a major bottleneck we were spending more time shuffling data around in memory than doing actual processing It was infuriating like trying to push a shopping cart through a revolving door And that's where we started diving into non-von neumann approaches Specifically we looked at dataflow architectures This involves moving data through the machine using what could be viewed like a pipeline where a calculation is only triggered when the data is actually available.

Think of it as a plumbing system where water flows only when a faucet is open instead of some central reservoir where each pipe has to ask to use it. So the whole architecture and paradigm changes from "bring the data to the cpu" to "send the operation to the data" in a very high level explanation.

So in one of my first attempts to implement something like this I used a simpler version based on threads that are created to execute operations on the data like a micro pipeline. So we defined a very small set of operations such as additions multiplications and divisions and then created threads to do these simple operations and linked the output of each thread to the next one. We weren’t using an actual hardware dataflow architecture in this case but we where using threads and channels to simulate the behaviour of it. The code could look something like this in python.

```python
import threading
import queue

class OperationThread(threading.Thread):
    def __init__(self, input_queue, output_queue, operation):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.operation = operation

    def run(self):
        while True:
            try:
                data = self.input_queue.get(timeout=0.1)  # Add timeout
                if data is None:  # Poison pill signal
                    break
                result = self.operation(data)
                self.output_queue.put(result)
            except queue.Empty:
                continue # just wait until the new queue has something

def add(a):
  return a + 1

def multiply(a):
  return a * 2


# setup the queues
queue_1 = queue.Queue()
queue_2 = queue.Queue()
queue_3 = queue.Queue()

# setup the threads
thread1 = OperationThread(queue_1,queue_2,add)
thread2 = OperationThread(queue_2,queue_3,multiply)

# start the threads
thread1.start()
thread2.start()

# Push the initial data
queue_1.put(10)

# Get the final data
result = queue_3.get()
print(result)


# add shutdown
queue_1.put(None)
thread1.join()

queue_2.put(None)
thread2.join()
```
This was a simplified version that was basically trying to showcase the basics of moving data around using queues and threads. It’s far from being a complete implementation of a dataflow architecture but it helped me in the early stages of understanding the main concepts of it.

Another architecture I've looked at is the reconfigurable architectures like FPGAs Field Programmable Gate Arrays It’s an entirely different beast than CPUs these are essentially hardware circuits that you can program to do specific tasks They're not great for general purpose stuff but when you have highly specific and repetitive operations they can be incredibly efficient

I ended up trying to re-implement a portion of the same image processing pipeline that I was working with the multithreaded approach. I used an high level synthesis tool to build the hardware part and after a while the result was impressive. The same image transformation was taking roughly 20 times less time to complete than the traditional approach. The code to do something similar looks something like the following in SystemVerilog
```systemverilog
module adder(
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] sum
);
    assign sum = a + b;
endmodule

module multiplier(
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] product
);
    assign product = a * b;
endmodule

module pipeline(
  input logic [31:0] input_data,
  output logic [31:0] output_data
);
  logic [31:0] stage_1_out;
  adder adder1 (
    .a(input_data),
    .b(3),
    .sum(stage_1_out)
  );

  multiplier mul1 (
    .a(stage_1_out),
    .b(2),
    .product(output_data)
  );
endmodule
```
This is of course a simple example with a simple addition and multiplication operation but this serves as an example of how you can describe hardware operations to perform specific tasks using hardware description languages such as SystemVerilog. Now we're not talking about general purpose CPUs but actual configurable hardware that performs those tasks. You define the modules like they are small operations such as adders and multipliers and then you wire those modules into the main pipeline module and then a synthesys tool will generate actual low-level hardware logic out of it. It is a very powerful tool if used with the right purposes.

Now there is also an area called neuromorphic computing which is very interesting This is where the hardware attempts to mimic the structure and function of the brain specifically using concepts like neural networks with spiking neurons. They are not meant to compete with current CPUs in speed or accuracy but they are designed to process things like sensory information in a more efficient way than the standard CPUs do for the same task.

I remember a friend of mine telling me a story of an attempt to perform handwritten digit recognition with a neuromorphic chip, and it was kind of cool how the chip was performing the classification with much less power than a traditional CPU or GPU. It's still very much research level at the moment but it's a fascinating area to keep an eye on. This would be a very simplified example of a Spiking Neural network implementation in Python

```python
import numpy as np
import matplotlib.pyplot as plt

class SpikingNeuron:
    def __init__(self, threshold=1, reset_potential=0):
        self.membrane_potential = 0
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.spikes = []
        self.time = 0

    def update(self, input_current):
        self.time += 1
        self.membrane_potential += input_current
        if self.membrane_potential >= self.threshold:
            self.spikes.append(self.time)
            self.membrane_potential = self.reset_potential
        return self.spikes


# setup the neuron
neuron = SpikingNeuron()

# simulate input current
input_current = [0.5, 0.6, 0.4, 0.7, 0.2, 0.9, 0.1, 0.8, 0.3, 0.5]
spikes = []
for current in input_current:
    spikes = neuron.update(current)


# plot the results
plt.figure(figsize=(10, 4))
plt.plot(range(len(input_current)), input_current, label="Input Current")
plt.vlines(spikes, ymin=0, ymax=1, colors='r', linestyles='dashed', label="Spike Times")
plt.xlabel("Time Steps")
plt.ylabel("Current/Spike")
plt.title("Spiking Neuron simulation")
plt.legend()
plt.grid(True)
plt.show()
```
This is a very simple example and in order to create a real use case would require a network of interconnected neurons and much more complex implementation but it’s a good starting point to explain the basic concepts.

This is a huge field and there are many different approaches to non-von neumann architectures from quantum computing to memristors and even optical computing Each has their own pros and cons and all are under research and development with different goals in mind. I would say that if you are trying to get deeper into it I would recommend starting by looking into "Parallel Computer Architecture A Hardware/Software Approach" by David E. Culler, Jaswinder Pal Singh, and Anoop Gupta and "Structure and Interpretation of Computer Programs" by Harold Abelson and Gerald Jay Sussman . Those will provide you a base knowledge of the computer architecture that's going to be very useful going forward.

The whole point of all of this is that the von Neumann architecture is a bottleneck and for certain use cases like AI we really need to start thinking outside the box literally. Non-von Neumann is not a single thing its an umbrella term for many different ways of computing and there is no “one better than the other” is more like finding the best tool for the job at hand. Also keep in mind this is constantly evolving as new research is being developed and published every day.

So yeah that's my take on non-von Neumann architectures I hope that helped.
