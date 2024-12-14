---
title: "Is this still considered a Neural Network?"
date: "2024-12-14"
id: "is-this-still-considered-a-neural-network"
---

alright, let's break this down. someone's asking if something they've got is *still* a neural network. that implies they might have altered it quite a bit, or they're working on something unconventional. it's a pretty common question when people start experimenting beyond the basic textbook examples. let me tell you, i've been there. more than once.

i recall this one project back in '08. i was messing with image recognition, long before it was the hot topic it is now. i started with a classic convolutional neural network, you know, the kind with stacked layers doing convolutions and pooling. it was... adequate. but i had this notion about incorporating spatial relationships more directly. so i started messing with the connections, making them non-local, some connections skipping entire layers. things got weird really fast. it worked.. sort of. accuracy improved marginally but debugging became a nightmare. i kept asking myself the same question, "is this even a neural network anymore?" since things were so far from the initial template i used from goodfellow's book. the layers looked like a plate of spaghetti. a beautiful spaghetti plate but spaghetti nonetheless.

the core of the question really comes down to this: what *defines* a neural network? it's not just about having layers and nodes. the key element is that it’s a network of interconnected nodes (neurons) where these connections have learnable weights. these weights are adjusted via backpropagation or some related optimization process using a loss function, generally derived from calculus chain rule. the nodes themselves compute an output based on a weighted sum of their inputs, usually passed through a non-linear activation function. these things together allow the network to fit patterns in the data.

so, if your structure deviates from the typical layout – maybe you've replaced the conventional activation functions, or introduced unconventional connection patterns, or a totally novel type of node – it's still a neural network, *if* it satisfies the conditions mentioned before. meaning, we've got weighted connections, an optimization process, and some sort of non-linearity in the node computation, it is still considered a neural net. a weird one perhaps, but a net nonetheless. you can call it your own custom brand of neural net.

it's funny, looking back, that particular experiment was mostly a dead-end. but it taught me a lot about how flexible and yet finicky these things could be. it taught me to embrace the experimentation. and to document everything! it became a running joke with my colleagues at the time, that every new model of mine needed its own dedicated flow chart because of how complex and experimental it was.

now, let's see some common examples where people tend to get confused.

first, what about replacing a typical sigmoid or relu with something unusual for the activation function? something like sine or absolute value? or even a piecewise function. if you did that, you will still be using a neural net. as long as you still follow the principle of computing a weighted sum of the inputs and putting that output through a function. in this case the activation function itself is a hyperparameter you can tune. for instance:

```python
import torch
import torch.nn as nn

class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return torch.abs(x) # Absolute value activation

class CustomNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomNeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = CustomActivation()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Example usage
input_size = 10
hidden_size = 20
output_size = 2
model = CustomNeuralNet(input_size, hidden_size, output_size)

# Dummy input for demonstration
dummy_input = torch.randn(1, input_size) # Batch size of 1
output = model(dummy_input)
print(output)
```

this snippet shows that even a rather unusual activation does not make it "not a neural network." in fact you might stumble upon some novel solutions by exploring these.

now, consider that you start removing connections between layers. say instead of a sequential fully-connected topology you introduce some skip connections or connections in a more general graph. well, you would still be using a neural net, as long as the principle of computing outputs by weighted connections is met. an example in code would be:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnectionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SkipConnectionNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear2(x1))
        x3 = self.linear3(x2 + x1) # Skip connection
        return x3

# Example usage
input_size = 10
hidden_size = 20
output_size = 2
model = SkipConnectionNet(input_size, hidden_size, output_size)

# Dummy input for demonstration
dummy_input = torch.randn(1, input_size) # Batch size of 1
output = model(dummy_input)
print(output)
```
as you can see, even adding or removing connections does not make your model "not a neural net". it is still an interconnected structure of nodes with weighted connections where outputs are computed in the forward pass and weights are updated in the backpropagation.

another thing that i commonly see is people mixing neural networks with other kinds of systems, lets say that the output of your network acts as a parameter for another system that is not a neural network, or vice versa. or lets say you use the output of a network as the weight of another one, it will be "tricky" to qualify the whole system as a neural network, but the core components are still. the neural network itself will still be a neural network.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# a "non-NN" component
class MathFunction:
    def __init__(self):
      pass
    def forward(self, x):
        return x**2 + 2*x + 1

# Combining both
class HybridSystem:
  def __init__(self, input_size, hidden_size, output_size):
    self.nn = SimpleNN(input_size, hidden_size, output_size)
    self.math = MathFunction()

  def forward(self, x):
    nn_output = self.nn(x)
    return self.math.forward(nn_output)


# Example usage:
input_size = 5
hidden_size = 10
output_size = 1
model = HybridSystem(input_size, hidden_size, output_size)

# Dummy input for demonstration
dummy_input = torch.randn(1, input_size)
output = model(dummy_input)
print(output)

```

in this case, only `SimpleNN` is a neural network. the `MathFunction` is just a regular mathematical function. but the key here is that neural network itself remains as it was. it did not become "not a neural network" just because you decided to use its output as part of a larger system.

if you are starting to go off the beaten path, you should probably explore the vast literature on the subject, instead of trying to re-invent the wheel. if you want a very solid theoretical background i recommend "deep learning" by goodfellow, bengio and courville, it will give you a very formal perspective on the topic. if you prefer the practical side of things i recommend "hands-on machine learning with scikit-learn, keras & tensorflow" by aurelien geron, it will provide a solid framework for experimentation. also, there are tons of high quality papers on the arxiv, the ones about "graph neural networks" will surely be insightful if you are working with unconventional connections or structures.

to summarize, if you got a structure of interconnected nodes with learnable weights optimized via gradient descent, with some kind of non-linearity in the neuron's computation, you are probably still within the realm of "neural networks" even if your structure is rather unconventional. dont be afraid to experiment but also do your research and documentation. you will be amazed by how far you can go with these things.
