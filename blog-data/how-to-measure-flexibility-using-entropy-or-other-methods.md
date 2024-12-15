---
title: "How to measure flexibility using entropy or other methods?"
date: "2024-12-15"
id: "how-to-measure-flexibility-using-entropy-or-other-methods"
---

so, you're looking into measuring flexibility, huh? interesting. i've been down this road before, not just in some abstract theoretical sense, but also when i was dealing with some gnarly issues in a large scale distributed system a few years back. it's less about physical flexibility and more about how easily things can adapt and change. it’s a cool area to explore.

when you talk about flexibility, what we're really getting at is the capacity of a system, or a process, or even a dataset to handle new situations without falling apart or needing a major overhaul. entropy comes into play here because it gives us a way to quantify disorder, randomness, or, in this case, how much 'wiggle room' a system has. higher entropy in a context can indicate more flexibility or potential for change because it suggests many possible states.

my first real encounter with this was when i was working on optimizing a content delivery network (cdn). we had this massively complex caching system that was supposed to adapt to changing user traffic patterns, but it was getting overwhelmed. the system was rigid, and any small deviation from expected traffic meant performance went down. i realized we needed a metric to actually measure how flexible it was, and that's when i started exploring entropy. initially it wasn't pretty; i mean my first calculations would give me a result, but i wasn't sure they were useful, i was kinda coding blind.

one way to approach this is by treating your system's state as a set of probabilities. imagine we have a system that has a limited number of configurations, like the cdn servers being assigned to different geographical locations based on usage, think of it as an array of different locations each associated to a server, lets call this array `server_configuration`. each state represents a different way the cdn is configured to handle the current load. we can determine this distribution by looking at how often each configuration is used, the idea is that if the configurations are highly distributed, it means that the system is adapting to different circumstances and that shows some flexibility.

the way to calculate the entropy *h* is this:

```python
import numpy as np

def calculate_entropy(probabilities):
  """
    calculates the shannon entropy from a probability distribution.

    Args:
      probabilities (list): list of floats representing the probability distribution.

    Returns:
      float: the entropy of the distribution
  """
  probabilities = np.asarray(probabilities)
  probabilities = probabilities[probabilities > 0]
  return -np.sum(probabilities * np.log2(probabilities))
```

so the idea is that you can collect data about the system's configuration distribution, for example using a system monitoring tools and use them as input, and calculate how this distribution changes over time, you can now have a system which is capable of quantifying how it is adapting to external changing factors.

let's say we measured our cdn's configuration changes, and we found three possible scenarios: one where the system is pretty predictable and uses 1 server configuration 80% of the time and the other 2 the remaining, another where the distribution is more uniform and the system has three configuration with probabilities 30%, 30% and 40% respectively and another one where the system almost always changes, each config is almost never used more than 15% of the time, we can represent this distribution in code and then run this:

```python
probabilities_rigid = [0.8, 0.1, 0.1]
probabilities_medium = [0.3, 0.3, 0.4]
probabilities_flexible = [0.1, 0.15, 0.1, 0.25, 0.05, 0.15, 0.20]

entropy_rigid = calculate_entropy(probabilities_rigid)
entropy_medium = calculate_entropy(probabilities_medium)
entropy_flexible = calculate_entropy(probabilities_flexible)


print(f"entropy of rigid system {entropy_rigid:.2f}")
print(f"entropy of medium system {entropy_medium:.2f}")
print(f"entropy of flexible system {entropy_flexible:.2f}")
```

as you can see, the higher the entropy, the more distributed the probabilities of our configuration are, in other words the system is more flexible in this case. it's a pretty clean way to quantify the 'randomness' of how the system is adapting to external factors. we started seeing that we could use this as a signal of how the system was behaving, and it became a key factor when we decided to optimize our network architecture.

but, entropy is not the whole story. we can also look at other methods, specifically those focused on process complexity. you could use something based on "kolmogorov complexity". it focuses on measuring the length of a program that produces a given data string. if a process is flexible it will require less specific instructions, a general set of instructions could adapt to a wide variety of situations, meaning that the algorithmic description of the process could be smaller.

for example, imagine we have a machine learning model for predicting user behavior. if our model is highly specific to one type of user interaction (it doesn't handle edge cases) the algorithm would need lots of conditions to handle the different cases in data, and that would increase the algorithmic description of the model. but if we build a more general model using transfer learning, the algorithm description could be much smaller and flexible to other inputs. it is hard to directly measure kolmogorov complexity in practice since that means determining the shortest program, which is an undecidable problem, so you need to approximate it.

one method is to look at the minimum description length (mdl). the idea here is that you try to find the simplest model (the one with the shortest algorithmic description) that accurately explains your data. if the model is complex (lots of specific parameters) it is not very flexible, but a simpler one is. measuring mdl can involve looking at the number of parameters in a machine learning model or the length of code needed to achieve a specific task. i had a colleague who once said, that “debugging code is like archaeology, except the artifacts are made of bugs”.

a simple example, not the most reliable one, but it gives the idea of mdl, we can measure the number of lines to describe different configurations of a given function

```python
def function_rigid():
    # very specific function hardcoded behavior with 100 lines of code
    # here would be 100 lines, but for brevity i will skip it
    pass

def function_flexible():
    # general function that can adapt with 50 lines of code
    # again skip it
    pass

def get_function_length(func):
    import inspect
    source_code = inspect.getsource(func)
    return len(source_code.splitlines())

length_rigid = get_function_length(function_rigid)
length_flexible = get_function_length(function_flexible)

print(f"length of rigid function {length_rigid}")
print(f"length of flexible function {length_flexible}")
```

this example provides an overview of how a less flexible function would involve more specific code, while a more flexible one needs less code. if your code is structured in many functions it might give you a better insight about flexibility.

so, to wrap it up, both entropy and methods related to process complexity like kolmogorov complexity, provide different views on the flexibility of a given system, process or model. if you want to dive deeper into the theoretical aspects of this i would recommend “information theory, inference, and learning algorithms” by david mackay, it provides a pretty comprehensive background on entropy, and for complexity theory and kolmogorov complexity “computability and logic” by george s. boolos, john p. burgess and richard c. jeffrey is a very complete book which provides the mathematical grounds to this area. i hope this helps.
