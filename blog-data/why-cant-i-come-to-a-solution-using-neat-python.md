---
title: "Why can't I come to a solution using NEAT-python?"
date: "2024-12-14"
id: "why-cant-i-come-to-a-solution-using-neat-python"
---

well, looks like you're hitting a wall with neat-python. i've been there, trust me. it's a powerful library, but it can be a real headache if you don't have the knobs dialed in correctly. from what you've said, it sounds like you are not getting any working solutions, meaning the fitness is not improving. let's break this down like we are debugging some nasty piece of code, shall we?

first off, it's super common to see neat not converge, especially if you're just getting started. the core issue with evolutionary algorithms, and neat in particular, is that they are massively sensitive to their configuration. it's not like training a neural network where you often see some steady progress, it's more like trying to find a specific grain of sand in the entire desert.

the first thing to check is your fitness function. you gotta be ruthlessly objective here. is it really measuring what you think it's measuring? i remember once, back in my early days, trying to get a bot to navigate a simple maze. my fitness function was rewarding bots that simply moved forward, regardless of whether they were getting closer to the goal. the algorithm did great at going in circles, it was just amazing. lesson learned - always, always double-check, triple-check your fitness function. make sure it's a function that rewards the behaviour you actually want.

another big one is the complexity of your problem space relative to the network's capacity. neat starts with fairly minimal networks, adding complexity through connections and nodes as needed. if your problem is just inherently too complex for it to figure out within a reasonable number of generations, it will just spin its wheels. try simplifying it. start with a really basic setup and then increase the problem space's complexity. maybe a few simple simulations first just to test that the fitness function is working perfectly.

you might also be encountering the dreaded "stagnation" problem. this happens when a population of genomes gets stuck in a local optimum, meaning that they are all sort of similar to each other and none of them are particularly good. there are many neat parameters that directly influence this, like the compatibility threshold, mutation rates and population size. a good place to start tweaking these is the config file.

speaking of the config file, i've seen a lot of folks ignore this critical part and just go with the defaults. don't do this, the defaults are there to start but they are not the best parameters for your specific problem. this is one of the places where your "debugging" time should be spent. the neat documentation has decent descriptions of what those parameters do, but you are better reading the original paper, "evolving neural networks through augmenting topologies". you'll get a very deep understanding of every single parameter and how it works. it will be a bit tough but it’s worth it, it helped me a lot. 

let's look at some code examples. first, here's a very simple fitness function example (replace it with your actual task, or simplify to see progress):

```python
import numpy as np

def simple_fitness(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # simulation
    x = np.random.uniform(-1,1)
    output = net.activate([x])
    target = x**2 # simple function to approximate
    fitness = 1/(1 + abs(target - output[0]))
    return fitness
```

this is a very basic example that tries to approximate x^2 using a feedforward network. if you have a function like this and you are not seeing progress, then there's something drastically wrong with your fitness itself, you need to re-think your approach.

now, for the configuration file. here's a basic starting point, note that this is in the file config-feedforward.txt:

```
[NEAT]
compatibility_threshold = 3.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.1
conn_delete_prob = 0.01
node_add_prob = 0.03
node_delete_prob = 0.001
mutation_power = 0.1
weight_mutation_power = 0.5
bias_mutation_power = 0.5
initial_connection = full_direct
population_size = 100
fitness_criterion = max
fitness_threshold = 100
reset_on_extinction = True

[DefaultGenome]
activation_default = sigmoid
activation_options = sigmoid
aggregation_default = sum
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_replace_rate = 0.1
compatibility_threshold = 3.0
conn_add_prob = 0.1
conn_delete_prob = 0.01
enabled_default = True
initial_connection = full_direct
num_inputs = 1
num_outputs = 1
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.5
weight_replace_rate = 0.1

[DefaultSpeciesSet]
elitism = 2
survival_threshold = 0.2

[DefaultStagnation]
species_fitness_func = mean
max_stagnation = 10
species_elitism = 2

[DefaultReproduction]
elitism = 2
```

this is a basic example, you can load this config using the following code:

```python
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward.txt')
```

remember, that configuration values should be tweaked according to your specific problem.

sometimes, even with the parameters and fitness function properly adjusted, neat simply needs some more time to find something useful. that's because, well, evolution is slow, even with computers. try increasing the population size or the number of generations. but remember, more time means more computational power, and i think we all know, computational power equals money. it's an important trade-off. sometimes a better configuration solves the problem instead of more time.

another trick, if everything else fails, is to try different activation functions. while sigmoid is a classic, it might not be optimal for your problem. tanh, relu or even custom activations are options. look at the neat-python examples to find good implementations of different types of activation functions or use the `activation_options` parameter in your config file.

and, of course, make sure you're actually using neat correctly. there's nothing like spending hours looking at a config file only to realize that i was loading the data incorrectly (true story, that happened to me once).

here is a full working example, that loads the config file and defines the simple_fitness function:

```python
import neat
import numpy as np

def simple_fitness(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # simulation
    x = np.random.uniform(-1,1)
    output = net.activate([x])
    target = x**2 # simple function to approximate
    fitness = 1/(1 + abs(target - output[0]))
    return fitness

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(simple_fitness, 500)

    print('\nBest genome:\n{!s}'.format(winner))
    return winner, config, stats

if __name__ == '__main__':
    winner, config, stats = run_neat('config-feedforward.txt')

    # show best fitness
    best_fitness = max(stats.get_fitnesses())
    print("Best fitness ever: {0}".format(best_fitness))
```

remember to save the config example I gave above in a file named `config-feedforward.txt` in the same folder as the python script. this script loads the config, runs the experiment and then displays the best fitness ever. if you are getting low fitnesses, then its time to go back and debug the fitness, parameter and check that you are using the correct implementation.

as i mentioned above, you should read the original neat paper, "evolving neural networks through augmenting topologies". this will really clear your mind about the nuances of this algorithm. another really good resource for understanding the different evolutionary algorithms is the book “introduction to evolutionary computing”, but that is more for learning the background than focusing on neat specific parameters.

oh and before i forget, why did the neat algorithm refuse to cross the road? because it couldn't find a better path, obviously.

to summarize, if neat isn't converging, it's usually a combination of a bad fitness function, poor configuration parameters, a complex problem space, stagnation, or simply using the library incorrectly. go through each one of these points, make sure you've understood them, and i promise you'll find a solution. good luck, and happy debugging.
