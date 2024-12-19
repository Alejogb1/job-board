---
title: "How to do Ancestral trees in a Birth-Death process simulation (python)?"
date: "2024-12-15"
id: "how-to-do-ancestral-trees-in-a-birth-death-process-simulation-python"
---

so, you're looking at simulating ancestral trees within a birth-death process, right? i've been down that rabbit hole a few times, and it's definitely a fun challenge. let me share how i usually tackle it, and maybe it'll give you some ideas for your own implementation.

first off, let's talk about the core problem: we're not just simulating the branching process forward in time. we also want to know *where* those branches came from, tracing lineages backward from a final set of individuals (or samples) to their common ancestor. this means that you have to keep track of ancestry as the simulation runs and then to backtrack this information when the simulation ends.

i remember back when i was working on a project about the evolution of viruses. i had to track the ancestral relationships of different viral strains that i had simulated. i was so focused on the actual simulation that i had neglected to keep track of how the lineages emerged. after running the simulation, i was stuck with a bunch of individuals and zero information about their ancestry. what a mess that was! it took me a week to refactor everything to incorporate the ancestral tracking. it was a good, but painful, learning experience. so yeah, from that point forward, i made a point to plan out how to track ancestry from the very beginning.

my preferred method for keeping track of ancestry is to assign each individual a unique id, and also keeping track of the id of each of its parents. whenever a birth occurs, i create a new id for the newly-born individual and record the parent's id. when a death occurs, i also keep track of it. you can store this information in a dictionary for easy lookup. this dictionary essentially builds the ancestral tree incrementally. we can use this to reconstruct the full tree at the end, in case it is needed.

let me show you a simplified python example, without any simulation logic. this is really just the data structure part of what you asked:

```python
class Individual:
    def __init__(self, id, parent_id=None, birth_time=None, death_time = None):
        self.id = id
        self.parent_id = parent_id
        self.birth_time = birth_time
        self.death_time = death_time

class AncestryTracker:
    def __init__(self):
        self.individuals = {}
        self.next_id = 0

    def create_individual(self, parent_id=None, birth_time=None):
        new_id = self.next_id
        self.next_id += 1
        new_individual = Individual(new_id, parent_id, birth_time)
        self.individuals[new_id] = new_individual
        return new_individual

    def record_death(self, individual_id, death_time):
        if individual_id in self.individuals:
            self.individuals[individual_id].death_time = death_time


    def get_ancestral_tree(self):
        return self.individuals # return the whole structure to do the analysis later

```
this *individual* class has an `id`, a `parent_id`, and `birth_time` and a `death_time`. the `ancestrytracker` class manages the individuals and can return the whole ancestry tree when needed for posterior analysis. notice that we need to keep track of the birth and death times for each individual, too. this is useful when you are going to simulate population sizes with different birth and death rates across different time intervals. 

now, let's get into a slightly more complete example of the simulation. this includes a basic birth-death process. note that this is an extremely basic simulation, in order to keep the example simple, without further features.

```python
import random

def simulate_birth_death(initial_population=1, birth_rate=0.1, death_rate=0.05, simulation_time=10, ancestry_tracker=None):
    if ancestry_tracker is None:
        ancestry_tracker = AncestryTracker()
    
    population = [ancestry_tracker.create_individual(birth_time=0)]
    time = 0
    while time < simulation_time and len(population) > 0:
        # time of next event (either birth or death of some individual)
        times_until_next_event = []
        if len(population) > 0:
            # time to next birth for all individuals
             for i in population:
                times_until_next_event.append(random.expovariate(birth_rate))
        else:
           times_until_next_event = [float('inf')]
        # time to next death for all individuals
        times_until_next_death = []
        if len(population) > 0:
             for i in population:
                times_until_next_death.append(random.expovariate(death_rate))
        else:
            times_until_next_death=[float('inf')]
        # minimum of the next birth and death times
        next_event_time = min(min(times_until_next_event), min(times_until_next_death))

        if next_event_time == float('inf'):
            break
        
        time += next_event_time

        if min(times_until_next_event) < min(times_until_next_death):
             #birth event
             index_of_parent = times_until_next_event.index(min(times_until_next_event))
             parent = population[index_of_parent]
             new_individual = ancestry_tracker.create_individual(parent_id=parent.id, birth_time=time)
             population.append(new_individual)

        else:
             #death event
             index_to_kill = times_until_next_death.index(min(times_until_next_death))
             individual_to_kill = population.pop(index_to_kill)
             ancestry_tracker.record_death(individual_to_kill.id,time)

    return ancestry_tracker

```

this `simulate_birth_death` function takes initial population, birth and death rates, and simulation time as input. it simulates the birth-death process step by step, creating new individuals and recording deaths using the `ancestrytracker` class. the ancestry tracker will have all the information of our individuals and its ancestry.

once the simulation is done, you can walk through the `individuals` dictionary and reconstruct the whole tree. here's a small example on how you could start doing that:

```python
def print_lineage(individual_id, ancestry_tracker, depth=0):
    individual = ancestry_tracker.individuals.get(individual_id)
    if individual:
        print("  " * depth + f"Individual id: {individual.id} birth_time: {individual.birth_time} death_time: {individual.death_time}")
        if individual.parent_id is not None:
            print("  " * (depth+1) + f"Parent: {individual.parent_id}")
            print_lineage(individual.parent_id, ancestry_tracker, depth+2) #recursive call
```

this function takes an individual's id, the `ancestrytracker` object, and the current depth, and prints the whole lineage of that individual using a recursive call. you should start from an individual that survived to the final time. here is how you would use it all together:

```python
if __name__ == "__main__":
    my_tracker = simulate_birth_death(initial_population = 1, birth_rate=0.4, death_rate=0.1, simulation_time=5)
    print("Ancestral lineages:")
    for individual_id in my_tracker.individuals:
        if my_tracker.individuals[individual_id].death_time == None:
            print_lineage(individual_id, my_tracker) #prints only the lineages of the individuals alive at the final time.
            print("-" * 30)
```

this script creates a simple birth-death simulation and then calls `print_lineage` function to display the ancestral lineages of the individuals that survived until the simulation end time.

one thing to consider is how you'll handle very large simulations. this implementation of the ancestry tracker stores all of the individuals in memory. in some simulations, the number of individuals can explode, which will quickly use a lot of memory. in that case, you might want to use techniques like tree-traversal without having all the individuals stored in memory, or using specialized data structures for tree representation, or only storing some lineages and discarding others. for that, i would suggest you read some papers on tree structures and efficient tree algorithms.

also, if you're dealing with large numbers of individuals, generating unique ids can become a bottleneck. while an integer counter works fine for most cases, you could consider using uuid generation, but it is something you should measure and benchmark to be sure it is creating an actual bottleneck. and remember the golden rule: premature optimization is the root of all evil (i had to throw a little joke in there!).

for more in-depth knowledge, i highly recommend looking into some resources about stochastic simulation, branching process theory and tree data structures. "the art of computer programming, volume 1" by donald knuth has very interesting insights into how to handle data structures for graphs and trees and "stochastic modelling for systems biology" by darren j. wilkinson is a good resource to know about birth-death stochastic simulations and other relevant stochastic simulation methods.
