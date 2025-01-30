---
title: "Why does the average population score decrease in a genetic algorithm as it approaches the optimal solution?"
date: "2025-01-30"
id: "why-does-the-average-population-score-decrease-in"
---
The phenomenon of decreasing average population scores in a genetic algorithm as it converges toward an optimal solution stems primarily from the selective pressure imposed by the fitness function and the inherent probabilistic nature of genetic operations like mutation and crossover. This isn't a paradox; it's a direct consequence of how populations evolve within this framework.

Let's establish some background based on my experience building optimization algorithms for signal processing. In a typical genetic algorithm, we represent potential solutions (individuals) as chromosomes (sequences of genes). Each chromosome is evaluated against a fitness function, which assigns a score reflecting how well the solution performs for the optimization problem. The process iterates through generations: individuals are selected for reproduction based on fitness (fitter individuals are more likely to become parents), their genetic material is combined (crossover), and random changes are introduced (mutation) to create new offspring. This sequence of events is designed to progressively shift the population toward higher fitness.

The decline in average score isn’t about the overall performance of *any* individual worsening, it's about the *distribution* of fitness scores tightening around the optimal, or near-optimal, solution. Initially, the population is diverse, with individuals scattered across the search space. Some perform poorly, some may be near good solutions, but the average is pulled down by the significant number of poorly adapted individuals. As selection progresses, the weaker individuals are gradually eliminated, and the population becomes dominated by higher-fitness individuals. This process doesn't necessarily mean every individual's score will increase monotonically across generations; specific individuals can have their scores decrease due to the stochastic nature of mutation. However, the distribution changes markedly.

The key point is that as the population clusters near an optimal solution, there is less room for extreme fitness values. The overall population becomes more homogenous. The individuals with very poor fitness will not reproduce often and eventually be eliminated. Therefore, the average is more reflective of that tightly grouped solution space. The peak fitness of the best individual or even the top 10 will likely increase, but the average, dragged down initially by low-fitness individuals, naturally drops as the lower performing population is removed. As a result, as the best score gets better, the average population performance gets worse when viewed across the whole population.

I’ve seen this consistently in my work, particularly when dealing with multimodal fitness landscapes, such as those encountered in antenna design where the optimization can get stuck on a local maxima or converge at a single point. To provide a more concrete illustration of this, I'll present three code examples, using Python with a minimal implementation that highlights the core concept.

**Code Example 1: Basic Population Tracking**

```python
import random

def fitness(individual):
  # Simplified fitness function for demonstration.
  # The closer to 10, the better
  return abs(10 - individual)

def create_population(size):
  return [random.randint(0, 20) for _ in range(size)]

def select_parents(population):
    # simple roulette wheel selection
  fitness_scores = [fitness(ind) for ind in population]
  total_fitness = sum(fitness_scores)
  probabilities = [score / total_fitness for score in fitness_scores]
  parents = random.choices(population, weights = probabilities, k = 2)
  return parents

def crossover(parents):
  return (parents[0] + parents[1]) // 2

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        return random.randint(0, 20)
    else:
        return individual


def next_generation(population, mutation_rate):
  new_population = []
  for _ in range(len(population)):
      parents = select_parents(population)
      offspring = crossover(parents)
      offspring = mutate(offspring, mutation_rate)
      new_population.append(offspring)
  return new_population

def calculate_average_fitness(population):
    return sum(fitness(ind) for ind in population) / len(population)

# Parameters
population_size = 100
mutation_rate = 0.1
generations = 50

# Initialize population
population = create_population(population_size)

for gen in range(generations):
    average_score = calculate_average_fitness(population)
    best_score = min(fitness(ind) for ind in population) # min since we're optimizing for a value close to 10
    print(f"Gen {gen}: Avg Score = {average_score:.2f}, Best Score = {best_score}")
    population = next_generation(population, mutation_rate)

```

*Commentary:* This example tracks the average fitness of the population across generations. As the algorithm progresses, the ‘best score’ (the individual closest to 10) will generally improve, while the average score will often *decrease*. The population converges towards optimal solution. The fitness function here is simplistic; however, it illustrates the core concept.

**Code Example 2: Explicit Fitness Tracking**

```python
import random

def fitness(individual):
  return abs(10 - individual)

def create_population(size):
  return [random.randint(0, 20) for _ in range(size)]

def select_parents(population):
    fitness_scores = [fitness(ind) for ind in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    parents = random.choices(population, weights = probabilities, k = 2)
    return parents

def crossover(parents):
  return (parents[0] + parents[1]) // 2

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        return random.randint(0, 20)
    else:
        return individual


def next_generation(population, mutation_rate):
  new_population = []
  for _ in range(len(population)):
      parents = select_parents(population)
      offspring = crossover(parents)
      offspring = mutate(offspring, mutation_rate)
      new_population.append(offspring)
  return new_population

def track_fitness(population):
  fitness_values = [fitness(ind) for ind in population]
  return max(fitness_values), sum(fitness_values)/len(fitness_values), min(fitness_values) #max avg min

# Parameters
population_size = 100
mutation_rate = 0.1
generations = 50

# Initialize population
population = create_population(population_size)

max_fitness_scores = []
average_fitness_scores = []
min_fitness_scores = []

for gen in range(generations):
    max_score, avg_score, min_score = track_fitness(population)
    max_fitness_scores.append(max_score)
    average_fitness_scores.append(avg_score)
    min_fitness_scores.append(min_score)

    print(f"Gen {gen}: Max = {max_score:.2f}, Avg = {avg_score:.2f}, Min = {min_score:.2f}")
    population = next_generation(population, mutation_rate)

import matplotlib.pyplot as plt

plt.plot(range(generations), max_fitness_scores, label='Max Fitness')
plt.plot(range(generations), average_fitness_scores, label='Average Fitness')
plt.plot(range(generations), min_fitness_scores, label='Min Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Evolution')
plt.legend()
plt.show()
```

*Commentary:* This expands upon the first example by explicitly tracking not just the average but also the maximum and minimum fitness scores in each generation. This more clearly demonstrates the dynamic - max scores tend to improve, min scores tend to be pushed towards the max, causing the average to compress. The matplotlib plot visualizes how the population evolves as the algorithm progresses. This method also uses a roulette wheel selection method.

**Code Example 3: Population Convergence**

```python
import random
import numpy as np


def fitness(individual):
  return abs(10 - individual)

def create_population(size):
  return [random.randint(0, 20) for _ in range(size)]

def select_parents(population):
    fitness_scores = [fitness(ind) for ind in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    parents = random.choices(population, weights = probabilities, k = 2)
    return parents

def crossover(parents):
  return (parents[0] + parents[1]) // 2

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        return random.randint(0, 20)
    else:
        return individual


def next_generation(population, mutation_rate):
  new_population = []
  for _ in range(len(population)):
      parents = select_parents(population)
      offspring = crossover(parents)
      offspring = mutate(offspring, mutation_rate)
      new_population.append(offspring)
  return new_population

def calculate_average_fitness(population):
    return sum(fitness(ind) for ind in population) / len(population)

def calculate_std_dev(population):
  fitness_scores = [fitness(ind) for ind in population]
  return np.std(fitness_scores)

# Parameters
population_size = 100
mutation_rate = 0.1
generations = 50

# Initialize population
population = create_population(population_size)

for gen in range(generations):
    average_score = calculate_average_fitness(population)
    std_dev = calculate_std_dev(population)
    best_score = min(fitness(ind) for ind in population)

    print(f"Gen {gen}: Avg Score = {average_score:.2f}, Best Score = {best_score}, Std Dev = {std_dev:.2f}")
    population = next_generation(population, mutation_rate)
```

*Commentary:* This example focuses on the standard deviation of the population's fitness scores. As the algorithm progresses, the standard deviation tends to decrease, demonstrating how the population converges. The standard deviation is the spread or dispersion of the data, and when it becomes small it reflects that the fitness distribution is tighter around the average.

In conclusion, the decline in average population score is not a deficiency; it's an indicator that the genetic algorithm is effectively narrowing its search towards higher performing individuals. This observation is expected behavior and a byproduct of the selection, crossover, and mutation operations when applied to a population. Understanding the distribution of fitness, not just the average value is crucial for effective algorithm design and evaluation. For further study, I recommend exploring texts on evolutionary computation, specifically those that delve into the theoretical underpinnings of genetic algorithms as well as practical applications with examples. Good resources include literature on the No Free Lunch theorem and advanced selection methods. Books detailing stochastic optimization approaches are also informative and useful.
