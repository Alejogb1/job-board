---
title: "Why does the differential evolution algorithm produce varying results across multiple runs?"
date: "2025-01-30"
id: "why-does-the-differential-evolution-algorithm-produce-varying"
---
The differential evolution (DE) algorithm, unlike deterministic optimization methods, possesses inherent stochasticity which directly contributes to its variable output across independent runs. This variability stems primarily from the random initialization of the population and the subsequent random selection of individuals for mutation and crossover operations. These factors, coupled with the algorithm's reliance on evolutionary heuristics, make it non-deterministic in nature.

Let's break down why this happens. DE operates on a population of candidate solutions (vectors), initially sampled uniformly or according to a predefined scheme within the search space. Because these starting points are randomized, each run starts from a different region of the parameter landscape. This initial variance significantly affects the trajectory the algorithm will take during its iterative process. Each individual's fitness, calculated by the objective function, is then compared within its generation, and only the fitter individuals will potentially survive to the next generation through mutation, crossover and selection.

The mutation stage introduces noise into the solution space. In standard DE, this process involves selecting a base vector and at least two others from the current generation. The weighted difference between two of these vectors is then added to the base vector to produce a mutant vector. The randomness comes from the random selection of the base vector and the vectors used to create the difference, and this influences the regions of the search space that get explored. Because the selections are stochastic, different mutant vectors are likely generated each time the algorithm is executed on the same problem.

Crossover, another key element, combines features of the mutant vector and a target vector (usually, the initial vector). DE uses binomial or exponential crossover. This step also has an inherent random component: for binomial crossover, elements of either the mutant or target vector are chosen to form a trial vector through a probability. Therefore, different trial vectors are created every run, even from the same set of parents. The degree of randomness inherent in crossover controls how much the parent vectors influence each other.

Finally, the selection stage compares the trial vector with the target vector. The fitter of the two survives, if at all, into the next generation. This process is deterministic based solely on fitness, but is wholly dependent on the outputs from both the mutation and crossover stages which, as already discussed, are stochastic. This iterative process of mutation, crossover, and selection continues for a predetermined number of generations or until some other stopping criterion is met.

The implication of this stochasticity is that the search trajectory, and therefore, the algorithm's output, is significantly influenced by chance at multiple points during execution. This is not an error; it's an intentional characteristic that allows DE to explore the search space widely and escape local minima. However, it leads to a different final approximation of the global optimum during each run of the algorithm. Given that there isn't a single, deterministic path to the optimum when the initial conditions and the processes themselves have random components, it becomes clear why we don't see consistent behavior between runs.

To illustrate this, consider a simplified Python implementation of DE, focusing on the core components.

```python
import numpy as np

def mutate(population, F, base_index, indices):
    base = population[base_index]
    diff = population[indices[0]] - population[indices[1]]
    mutant = base + F * diff
    return mutant

def crossover(target, mutant, CR, D):
    trial = target.copy()
    j_rand = np.random.randint(0,D)
    for j in range(D):
        if np.random.rand() < CR or j == j_rand:
            trial[j] = mutant[j]
    return trial

def select(target, trial, objective_function):
  if objective_function(trial) < objective_function(target):
    return trial
  return target

def differential_evolution(objective_function, bounds, pop_size=20, generations=100, F=0.8, CR=0.9):
    D = len(bounds)
    population = np.random.uniform(bounds[:,0], bounds[:,1], size=(pop_size, D))
    for _ in range(generations):
        new_population = []
        for i in range(pop_size):
            indices = np.random.choice(np.delete(np.arange(pop_size), i), size=2, replace=False)
            mutant = mutate(population, F, i, indices)
            trial = crossover(population[i], mutant, CR, D)
            new_population.append(select(population[i],trial, objective_function))
        population = np.array(new_population)

    best_index = np.argmin([objective_function(x) for x in population])
    return population[best_index]

# Example use
def objective_function(x):
  return x[0]**2 + x[1]**2  # Example objective, the sum of squares

bounds = np.array([[-5, 5], [-5, 5]])  # Bounds for two variables
result = differential_evolution(objective_function, bounds)
print(result)
```

This code provides a basic framework for understanding DE. The `mutate`, `crossover` and `select` functions highlight the use of randomness during both mutation and crossover. Each run, even on the same objective function and with the same parameters, will produce a different outcome.

Let's modify the above example to demonstrate how the results differ, we will execute the function multiple times with identical parameters.

```python
import numpy as np

def mutate(population, F, base_index, indices):
    base = population[base_index]
    diff = population[indices[0]] - population[indices[1]]
    mutant = base + F * diff
    return mutant

def crossover(target, mutant, CR, D):
    trial = target.copy()
    j_rand = np.random.randint(0,D)
    for j in range(D):
        if np.random.rand() < CR or j == j_rand:
            trial[j] = mutant[j]
    return trial

def select(target, trial, objective_function):
  if objective_function(trial) < objective_function(target):
    return trial
  return target

def differential_evolution(objective_function, bounds, pop_size=20, generations=100, F=0.8, CR=0.9):
    D = len(bounds)
    population = np.random.uniform(bounds[:,0], bounds[:,1], size=(pop_size, D))
    for _ in range(generations):
        new_population = []
        for i in range(pop_size):
            indices = np.random.choice(np.delete(np.arange(pop_size), i), size=2, replace=False)
            mutant = mutate(population, F, i, indices)
            trial = crossover(population[i], mutant, CR, D)
            new_population.append(select(population[i],trial, objective_function))
        population = np.array(new_population)

    best_index = np.argmin([objective_function(x) for x in population])
    return population[best_index]

# Example use
def objective_function(x):
  return x[0]**2 + x[1]**2  # Example objective, the sum of squares

bounds = np.array([[-5, 5], [-5, 5]])  # Bounds for two variables

num_runs = 5
for i in range(num_runs):
    result = differential_evolution(objective_function, bounds)
    print(f"Run {i+1}: {result}")
```
This code executes the same `differential_evolution` function multiple times, printing each result. This will demonstrate the variations between runs.

Finally, let’s consider a case where we control the random seed to highlight the deterministic behavior that occurs *given* a specific random initialization.

```python
import numpy as np

def mutate(population, F, base_index, indices):
    base = population[base_index]
    diff = population[indices[0]] - population[indices[1]]
    mutant = base + F * diff
    return mutant

def crossover(target, mutant, CR, D):
    trial = target.copy()
    j_rand = np.random.randint(0,D)
    for j in range(D):
        if np.random.rand() < CR or j == j_rand:
            trial[j] = mutant[j]
    return trial

def select(target, trial, objective_function):
  if objective_function(trial) < objective_function(target):
    return trial
  return target

def differential_evolution(objective_function, bounds, pop_size=20, generations=100, F=0.8, CR=0.9, seed=None):
    D = len(bounds)
    if seed is not None:
        np.random.seed(seed)
    population = np.random.uniform(bounds[:,0], bounds[:,1], size=(pop_size, D))
    for _ in range(generations):
        new_population = []
        for i in range(pop_size):
            indices = np.random.choice(np.delete(np.arange(pop_size), i), size=2, replace=False)
            mutant = mutate(population, F, i, indices)
            trial = crossover(population[i], mutant, CR, D)
            new_population.append(select(population[i],trial, objective_function))
        population = np.array(new_population)

    best_index = np.argmin([objective_function(x) for x in population])
    return population[best_index]

# Example use
def objective_function(x):
  return x[0]**2 + x[1]**2  # Example objective, the sum of squares

bounds = np.array([[-5, 5], [-5, 5]])  # Bounds for two variables

seed_val = 42
num_runs = 3
for i in range(num_runs):
    result = differential_evolution(objective_function, bounds, seed=seed_val)
    print(f"Run {i+1} with seed {seed_val}: {result}")

```

In this code, we have introduced the `seed` parameter which allows us to fix the random number generator's seed. Running with the same seed should produce the same result every time.

For anyone who wants to improve their DE understanding and usage, I would recommend exploring literature on evolutionary algorithms, focusing on papers detailing the effects of different mutation strategies, crossover parameters, and population size. Also consider reviewing books with sections on numerical optimization or metaheuristics. Experimenting with varied settings of F and CR will highlight their impact on the search. Finally, visualizing the trajectories of solutions during the algorithm's run can provide invaluable intuition into the search behavior of this optimization approach. It is important to acknowledge and expect variability in DE results, and adapt workflows accordingly.
