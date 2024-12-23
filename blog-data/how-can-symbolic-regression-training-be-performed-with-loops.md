---
title: "How can symbolic regression training be performed with loops?"
date: "2024-12-23"
id: "how-can-symbolic-regression-training-be-performed-with-loops"
---

Alright, let's talk about symbolic regression training with loops. It’s a topic that I've spent a fair amount of time grappling with, having built a few bespoke systems for material property prediction back in the day. Specifically, we were trying to derive governing equations from experimental data, a task where symbolic regression shines when traditional methods fall short. Instead of relying on pre-defined function forms, we could let the algorithm discover them. It wasn't always smooth sailing, and one of the initial challenges was dealing with the computational cost. Loops, when used judiciously, can be part of the solution, but they can also be a bottleneck if approached naively.

Symbolic regression, at its core, involves searching for the mathematical expression that best fits a given dataset. Most techniques operate within a generative framework, typically a tree-based structure. The nodes represent operators (like addition, multiplication, sine, etc.), and the leaves are variables or constants. Now, the training loop, usually implemented as a population-based evolutionary process (e.g., genetic programming), iteratively modifies and evaluates these expression trees.

You might think of it this way: we begin with a population of random trees, each representing a candidate equation. The loop then evaluates each tree against the data, assigning a fitness score. Based on this fitness, we select trees to breed, mutate, and crossover, thereby creating the next generation of candidate solutions. This process continues until a satisfactory solution is found or a certain number of iterations are completed. This whole process is inherently iterative and benefits from controlled loop structures.

The critical point is *how* you implement these loops, particularly when considering efficiency. In a first iteration, we used to implement a fairly straightforward nested loop structure. A ‘generation’ loop encompassed a loop that iterated through each individual in the population, calculating the fitness. That worked for small datasets, but it quickly became an issue with more complex models and larger datasets. It was primarily bottlenecked by the individual fitness evaluation step within the inner loop.

Let's look at some specific examples to illustrate these points. First, consider a basic, albeit inefficient, Python implementation, without the overhead of a full symbolic regression library, just to get the gist.

```python
import numpy as np
import random

def evaluate_expression(expression_tree, data):
    # simplified evaluator, normally more complex
    results = []
    for x in data:
        try:
            result = eval(expression_tree.replace('x', str(x))) #Avoid using eval in production code
            results.append(result)
        except:
            results.append(np.nan)
    return np.array(results)

def fitness_function(expression_tree, data, target):
    results = evaluate_expression(expression_tree, data)
    if np.any(np.isnan(results)):
        return np.inf # Assign high penalty for invalid expressions
    return np.mean(np.abs(results - target))

def generate_random_expression(variables = ['x'], operators = ['+', '-', '*', '/'], depth = 3):
    if depth == 0 or random.random() < 0.2:
        return random.choice(variables + [str(random.randint(-5, 5))])
    else:
        op = random.choice(operators)
        return f"({generate_random_expression(variables,operators, depth-1)} {op} {generate_random_expression(variables, operators, depth-1)})"

def basic_training_loop(data, target, population_size=50, generations=50):
    population = [generate_random_expression() for _ in range(population_size)]
    best_fitness = np.inf
    best_expression = None

    for gen in range(generations):
      for i in range(population_size):
        fitness = fitness_function(population[i], data, target)
        if fitness < best_fitness:
            best_fitness = fitness
            best_expression = population[i]
      print(f"Generation: {gen}, Best Fitness: {best_fitness}")
      # simplified selection and mutation mechanism
      population.sort(key=lambda expr: fitness_function(expr, data, target))
      population = population[:population_size//2] + [generate_random_expression() for _ in range (population_size//2)]

    return best_expression

data = np.linspace(0,10,100)
target = 2*data + 3 + np.random.normal(0, 1, 100)
best_expr = basic_training_loop(data, target)
print(f'Best Expression found: {best_expr}')
```

This first code snippet showcases a basic implementation where the main loops are clearly visible – one iterating through generations and the other through individuals within the population. However, fitness evaluations are performed iteratively. This is suboptimal as it’s serial.

Our second attempt was a significant improvement. We adopted vectorized operations, leveraging the capabilities of libraries like numpy, wherever possible. Instead of processing data points one at a time, we operated on entire arrays. This led to dramatic speedups. It’s worth noting that the `eval` function here is dangerous, using `ast.literal_eval` and constructing the tree carefully is the best approach. However, for illustration purposes, I'm keeping this simpler.

```python
import numpy as np
import random

def vectorized_evaluate_expression(expression_tree, data):
    try:
        return eval(expression_tree.replace('x', 'data')) # avoid eval in production!
    except:
        return np.full(len(data), np.nan)

def vectorized_fitness_function(expression_tree, data, target):
    results = vectorized_evaluate_expression(expression_tree, data)
    if np.any(np.isnan(results)):
       return np.inf
    return np.mean(np.abs(results - target))

def generate_random_expression_vec(variables = ['x'], operators = ['+', '-', '*', '/'], depth = 3):
    if depth == 0 or random.random() < 0.2:
        return random.choice(variables + [str(random.randint(-5, 5))])
    else:
        op = random.choice(operators)
        return f"({generate_random_expression_vec(variables, operators,depth-1)} {op} {generate_random_expression_vec(variables, operators,depth-1)})"

def vectorized_training_loop(data, target, population_size=50, generations=50):
    population = [generate_random_expression_vec() for _ in range(population_size)]
    best_fitness = np.inf
    best_expression = None
    for gen in range(generations):
      fitnesses = [vectorized_fitness_function(expr, data, target) for expr in population]
      min_fitness_idx = np.argmin(fitnesses)
      if fitnesses[min_fitness_idx] < best_fitness:
        best_fitness = fitnesses[min_fitness_idx]
        best_expression = population[min_fitness_idx]
      print(f"Generation: {gen}, Best Fitness: {best_fitness}")
      population = np.array(population)[np.argsort(fitnesses)][:population_size//2].tolist() + [generate_random_expression_vec() for _ in range (population_size//2)]
    return best_expression

data = np.linspace(0,10,100)
target = 2*data + 3 + np.random.normal(0, 1, 100)
best_expr = vectorized_training_loop(data, target)
print(f'Best Expression found: {best_expr}')
```

Here, the key change is in `vectorized_evaluate_expression` where `eval` now processes the data in one go rather than an iterative approach. This allows significant speed improvements.

Further enhancements involved parallelizing the fitness evaluation step. Using libraries like `multiprocessing` or `concurrent.futures` to farm out individual evaluations to separate processes allowed for another significant speedup. We saw near-linear scaling with the number of cores. This is crucial as even a vectorized fitness evaluation can become slow if the population size is large.

```python
import numpy as np
import random
import concurrent.futures

def vectorized_evaluate_expression_par(expression_tree, data):
    try:
        return eval(expression_tree.replace('x', 'data'))
    except:
        return np.full(len(data), np.nan)

def vectorized_fitness_function_par(expression_tree, data, target):
    results = vectorized_evaluate_expression_par(expression_tree, data)
    if np.any(np.isnan(results)):
        return np.inf
    return np.mean(np.abs(results - target))

def generate_random_expression_par(variables = ['x'], operators = ['+', '-', '*', '/'], depth = 3):
    if depth == 0 or random.random() < 0.2:
        return random.choice(variables + [str(random.randint(-5, 5))])
    else:
        op = random.choice(operators)
        return f"({generate_random_expression_par(variables,operators,depth-1)} {op} {generate_random_expression_par(variables,operators,depth-1)})"


def parallel_training_loop(data, target, population_size=50, generations=50):
    population = [generate_random_expression_par() for _ in range(population_size)]
    best_fitness = np.inf
    best_expression = None
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for gen in range(generations):
            fitnesses = list(executor.map(vectorized_fitness_function_par, population, [data] * population_size, [target] * population_size))
            min_fitness_idx = np.argmin(fitnesses)
            if fitnesses[min_fitness_idx] < best_fitness:
                best_fitness = fitnesses[min_fitness_idx]
                best_expression = population[min_fitness_idx]
            print(f"Generation: {gen}, Best Fitness: {best_fitness}")
            population = np.array(population)[np.argsort(fitnesses)][:population_size//2].tolist() + [generate_random_expression_par() for _ in range (population_size//2)]

    return best_expression


data = np.linspace(0,10,100)
target = 2*data + 3 + np.random.normal(0, 1, 100)
best_expr = parallel_training_loop(data, target)
print(f'Best Expression found: {best_expr}')
```

This snippet now parallelizes fitness calculations using `concurrent.futures`, making the code execute significantly faster, especially on multi-core processors.

Beyond these code-level optimizations, it's worth considering algorithm improvements too. The choice of evolutionary operators, tree representation, and even the initial population setup can impact loop execution time by affecting the convergence speed. Furthermore, libraries specifically designed for symbolic regression, such as gplearn or the DEAP framework (used in many academic papers, like those by Dr. Riccardo Poli), can provide more sophisticated and optimized implementations of the underlying loops. For a deeper dive into the theory of genetic programming and its use in symbolic regression, I'd suggest checking out 'Genetic Programming' by John R. Koza, it is a foundational text in the field. Another excellent source is 'A Field Guide to Genetic Programming' by Riccardo Poli et al. These resources offer insights into the core algorithms which are implemented within the training loops.

In closing, while loops are fundamental to symbolic regression training, it's crucial to be mindful of how they're implemented. Vectorized operations, parallelization, and smart algorithm selection are key to making this powerful technique computationally tractable for real-world problems. It's not just about the loops, it's about *how* you use them.
