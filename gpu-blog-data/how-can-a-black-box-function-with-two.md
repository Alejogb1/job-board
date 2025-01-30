---
title: "How can a black box function with two inputs be optimized?"
date: "2025-01-30"
id: "how-can-a-black-box-function-with-two"
---
The performance of a black box function with two inputs is critically dependent on the nature of the function itself and the characteristics of its input domain; without internal visibility, optimization strategies must focus on techniques such as input sampling, surrogate modeling, and potentially, heuristic algorithms. In my experience working on optimization problems for embedded systems, I've frequently encountered situations where the underlying function was completely opaque, demanding a pragmatic and data-driven approach.

The inherent challenge with optimizing a black box function lies in the lack of gradient information. Traditional optimization methods like gradient descent are rendered unusable. Therefore, optimization typically relies on evaluating the function at different points in the input space and then using the results to guide future explorations. Since two inputs define a 2D input space, the chosen strategies must efficiently navigate and exploit any structure present in this space. The efficiency of the chosen strategy is vital, especially when evaluating the function is computationally expensive or time-consuming, which is often the case with hardware simulations or complex numerical models. The goal is to find the input combination (x,y) that minimizes or maximizes the output value f(x, y), with the caveat that f is only accessible through function evaluations; its internal mechanism is hidden.

One strategy is to use a sampling approach, covering the 2D input space in a structured manner. Grid sampling, while straightforward, quickly becomes computationally intensive as the desired resolution increases. Instead, a quasi-random sampling approach, such as using a Sobol sequence or a Halton sequence, often provides better coverage with fewer samples. These sequences are designed to minimize clustering and ensure that each region of the input space is represented. After sampling, one can identify the point associated with the best (minimum or maximum) result, thus obtaining an initial estimation of the optimum location.

Another method involves constructing a surrogate model of the black box function. A surrogate model is an approximation of the original function that is computationally inexpensive to evaluate. Gaussian processes, also known as Kriging, are often effective choices, especially when the underlying function is assumed to be smooth. These methods model the function as a distribution, capturing the uncertainty inherent in the black-box evaluation and guiding subsequent sampling efforts towards the areas with high probability of improvement. The surrogate model is progressively refined by re-evaluating the actual black-box function at carefully selected points that are predicted by the model to lead to better outputs. This iteratively builds a better understanding of the function landscape, allowing for faster convergence towards optimal parameters.

Finally, more advanced optimization algorithms like genetic algorithms or particle swarm optimization can be effective. These techniques are heuristic search algorithms that mimic natural processes, exploring the solution space guided by the results of previous evaluations. They often work well in situations with complex, multimodal landscapes where many local minima exist, but they can also be less effective compared to surrogate methods on smoother functions. Their effectiveness relies on carefully tuned parameters which can introduce new areas of experimentation.

Here are a few code examples, focusing on Python due to its popularity in scientific computing, to illustrate the concepts discussed:

**Example 1: Quasi-random sampling using Sobol sequence and basic grid sampling for comparison.**

```python
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt

def black_box_function(x, y):
    # Placeholder black box function. In real scenario, this would
    # be an external, opaque function.
    return (x**2 + y**2) + np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

# Define input space range
x_range = [0, 5]
y_range = [0, 5]

# Generate Sobol samples
sampler = qmc.Sobol(d=2, scramble=False) # Use scramble = True in more complex cases
num_samples = 100
sobol_samples = sampler.random(num_samples)
sobol_samples = sobol_samples * (np.array(x_range) - np.array([0, 0])) + np.array([0,0])

# Evaluate the black-box
sobol_values = [black_box_function(x[0], x[1]) for x in sobol_samples]
best_sobol_idx = np.argmin(sobol_values)
best_sobol_x = sobol_samples[best_sobol_idx][0]
best_sobol_y = sobol_samples[best_sobol_idx][1]
best_sobol_val = sobol_values[best_sobol_idx]

# Generate Grid samples for comparision
x_grid = np.linspace(x_range[0], x_range[1], int(np.sqrt(num_samples)))
y_grid = np.linspace(y_range[0], y_range[1], int(np.sqrt(num_samples)))
grid_x, grid_y = np.meshgrid(x_grid, y_grid)
grid_samples = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
grid_values = [black_box_function(x[0], x[1]) for x in grid_samples]
best_grid_idx = np.argmin(grid_values)
best_grid_x = grid_samples[best_grid_idx][0]
best_grid_y = grid_samples[best_grid_idx][1]
best_grid_val = grid_values[best_grid_idx]

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(sobol_samples[:, 0], sobol_samples[:, 1], c=sobol_values, cmap='viridis')
plt.scatter(best_sobol_x, best_sobol_y, color='red', marker='*', s=200, label='Best')
plt.title(f"Sobol Sampling (Best = {best_sobol_val:.2f})")
plt.xlabel("Input X")
plt.ylabel("Input Y")
plt.colorbar(label="Output")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(grid_samples[:, 0], grid_samples[:, 1], c=grid_values, cmap='viridis')
plt.scatter(best_grid_x, best_grid_y, color='red', marker='*', s=200, label='Best')
plt.title(f"Grid Sampling (Best = {best_grid_val:.2f})")
plt.xlabel("Input X")
plt.ylabel("Input Y")
plt.colorbar(label="Output")
plt.legend()


plt.tight_layout()
plt.show()


print(f"Best Sobol result: x={best_sobol_x:.2f}, y={best_sobol_y:.2f}, value={best_sobol_val:.2f}")
print(f"Best Grid result: x={best_grid_x:.2f}, y={best_grid_y:.2f}, value={best_grid_val:.2f}")
```
This example demonstrates how to generate quasi-random samples and compare this against a simple grid sampling technique, using a dummy black box function. The Sobol sequence provides a more uniform distribution of samples than a regular grid with the same number of points, leading to a potentially better exploration of the input space, and is visible in the output plot. The color represents the function values at those locations. This example provides a baseline for comparing more sophisticated methods.

**Example 2: Surrogate modeling using Gaussian Process Regression.**

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def black_box_function(x, y):
    # Placeholder black box function.
    return (x**2 + y**2) + np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

#Initial Samples
x_range = [0, 5]
y_range = [0, 5]

initial_samples = np.random.rand(5, 2)
initial_samples[:, 0] = initial_samples[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
initial_samples[:, 1] = initial_samples[:, 1] * (y_range[1] - y_range[0]) + y_range[0]

initial_values = np.array([black_box_function(x[0], x[1]) for x in initial_samples])

# Kernel
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gp.fit(initial_samples, initial_values)


# Acquisition function - Expected Improvement
def expected_improvement(x, model, y_opt, xi=0.01):
    x = np.array(x).reshape(1, -1)
    mu, sigma = model.predict(x, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = np.maximum(0, y_opt - mu - xi)
    z = improvement / sigma
    ei = improvement*np.cdf(z) + sigma*np.exp(-0.5*z**2)/np.sqrt(2*np.pi)
    return -ei # We minimize the negative

# Sequential optimization
num_iterations = 5
for _ in range(num_iterations):

    # Optimization
    bounds = np.array([x_range, y_range])
    opt_res = minimize(expected_improvement, x0=np.random.rand(2), args=(gp, min(initial_values)),
                        method='L-BFGS-B', bounds=bounds.T)

    # New evaluation
    next_point = opt_res.x
    new_value = black_box_function(next_point[0], next_point[1])

    #Update model
    initial_samples = np.vstack((initial_samples, next_point))
    initial_values = np.append(initial_values, new_value)
    gp.fit(initial_samples, initial_values)

# Visualization of the surrogate model
x_plot = np.linspace(x_range[0], x_range[1], 100)
y_plot = np.linspace(y_range[0], y_range[1], 100)
X, Y = np.meshgrid(x_plot, y_plot)
XX = np.stack([X.flatten(), Y.flatten()], axis=-1)

Z, sigma = gp.predict(XX, return_std=True)

Z = Z.reshape(X.shape)
sigma = sigma.reshape(X.shape)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, label='Surrogate Model')
ax.scatter(initial_samples[:, 0], initial_samples[:, 1], initial_values, color='red', marker='o', s=100, label='Sampled Points')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Predicted Value")
ax.legend()
ax.set_title("Surrogate Model Approximation")
plt.show()


best_index = np.argmin(initial_values)
best_x = initial_samples[best_index][0]
best_y = initial_samples[best_index][1]
best_val = initial_values[best_index]
print(f"Best result: x={best_x:.2f}, y={best_y:.2f}, value={best_val:.2f}")
```
This example demonstrates Gaussian process-based Bayesian optimization using an Expected Improvement (EI) acquisition function. The plot visualizes the current surrogate model and sample locations. A key feature of surrogate modeling is that it seeks a trade-off between exploration (regions of high uncertainty) and exploitation (regions of low expected value). This example shows the iterative refinement of the surrogate model by adding new evaluation points based on the optimized acquisition function.

**Example 3: Simple genetic algorithm.**

```python
import numpy as np
import random

def black_box_function(x, y):
    # Placeholder black box function.
    return (x**2 + y**2) + np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

def create_initial_population(population_size, x_range, y_range):
    population = []
    for _ in range(population_size):
      x = random.uniform(x_range[0], x_range[1])
      y = random.uniform(y_range[0], y_range[1])
      population.append((x, y))
    return population

def fitness_function(individual):
    return black_box_function(individual[0], individual[1])

def select_parents(population, fitnesses, num_parents):
    population_with_fitness = list(zip(population, fitnesses))
    population_with_fitness.sort(key=lambda x: x[1]) # Assuming minimization problem
    parents = [ind[0] for ind in population_with_fitness[:num_parents]]
    return parents

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        alpha = random.random()
        child_x = alpha * parent1[0] + (1 - alpha) * parent2[0]
        child_y = alpha * parent1[1] + (1 - alpha) * parent2[1]
        return (child_x, child_y)
    return parent1

def mutate(individual, mutation_rate, x_range, y_range):
    mutated_x = individual[0]
    mutated_y = individual[1]
    if random.random() < mutation_rate:
      mutated_x += random.gauss(0, 0.5)
      mutated_y += random.gauss(0, 0.5)
      mutated_x = np.clip(mutated_x, x_range[0], x_range[1])
      mutated_y = np.clip(mutated_y, y_range[0], y_range[1])
    return (mutated_x, mutated_y)

# Parameters
population_size = 50
num_generations = 30
num_parents = 10
mutation_rate = 0.1
crossover_rate = 0.8
x_range = [0, 5]
y_range = [0, 5]

# Genetic algorithm
population = create_initial_population(population_size, x_range, y_range)

for generation in range(num_generations):
  fitnesses = [fitness_function(individual) for individual in population]
  parents = select_parents(population, fitnesses, num_parents)
  new_population = parents.copy()

  while len(new_population) < population_size:
      parent1 = random.choice(parents)
      parent2 = random.choice(parents)
      child = crossover(parent1, parent2, crossover_rate)
      mutated_child = mutate(child, mutation_rate, x_range, y_range)
      new_population.append(mutated_child)
  population = new_population

best_individual = min(population, key=fitness_function)
best_value = fitness_function(best_individual)
print(f"Best result: x={best_individual[0]:.2f}, y={best_individual[1]:.2f}, value={best_value:.2f}")
```
This last example gives a rudimentary genetic algorithm approach. It shows the basic mechanism of initialization, parent selection, crossover, and mutation. The code is a very basic implementation that does not cover all complexities of a GA, such as advanced selection methods, elitism, or adaptive parameters, but it provides a simple way to approach the problem when more standard gradient-based optimizations are not applicable.

For those interested in further study, I would recommend exploring resources on Bayesian optimization, surrogate modeling with Gaussian processes, and the basics of evolutionary algorithms. Researching statistical methods on design of experiments, such as Latin hypercube sampling is also valuable in this context. For an understanding of numerical optimization, consult introductory texts in that area. Exploring dedicated Python libraries such as *scikit-optimize* would also provide a wealth of practical examples. Finally, consider reviewing materials on sensitivity analysis; while not directly addressing optimization, it helps in identifying important input parameters, which indirectly influences optimization strategies.
