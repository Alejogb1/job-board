---
title: "Can PyTorch accelerate the Particle Swarm Optimization algorithm?"
date: "2025-01-30"
id: "can-pytorch-accelerate-the-particle-swarm-optimization-algorithm"
---
Particle Swarm Optimization (PSO), while conceptually simple, often encounters computational bottlenecks, particularly when evaluating complex objective functions or dealing with high-dimensional parameter spaces. Utilizing PyTorch, specifically its ability to perform tensor-based calculations on GPUs, can indeed offer substantial acceleration to PSO implementations. My experience optimizing various machine learning models, notably through large-scale hyperparameter searches that relied on adaptations of PSO, has shown that the algorithmic structure lends itself well to parallelization using tensor operations.

The core of PSO involves iterative updates of particle positions and velocities based on the best-known positions encountered, both individually and within the swarm. These updates inherently involve calculations over multiple data points, making vectorization an immediately applicable strategy. PyTorch provides the necessary tools to structure the particle positions, velocities, and personal best positions as tensors. By representing these elements as tensors rather than individual scalars or lists of values, we enable the underlying C++ implementations within PyTorch and potential GPU acceleration to execute computations in parallel across all particles concurrently. The benefit grows multiplicatively with the swarm size and parameter dimension. While the PSO algorithm itself remains sequential, the update calculations can transition to a parallelizable form.

Let's look at a basic implementation and then show some PyTorch adjustments:

**Example 1: Standard Python PSO (CPU Bound)**

This example demonstrates the foundational mechanics of PSO in a standard procedural manner using numpy, which often runs on the CPU.

```python
import numpy as np

def fitness_function(x):
    return np.sum(x**2)

def pso_step(positions, velocities, pbest_positions, pbest_values, gbest_position, inertia, cognitive, social):
    for i in range(positions.shape[0]): # Iterating over each particle individually
        r1 = np.random.rand(positions.shape[1])
        r2 = np.random.rand(positions.shape[1])
        velocities[i] = inertia * velocities[i] + cognitive * r1 * (pbest_positions[i] - positions[i]) + social * r2 * (gbest_position - positions[i])
        positions[i] = positions[i] + velocities[i]

        current_value = fitness_function(positions[i])
        if current_value < pbest_values[i]:
            pbest_values[i] = current_value
            pbest_positions[i] = positions[i].copy()
        
        if current_value < fitness_function(gbest_position):
            gbest_position = positions[i].copy()

    return positions, velocities, pbest_positions, pbest_values, gbest_position

# Parameters
num_particles = 50
dimension = 2
iterations = 100
inertia = 0.7
cognitive = 1.4
social = 1.4

# Initialization
positions = np.random.rand(num_particles, dimension)
velocities = np.random.rand(num_particles, dimension) * 0.1
pbest_positions = positions.copy()
pbest_values = np.array([fitness_function(positions[i]) for i in range(num_particles)])
gbest_position = pbest_positions[np.argmin(pbest_values)].copy()

# PSO loop
for _ in range(iterations):
    positions, velocities, pbest_positions, pbest_values, gbest_position = pso_step(positions, velocities, pbest_positions, pbest_values, gbest_position, inertia, cognitive, social)

print("Best position:", gbest_position)
print("Best fitness value:", fitness_function(gbest_position))

```

This code exhibits typical PSO update mechanics, looping through every particle within the swarm. This is the area where PyTorch can greatly improve performance. The loop prevents the use of GPU acceleration on the particle update step and also on the fitness evaluation step.

**Example 2: PyTorch PSO (CPU, Tensor Operations)**

Now, we shift the core operations to utilize PyTorch tensors, but keep computation on the CPU. This will highlight the vectorization, but not the GPU acceleration benefits.

```python
import torch

def fitness_function(x):
    return torch.sum(x**2, dim=1) # sum over each particle

def pso_step(positions, velocities, pbest_positions, pbest_values, gbest_position, inertia, cognitive, social):
    r1 = torch.rand(positions.shape, dtype=torch.float32)
    r2 = torch.rand(positions.shape, dtype=torch.float32)
    velocities = inertia * velocities + cognitive * r1 * (pbest_positions - positions) + social * r2 * (gbest_position - positions)
    positions = positions + velocities

    current_values = fitness_function(positions)
    improved_indices = current_values < pbest_values
    pbest_values[improved_indices] = current_values[improved_indices]
    pbest_positions[improved_indices] = positions[improved_indices].clone()

    best_particle_index = torch.argmin(current_values)
    if current_values[best_particle_index] < fitness_function(gbest_position.unsqueeze(0))[0]:
        gbest_position = positions[best_particle_index].clone()

    return positions, velocities, pbest_positions, pbest_values, gbest_position


# Parameters (same as above)
num_particles = 50
dimension = 2
iterations = 100
inertia = 0.7
cognitive = 1.4
social = 1.4

# Initialization (converted to tensors)
positions = torch.rand(num_particles, dimension, dtype=torch.float32)
velocities = torch.rand(num_particles, dimension, dtype=torch.float32) * 0.1
pbest_positions = positions.clone()
pbest_values = fitness_function(positions)
gbest_position = pbest_positions[torch.argmin(pbest_values)].clone()

# PSO loop (similar to before)
for _ in range(iterations):
    positions, velocities, pbest_positions, pbest_values, gbest_position = pso_step(positions, velocities, pbest_positions, pbest_values, gbest_position, inertia, cognitive, social)

print("Best position:", gbest_position)
print("Best fitness value:", fitness_function(gbest_position.unsqueeze(0))[0])
```

In this refined version, the loop on each particle is removed. We use vectorized calculations on PyTorch tensors. Notice how calculating `current_values` calculates them across all particles in parallel. We also replace the scalar min with a parallelized equivalent using `argmin` and comparisons with boolean tensor indexing for vectorized assignment of `pbest_values` and `pbest_positions`. This is a significant shift in computational style. Even on the CPU, it will show benefits over the numpy version.

**Example 3: PyTorch PSO (GPU Accelerated)**

Now we fully utilize PyTorch by adding GPU acceleration. This is accomplished via a simple `.to("cuda")` operation if a GPU is available. The core logic remains similar to Example 2.

```python
import torch

def fitness_function(x):
    return torch.sum(x**2, dim=1)

def pso_step(positions, velocities, pbest_positions, pbest_values, gbest_position, inertia, cognitive, social):
    r1 = torch.rand(positions.shape, dtype=torch.float32, device=positions.device)
    r2 = torch.rand(positions.shape, dtype=torch.float32, device=positions.device)
    velocities = inertia * velocities + cognitive * r1 * (pbest_positions - positions) + social * r2 * (gbest_position - positions)
    positions = positions + velocities

    current_values = fitness_function(positions)
    improved_indices = current_values < pbest_values
    pbest_values[improved_indices] = current_values[improved_indices]
    pbest_positions[improved_indices] = positions[improved_indices].clone()

    best_particle_index = torch.argmin(current_values)
    if current_values[best_particle_index] < fitness_function(gbest_position.unsqueeze(0))[0]:
        gbest_position = positions[best_particle_index].clone()

    return positions, velocities, pbest_positions, pbest_values, gbest_position

# Parameters
num_particles = 50
dimension = 2
iterations = 100
inertia = 0.7
cognitive = 1.4
social = 1.4

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # automatically pick GPU if available
positions = torch.rand(num_particles, dimension, dtype=torch.float32, device=device)
velocities = torch.rand(num_particles, dimension, dtype=torch.float32, device=device) * 0.1
pbest_positions = positions.clone()
pbest_values = fitness_function(positions)
gbest_position = pbest_positions[torch.argmin(pbest_values)].clone()

# PSO loop
for _ in range(iterations):
    positions, velocities, pbest_positions, pbest_values, gbest_position = pso_step(positions, velocities, pbest_positions, pbest_values, gbest_position, inertia, cognitive, social)

print("Best position:", gbest_position.cpu().numpy()) # move back to cpu for printing
print("Best fitness value:", fitness_function(gbest_position.unsqueeze(0))[0].cpu().numpy())
```

The key change here is the automatic device selection and conversion of all tensors to the GPU if available. The speed improvement over Example 2 can be substantial for larger swarm sizes and more complex objective functions. The changes in this example are localized to device placement, everything else utilizes the same logic as the prior vectorized example.

To learn more about using PyTorch effectively for numerical computation I recommend focusing on several resources. The official PyTorch documentation is comprehensive and includes tutorials on tensors, optimization, and GPU usage. Additionally, the book "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann offers an in-depth look at building and deploying models with PyTorch and goes into details about efficient tensor manipulation. Finally, research papers focused on high-performance computing using tensor algebra frequently highlight the same principles used in these example optimizations. Focusing on the core functionality offered by the PyTorch library, specifically its vectorization capabilities and GPU usage, will quickly translate into faster particle swarm optimizations.
