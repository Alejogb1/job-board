---
title: "How can uncertainty be incorporated into dynamic optimization?"
date: "2025-01-30"
id: "how-can-uncertainty-be-incorporated-into-dynamic-optimization"
---
The core challenge in incorporating uncertainty into dynamic optimization lies in the inherent tension between the need for precise, deterministic models and the stochastic nature of real-world systems.  My experience working on stochastic control problems for aerospace trajectory optimization underscored this –  deterministic solutions often failed catastrophically when confronted with even minor deviations from the planned trajectory.  This necessitates a move beyond deterministic approaches and the adoption of methodologies designed to handle probabilistic information.

The most effective strategies generally involve reformulating the optimization problem to explicitly consider the uncertainty inherent in the system's dynamics and/or objective function. This is typically achieved through probabilistic modeling of uncertain parameters and the subsequent integration of this uncertainty into the optimization process itself.

There are several approaches to achieve this, the most prevalent being robust optimization, stochastic programming, and Markov Decision Processes (MDPs).

**1. Robust Optimization:** This approach aims to find solutions that are feasible and near-optimal for all possible realizations of the uncertain parameters within a predefined uncertainty set.  The key is defining this uncertainty set appropriately – too small, and the solution lacks robustness; too large, and the solution may be overly conservative and suboptimal.  In my work optimizing satellite constellations, we leveraged a polyhedral uncertainty set, bounding the uncertainty in orbital parameters. This offered a good balance between computational tractability and sufficient robustness.

**Code Example 1 (Robust Optimization using Python and CVXPY):**

```python
import cvxpy as cp
import numpy as np

# Define uncertain parameters with uncertainty set
uncertainty_set = cp.bmat([[cp.Parameter(nonneg=True), cp.Parameter()],
                            [cp.Parameter(), cp.Parameter(nonneg=True)]])

# Define optimization variables
x = cp.Variable(2)

# Define objective function
objective = cp.Minimize(cp.quad_form(x, np.array([[1, 0], [0, 1]])))

# Define constraints – incorporating uncertain parameters
constraints = [x >= uncertainty_set @ np.array([1,1])]

# Solve robust optimization problem
problem = cp.Problem(objective, constraints)

# Iterate through possible uncertainty realizations
num_realizations = 100
optimal_x = []
for i in range(num_realizations):
    uncertainty_set[0,0].value = np.random.uniform(0.0, 1.0)
    uncertainty_set[1,1].value = np.random.uniform(0.0, 1.0)
    uncertainty_set[0,1].value = np.random.uniform(-0.5, 0.5)
    uncertainty_set[1,0].value = uncertainty_set[0,1].value # Symmetric for simplicity
    problem.solve()
    optimal_x.append(x.value)

# Analyze the range of optimal solutions

```

This code snippet demonstrates a basic robust optimization problem.  The key is the use of `cp.Parameter` to represent uncertain parameters within the constraints. The loop iterates over different realizations within the uncertainty set, evaluating the optimal solution for each.  Analyzing the `optimal_x` array then reveals the robustness of the solution. Note that this example uses a simple quadratic objective and linear constraints; more complex problems may necessitate different solvers.


**2. Stochastic Programming:** This approach incorporates probabilistic information about the uncertain parameters directly into the optimization problem.  This can be done through chance constraints, which limit the probability of constraint violations, or by optimizing expected values or other risk measures.  In my past work on power grid management, we utilized stochastic programming with chance constraints to handle unpredictable fluctuations in renewable energy generation.

**Code Example 2 (Stochastic Programming with Chance Constraints):**

```python
import cvxpy as cp
import numpy as np

# Define uncertain parameter
uncertainty = cp.Parameter(nonneg=True)

# Define optimization variable
x = cp.Variable()

# Define objective function
objective = cp.Minimize(x)

# Define chance constraint
chance_constraint = cp.prob(x >= uncertainty) >= 0.95  # Probability of constraint satisfaction >= 95%

# Define problem and solve
problem = cp.Problem(objective, [chance_constraint])
uncertainty.value = np.random.normal(5,2) # Example uncertainty distribution
problem.solve()
print(x.value)

```
This example showcases a chance constraint. The `cp.prob` function calculates the probability that `x >= uncertainty`. The constraint ensures this probability remains above 0.95, guaranteeing a high level of constraint satisfaction despite uncertainty in `uncertainty`.  Note that solving chance-constrained problems might require specialized techniques and solvers beyond those readily available in basic optimization libraries.



**3. Markov Decision Processes (MDPs):**  For problems where uncertainty unfolds sequentially over time, MDPs provide a powerful framework. They model the system as a state-transition system where actions affect the probability of transitioning to different states, and rewards (or costs) are associated with these transitions. Dynamic programming or reinforcement learning techniques can then be employed to find optimal policies. During my research into autonomous vehicle navigation, I used MDPs to handle uncertainty in traffic patterns and pedestrian behavior.

**Code Example 3 (Simplified MDP using Python):**

```python
import numpy as np

# Define states, actions, transition probabilities, and rewards
states = ['A', 'B', 'C']
actions = ['left', 'right']
transition_probabilities = {
    'A': {'left': {'A': 0.8, 'B': 0.2}, 'right': {'A': 0.6, 'C': 0.4}},
    'B': {'left': {'A': 0.7, 'B': 0.3}, 'right': {'C': 1.0}},
    'C': {'left': {'B': 0.9, 'C': 0.1}, 'right': {'C': 0.8, 'A': 0.2}}
}
rewards = {
    'A': {'left': 10, 'right': 5},
    'B': {'left': 20, 'right': 15},
    'C': {'left': 2, 'right': 10}
}

# Value iteration (simplified)
gamma = 0.9  # Discount factor
V = np.zeros(len(states))
for _ in range(1000):  #Iterations for convergence
    V_new = np.zeros(len(states))
    for s in range(len(states)):
        V_new[s] = max([sum([transition_probabilities[states[s]][a][states[s2]] * (rewards[states[s]][a] + gamma * V[s2])
                            for s2 in range(len(states))]) for a in actions])
    V = V_new

print(V) #Optimal value function for each state


```

This example demonstrates a simplified value iteration algorithm for an MDP.  The transition probabilities and rewards incorporate the uncertainty in the system's dynamics.  A more sophisticated implementation would be required for larger state and action spaces, possibly using techniques like Monte Carlo simulation or approximate dynamic programming.  In realistic scenarios, deep reinforcement learning algorithms would be employed to handle the complexity.


**Resource Recommendations:**  "Stochastic Optimization" by Shapiro, Dentcheva, and Ruszczyński;  "Dynamic Programming and Optimal Control" by Dimitri P. Bertsekas; "Reinforcement Learning: An Introduction" by Sutton and Barto.  Furthermore,  a solid grounding in probability theory and optimization techniques is crucial.  Specialized solvers for robust optimization and stochastic programming are also invaluable.
