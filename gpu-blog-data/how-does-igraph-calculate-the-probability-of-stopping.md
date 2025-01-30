---
title: "How does `igraph` calculate the probability of stopping at a specific vertex?"
date: "2025-01-30"
id: "how-does-igraph-calculate-the-probability-of-stopping"
---
The probability of stopping at a specific vertex in an `igraph` random walk calculation, often employed in network analysis, isn't a single, universally applicable formula. Instead, it emerges from iterative simulations or mathematical calculations based on the transition probabilities defined by the graph's structure. I've spent considerable time debugging community detection algorithms leveraging random walks within `igraph` and have firsthand experience with the nuances. This isn’t about magically arriving at a final probability; it’s about understanding the underlying process.

The core concept involves modeling a walker's movement from vertex to vertex. In `igraph`, this movement is dictated by a transition probability matrix derived from the graph’s adjacency matrix or, if weighted, modified by edge weights. If we are considering a simple, unweighted, and undirected graph, then the transition probability is simply the inverse of the number of neighbours of the current node. In the case of weighted graphs, transition probabilities are calculated based on the sum of weights of the outgoing edges. Therefore, from a given vertex *i*, the probability of transition to its neighbour *j* will depend on the weight of that edge divided by the sum of all outgoing weights.

When simulating a random walk, the walker will step from a given node to a connected node with the probability described. The calculation of the stopping probability for a vertex is typically done through Monte Carlo simulations. Many simulations of random walks starting from random vertices will be generated, and in these simulations, we will be recording the times the random walker steps into our target vertex. Over a high number of simulations, the relative frequency of visits to this vertex should converge to the probability of stopping in that vertex. 

It’s crucial to note the difference between “hitting” a vertex (the first time the random walk reaches that vertex) and the stationary distribution (the long-term probability of finding the walker at a vertex at any given time). A stopping probability, as the question implies, is closer to the notion of the hitting probability, or the probability a random walk *eventually* lands on a specific vertex. However, if we are talking about the stationary distribution, we must consider the long-term behavior of the random walker, where we allow the random walk to iterate many times, until the probability distribution of the random walker position converges. 

Let's illustrate with some code.

**Example 1: Unweighted Graph - Monte Carlo Simulation**

Here, we demonstrate using simulations to estimate the stopping probability in an unweighted graph:

```python
import igraph as ig
import random

def simulate_random_walk(graph, start_vertex, max_steps, target_vertex):
    current_vertex = start_vertex
    for _ in range(max_steps):
      if current_vertex == target_vertex:
        return True
      neighbors = graph.neighbors(current_vertex)
      if not neighbors:
        return False
      current_vertex = random.choice(neighbors)
    return False


def estimate_stopping_probability(graph, target_vertex, num_simulations, max_steps):
    successful_hits = 0
    for _ in range(num_simulations):
        start_vertex = random.choice(range(graph.vcount()))
        if simulate_random_walk(graph, start_vertex, max_steps, target_vertex):
             successful_hits += 1
    return successful_hits / num_simulations


# Example graph
g = ig.Graph([(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,5)])
target_vertex = 5
num_simulations = 10000
max_steps = 100

stopping_probability = estimate_stopping_probability(g, target_vertex, num_simulations, max_steps)
print(f"Estimated stopping probability for vertex {target_vertex}: {stopping_probability}")
```

In this example, the `simulate_random_walk` function simulates one random walk. We start from a given node and then iteratively step to a random neighbour. We will return `True` if the random walker steps on the `target_vertex`. Otherwise, after a given number of steps we return `False`. The function `estimate_stopping_probability` will simulate many random walks, and keep a tally of how many times the walker landed in the `target_vertex`. The final `stopping_probability` corresponds to how many of the simulated walks landed in the `target_vertex` over the total walks we simulated. This is, essentially, a monte carlo estimation of the hitting probability.

**Example 2: Weighted Graph - Transition Matrix and Power Iteration**

For weighted graphs, we typically use matrix calculations to determine the long-term behaviour of the random walker (i.e., stationary distribution):

```python
import igraph as ig
import numpy as np

def compute_transition_matrix(graph):
    num_vertices = graph.vcount()
    transition_matrix = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        neighbors = graph.neighbors(i)
        if neighbors:
            outgoing_weights = np.array([graph.es[graph.get_eid(i, j)]["weight"] for j in neighbors])
            total_weight = np.sum(outgoing_weights)
            for k, neighbor in enumerate(neighbors):
                transition_matrix[i, neighbor] = outgoing_weights[k] / total_weight

    return transition_matrix

def power_iteration(transition_matrix, num_iterations):
    num_vertices = transition_matrix.shape[0]
    initial_vector = np.ones(num_vertices) / num_vertices # Uniform initial distribution
    current_vector = initial_vector
    for _ in range(num_iterations):
        current_vector = np.dot(current_vector, transition_matrix)
    return current_vector

# Example weighted graph
g = ig.Graph([(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,5)],
              edge_attrs={'weight': [1.0, 2.0, 0.5, 2.0, 1.0, 3.0, 1.0]})

transition_matrix = compute_transition_matrix(g)
num_iterations = 1000
stationary_distribution = power_iteration(transition_matrix, num_iterations)

print("Stationary distribution: ", stationary_distribution)

```

In the second example, we directly calculate the transition probability matrix based on the edge weights of the graph. We start by initializing a matrix full of zeros, and for each vertex, we go over all of its neighbours to fill the transition matrix. The matrix will describe the probability that a random walker in a given node will step into another given node. After calculating the transition matrix, we can estimate the stationary distribution through power iteration. We start from an initial uniform probability distribution over the vertices, then iteratively, we multiply that vector by our transition matrix. The resulting vector represents the updated probabilities of finding the random walker on a specific node. After repeating this for many iterations, the probability distribution converges. Note that this stationary distribution vector is not the same as the hitting probability.

**Example 3: Focusing on a specific Starting Node**

The stationary distribution gives us the long-term probability distribution for the walker, assuming the starting vertex is picked with uniform probability. If we want the stopping probability if we always start from a specific vertex, we can modify the power iteration. Also, we will now estimate a probability distribution over the stopping node by using simulations:

```python
import igraph as ig
import numpy as np
import random

def simulate_random_walk_specific_start(graph, start_vertex, max_steps):
    current_vertex = start_vertex
    for _ in range(max_steps):
      neighbors = graph.neighbors(current_vertex)
      if not neighbors:
        return current_vertex
      current_vertex = random.choice(neighbors)
    return current_vertex


def estimate_stopping_probability_specific_start(graph, start_vertex, num_simulations, max_steps):
    num_vertices = graph.vcount()
    stop_distribution = np.zeros(num_vertices)
    for _ in range(num_simulations):
      stop_node = simulate_random_walk_specific_start(graph, start_vertex, max_steps)
      stop_distribution[stop_node] +=1
    return stop_distribution / num_simulations


# Example graph
g = ig.Graph([(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,5)])
start_vertex = 0
num_simulations = 10000
max_steps = 100

stopping_distribution = estimate_stopping_probability_specific_start(g, start_vertex, num_simulations, max_steps)

print("Stopping distribution from node", start_vertex, ": ", stopping_distribution)
```

This final example focuses on the probability of stopping in different nodes, if we always start the walk in the same, predefined node. The function `simulate_random_walk_specific_start` simulates a random walk starting from a given node, until the maximum number of steps has been reached, or the random walk is trapped in a node with no outgoing edges. The `estimate_stopping_probability_specific_start` will then simulate many random walks from the same node, and tally the ending node of each simulation. In the end, the frequency of stopping on different nodes gives us the stopping distribution when starting from our given node.

`igraph` does not expose the full detail of the algorithms for random walk analysis directly, which can be seen as a black box. However, it exposes several functions like `random_walk()` that allow for direct simulation of random walks in a graph. Based on the underlying code (typically written in C), it's likely that the library leverages similar computational principles to what I've described here.

For deeper exploration, I recommend focusing on advanced network science textbooks covering Markov Chains, and books on numerical linear algebra when considering the matrix-based approach. Additionally, delving into scholarly publications on random walks on graphs will offer a comprehensive treatment of the underlying mathematics and algorithms. Finally, research into the theory of stationary distributions of markov chains can offer valuable theoretical underpinnings of the random walk approach. By examining these sources, the inner workings of stochastic random walk processes can be fully understood.
