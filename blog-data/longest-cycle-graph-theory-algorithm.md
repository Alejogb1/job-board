---
title: "longest cycle graph theory algorithm?"
date: "2024-12-13"
id: "longest-cycle-graph-theory-algorithm"
---

Alright so you wanna find the longest cycle in a graph right Been there done that a few times back in my day let me tell you its not always a walk in the park but lets break it down

First thing first you gotta understand that finding the absolute *longest* cycle in a general graph is NP-hard it's like trying to find a needle in a haystack but the needle is also made of hay and that hay is also hay its a mess this basically means no one has found a super-efficient algorithm that works for *every* type of graph out there that guarantees finding the absolute longest cycle in a reasonable timeframe if the graphs are large enough so we have to do some clever stuff depending on what kind of graph we're dealing with

Now you didn't specify *what* kind of graph you have so I'll assume it's a general undirected graph think of it like a network of nodes (think computers) connected by edges (network cables) So without more constraint it's tricky

**Here is what we know so far and what we should do**

1.  **NP-Hardness:** We've established it's NP-hard so a guaranteed perfect answer for every graph in reasonable time is off the table We're looking for heuristics or special case solutions depending on the specific case

2.  **Specific Graph Types:** If you have a specific type of graph like a planar graph or a bipartite graph or a directed acyclic graph (DAG) you might have some more specific solutions that may run faster but for the case of any arbitrary graph it's a no go so you need a more general approach

3.  **Heuristics:** Since we can't get perfection we use heuristics They're like shortcuts they don't always give you the best answer but they give you a good answer in a reasonable time its like approximating for example finding the largest circle that can be inside a square you cant find the exact value but approximation would yield a good result This is especially useful when you have a large graph

So what can we do in practice? Here are a few things I've messed around with in the past and they may or may not work for your specific needs depending on how big your graph is and what its topology looks like

**1. Depth-First Search (DFS) Based Approach**

This is your go-to for finding *any* cycle and can be adapted to try and find *longer* ones The idea is you explore paths in the graph and if you hit a node you already visited you have a cycle Keep track of how long the cycles are and pick the longest one you found.

Here is an example code snippet in python just to give you an idea this code is a bit naive and may not be very efficient if your graphs are big

```python
def dfs_longest_cycle(graph):
    max_cycle_length = 0
    max_cycle_path = []

    def dfs(node, path, visited):
        nonlocal max_cycle_length
        nonlocal max_cycle_path

        visited.add(node)
        path.append(node)

        for neighbor in graph[node]:
            if neighbor in path:
                cycle_start_index = path.index(neighbor)
                cycle = path[cycle_start_index:]
                if len(cycle) > max_cycle_length:
                    max_cycle_length = len(cycle)
                    max_cycle_path = cycle
                continue
            if neighbor not in visited:
                dfs(neighbor, path, visited)


        path.pop()

    for node in graph:
       dfs(node, [], set())

    return max_cycle_path

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'E'],
    'D': ['B', 'F'],
    'E': ['C', 'F'],
    'F': ['D','E']
}

longest_cycle = dfs_longest_cycle(graph)
print("Longest cycle:", longest_cycle)
```

**2. Approximation Algorithms using Local Search**

This one is more of a heuristic approach. You start with some initial cycle (you can pick any cycle) then you try to make it better. You do this by swapping nodes edges or other modifications. This is similar to finding the top of a hill you can climb from any initial position so to say It's not perfect but it can give decent results in a decent amount of time Here is a naive version of that one:

```python
import random

def find_initial_cycle(graph):
    start_node = random.choice(list(graph.keys()))
    path = [start_node]
    visited = {start_node}
    current_node = start_node

    while True:
        neighbors = [neighbor for neighbor in graph[current_node] if neighbor != start_node]
        if not neighbors:
             return None
        next_node = random.choice(neighbors)

        if next_node in visited:
            path.append(next_node)
            cycle_start_index = path.index(next_node)
            return path[cycle_start_index:]
        
        path.append(next_node)
        visited.add(next_node)
        current_node = next_node

def improve_cycle(graph, cycle):
    for _ in range(100):
      if len(cycle) < 3:
          return cycle # can't really improve
      
      a_index = random.randint(0,len(cycle)-1)
      b_index = random.randint(0,len(cycle)-1)
      if a_index == b_index:
          continue

      a = cycle[a_index]
      b = cycle[b_index]
      
      neighbors_of_a = graph[a]
      neighbors_of_b = graph[b]
      
      for c in neighbors_of_a:
          for d in neighbors_of_b:
              if c != a and c != b and d!= a and d!= b and (c not in cycle) and (d not in cycle):
                    
                    new_cycle = cycle[:]
                    new_cycle.insert(a_index+1,c)
                    new_cycle.insert(b_index+1,d)
                    
                    if len(new_cycle) > len(cycle) :
                         return new_cycle
      
    return cycle

def find_longest_cycle_local_search(graph):
    longest_cycle = []

    for _ in range(100):
        initial_cycle = find_initial_cycle(graph)
        if initial_cycle is None:
            continue
        
        current_cycle = initial_cycle
        for _ in range(10):
            current_cycle = improve_cycle(graph, current_cycle)
        
        if len(current_cycle) > len(longest_cycle):
            longest_cycle = current_cycle

    return longest_cycle

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B','F','G'],
    'F': ['C', 'E'],
    'G': ['E']
}
longest_cycle = find_longest_cycle_local_search(graph)
print("Longest cycle:", longest_cycle)
```

**3. Branch and Bound (if you need to be more precise)**

If you absolutely need the best solution or want to try for a *better* solution in a big graph and willing to sacrifice some processing time you can try a branch and bound approach This involves exploring possible cycles while keeping track of the best cycle found so far It's computationally intensive but can yield better results than a basic DFS or local search if you give it enough time. It is really the most thorough approach

The idea is:
1. **Initialization:** Start by finding a valid cycle and save this as the current best cycle.
2. **Branching:** Explore all options for extending the path by adding possible nodes one by one.
3. **Bounding:** Every time you branch, check if the current partial path has a potential for improving the best cycle found so far by calculating an upper bound of the path. If not prune that branch.
4. **Termination:** Stop when the entire graph has been explored or the processing time reaches a maximum amount.

Here is an example:

```python
def branch_and_bound_longest_cycle(graph):
    max_cycle_length = 0
    max_cycle_path = []

    def is_promising(path, max_cycle_length):
        if len(path) == 0:
            return True
        
        max_possible_length = len(path) + len(graph) - len(set(path))
        return max_possible_length > max_cycle_length

    def find_cycle_recursive(current_node, path, visited):
        nonlocal max_cycle_length
        nonlocal max_cycle_path

        visited.add(current_node)
        path.append(current_node)
        
        for neighbor in graph[current_node]:
            if neighbor in path:
                cycle_start_index = path.index(neighbor)
                cycle = path[cycle_start_index:]
                if len(cycle) > max_cycle_length:
                    max_cycle_length = len(cycle)
                    max_cycle_path = cycle
                
                continue

            if neighbor not in visited and is_promising(path, max_cycle_length):
                find_cycle_recursive(neighbor, path, visited.copy())

        
        path.pop()
        

    for node in graph:
        find_cycle_recursive(node, [], set())

    return max_cycle_path

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D','E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
longest_cycle = branch_and_bound_longest_cycle(graph)
print("Longest cycle:", longest_cycle)
```
**Things to keep in mind**

*   **Performance:** DFS and the local search approach are generally faster but they may not be the most efficient but if your graph is large it's better to start with a fast approach The branch and bound approach will be the slowest in most cases but it should give you better results than the other approaches.
*   **Graph Size:** As I said if your graph gets larger any of the above approaches can slow down quite a bit You'll need to be more clever with caching and other optimization tricks but remember NP-hardness is a pain in the butt.
*   **Graph Type:** Remember that the graph type is key it may provide ways of optimization like for bipartite graphs.
*   **Randomness:** As you saw the local search has an element of randomness and can yield different results each time so you might need to run it several times to get the best result you can

**Resources**

For a more in depth look you might want to check out "Introduction to Algorithms" by Thomas H Cormen et al It's a classic in the field and covers graph algorithms in detail Also if you are trying to find advanced algorithm for finding long cycles or paths I would recommend a more recent book named "Algorithm Design" by Jon Kleinberg and Éva Tardos This one also goes over a lot of approximation algorithms

So yeah that's basically the gist of it finding the longest cycle is a bit of a challenge but with these tricks in your tool belt you have a good starting point Hope this makes things a bit clearer! Let me know if you have any more questions
