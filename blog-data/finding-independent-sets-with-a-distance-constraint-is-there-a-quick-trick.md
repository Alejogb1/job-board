---
title: "Finding Independent Sets with a Distance Constraint: Is There a Quick Trick?"
date: '2024-11-08'
id: 'finding-independent-sets-with-a-distance-constraint-is-there-a-quick-trick'
---

```python
def reduce_to_independent_set(graph, k):
  """Reduces the distance-3 independent set problem to the independent set problem.

  Args:
    graph: A graph represented as an adjacency list.
    k: The desired size of the independent set.

  Returns:
    A new graph and the desired size of the independent set in the new graph.
  """
  new_graph = {}
  for u in graph:
    new_graph[u] = set()
    for v in graph[u]:
      # Add a new vertex in the middle of every edge.
      new_vertex = f"{u}-{v}"
      new_graph[u].add(new_vertex)
      new_graph[v].add(new_vertex)
      new_graph[new_vertex] = {u, v}

  # Add a new vertex and connect it to all the new vertices.
  new_graph["*"] = set(new_graph.keys() - graph.keys())
  for vertex in new_graph["*"]:
    new_graph[vertex].add("*")

  return new_graph, k + 1
```
