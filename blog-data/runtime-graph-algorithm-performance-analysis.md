---
title: "runtime graph algorithm performance analysis?"
date: "2024-12-13"
id: "runtime-graph-algorithm-performance-analysis"
---

Okay so runtime graph algorithm performance analysis yeah I’ve been down that rabbit hole a few times let’s talk about it

First off runtime analysis isn’t magic it’s all about understanding how the time it takes for an algorithm to run scales with the size of the input that size in graph algorithms usually means number of nodes and edges so we're talking *n* and *m* here

My first real tango with this was back in 09 I was working on a social network analysis project yeah before it was cool we needed to find influencer nodes basically central nodes in a large friend graph we used a basic breadth first search BFS and well it was slow really slow the graph had like millions of nodes and the performance was a disaster we were seeing runtime that was like n squared maybe even worse something was clearly wrong because bfs should be in the ballpark of O(n + m) after profiling I realized my naive implementation was copying the entire queue instead of just using pointers or references which added a bunch of overhead it also involved needless dynamic memory allocation it was a rookie mistake for sure I was just out of university back then

We fixed it eventually we rewrote the algorithm with queues using linked lists the runtime went from painfully slow to surprisingly acceptable it was a major lesson on how important data structures choices are to performance it's a classic example of implementation details having a massive effect on big O complexity in practice

So the thing you need to keep in mind is that theoretical runtime big-O notation that’s just a guide it’s like a roadmap but it doesn't tell you about traffic lights or construction zones or potholes on the way it's asymptotic behavior we’re talking about what happens when input gets really big but in the real world small constants can totally change things

Another classic example is Dijkstra's algorithm for finding shortest paths I've used that thing countless times on routing problems from GPS mapping apps to network pathfinding and in other places as well I was working on a network security project last year trying to find the shortest path between vulnerable servers when we encountered some issues Initially we tried the naive implementation using a priority queue built with an array but it was too slow even with relatively small graphs the big O of Dijkstra with an array-based priority queue is something like O(n^2) which is fine for smaller graphs but quickly becomes a problem and a source of headache for large networks

The solution of course was the classic priority queue implementation based on a min-heap which gives you an average runtime closer to O(m log n) yeah that's a big jump and it has a huge impact on practical terms we also experimented with Fibonacci heaps which have an even better asymptotic runtime of O(m + n log n) but in reality on typical network graphs the overhead of Fibonacci heaps makes them less performant than regular binary heaps It’s like needing an F1 car for a trip to the grocery store you are overkilling the solution this just proved that big-O is not the only thing to consider

Now for some code examples because we all like seeing code and this is after all the point here

Okay so here is a Python implementation of BFS on a graph represented with an adjacency list it’s a common thing for graph algorithms

```python
def bfs(graph, start_node):
    visited = set()
    queue = [start_node]
    visited.add(start_node)

    while queue:
        node = queue.pop(0)
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example graph representation
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

bfs(graph, 'A')
```

This is standard its what many would start with when considering BFS and it gives you some feeling for the basic structure

Now here is a basic implementation of Dijkstra's using a standard min-heap in Python for a weighted graph this uses `heapq` from python

```python
import heapq

def dijkstra(graph, start_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]

    while priority_queue:
        dist, current_node = heapq.heappop(priority_queue)

        if dist > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# Example weighted graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 5, 'E': 2},
    'C': {'A': 4, 'F': 2},
    'D': {'B': 5},
    'E': {'B': 2, 'F': 1},
    'F': {'C': 2, 'E': 1}
}
distances = dijkstra(graph, 'A')
print(distances)

```

This is what happens if you want to use an efficient implementation of Dijkstra with weights and you can see the heapq being employed which gives an improvement of the run time

And to really drive the point home here's a slightly different Dijkstra implementation with a binary heap implementation without relying on a built-in python library it requires more code but shows you how it works under the hood. Because this is what happens when you are trying to really get deep into the issue

```python
class BinaryHeap:
    def __init__(self):
        self.heap_array = []

    def _get_parent_index(self, index):
        return (index - 1) // 2

    def _get_left_child_index(self, index):
        return 2 * index + 1

    def _get_right_child_index(self, index):
        return 2 * index + 2

    def _has_parent(self, index):
        return self._get_parent_index(index) >= 0

    def _has_left_child(self, index):
        return self._get_left_child_index(index) < len(self.heap_array)

    def _has_right_child(self, index):
        return self._get_right_child_index(index) < len(self.heap_array)

    def _get_parent(self, index):
        return self.heap_array[self._get_parent_index(index)]

    def _get_left_child(self, index):
        return self.heap_array[self._get_left_child_index(index)]

    def _get_right_child(self, index):
        return self.heap_array[self._get_right_child_index(index)]

    def _swap(self, index1, index2):
        self.heap_array[index1], self.heap_array[index2] = self.heap_array[index2], self.heap_array[index1]

    def _heapify_up(self):
        index = len(self.heap_array) - 1
        while self._has_parent(index) and self._get_parent(index)[0] > self.heap_array[index][0]:
            self._swap(index, self._get_parent_index(index))
            index = self._get_parent_index(index)

    def _heapify_down(self):
        index = 0
        while self._has_left_child(index):
            smaller_child_index = self._get_left_child_index(index)
            if self._has_right_child(index) and self._get_right_child(index)[0] < self._get_left_child(index)[0]:
                smaller_child_index = self._get_right_child_index(index)

            if self.heap_array[index][0] < self.heap_array[smaller_child_index][0]:
                break

            self._swap(index, smaller_child_index)
            index = smaller_child_index


    def insert(self, priority_value, value):
         self.heap_array.append((priority_value, value))
         self._heapify_up()

    def extract_min(self):
        if len(self.heap_array) == 0:
            return None
        if len(self.heap_array) == 1:
            return self.heap_array.pop()
        min_val = self.heap_array[0]
        self.heap_array[0] = self.heap_array.pop()
        self._heapify_down()
        return min_val
    def is_empty(self):
        return len(self.heap_array) == 0

def dijkstra_custom_heap(graph, start_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    priority_queue = BinaryHeap()
    priority_queue.insert(0,start_node)


    while not priority_queue.is_empty():
        dist, current_node = priority_queue.extract_min()
        if dist > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                priority_queue.insert(distance, neighbor)

    return distances
# Example weighted graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 5, 'E': 2},
    'C': {'A': 4, 'F': 2},
    'D': {'B': 5},
    'E': {'B': 2, 'F': 1},
    'F': {'C': 2, 'E': 1}
}
distances = dijkstra_custom_heap(graph, 'A')
print(distances)
```

This is what it would take to actually implement a heap from scratch it’s a lot more code but it shows what's going on and how a priority queue would work behind the scenes

For resources you are going to need some background so I would suggest not looking at blog posts or random websites but more robust content
"Introduction to Algorithms" by Cormen et al. is a classic for a reason Its an extremely complete book and everything in here is gold
Also "Algorithms" by Sedgewick and Wayne is a good option especially if you prefer a more practical approach with Java examples they give a lot of information
Another good option is "Algorithm Design" by Kleinberg and Tardos it provides more focus on algorithm design and theoretical analysis

One more thing to keep in mind is input distribution its a simple one but worth knowing real world graphs aren’t always random sometimes they have power-law distributions or specific network topologies that can affect the performance of algorithms even if theoretically it's the same so you should always test with realistic graph examples not just random graphs you should try different examples to see how the algorithm behaves in various situations

Okay so what was I talking about oh yeah runtime graph algorithm analysis so yes in conclusion always profile always measure and don’t assume that big-O will always be the most important factor in your real-world implementation And always remember to choose the right data structure for the job not everything can be treated like a hammer it depends on the task that you are doing

That’s it I guess yeah good luck and happy coding
