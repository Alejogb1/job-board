---
title: "3 sat to independent set reduction?"
date: "2024-12-13"
id: "3-sat-to-independent-set-reduction"
---

Okay so 3-SAT to Independent Set reduction right I've seen this rodeo before a few times back in the day when I was neck deep in my undergrad algorithms course feels like a lifetime ago honestly anyway the gist of it is that we're trying to show that the Independent Set problem is NP-complete which we do by showing it's at least as hard as 3-SAT which we know is NP-complete

Basically what we gotta do is construct an instance of the Independent Set problem from an arbitrary 3-SAT instance This construction has to be polynomial time otherwise the reduction is kinda pointless because it means that we could solve something really hard really quickly but we couldn't even turn it into the thing we want to solve in the first place It also needs to be correct which is to say if the 3-SAT instance has a satisfying assignment then the independent set instance has an independent set of a specific size and vice versa

So let's say we have a 3-SAT formula I'll use the standard variables x1 x2 x3 and their negations and so on and we have clauses of the form (a or b or c) where a b and c are literals which is either a variable or the negation of a variable

For example lets say we have a formula like this

(x1 or x2 or not x3) and (not x1 or x3 or x4) and (x2 or not x4 or not x1)

Ok so what we do is for every clause we create a group of three vertices in our graph think of them as triangles one for each clause In our example we will have three triangles

Then within each triangle we connect the three vertices all to each other This establishes that within each triangle we can pick only one vertex if we want to form a Independent set since all of the vertices are connected to each other

Now we need to connect the vertices from different clauses together which represents the fact that different choices of literals across the clauses shouldnt contradict one another

So we create an edge between any two vertices if their labels represent opposite literals which means variable x1 and not x1 for example

Using the example i used earlier here is how the reduction would create the graph for Independent set problem

Vertex labels 1: x1 2: x2 3: not x3 within the clause (x1 or x2 or not x3)

Vertex labels 4: not x1 5: x3 6: x4 within the clause (not x1 or x3 or x4)

Vertex labels 7: x2 8: not x4 9: not x1 within the clause (x2 or not x4 or not x1)

Edges exist in each triangle like 1-2 1-3 2-3 etc and between 1-4 1-9 2-7 3-5 4-9 6-8

Ok now what's the size of the independent set we're looking for well it's the number of clauses in our formula In our example it's three If we can find an independent set of size 3 it means there exists a satisfying assignment for our 3-SAT formula

The idea is that picking one vertex in each triangle corresponds to picking one literal to be true in that clause and by having edges between contradictory literals if we have an independent set this means we haven't chosen literals that would create a contradiction such as choosing both x1 and not x1 at the same time

Okay let me show you some code This is in python because that's what I use most of the time You will need to install networkx to run the code it should be `pip install networkx`

```python
import networkx as nx

def sat_to_independent_set(clauses):
    """
    Converts a 3-SAT instance to an Independent Set instance.

    Args:
        clauses: A list of clauses, where each clause is a tuple of literals.
                 A literal is a string like 'x1' or '~x2'.

    Returns:
        A tuple (graph, k), where graph is a networkx Graph object representing the
        Independent Set instance and k is the size of the Independent Set.
    """
    G = nx.Graph()
    vertex_map = {}
    vertex_count = 0
    
    for clause_index, clause in enumerate(clauses):
        for i, literal in enumerate(clause):
            vertex_id = vertex_count
            vertex_map[vertex_id] = literal
            G.add_node(vertex_id)
            vertex_count += 1
    
        # Add edges within a triangle
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
               G.add_edge(vertex_count - len(clause) + i, vertex_count - len(clause) + j)
        
    #Add edges for contradictory literals
    for i in range(len(clauses)*3):
        for j in range (i+1, len(clauses)*3):
            
            if (i in vertex_map and j in vertex_map):
                
                lit1 = vertex_map[i]
                lit2 = vertex_map[j]

                if lit1[0] == '~' and lit1[1:] == lit2:
                   
                   G.add_edge(i,j)
                if lit2[0] == '~' and lit2[1:] == lit1:
                   G.add_edge(i,j)
    
    k = len(clauses)
    
    return G,k


#Example usage
clauses = [('x1', 'x2', '~x3'), ('~x1', 'x3', 'x4'), ('x2', '~x4', '~x1')]
graph, k = sat_to_independent_set(clauses)


print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
print(f"Size of the required Independent Set {k}")

#For simple visualization
#nx.draw(graph, with_labels=True)
#import matplotlib.pyplot as plt
#plt.show()
```
The above code makes the graph and the k value for the independent set problem it is not meant to solve the independent set problem only construct it.

So the core idea is we have this polynomial time reduction from 3SAT to Independent set and since 3-SAT is NP-Complete that means independent set is NP-hard And since independent set is in NP we can conclude that Independent Set is NP complete

I remember when I was studying this stuff and my professor said that reductions are hard because you have to prove two directions If sat is satisfied then your constructed independent set has k size and also if your independent set has size k then there exists a satisfying assignment for SAT I really didnt get this for a while and it was so annoying

Actually the biggest problem I had was my code used to be incorrect because I was not making a new count variable each time for the vertices and my edges were all messed up so always check your edge cases when writing code and always think about the math behind what you are doing not only about the code

And sometimes when i'm trying to visualize this stuff i feel like i'm not only reducing the problem but also my own brain cells (is that considered a joke?) okay let's move on.

Here is another snippet to show how you would find independent sets given a graph using python this will be useful if you need to test if the independent set exists

```python
import networkx as nx
def find_independent_set(graph, k):
    """
    Tries to find an independent set of size k.

    Args:
        graph: A networkx Graph object.
        k: The desired size of the independent set.

    Returns:
        An independent set if one exists or None
    """
    for nodes in combinations(graph.nodes(), k):
        subgraph = graph.subgraph(nodes)
        if subgraph.number_of_edges() == 0:
            return list(nodes)
    return None

from itertools import combinations

#Example usage
clauses = [('x1', 'x2', '~x3'), ('~x1', 'x3', 'x4'), ('x2', '~x4', '~x1')]
graph, k = sat_to_independent_set(clauses)

independent_set = find_independent_set(graph,k)

if independent_set:
  print(f"Found independent set: {independent_set}")
else:
    print("No independent set found of the specified size")
```

This second example uses the itertools module to check all combinations of k vertices and checks if those combinations form an independent set by checking if they have no edges. Now it is important to notice that this is an algorithm that runs with exponential time to the size of the graph in general you shouldn't actually use this algorithm for big graphs we only use it to verify small ones

Finally here is the last snippet that shows you how you can get a boolean assignment from the independent set if it is found

```python
import networkx as nx
from itertools import combinations

def get_assignment_from_independent_set(independent_set, vertex_map):
    assignment = {}
    for node in independent_set:
        lit = vertex_map[node]
        if lit[0] == '~':
            assignment[lit[1:]] = False
        else:
            assignment[lit] = True
    return assignment

#Example usage
clauses = [('x1', 'x2', '~x3'), ('~x1', 'x3', 'x4'), ('x2', '~x4', '~x1')]
graph, k = sat_to_independent_set(clauses)
vertex_map = {}
vertex_count = 0
for clause_index, clause in enumerate(clauses):
        for i, literal in enumerate(clause):
            vertex_id = vertex_count
            vertex_map[vertex_id] = literal
            vertex_count += 1

independent_set = find_independent_set(graph,k)
if independent_set:

  assignment = get_assignment_from_independent_set(independent_set,vertex_map)
  print(f"Found a satisfying assignment {assignment}")
else:
    print("No independent set found of the specified size")
```

This last code gets the boolean values from the independent set we find this is the last piece of the puzzle that shows the satisfying assignment from the independent set itself.

For a deeper dive into this topic I would recommend reading "Computers and Intractability: A Guide to the Theory of NP-Completeness" by Michael R Garey and David S Johnson it's a classic text that really explains everything in detail Also "Algorithm Design" by Jon Kleinberg and Eva Tardos is also very good for all things algorithms related and provides a solid background in this topic And for something more concise and less dense I would suggest you to take a look at "Introduction to Algorithms" by Thomas H. Cormen Charles E. Leiserson Ronald L. Rivest and Clifford Stein it's also known as CLRS and its a very well-known algorithms book that covers this topic pretty good I hope this helps and if you have any more questions just post them I will try to answer as quick as possible.
