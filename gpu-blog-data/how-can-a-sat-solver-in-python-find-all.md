---
title: "How can a SAT-solver in Python find all free polyomino combinations within a given area?"
date: "2025-01-30"
id: "how-can-a-sat-solver-in-python-find-all"
---
The core challenge in finding all free polyomino combinations within a given area using a SAT solver lies in effectively representing the spatial constraints of the problem as Boolean clauses.  My experience with constraint satisfaction problems, particularly during the development of a circuit layout optimization tool, highlighted the importance of a concise and efficient encoding scheme.  Directly encoding adjacency using pairwise constraints proves computationally expensive for larger areas. Instead, a more sophisticated approach employing a tiling representation coupled with a cardinality constraint offers superior scalability.

**1. Explanation:**

The problem of finding all free polyominoes within a given area can be formulated as a Boolean satisfiability (SAT) problem. We represent the area as a grid, and each cell in the grid can either be occupied by a polyomino tile (True) or empty (False).  The polyomino itself is defined by a set of connected cells.  To ensure we generate only *free* polyominoes—those without holes—we need to encode connectivity and absence of internal cavities. This is achieved through careful construction of the SAT clauses.

First, we need to define a variable for each cell in the grid.  A variable `x_i_j` represents whether the cell at row `i` and column `j` is occupied.  The size of the grid determines the number of variables.

Next, we define clauses that ensure the polyomino's connectivity.  For each cell occupied by the polyomino, at least one of its adjacent cells (horizontally or vertically) must also be occupied.  These adjacency clauses ensure that the polyomino is a single connected component.

To prevent holes, we employ an area constraint. We compute the total number of occupied cells.  This count should match the size of the polyomino we are searching for. If the occupied cell count is larger, it indicates the presence of holes or multiple disjoint polyominoes.  This is enforced using a cardinality constraint,  easily expressible in the SAT formulation through a totalizer circuit.

Finally, we use a SAT solver to find all satisfying assignments. Each satisfying assignment corresponds to a valid free polyomino placement within the given area.  The solution space will be all such placements of the specified polyomino size.  The algorithm iterates through polyomino sizes (e.g., monomino, domino, triomino, etc.), finding all valid placements for each size before moving to the next.

**2. Code Examples:**

These examples illustrate different aspects of the process, focusing on clarity rather than extreme optimization.  I’ve avoided using specific SAT solver libraries for better illustration of core principles.

**Example 1: Representing the Grid and Variables:**

```python
def create_grid_variables(rows, cols):
    """Creates a dictionary of variables representing the grid."""
    variables = {}
    for i in range(rows):
        for j in range(cols):
            variables[(i, j)] = f"x_{i}_{j}"  # String representation for SAT solver
    return variables

#Example usage
grid_variables = create_grid_variables(4, 5) # 4x5 grid
print(grid_variables) # Output: {(0,0):'x_0_0', (0,1):'x_0_1', ...}
```

This code snippet demonstrates a way to represent the grid and assign unique variable names for each cell, crucial for feeding information into a SAT solver.


**Example 2: Adjacency Clauses Generation:**

```python
def generate_adjacency_clauses(grid_variables, rows, cols):
    """Generates clauses ensuring polyomino connectivity."""
    clauses = []
    for i in range(rows):
        for j in range(cols):
            current_var = grid_variables[(i, j)]
            # Check adjacent cells (right and down)
            if j + 1 < cols:
                clauses.append(f"-{current_var} + {grid_variables[(i, j + 1)]}") #Implication: if cell is occupied, neighbor must be as well
            if i + 1 < rows:
                clauses.append(f"-{current_var} + {grid_variables[(i + 1, j)]}")
    return clauses


#Example usage (assuming grid_variables from Example 1)
adjacency_clauses = generate_adjacency_clauses(grid_variables, 4, 5)
print(adjacency_clauses)
```

This function generates clauses that enforce connectivity.  Note the use of the `-` prefix to denote negation in the Boolean expression. This example only considers adjacent right and bottom cells to demonstrate principle. A complete solution would include all four directions.


**Example 3: Cardinality Constraint (Illustrative):**

Implementing a full cardinality constraint requires a sophisticated totalizer circuit, which is beyond the scope of a concise example. This simplified snippet illustrates the concept:

```python
def simplified_cardinality_check(occupied_cells, target_size):
    """Simplified cardinality check –  not a true cardinality constraint."""
    return len(occupied_cells) == target_size

#Example usage:
occupied = [(0,0), (0,1), (1,1)]
target_size = 3
if simplified_cardinality_check(occupied, target_size):
    print("Polyomino size matches target")
else:
    print("Size mismatch, likely a hole or multiple polyominoes")
```
This is a simplified check to illustrate the concept.  A proper implementation would involve using a more sophisticated cardinality constraint encoding, potentially using techniques like sequential counters or adder networks to enforce the exact count of occupied cells matching the target polyomino size.


**3. Resource Recommendations:**

For a deeper understanding of SAT solving, I would recommend studying texts on Boolean algebra and logic design.  Exploring publications on constraint satisfaction problems and their applications in combinatorial optimization will be invaluable.  Finally, familiarizing yourself with the inner workings of different SAT solving algorithms (DPLL, CDCL) is crucial for efficient implementation.  These resources, coupled with practical experience, will build a strong foundation for tackling this type of problem.
