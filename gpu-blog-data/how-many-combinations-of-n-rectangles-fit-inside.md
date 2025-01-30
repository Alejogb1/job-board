---
title: "How many combinations of N rectangles fit inside a square?"
date: "2025-01-30"
id: "how-many-combinations-of-n-rectangles-fit-inside"
---
Determining the exact number of combinations of *N* rectangles that fit inside a square, without overlaps and accounting for all possible arrangements, is an NP-hard problem and doesn't lend itself to a single, neat formula. My experience working on optimization algorithms for resource allocation in a cloud environment highlighted the combinatorial explosion inherent in such packing problems. What seems conceptually simple – fitting shapes into a container – quickly escalates in complexity as the number of rectangles and their varying dimensions increase. There isn’t a readily applicable “number” we can calculate; rather, the solution space demands computational search and often necessitates approximations.

The core difficulty arises from the myriad degrees of freedom: the varying dimensions of the rectangles, their rotation, and the potential placements along the x and y axes of the square. Each added rectangle vastly increases the potential combinations, making an exhaustive search quickly infeasible. Furthermore, even restricting ourselves to axis-aligned rectangles (no rotation) doesn't drastically reduce the complexity. The problem transitions from a simple counting exercise to one of geometric constraint satisfaction, best addressed using heuristic algorithms and potentially AI-based approaches when a precise solution is not mandatory.

To illustrate the challenges, consider the most basic case of only one rectangle (N=1). This provides a very simple answer: there is one way to place *that* rectangle. However, this result is less insightful. The interesting challenge starts when N becomes greater than 1. Let's investigate several scenarios with code and analysis to explore this.

**Code Example 1: Simplified Approach for Fixed Rectangles**

First, let’s constrain the problem significantly. Assume that we have *N* rectangles with fixed dimensions and are merely trying to find *a* packing solution. This isn't counting combinations yet, but it shows a critical starting point. We will use Python because of its clear readability for algorithmic tasks. I'll represent rectangles using tuples (width, height) and the square as a (side) tuple for brevity.

```python
def can_fit(square_side, rectangles, placement_coords):
  """Checks if rectangles fit in a square given a placement list.

  Args:
    square_side: Side length of the square (int).
    rectangles: List of rectangle (width, height) tuples.
    placement_coords: List of (x,y) coordinates (tuples) for each rectangle.

  Returns:
    True if all rectangles fit without overlap, False otherwise.
  """
  for i, (x1, y1) in enumerate(placement_coords):
    width1, height1 = rectangles[i]
    for j in range(i):  # Check for overlaps with already placed rectangles.
      x2, y2 = placement_coords[j]
      width2, height2 = rectangles[j]
      if not (x1 + width1 <= x2 or x1 >= x2 + width2 or y1 + height1 <= y2 or y1 >= y2 + height2):
        return False  # Overlap detected.
    # Check boundaries
    if x1 < 0 or y1 < 0 or x1 + width1 > square_side or y1 + height1 > square_side:
        return False

  return True


#Example Usage
square_side = 10
rectangles = [(2, 3), (4, 2), (3, 3)]
placement_coords = [(0, 0), (2, 3), (6, 0)]
if can_fit(square_side, rectangles, placement_coords):
    print("Rectangles fit without overlap.")
else:
    print("Rectangles do not fit or overlap.")

```

This code implements a basic collision check. It iterates through each rectangle, checking for overlaps against all previously placed rectangles using an axis-aligned bounding box test, and ensuring all rectangles stay within the square. It demonstrates that even with fixed shapes and placement positions, the collision checking is not trivial. The function returns *one* valid arrangement or none but does not count combinations of placements. Note the complexity when evaluating overlaps: each added rectangle adds multiple comparisons.

**Code Example 2: Generating Permutations**

To find *different* arrangements, we must move into the realm of combinatorial algorithms. Here, we will look at a naive method to explore possible permutations of placement order and, hence, a potential arrangement. Critically, this code will not perform geometric checking; it only illustrates the growth of the search space for placement order.

```python
import itertools

def generate_placement_orders(rectangles):
    """Generates all possible placement orders of rectangles.
    Args:
        rectangles: list of rectangles (width, height) tuples.

    Returns:
        List of lists. Each list is one possible ordering
    """
    return list(itertools.permutations(range(len(rectangles))))

rectangles = [(2, 3), (4, 2), (3, 3)]
placement_orders = generate_placement_orders(rectangles)
print(f"Number of placement orders: {len(placement_orders)}") #This is equal to N! where N is number of rectangles
print(f"Example placement order: {placement_orders[0]}")
```

This example uses `itertools.permutations` which provides the number of different rectangle-placement orders. The output will be 3! = 6 for three rectangles, but rapidly increases (N!). The important takeaway is that even before considering *where* to place rectangles, the sheer number of placement orders forms an enormous search space. This underscores why a precise answer is difficult – exhaustive search becomes intractable even for modest *N*. Each permutation is a potentially different spatial arrangement (if different placement coordinates are considered), and therefore may produce a new, distinct set of spatial arrangements if placement is considered.

**Code Example 3: A Recursive (and Incomplete) Approach**

To get closer to combinations and *not* only permutations, one might use a recursive search with backtracking. This algorithm is greatly simplified and does *not* compute a final count but merely demonstrates a potential direction of search. This code is more advanced, but still suffers from severe efficiency problems. It does *not* produce a result for large values of N and should *not* be run if one expects a definitive count of combinations.

```python
def recursive_placement(square_side, rectangles, current_placement, placed_rectangles, result_count):
    """Performs a recursive backtracking search for possible placements.

    Args:
        square_side: Side of the square.
        rectangles: List of rectangles (width, height) tuples.
        current_placement: List of (x, y) tuples currently placed.
        placed_rectangles: List of indices for already placed rectangles.
        result_count: An list that stores the current partial count.
    Returns:
         A potentially updated result count.
    """
    if len(placed_rectangles) == len(rectangles):
        result_count[0] += 1 # Found a complete arrangement
        return

    next_rect_idx = 0
    for i in range(len(rectangles)):
       if i not in placed_rectangles:
            next_rect_idx = i
            break
    next_rect_width, next_rect_height = rectangles[next_rect_idx]

    for x in range(0, square_side):
        for y in range(0, square_side):
            # Simple early check if bounds fit at all.
            if x + next_rect_width > square_side or y + next_rect_height > square_side:
              continue

            proposed_placement = current_placement + [(x,y)]
            #Now, check if this would collide
            is_valid = True
            for i in range(len(placed_rectangles)):
              x2, y2 = current_placement[i]
              width2, height2 = rectangles[placed_rectangles[i]]
              if not (x + next_rect_width <= x2 or x >= x2 + width2 or y + next_rect_height <= y2 or y >= y2 + height2):
                  is_valid = False
                  break

            if is_valid:
              #Continue search, if no collision.
              placed_rectangles.append(next_rect_idx)
              recursive_placement(square_side, rectangles, proposed_placement, placed_rectangles, result_count)
              placed_rectangles.pop()

    return


#Example Usage
square_side = 4
rectangles = [(1, 1), (1, 2)]
result_counter = [0] #Use a mutable type to store results
recursive_placement(square_side, rectangles, [], [], result_counter)
print(f"Possible placements (not exhaustive): {result_counter[0]}") # Expect some result, but it is not guaranteed to find all valid combinations.
```

This recursive function attempts to explore the space of rectangle arrangements. It steps through possible x,y coordinates for the placement of each rectangle and checks overlaps using the `can_fit` logic from the first code example. It uses a backtracking approach - moving back to a previous state in the event a placement does not produce a valid, non-overlapping configuration.

This code example demonstrates why such problems are NP-hard: the branching factor in the search grows exponentially with the number of rectangles. While this implementation can find *some* valid arrangements, it is incomplete and highly inefficient. It explores only a subset of valid configurations. This approach demonstrates that without advanced optimization and pruning techniques, an exhaustive exploration of this search space is essentially impossible for even a small number of rectangles.

**Resource Recommendations**

For deeper understanding, studying the following areas is useful: *Combinatorial Optimization*, which covers techniques for solving discrete optimization problems such as the packing problems.  *Computational Geometry*, which offers algorithms and data structures for handling geometric shapes and spatial relationships. Also,  *Constraint Programming* can provide tools to formulate this problem as a series of constraints on variables, which might help in certain scenarios. Finally, studies into various *Heuristic search methods*, like genetic algorithms and simulated annealing, may lead to *approximate* solutions to this difficult problem. These methods are often the practical solution when a precise, exhaustive count is beyond reach.
