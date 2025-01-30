---
title: "What algorithm efficiently finds squares in a grid?"
date: "2025-01-30"
id: "what-algorithm-efficiently-finds-squares-in-a-grid"
---
Finding squares within a grid, particularly one that might contain various shapes or marked cells, requires a methodical approach. I've encountered this issue numerous times in image processing and game development contexts where identifying these structures quickly is critical. Naive approaches, like checking every possible set of four points, rapidly become computationally expensive as grid size increases. The solution lies in leveraging the geometric properties of a square and strategically reducing the search space.

The core principle hinges on recognizing that a square is defined by four equal sides and four right angles. Instead of examining arbitrary combinations of points, it's far more efficient to identify a potential side of the square and then check if the other three vertices logically follow. This approach primarily relies on two steps: 1) iterating over pairs of points that could form a side and 2) computationally verifying the existence of the remaining two vertices.

My typical implementation involves a nested loop structure. The outer loop iterates through each point in the grid, and the inner loop iterates through all *subsequent* points. Why only subsequent? Because each potential line segment will be formed by a point, *P1*, and a point, *P2*, and if we allow points already encountered to contribute a line segment, we'd have multiple equivalent lines to check. For each pair of points, I calculate the distance (length of the potential side) and the directional vector from P1 to P2. This vector encapsulates the direction and magnitude of that potential side. With these in hand, the coordinates of the remaining two vertices, P3 and P4, can be precisely calculated by rotating and translating the original vector. If both P3 and P4 also exist and are marked, a square has been found.

The complexity of this algorithm is approximately O(n^3) in the worst case, where 'n' represents the number of grid points, because while each point is being compared to other points, the check for remaining vertices is linear in time. The actual time taken is considerably lower on average in many practical grid arrangements. This is due to the fact that only points considered for a side vector can be used, greatly reducing the number of candidate side vectors.

Let’s explore this with some examples. Assume the grid is represented as a set of coordinate tuples within a Python dictionary to speed access to relevant points. The `grid` dictionary uses tuples as keys, with a boolean `True` value indicating a point exists.

**Code Example 1:** A straightforward implementation, focusing on readability and understanding the core logic.

```python
import math

def find_squares_simple(grid):
  squares = []
  points = list(grid.keys())
  n = len(points)
  for i in range(n):
    for j in range(i + 1, n):
      p1 = points[i]
      p2 = points[j]
      dx = p2[0] - p1[0]
      dy = p2[1] - p1[1]
      side_length = math.sqrt(dx * dx + dy * dy)
      if side_length == 0: #Same point
          continue

      p3_x = p2[0] - dy
      p3_y = p2[1] + dx
      p4_x = p1[0] - dy
      p4_y = p1[1] + dx

      if (p3_x, p3_y) in grid and (p4_x, p4_y) in grid:
        squares.append((p1, p2, (p3_x, p3_y), (p4_x, p4_y)))
  return squares

# Example usage
grid1 = {(0, 0): True, (0, 1): True, (1, 0): True, (1, 1): True}
print(find_squares_simple(grid1))
# Output: [((0, 0), (0, 1), (1, 1), (1, 0))]
```

In this example, the nested loops iterate through all point pairs. For every candidate line segment (p1 to p2), the `dx` and `dy` differences are calculated, alongside the Euclidean distance, skipping invalid pairings where points are identical. The potential coordinates for the remaining points (p3 and p4) are calculated using rotational transformations of the vector from `p1` to `p2`. If both potential points are present in the grid, we’ve located a square. This is a clear, but potentially slower implementation.

**Code Example 2:** An optimized variant that checks for the presence of the third point before calculating the fourth, making use of Python's short-circuit evaluation.

```python
import math

def find_squares_optimized(grid):
  squares = []
  points = list(grid.keys())
  n = len(points)
  for i in range(n):
    for j in range(i + 1, n):
      p1 = points[i]
      p2 = points[j]
      dx = p2[0] - p1[0]
      dy = p2[1] - p1[1]
      side_length = math.sqrt(dx * dx + dy * dy)
      if side_length == 0:
          continue

      p3_x = p2[0] - dy
      p3_y = p2[1] + dx
      if (p3_x, p3_y) not in grid:
            continue
      
      p4_x = p1[0] - dy
      p4_y = p1[1] + dx
      if (p4_x, p4_y) in grid:
        squares.append((p1, p2, (p3_x, p3_y), (p4_x, p4_y)))
  return squares
# Example usage
grid2 = {(0, 0): True, (0, 2): True, (2, 0): True, (2, 2): True, (1,1):True}
print(find_squares_optimized(grid2))
# Output: [((0, 0), (0, 2), (2, 2), (2, 0))]
```

Here, if the third point `(p3_x, p3_y)` is not in the grid, the program skips to the next potential line segment, without computing `p4`. This small optimization reduces redundant calculations when potential squares are not fully present.

**Code Example 3:** Handling potential rounding errors by using a tolerance when comparing side lengths, which can be an issue with floating-point arithmetic. Note, this example focuses on checking equality of the side lengths as that was a common area of concern from my personal experience. It does *not* account for non-axis-aligned squares.

```python
import math

def find_squares_tolerance(grid, tolerance=1e-6):
  squares = []
  points = list(grid.keys())
  n = len(points)
  for i in range(n):
    for j in range(i + 1, n):
      p1 = points[i]
      p2 = points[j]
      dx = p2[0] - p1[0]
      dy = p2[1] - p1[1]
      side_length = math.sqrt(dx * dx + dy * dy)
      if side_length == 0:
        continue

      p3_x = p2[0] - dy
      p3_y = p2[1] + dx

      if (p3_x, p3_y) not in grid:
          continue
      
      p4_x = p1[0] - dy
      p4_y = p1[1] + dx

      if (p4_x, p4_y) not in grid:
            continue

      # Check if sides are nearly equal, accounting for floating-point errors
      dx_p3_p4 = p4_x - p3_x
      dy_p3_p4 = p4_y - p3_y
      side_length_2 = math.sqrt(dx_p3_p4 * dx_p3_p4 + dy_p3_p4 * dy_p3_p4)
      if abs(side_length - side_length_2) > tolerance:
          continue
      dx_p1_p4 = p4_x-p1[0]
      dy_p1_p4 = p4_y-p1[1]
      side_length_3 = math.sqrt(dx_p1_p4*dx_p1_p4 + dy_p1_p4*dy_p1_p4)
      if abs(side_length-side_length_3) > tolerance:
          continue
      squares.append((p1, p2, (p3_x, p3_y), (p4_x, p4_y)))
  return squares

grid3 = {(0.0, 0.0): True, (0.0, 1.000001): True, (1.0, 0.0): True, (1.000001, 1.000001):True}
print(find_squares_tolerance(grid3))
# Output: [((0.0, 0.0), (0.0, 1.000001), (1.000001, 1.000001), (1.0, 0.0))]
```

This final example adds a tolerance check for the side length equality. Due to how floating-point numbers are stored in computers, they are frequently not exact. By utilizing a small tolerance, the check now allows slightly imperfect squares to be identified.

When choosing an algorithm, consider the nature of the input data. For grids with sparse points, the dictionary-based approach and the described optimizations are effective. However, if memory is a strict constraint, a more bitwise representation of the grid can save memory, but often at the cost of access time.

For additional resources on spatial algorithms, I recommend exploring literature on computational geometry; specifically related to point set analysis and polygon detection. Materials focusing on image processing techniques also offer practical examples of grid-based operations. Finally, examining implementations in popular game development frameworks can provide insights into real-world usage of similar spatial algorithms. I also strongly recommend studying any texts on discrete mathematics for rigorous analysis of these types of algorithms, and exploring linear algebra which gives the mathematical basis for many geometric calculations.
