---
title: "How to avoid TLE when solving the Number of Enclaves LeetCode problem?"
date: "2025-01-30"
id: "how-to-avoid-tle-when-solving-the-number"
---
The most significant performance bottleneck encountered while solving LeetCode's "Number of Enclaves" problem often arises from inefficient traversal of the grid. A naive recursive or iterative depth-first search (DFS) without proper optimization can easily result in a Time Limit Exceeded (TLE) error, especially for larger grids. The problem requires counting the number of land cells (value 1) that are not connected to the boundary of the grid, essentially identifying 'enclaves'. The core challenge lies in the fact that a standard DFS or BFS, if not implemented carefully, might repeatedly traverse already visited cells, consuming valuable CPU time.

I've personally encountered this TLE issue multiple times during coding challenges and have found that a focused strategy involving in-place modification of the input grid and strategic exploration is critical for achieving optimal runtime. The key optimization stems from a shift in perspective: instead of searching for enclaves, the approach focuses on marking land cells connected to the boundary. By doing this, all remaining land cells are implicitly identified as belonging to enclaves, eliminating the need for a second full traversal.

The fundamental principle is that any land cell connected to the grid's boundary (top, bottom, left, or right edges) is not part of an enclave. Consequently, I start by iterating over the grid's border cells. When encountering a land cell (grid[i][j] == 1), I initiate a depth-first search to systematically mark all connected land cells as visited. This process, achieved by modifying the original grid, effectively 'drains' land connected to the boundary. I've observed this strategy consistently cut down on runtime as it directly addresses the problem of redundant searches by limiting traversal only to boundary-connected areas. After this initial phase, a simple traversal through the grid can count the remaining land cells – these represent the enclave cells and comprise the final solution.

Let's illustrate this with three different code examples implemented in Python, highlighting the approach, variations, and the underlying optimization:

**Example 1: Recursive DFS with In-Place Modification**

```python
def numEnclaves(grid):
    rows, cols = len(grid), len(grid[0])

    def dfs(row, col):
        if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] != 1:
            return
        grid[row][col] = 0  # Mark visited land
        dfs(row + 1, col)
        dfs(row - 1, col)
        dfs(row, col + 1)
        dfs(row, col - 1)

    # Process boundary cells
    for row in range(rows):
      for col in range(cols):
        if (row == 0 or row == rows - 1 or col == 0 or col == cols-1) and grid[row][col] == 1:
           dfs(row,col)
    
    enclaves = 0
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                enclaves += 1
    return enclaves
```

This code initiates the depth-first search `dfs` from each boundary land cell. The crucial part is the `grid[row][col] = 0` line, which directly modifies the grid in-place. This effectively marks the explored land, preventing revisits during subsequent DFS calls or when traversing to calculate enclaves. This approach, although straightforward, can trigger Python's recursion limit on exceedingly large input grids. The in-place grid modification is the core optimization, eliminating redundant exploration, and directly addresses the TLE concern.

**Example 2: Iterative DFS with Stack (Avoids recursion limits)**

```python
def numEnclaves(grid):
    rows, cols = len(grid), len(grid[0])

    def iterative_dfs(row, col):
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
                continue
            grid[r][c] = 0
            stack.append((r + 1, c))
            stack.append((r - 1, c))
            stack.append((r, c + 1))
            stack.append((r, c - 1))


    # Process boundary cells
    for row in range(rows):
      for col in range(cols):
         if (row == 0 or row == rows - 1 or col == 0 or col == cols-1) and grid[row][col] == 1:
           iterative_dfs(row,col)

    enclaves = 0
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                enclaves += 1
    return enclaves
```

This example presents a transformation of the recursive DFS into its iterative counterpart by using a stack. The logic remains the same: iterate through the border, and when you find a land cell, you use a stack to perform a DFS that marks all connected lands as visited. The main advantage here is that the stack-based DFS bypasses Python's recursion limit, making it suitable for processing very large grid inputs and thereby improving robustness against TLE. While both recursive and stack based DFS achieve similar runtimes for reasonable inputs, for very large inputs, the iterative approach offers more stability.

**Example 3: Boundary-First Exploration (Slight Variation)**

```python
def numEnclaves(grid):
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        dfs(grid,r,0)
        dfs(grid,r, cols-1)

    for c in range (cols):
        dfs(grid,0,c)
        dfs(grid,rows -1, c)

    def dfs(grid, row, col):
      if 0<= row < rows and 0<= col <cols and grid[row][col] ==1:
          grid[row][col] = 0
          dfs(grid,row+1,col)
          dfs(grid,row-1,col)
          dfs(grid,row,col+1)
          dfs(grid,row,col-1)

    enclaves = 0
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                enclaves += 1
    return enclaves
```

This code variation refactors how the boundary cells are addressed. Instead of checking boundary condition during grid iteration, it performs the exploration directly from boundary cells using for loops. This can be more concise, and it highlights an alternative way to ensure all boundary land cells get their exploration initiated first. Functionally, this method achieves the same end result as the first example, with an equivalent time complexity. It just organizes the loop structure to specifically target the grid edges before attempting the full scan. It's also worth noting the use of a helper method which encapsulates the depth-first search logic.

These examples, while seemingly simple, address the core cause of TLE by optimizing grid traversal. The key takeaway is that in-place modification and systematic border exploration are indispensable for efficiently solving this problem.

For continued learning and improvement in similar challenges, I recommend consulting algorithmic textbooks focused on graph traversal techniques and practicing on coding platforms that provide specific performance feedback. Resources that outline different ways to implement graph traversal such as BFS (breadth-first search) and understanding the difference between recursion versus iteration can also be beneficial. Additionally, exploring resources focused on time complexity analysis is critical to understanding how code optimization affects performance especially for larger input sizes. Understanding different algorithmic complexities, such as O(M*N) versus O(M+N), will enable more effective optimization strategies and aid in avoiding TLE errors in general.
