---
title: "What is the largest square that can be found in an n x n grid?"
date: "2025-01-30"
id: "what-is-the-largest-square-that-can-be"
---
The problem of finding the largest square within an *n x n* grid, where cells may or may not be “filled,” is a dynamic programming classic, often encountered when dealing with image processing or game board analysis. Specifically, we are searching for the square of the greatest side length composed entirely of filled cells (typically represented as 1s) within a matrix of 0s and 1s. The naive approach, involving repeated checks of every possible square size at every cell location, carries a substantial time complexity, rendering it impractical for large grids.

The core insight lies in understanding that the size of the largest square ending at cell *(i, j)* is directly dependent on the sizes of the largest squares ending at cells *(i-1, j)*, *(i, j-1)*, and *(i-1, j-1)*. This overlapping subproblem structure immediately suggests a dynamic programming solution. I've personally implemented this on embedded systems with limited memory where efficiency was paramount. Instead of recomputing the largest square for each cell, I incrementally build a table storing these intermediate results. This memoization strategy drastically reduces the computational overhead.

The solution involves creating a dynamic programming table of the same dimensions as the input grid. Each cell in this table, at index *(i, j)*, will store the size (side length) of the largest square that *ends* at the corresponding cell *(i, j)* in the original input grid. The value at a table cell is determined by considering the following:

1.  **Base Cases:** If the input cell at position *(i, j)* is 0, then the corresponding cell in the DP table at *(i, j)* will also be 0, as no square can end at an unfilled cell. Similarly, for the first row and column, if the original input cell contains a 1, then the corresponding cell in the DP table will be 1, as that single cell constitutes a square of size 1.

2.  **General Case:** If the input cell at position *(i, j)* is 1, then the corresponding cell in the DP table at *(i, j)* is calculated as the minimum of three cells: the largest square ending at the cell above (i-1, j), to the left (i, j-1), and diagonally above-left (i-1, j-1), and then incremented by one. The rationale is that we extend the existing square by one in each direction if possible. Taking the minimum ensures that a square can indeed be expanded without including any ‘0’ cells. The final result, the size of the largest square in the original grid, will be the maximum value in the dynamic programming table.

Below are three implementations to illustrate this approach in different languages:

**Python Implementation:**

```python
def find_largest_square(grid):
    if not grid or not grid[0]:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    dp = [[0 for _ in range(cols)] for _ in range(rows)]
    max_square_size = 0

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                if i == 0 or j == 0:
                   dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_square_size = max(max_square_size, dp[i][j])

    return max_square_size

# Example Usage:
grid = [
    [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0]
]
print(find_largest_square(grid))  # Output: 3
```

This Python implementation initializes a DP table with zeros. It then iterates through the input grid. If a cell in the grid contains a ‘1’, it updates the corresponding cell in the DP table with the minimum value from its neighbors plus one if possible. The base cases, where the cell is in the first row or first column, are handled separately. The function maintains a `max_square_size` variable to track the largest size found so far, returning this value at the end.

**Java Implementation:**

```java
public class LargestSquare {

    public static int findLargestSquare(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int rows = grid.length;
        int cols = grid[0].length;
        int[][] dp = new int[rows][cols];
        int maxSquareSize = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    maxSquareSize = Math.max(maxSquareSize, dp[i][j]);
                }
            }
        }

        return maxSquareSize;
    }


    public static void main(String[] args) {
        int[][] grid = {
            {1, 0, 1, 0, 0},
            {1, 0, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 0, 0, 1, 0}
        };
        System.out.println(findLargestSquare(grid));  // Output: 3
    }
}
```

The Java code mirrors the Python logic. It uses a 2D array `dp` as its dynamic programming table. The nested for-loops implement the same iterative processing, checking cell values in the `grid` and updating the `dp` array accordingly, also updating the `maxSquareSize`. It makes use of `Math.min` and `Math.max` functions to achieve comparisons effectively. The main method provides an example of calling `findLargestSquare` and printing the result.

**C++ Implementation:**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int findLargestSquare(const vector<vector<int>>& grid) {
    if (grid.empty() || grid[0].empty()) {
        return 0;
    }

    int rows = grid.size();
    int cols = grid[0].size();
    vector<vector<int>> dp(rows, vector<int>(cols, 0));
    int maxSquareSize = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == 1) {
                if (i == 0 || j == 0) {
                   dp[i][j] = 1;
                } else {
                    dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                }
               maxSquareSize = max(maxSquareSize, dp[i][j]);
           }
        }
    }
    return maxSquareSize;
}

int main() {
    vector<vector<int>> grid = {
        {1, 0, 1, 0, 0},
        {1, 0, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 1, 0}
    };
    cout << findLargestSquare(grid) << endl; // Output: 3
    return 0;
}
```

This C++ implementation showcases similar logic using vectors to represent the grid and DP table. It leverages the standard `<algorithm>` library's `min` and `max` functions.  Like the other implementations, it iterates through the matrix, filling the `dp` matrix based on the same rules. The example usage in the main function mirrors that of the Python and Java versions.

Regarding resources, I recommend consulting texts on algorithmic design and dynamic programming. Specific chapters covering matrix traversal and memoization within these books will provide a deeper theoretical understanding and potentially offer alternative perspectives on optimization. Additionally, reviewing solutions and discussions in advanced algorithm problem-solving platforms can expose nuances and edge cases that may not be immediately apparent. Finally, exploring tutorials focused specifically on dynamic programming, and especially those that provide step-by-step derivations of common DP approaches will help solidify understanding.
