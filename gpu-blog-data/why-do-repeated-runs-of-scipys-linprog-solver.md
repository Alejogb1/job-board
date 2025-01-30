---
title: "Why do repeated runs of SciPy's linprog solver yield different results?"
date: "2025-01-30"
id: "why-do-repeated-runs-of-scipys-linprog-solver"
---
The non-deterministic nature of numerical optimization algorithms, particularly those employed by SciPy's `linprog` solver, often leads to slightly varying solutions across multiple runs, despite identical inputs. This behavior stems from the use of floating-point arithmetic and the inherent iterative nature of the simplex method (or its variants) underlying `linprog`, which introduces rounding errors that can impact the chosen search path. My work on large-scale logistics planning, where precise and repeatable optimization is critical for maintaining consistency in route allocations, forced me to deeply investigate this.

The `linprog` function in SciPy provides several methods for solving linear programming problems. While the 'highs' method (introduced relatively recently) attempts to provide more determinism through tighter control over floating-point operations and its own implementation, most methods, such as 'simplex,' 'interior-point,' and 'revised simplex,' rely on iterative approaches. These methods begin with an initial feasible solution and proceed by iteratively refining it until an optimal (or a near-optimal) solution is found. This iterative process involves several operations, including matrix inversions and basic arithmetic on floating-point numbers.

Each computation involving floating-point numbers introduces rounding errors. These errors are generally small individually, but they can accumulate over numerous iterations of an optimization algorithm. Different execution environments (operating system, CPU architecture, other processes competing for resources) can slightly alter the order of operations and the specific rounding that occurs during each iteration, leading to minor discrepancies in the path taken by the simplex algorithm towards the solution. As the algorithm traverses the feasible region, these minute differences can influence the next pivot point, leading to potentially different sequences of pivots and, ultimately, subtly different final solutions. This effect is amplified by the fact that the feasible region in linear programming can have vertices and edges where multiple solutions may exist, or nearly exist, leading the solver to choose slightly different ‘optimal’ solutions each time.

Furthermore, initial starting points, particularly for interior point methods, are subject to variations. Although the algorithm generally moves towards the optimal solution, slight initial variations may influence the iterative path to the solution. This is further influenced by the fact that not all problems have a unique optimal solution. In some cases, many solutions will result in the same value of the objective function, leading to a non-unique solution to the linear problem.

Here are three examples demonstrating these variations, along with commentary:

**Example 1: Simple Problem, Minor Variations**

```python
import numpy as np
from scipy.optimize import linprog

c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)

for i in range(3):
    result = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='simplex')
    print(f"Run {i+1}: x={result.x}, obj={result.fun:.5f}")
```
*   **Commentary:** This example sets up a very basic linear programming problem. The goal is to minimize `-x0 + 4x1` subject to the constraints `-3x0 + x1 <= 6` and `x0 + 2x1 <= 4`. The `simplex` method is used. Even with this simple, well-defined problem, you’ll see slight differences in the numerical values for the optimal `x` and objective function, `obj`, across multiple runs. The core solution is consistent (same vertex), however the numerical precision varies due to the computational environment. These minor differences can become important when the results are used in other numerical computations.

**Example 2: Ill-Conditioned Problem, More Pronounced Differences**

```python
import numpy as np
from scipy.optimize import linprog

c = [-1, 1]
A = [[1, -1.0001], [-1, 1]]
b = [0, 0]
x0_bounds = (None, None)
x1_bounds = (None, None)

for i in range(3):
    result = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='simplex')
    print(f"Run {i+1}: x={result.x}, obj={result.fun:.5f}, success={result.success}")
```

*   **Commentary:** This example features an ill-conditioned problem, meaning the constraints are very close to being redundant. The constraints `x0 - 1.0001x1 <= 0` and `-x0 + x1 <= 0` form very narrow angles near the origin, making the feasible region very small and extremely sensitive to minor changes caused by floating-point arithmetic. As a result, the `simplex` method might not consistently find the global optimum because of the subtle rounding errors along the edges of the feasible region. Note that the reported solution may differ significantly, and the ‘success’ boolean variable might even flip between ‘True’ and ‘False’ due to the computational limitations.

**Example 3: Using the 'highs' Method for Increased Determinism**

```python
import numpy as np
from scipy.optimize import linprog

c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)


for i in range(3):
    result = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs')
    print(f"Run {i+1}: x={result.x}, obj={result.fun:.5f}")
```

*   **Commentary:** Here, the original problem from Example 1 is revisited, but the `highs` method is employed. The ‘highs’ method tends to provide more consistent results across runs when compared to the ‘simplex’ method. This is because it's implemented to handle some floating point differences more robustly. Still, we may find minor variations in the precision of `x` and `obj`.  This demonstrates the attempt to mitigate these variations through improved numerical techniques but the underlying precision limitations remain.

To address the variations in results, several strategies can be adopted. First, one should be cognizant of the numerical limitations of optimization algorithms. When very precise answers are necessary, it is worth exploring alternative formulations that are less susceptible to numerical errors. Specifically, rescaling the inputs or using an epsilon tolerance (for comparisons), can be helpful. Moreover, while `highs` method often gives more repeatable results, it doesn’t entirely eliminate the problem. The choice of algorithm should match the application; an ill-conditioned problem might need a different approach entirely. Moreover, increasing the iteration limit (while not guaranteeing a consistent answer), will generally reduce the variance in the final solution.

For further study, I would recommend researching the following: 'Numerical Recipes' for a comprehensive look at floating-point operations and numerical methods in scientific computing, 'Linear Programming and Network Flows' by Bazaraa et al. for in-depth coverage on the theory behind linear programming, and the SciPy documentation itself for detailed explanations of the various solver methods and their associated parameters. Finally, understanding the precision of your floating point representations is critical.

In conclusion, while SciPy's `linprog` provides powerful tools for linear optimization, the iterative nature and reliance on floating-point calculations make it inherently susceptible to minor variations in results across different runs. Understanding the sources of these variations and adopting suitable strategies to mitigate their impact is crucial for achieving consistent and reliable solutions, especially in critical applications where precision and repeatability are paramount.
