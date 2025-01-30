---
title: "How can GEKKO be used to count occurrences in a 2D array based on a condition?"
date: "2025-01-30"
id: "how-can-gekko-be-used-to-count-occurrences"
---
The effective counting of elements within a 2D array, conditional on specific criteria, is a frequent task in diverse numerical and symbolic computation scenarios. GEKKO, while primarily a tool for optimization and differential equations, possesses sufficient flexibility via its symbolic variables and constraint-handling capabilities to achieve this objective, albeit with a slightly unconventional approach compared to methods relying on direct looping. My experience in developing simulation and analysis tools within a manufacturing environment has often required this exact operation – analyzing sensor readings across spatial grids, for example. Direct iteration is frequently computationally expensive when coupled with additional optimization; thus, a constraint-based approach can be more appropriate within the GEKKO ecosystem.

GEKKO excels at manipulating symbolic variables, constructing algebraic relationships between them, and solving for those variables that satisfy a given set of constraints. The core idea to count conditional occurrences within a 2D array in GEKKO is to map the array into a set of binary variables, each associated with a single element. These binary variables will act as indicators, switching to one if the condition is met, and staying zero otherwise. Subsequently, the sum of these binary indicator variables provides the count, which can be accessed via GEKKO’s model solving. The conditional test is enforced through mathematical constraints involving the array element’s value and the intended condition.

To demonstrate, consider a 2D NumPy array as input. We can use the `gekko.Var` class to create symbolic binary variables, equal to the dimensions of the array. Consider the array *A* below, representing, for illustrative purposes, temperature readings on a grid:

```python
import numpy as np
from gekko import GEKKO

A = np.array([[25, 30, 35],
              [28, 32, 29],
              [33, 26, 31]])

rows, cols = A.shape

m = GEKKO(remote = False)

count_vars = [[m.Var(lb=0, ub=1, integer = True) for _ in range(cols)] for _ in range(rows)]

count_sum = m.Var(lb=0, integer=True)

m.Equation(count_sum == m.sum([m.sum(row) for row in count_vars]))

target_temp = 30

for i in range(rows):
  for j in range(cols):
    m.Equation((A[i,j] <= target_temp) * count_vars[i][j] == 0)
    m.Equation((A[i,j] > target_temp) * (1-count_vars[i][j]) == 0)


m.options.SOLVER = 1
m.solve(disp=False)

print(f"The count of elements above {target_temp} is: {count_sum.value[0]}")
```

Here, the code initializes `GEKKO`, defines the array *A*, and creates a matrix of binary `GEKKO.Var` objects named `count_vars` mirroring *A*. The variable `count_sum` is initialized to act as the accumulator. Importantly, the code iterates through each element of *A*, adding two critical constraints. The first, `(A[i,j] <= target_temp) * count_vars[i][j] == 0`, forces the corresponding `count_vars` element to be zero if the value at *A*[i,j] is not greater than *target_temp*. The second, `(A[i,j] > target_temp) * (1-count_vars[i][j]) == 0` forces `count_vars` to one otherwise. The summation of all variables in `count_vars` is enforced to equal `count_sum`. Finally, the optimization process yields the final count, stored in `count_sum.value[0]`. Note that setting SOLVER to 1 activates the integer-capable optimizer, allowing correct manipulation of the binary variables.

The preceding example demonstrated counting occurrences *above* a particular value. We can adapt the code for different conditions. The following example modifies the code to count how many elements in a grid fall *within* a particular range:

```python
import numpy as np
from gekko import GEKKO

A = np.array([[25, 30, 35],
              [28, 32, 29],
              [33, 26, 31]])

rows, cols = A.shape

m = GEKKO(remote = False)

count_vars = [[m.Var(lb=0, ub=1, integer = True) for _ in range(cols)] for _ in range(rows)]

count_sum = m.Var(lb=0, integer=True)

m.Equation(count_sum == m.sum([m.sum(row) for row in count_vars]))

lower_bound = 28
upper_bound = 32

for i in range(rows):
    for j in range(cols):
        m.Equation((A[i,j] < lower_bound) * count_vars[i][j] == 0)
        m.Equation((A[i,j] > upper_bound) * count_vars[i][j] == 0)
        m.Equation((A[i,j] >= lower_bound) * (A[i,j] <= upper_bound) * (1-count_vars[i][j]) == 0)

m.options.SOLVER = 1
m.solve(disp=False)

print(f"The count of elements between {lower_bound} and {upper_bound} is: {count_sum.value[0]}")

```
Here, we now set an upper and lower bound. Three constraints are now imposed. The first, `(A[i,j] < lower_bound) * count_vars[i][j] == 0` ensures that count_vars equals zero if A[i,j] is lower than the bound. The second, `(A[i,j] > upper_bound) * count_vars[i][j] == 0` ensures that count_vars equals zero if A[i,j] is higher than the bound. The third constraint, `(A[i,j] >= lower_bound) * (A[i,j] <= upper_bound) * (1-count_vars[i][j]) == 0`, ensures count_vars equals one if the element is within the bound. These constraints collectively enforce the desired behavior – `count_vars` is one if the element is in the given range, otherwise zero. Again, the final count is found in `count_sum.value[0]`.

The versatility of GEKKO’s constraint system allows for arbitrarily complex conditions to be applied. Consider an example that counts only even numbers within the array:

```python
import numpy as np
from gekko import GEKKO

A = np.array([[25, 30, 35],
              [28, 32, 29],
              [33, 26, 31]])

rows, cols = A.shape

m = GEKKO(remote = False)

count_vars = [[m.Var(lb=0, ub=1, integer = True) for _ in range(cols)] for _ in range(rows)]

count_sum = m.Var(lb=0, integer=True)

m.Equation(count_sum == m.sum([m.sum(row) for row in count_vars]))


for i in range(rows):
  for j in range(cols):
    m.Equation((A[i,j] % 2 != 0) * count_vars[i][j] == 0)
    m.Equation((A[i,j] % 2 == 0) * (1-count_vars[i][j]) == 0)

m.options.SOLVER = 1
m.solve(disp=False)

print(f"The count of even numbers is: {count_sum.value[0]}")
```
This example checks for even numbers using the modulus operator. If `A[i,j] % 2 != 0` (meaning the number is odd), `count_vars` is forced to zero. If `A[i,j] % 2 == 0` (even), `count_vars` is forced to one. This illustrates that arithmetic operations can be directly included within GEKKO’s equations to define intricate conditions.

While these examples are relatively simple, they highlight the methodology. This approach is particularly advantageous within larger systems, allowing the conditional counts to be directly incorporated as part of a larger constrained optimization or simulation model within the GEKKO framework. The explicit construction of binary indicator variables allows for simultaneous optimization of the count variable, alongside any other related continuous or integer variables, thus removing the computational overhead of iterative counting as a preliminary operation.

For further exploration of GEKKO’s capabilities, I recommend consulting the official GEKKO documentation. Resources covering linear and nonlinear programming, mixed-integer programming, and dynamic optimization problems will offer more insight into the foundations of the tool, and reveal further practical applications of this strategy.  Furthermore, materials covering symbolic computation and constraint satisfaction programming would be useful for gaining a deeper appreciation of the techniques used in these examples. The ability to manipulate equations to satisfy conditions and count instances simultaneously can be a very valuable aspect of numerical modeling.
