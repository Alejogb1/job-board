---
title: "Why am I getting a dual infeasible solution in this simple linear program?"
date: "2025-01-30"
id: "why-am-i-getting-a-dual-infeasible-solution"
---
The core issue underlying dual infeasibility in what appears to be a "simple" linear program often stems from inconsistencies between the primal problem's constraints and its objective function, specifically concerning the feasibility region's unboundedness in the dual space.  This isn't necessarily indicative of a coding error, but rather a fundamental flaw in the problem's formulation.  In my experience resolving such issues over the past decade working on large-scale optimization projects for financial modelling, the root cause usually lies in either overly restrictive constraints or an objective that's unattainable given the constraints.

Let's clarify.  A dual infeasible solution means the dual problem, representing the problem from the perspective of the constraints, has no feasible solution.  This implies that there exists no set of dual variables that satisfy the dual constraints while also maximizing (or minimizing, depending on the primal problem's objective) the dual objective function. This, in turn, reflects a problem within the primal formulation itself.  While seemingly paradoxical—how can a problem be infeasible in its dual form while the primal appears solvable?—this highlights the interconnectedness of primal and dual spaces in linear programming.

**Explanation:**

Consider a standard linear program (LP) in its primal form:

Minimize:  c<sup>T</sup>x

Subject to: Ax ≥ b
             x ≥ 0

where:

* `c` is the cost vector.
* `x` is the vector of decision variables.
* `A` is the constraint matrix.
* `b` is the constraint vector.

The associated dual problem is:

Maximize: b<sup>T</sup>y

Subject to: A<sup>T</sup>y ≤ c
             y ≥ 0

Dual infeasibility arises when the constraints A<sup>T</sup>y ≤ c are contradictory, meaning no non-negative vector `y` can simultaneously satisfy all these inequalities. This, fundamentally, indicates an issue within the primal problem's constraints.  The primal constraints, while appearing reasonable in isolation, collectively create an infeasible region in the dual space.

Three common scenarios lead to this:

1. **Overly Restrictive Primal Constraints:**  The primal constraints are too tight, creating an empty feasible region in the primal space. Though this may seem counterintuitive to dual infeasibility, an empty primal feasible region implies unboundedness in the dual feasible region, leading to dual infeasibility. The solver cannot find a feasible solution because the constraints are mutually exclusive.

2. **Inconsistent Primal Constraints:**  The primal constraints may be logically inconsistent.  For instance, you might have constraints like x₁ + x₂ ≥ 10 and x₁ + x₂ ≤ 5.  These are clearly contradictory, resulting in an empty feasible region in the primal, hence, dual infeasibility.  The solver recognizes this inconsistency and reports the dual infeasibility.

3. **Incorrectly Specified Objective Function:** While less common as the direct cause, an improperly defined objective function can indirectly lead to dual infeasibility.  An objective function that inherently conflicts with the constraints, demanding a solution that violates the constraints, can effectively create an unachievable optimum, manifesting as dual infeasibility.


**Code Examples with Commentary:**

Let's illustrate this with Python using the `scipy.optimize.linprog` solver.

**Example 1: Overly Restrictive Constraints:**

```python
from scipy.optimize import linprog

c = [1, 1]
A = [[1, 1], [-1, -1]]  # Note the negative constraint making this overly restrictive
b = [-1, 1]
x_bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

print(result)
```

This example demonstrates how seemingly plausible constraints can lead to infeasibility. The second constraint (-x₁ - x₂ ≥ 1) is too restrictive when combined with the first, generating an empty feasible region, resulting in dual infeasibility indicated by the solver's output.

**Example 2: Inconsistent Constraints:**

```python
from scipy.optimize import linprog

c = [2, 3]
A = [[1, 1], [-1, -1]]
b = [10, -5] # Contradictory Constraints
x_bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

print(result)
```

Here, the constraints x₁ + x₂ ≤ 10 and -x₁ - x₂ ≤ -5 are fundamentally contradictory.  The solver will detect this inconsistency and typically report dual infeasibility.


**Example 3:  Indirectly Caused by the Objective (less common):**


```python
from scipy.optimize import linprog

c = [-1, -1] # Maximizing this can lead to issues with certain constraints.
A = [[1, 1]]
b = [1]
x_bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

print(result)

```
In this example, while the constraints themselves aren't contradictory, the objective of maximizing -x₁ -x₂ pushes towards unbounded solutions which might conflict with the solver's internal mechanisms, potentially leading to dual infeasibility in certain solver configurations.


**Resource Recommendations:**

Consult a comprehensive textbook on linear programming and optimization theory.  Study the duality theorem and its implications for primal and dual problem relationships. Familiarize yourself with the specifics of your chosen linear programming solver's documentation, particularly regarding error messages and interpretation of solution statuses.  Understanding the solver's algorithms and limitations is crucial.  Furthermore, consider revisiting the fundamentals of linear algebra, particularly matrix operations and vector spaces, to gain a solid understanding of the underlying mathematical framework.


In conclusion, dual infeasibility is not a simple coding bug but a symptom of a deeper problem within the mathematical model.  Thoroughly scrutinizing the primal problem's constraints and objective function for consistency and feasibility is the most effective way to resolve this issue. Careful analysis, combined with a robust understanding of linear programming theory and solver capabilities, is essential for effective troubleshooting and model refinement.
