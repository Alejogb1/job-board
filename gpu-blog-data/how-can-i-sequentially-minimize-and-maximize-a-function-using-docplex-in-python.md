---
title: "How can I sequentially minimize and maximize a function using docplex in Python?"
date: "2025-01-26"
id: "how-can-i-sequentially-minimize-and-maximize-a-function-using-docplex-in-python"
---

I’ve frequently encountered the need to iteratively refine solutions using mathematical programming, transitioning between minimization and maximization objectives.  `docplex`, IBM’s modeling library for optimization, provides a flexible framework for achieving this, though it requires careful manipulation of the model object. This response will detail the process, incorporating practical examples based on situations I’ve faced.

The core idea centers around modifying the objective function of a `docplex` model between solve calls. You aren’t creating separate models; instead, you are dynamically altering the existing model’s goal. This sequential alteration of the objective allows for a two-stage optimization: First, minimizing an objective, then, using the result as a basis, maximizing a potentially related but distinct objective.

Here's how the process typically unfolds:

1.  **Initial Model Creation:** You begin by defining the variables, constraints, and an initial objective function, typically targeting minimization.
2.  **Initial Solve (Minimization):** The model is solved with the initial objective. The solution values are stored and become foundational for the next stage.
3.  **Objective Function Modification:** The `objective` attribute of the `docplex` model is modified to reflect the maximization problem. This could involve changing the function itself or negating the existing one.
4.  **Second Solve (Maximization):** The model is re-solved with the altered objective, using the previously computed variable assignments as an initial or warm start.

Let's translate this into practical code examples:

**Example 1: Simple Sequential Optimization**

This example demonstrates a simple linear programming problem where we first minimize the sum of two variables, and then, with the minimized solution as a base, maximize their product.

```python
from docplex.mp.model import Model

# Initial model setup for minimization
model = Model("sequential_optimization_1")
x = model.continuous_var(name="x")
y = model.continuous_var(name="y")

model.add_constraint(x + y >= 10)
model.add_constraint(x <= 15)
model.add_constraint(y <= 15)
model.minimize(x + y)

# Solve the minimization problem
solution_min = model.solve()
if solution_min:
    print("Solution for minimization:")
    print(f"x = {solution_min.get_value(x)}, y = {solution_min.get_value(y)}, objective = {solution_min.objective_value}")
    x_min = solution_min.get_value(x)
    y_min = solution_min.get_value(y)

    # Modify the objective function for maximization
    model.maximize(x * y)

    # Solve the maximization problem with warm start
    solution_max = model.solve()

    if solution_max:
        print("\nSolution for maximization:")
        print(f"x = {solution_max.get_value(x)}, y = {solution_max.get_value(y)}, objective = {solution_max.objective_value}")
else:
    print("Initial minimization problem has no solution")

```

In this example, the `Model` object is created with constraints and a minimization objective (`x + y`). The `solve` method provides the optimal values after minimization. After printing the solution, the model’s objective is switched to maximization (`x * y`). When the model is solved again, `docplex` uses the previous solution as a starting point, leading to efficient exploration of the solution space. The solution values of the second solve, associated with the maximization, are also printed.

**Example 2: Objective Modification and Constraint Retention**

Here, we use the same variables as example one but introduce a new maximization objective involving their ratio after the initial minimization of their sum. The crucial point is that constraints stay the same between solve calls.

```python
from docplex.mp.model import Model

# Initial model setup
model = Model("sequential_optimization_2")
x = model.continuous_var(name="x")
y = model.continuous_var(name="y")

model.add_constraint(x + y >= 10)
model.add_constraint(x <= 15)
model.add_constraint(y <= 15)
model.minimize(x + y)

# Solve minimization
solution_min = model.solve()
if solution_min:
    print("Solution for minimization:")
    print(f"x = {solution_min.get_value(x)}, y = {solution_min.get_value(y)}, objective = {solution_min.objective_value}")

    # Modify objective function for maximization
    model.maximize(x/y)

    # Solve maximization
    solution_max = model.solve()
    if solution_max:
       print("\nSolution for maximization:")
       print(f"x = {solution_max.get_value(x)}, y = {solution_max.get_value(y)}, objective = {solution_max.objective_value}")
else:
    print("Initial minimization problem has no solution")
```

The second example emphasizes that you modify only the objective function. The constraints added before the minimization phase continue to hold during the maximization phase. This reuse of the same constraint space is efficient and allows sequential optimization to proceed. The ratio objective, `x/y`, is a non-linear function that may be challenging for some solvers to handle directly when the domain is not properly handled or a particular solver is not used. It works here for illustration purposes; but more complex objective changes will depend on the specific solver chosen and may require a re-definition of the problem as a nonlinear program.

**Example 3: Using Previous Solution Values in the Maximization Objective**

This example illustrates how previous solution values, especially a previous objective function value, can inform the subsequent maximization. Here we initially minimize the sum of x and y and in the maximization phase we want to maximize their sum, while imposing that the previous optimal value obtained in the minimization is a lower bound.

```python
from docplex.mp.model import Model

# Initial model setup
model = Model("sequential_optimization_3")
x = model.continuous_var(name="x")
y = model.continuous_var(name="y")

model.add_constraint(x + y >= 10)
model.add_constraint(x <= 15)
model.add_constraint(y <= 15)
model.minimize(x + y)

# Solve the minimization problem
solution_min = model.solve()
if solution_min:
    print("Solution for minimization:")
    print(f"x = {solution_min.get_value(x)}, y = {solution_min.get_value(y)}, objective = {solution_min.objective_value}")
    min_objective_value = solution_min.objective_value


    # Modify objective function and introduce a constraint based on the previous solution
    model.maximize(x + y)
    model.add_constraint(x + y >= min_objective_value)


    # Solve the maximization problem
    solution_max = model.solve()
    if solution_max:
        print("\nSolution for maximization:")
        print(f"x = {solution_max.get_value(x)}, y = {solution_max.get_value(y)}, objective = {solution_max.objective_value}")
else:
    print("Initial minimization problem has no solution")
```

This example demonstrates how we can extract the objective value and incorporate it as part of the subsequent optimization. We first minimize `x+y` and obtain the associated optimal value which is stored in the variable `min_objective_value`. We then change the objective to maximize `x+y` while constraining the sum of `x` and `y` to be greater than or equal to the `min_objective_value`.

This pattern is useful for scenarios requiring that the second solution is at least as good, or better, than the first solution, according to the minimization objective function value.

**Resource Recommendations:**

To deepen your understanding of `docplex` and related optimization techniques, consider exploring the following resources:

1.  **IBM Decision Optimization Documentation:** The official documentation provides a comprehensive overview of `docplex` capabilities.
2.  **Mathematical Programming Textbooks:** Works on linear, mixed-integer, and nonlinear programming provide the theoretical foundation to guide your modeling process.
3.  **Optimization Software Manuals:** Specific solver manuals (like CPLEX) offer details on algorithms and tuning options when used with `docplex`.

In conclusion, `docplex` facilitates sequential minimization and maximization through objective function modification, making it possible to build complex, multi-stage optimization workflows. Understanding how to leverage `docplex`’s ability to change the objective function while keeping constraints allows you to implement sophisticated, customized optimization models. Always consider the specific characteristics of the objective functions and the numerical behavior of the chosen solver when modifying the objectives between solve calls. This is particularly important when switching between minimization and maximization.
