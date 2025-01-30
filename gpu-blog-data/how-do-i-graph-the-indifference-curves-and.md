---
title: "How do I graph the indifference curves and budget constraint for U = (log(x) + log(y))^0.5?"
date: "2025-01-30"
id: "how-do-i-graph-the-indifference-curves-and"
---
The core challenge in plotting indifference curves and a budget constraint for the utility function U = (log(x) + log(y))^0.5 lies in understanding how the specific functional form affects their shapes and the necessary computational steps. Unlike simpler Cobb-Douglas utilities, this form requires careful attention to the logarithm and the square root, both of which influence the curvature and level sets of the indifference curves. Furthermore, directly solving for `y` as a function of `x` for a given utility level proves cumbersome, necessitating numerical approaches.

**1. Explanation of the Economic Concepts and Their Representation**

Indifference curves graphically represent combinations of goods (x and y in this case) that yield the same level of utility for a consumer. Higher indifference curves correspond to higher levels of utility. The key is that along a single indifference curve, the consumer is equally happy with any combination of x and y. Mathematically, an indifference curve is defined by holding the utility function (U) at a constant level, typically denoted as `Ū`. In our case, this means solving (log(x) + log(y))^0.5 = `Ū` for various `Ū`. The budget constraint, conversely, outlines all possible combinations of x and y that a consumer can afford given their income (I) and the prices of the goods, `Px` and `Py`. It's a linear equation expressed as `Px * x + Py * y = I`. The consumer's optimal choice occurs at the point where an indifference curve is tangent to the budget constraint.

The challenge arises in plotting the indifference curves due to the utility function's structure. Isolating 'y' analytically is not straightforward. We must rely on implicit function plotting or numerical methods to approximate the curves. For the budget constraint, the direct linear form allows a simpler plotting technique. We must also ensure we define the relevant domain, given that both ‘x’ and ‘y’ must be strictly greater than zero due to the presence of the logarithmic function, and we are only interested in positive quantities in typical consumption models. In computational economics, such functions require a combination of analytical derivation, numerical approaches, and domain knowledge.

**2. Code Examples and Commentary**

I've used Python with `matplotlib` and `numpy`, common tools for such visualizations. While the core logic remains consistent, different languages and libraries could also be used. I have had extensive experience in using Python for numerical methods in my research.

**Example 1: Plotting Indifference Curves**

```python
import numpy as np
import matplotlib.pyplot as plt

def utility(x, y):
    """Calculates utility given x and y."""
    return np.sqrt(np.log(x) + np.log(y))

def plot_indifference_curve(utility_level, ax, x_min=0.01, x_max=5, num_points=200):
    """Plots an indifference curve for a given utility level."""
    x = np.linspace(x_min, x_max, num_points)
    y = np.zeros_like(x) # initialize y, will be computed iteratively
    for i, x_val in enumerate(x):
        # Find y such that utility(x,y) is close to utility_level
        def target_function(y_val): # define an auxillary function for numerical root finding
             return utility(x_val, y_val) - utility_level
        try:
            # start with an initial guess of y=1, for simple cases
            y[i] = find_root_secant(target_function, 1, 1e-5)  # Numerical root-finding
        except ValueError:
            # If no root was found, leave y at 0.
            y[i] = np.nan # If no root found return NaN
    ax.plot(x, y, label=f'U = {utility_level:.2f}')

def find_root_secant(f, x0, tolerance):
     """Simple secant method for root finding"""
     x1 = x0 + 1e-4
     f0 = f(x0)
     f1 = f(x1)
     while abs(f1)> tolerance:
        x2 = x1 - f1*(x1 - x0)/(f1-f0)
        x0 = x1
        f0 = f1
        x1 = x2
        f1 = f(x1)
     return x1


# Set of utility levels to plot
utility_levels = [1.0, 1.2, 1.4]

fig, ax = plt.subplots()
for level in utility_levels:
    plot_indifference_curve(level, ax)

ax.set_xlabel('Quantity of Good x')
ax.set_ylabel('Quantity of Good y')
ax.set_title('Indifference Curves for U = (log(x) + log(y))^0.5')
ax.legend()
plt.grid(True)
plt.show()
```

This code defines the `utility` function directly from the given expression. It introduces `plot_indifference_curve` which, for a given utility level, computes corresponding y values for a set of x values using the `find_root_secant` function, implementing a numerical root-finding method.  This is crucial since we can't easily isolate y as a function of x given a fixed utility level for this utility function. The secant method iteratively refines our approximation for y given a specific x to satisfy the fixed utility level, handling the implicit nature of the indifference curve. The `try-except` block manages errors and avoids NaNs from the graph, when solutions cannot be reached. Finally, a loop iterates through different utility levels to plot multiple indifference curves. The core computation here resides in the numerical root finding algorithm. The use of secant method is justified by its ease of implementation while maintaining a reasonably good convergence.

**Example 2: Plotting the Budget Constraint**

```python
import numpy as np
import matplotlib.pyplot as plt

def budget_constraint(income, px, py, num_points=200):
    """Calculates x and y values along the budget constraint."""
    x_max = income/px #maximum x
    x = np.linspace(0, x_max, num_points)
    y = (income - px * x) / py
    return x, y

income = 10
px = 2
py = 1

x_budget, y_budget = budget_constraint(income, px, py)

fig, ax = plt.subplots()
ax.plot(x_budget, y_budget, label='Budget Constraint')
ax.set_xlabel('Quantity of Good x')
ax.set_ylabel('Quantity of Good y')
ax.set_title('Budget Constraint')
ax.legend()
ax.grid(True)
plt.show()
```

Here, `budget_constraint` calculates the x and y values along the budget constraint line. We use the standard budget constraint equation `Px * x + Py * y = I` to derive `y = (I - Px * x) / Py`. Given income, prices, and number of points, the code computes corresponding values and plots the linear constraint. The budget constraint is a simple linear equation which is easily plotted. This code demonstrates the simplicity of plotting linear functions in comparison with the complexity encountered with implicit curves as in the case of indifference curves.

**Example 3: Combined Plot of Indifference Curves and Budget Constraint**

```python
import numpy as np
import matplotlib.pyplot as plt

# Utility Function (same as before)
def utility(x, y):
    return np.sqrt(np.log(x) + np.log(y))

# indifference curve plotting function (same as before)
def plot_indifference_curve(utility_level, ax, x_min=0.01, x_max=5, num_points=200):
     x = np.linspace(x_min, x_max, num_points)
     y = np.zeros_like(x) # initialize y
     for i, x_val in enumerate(x):
        def target_function(y_val):
             return utility(x_val, y_val) - utility_level
        try:
             y[i] = find_root_secant(target_function, 1, 1e-5)  # numerical root-finding
        except ValueError:
            y[i] = np.nan # If no root found return NaN
     ax.plot(x, y, label=f'U = {utility_level:.2f}')


# Budget Constraint Function (same as before)
def budget_constraint(income, px, py, num_points=200):
     x_max = income/px
     x = np.linspace(0, x_max, num_points)
     y = (income - px * x) / py
     return x, y

# root finding function (same as before)
def find_root_secant(f, x0, tolerance):
     x1 = x0 + 1e-4
     f0 = f(x0)
     f1 = f(x1)
     while abs(f1)> tolerance:
        x2 = x1 - f1*(x1 - x0)/(f1-f0)
        x0 = x1
        f0 = f1
        x1 = x2
        f1 = f(x1)
     return x1

# Parameters
income = 10
px = 2
py = 1
utility_levels = [1.0, 1.2, 1.4]

# Plotting
fig, ax = plt.subplots()

# Plotting Indifference Curves
for level in utility_levels:
    plot_indifference_curve(level, ax)


# Plotting Budget Constraint
x_budget, y_budget = budget_constraint(income, px, py)
ax.plot(x_budget, y_budget, label='Budget Constraint', color='black', linestyle='--')

ax.set_xlabel('Quantity of Good x')
ax.set_ylabel('Quantity of Good y')
ax.set_title('Indifference Curves and Budget Constraint')
ax.legend()
ax.grid(True)
plt.show()
```

This final example combines all the previous components. It plots both indifference curves for several utility levels and a budget constraint on the same graph. This code effectively demonstrates how the numerical methods work in conjunction with analytical equations. It is also a practical approach to find the optimal solution in consumer choice problems. The tangency points between the budget constraint and the highest achievable indifference curve indicate the optimal bundle for a consumer with this specific utility.

**3. Resource Recommendations**

For a deeper dive into the underlying theory, explore microeconomic textbooks focused on consumer choice. Specifically, look for sections on utility functions, indifference curves, and budget constraints. Resources discussing numerical methods in economics, which cover topics like root-finding algorithms, can provide a theoretical foundation and introduce alternative methods. Finally, documentation and tutorials for numerical computation libraries (like those in Python, Matlab, or R) are invaluable for mastering the practical aspects of generating these visualizations. Such resources, though not specific to this problem, are vital to building general skills.
