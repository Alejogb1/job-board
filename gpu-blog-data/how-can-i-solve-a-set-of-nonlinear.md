---
title: "How can I solve a set of nonlinear equations using fsolve in Python 3?"
date: "2025-01-30"
id: "how-can-i-solve-a-set-of-nonlinear"
---
`scipy.optimize.fsolve` implements a method for numerically approximating the roots of a set of nonlinear equations, a task frequently encountered in modeling and simulation. I've relied on this function extensively when, for example, modeling complex fluid dynamics where analytical solutions were impossible. The key to successful use lies in correctly formulating your system of equations into a form `f(x) = 0` where `x` is a vector of unknowns, and providing `fsolve` with an initial guess for the solution. The function then iteratively refines this guess using a variant of the Powell hybrid method, seeking to minimize the magnitude of the function `f`.

The core idea is that `fsolve` seeks values of `x` that make all the equations in your system equal to zero *simultaneously*. Crucially, since we're dealing with nonlinear systems, there's no guarantee of a unique solution. The method may converge to a local minimum, not a global one, or might even diverge, depending on the initial guess and the specific form of your equations. This means careful consideration is needed in setting up both the equation system and the initial guess.

Here's a breakdown of the necessary steps and some concrete examples based on problems I've encountered:

First, you need to define your system of nonlinear equations as a Python function. This function should accept a single argument, a NumPy array representing the vector of unknowns `x`, and must return a NumPy array that represents the evaluation of each equation in your system. The order of returned values *must* align with the unknowns in your input array `x`.

```python
import numpy as np
from scipy.optimize import fsolve

def equations1(x):
    """
    Defines a system of two nonlinear equations:
    x[0]^2 + x[1]^2 = 5
    x[0] - x[1] = 1
    """
    eq1 = x[0]**2 + x[1]**2 - 5
    eq2 = x[0] - x[1] - 1
    return np.array([eq1, eq2])

# Initial guess:
initial_guess = np.array([1.0, 1.0])

# Solve:
solution = fsolve(equations1, initial_guess)

print("Solution:", solution)
print("Equations evaluated at the solution:", equations1(solution))
```

In this first example, `equations1` encapsulates the system. `x[0]` represents the first unknown, and `x[1]` the second. The returned values are then the evaluation of `x[0]² + x[1]² - 5` and `x[0] - x[1] - 1` at the input `x`. I've chosen `[1.0, 1.0]` as the initial guess, a somewhat arbitrary starting point. `fsolve` returns the solution, which, if you were to evaluate the `equations1` function there, would be very close to `[0, 0]`. The print statement shows this, indicating that the system has converged. This example is a relatively simple case where the solution is straightforward to confirm through substitution. However, real-world problems rarely allow such manual verification.

A critical detail that frequently causes problems is the initial guess, which directly impacts convergence. A poor guess may lead to either convergence to a different root, or complete failure to find a solution. This is where problem-specific knowledge becomes essential.

Consider this second example which was adapted from a problem where I was modeling the equilibrium concentrations of a chemical reaction.

```python
def equations2(x):
    """
    Defines a system of three equations.
    x[0] + x[1] + x[2] = 1
    x[0]*x[1] = 0.2
    x[0]/x[2] = 0.1
    """
    eq1 = x[0] + x[1] + x[2] - 1
    eq2 = x[0] * x[1] - 0.2
    eq3 = x[0] / x[2] - 0.1
    return np.array([eq1, eq2, eq3])

# Initial guess, this time adjusted to the nature of the problem:
initial_guess2 = np.array([0.5, 0.5, 0.5])

solution2 = fsolve(equations2, initial_guess2)

print("Solution:", solution2)
print("Equations evaluated at the solution:", equations2(solution2))
```

Here, the initial guess of `[0.5, 0.5, 0.5]` is, in my experience, more sensible than, say, `[100, 100, 100]` because we are implicitly trying to solve for fractions that add up to one. The same principle applies even if the unknowns are not probabilities.  Note also that the equations are reformulated so that we're always looking for where each expression is zero, not necessarily where they are equal to another value. I've observed that it is much easier to debug when each equation follows this convention.

A problem I commonly see is when the equation system contains division by variables that could be zero. `fsolve` itself does not check for such divisions, and the division by zero will cause a run-time error *outside* the function itself. To demonstrate this, consider a modified system of equations, where we have to be careful about division, and also show how to deal with systems that don't quite converge to zero, since machine precision has limitations.

```python
def equations3(x):
  """
  Defines a system of three equations.
  x[0] + x[1] + x[2] = 1
  x[0]*x[1] = 0.2
  (x[0]+ 0.0001)/(x[2] + 0.0001) = 0.1 # Avoid division by zero
  """
  eq1 = x[0] + x[1] + x[2] - 1
  eq2 = x[0] * x[1] - 0.2
  eq3 = (x[0] + 0.0001) / (x[2] + 0.0001) - 0.1
  return np.array([eq1, eq2, eq3])

initial_guess3 = np.array([0.5, 0.5, 0.5])

solution3 = fsolve(equations3, initial_guess3)

print("Solution:", solution3)
print("Equations evaluated at the solution:", equations3(solution3))

# Check the result:
tolerance = 1e-8
if np.all(np.abs(equations3(solution3)) < tolerance):
    print("Solution is within tolerance.")
else:
    print("Solution did not converge to desired tolerance.")

```

Here, I've added a small constant to each variable that is involved in a division, preventing a division by zero.  Additionally, after calling `fsolve`, I also recommend explicitly checking that the function output `equations3` is within a pre-defined tolerance of zero. The check uses `np.all` to see if *all* the values returned are below our defined `tolerance`.  This is a good practice, especially with complex systems, since the result is still numerically approximated. When I was working on my heat transfer problems, this explicit check was vital for ensuring that the results I obtained were physically meaningful, rather than numerical artifacts.

In summary, using `fsolve` effectively requires:

1.  Careful formulation of your equation system into a function returning the evaluations of `f(x)=0` for each equation.
2.  Provision of a good initial guess for the solution. Domain expertise is necessary to guide this.
3.  Awareness of potential division-by-zero issues within the defined equation system and preventing this.
4.  Explicit checking if the solution has converged to an adequate level of tolerance after running fsolve, especially in sensitive simulations.

For further exploration of numerical root-finding methods, I would recommend exploring textbooks on numerical analysis and optimization.  Additionally, delving into the SciPy documentation for `optimize` and related modules like `root` can be quite useful.  It is often beneficial to consult literature related to Powell's hybrid method, since that is what's used by `fsolve`.  Also investigate the `scipy.optimize.root` function which is a more flexible method with many options, although I find that for my typical problems, `fsolve` is more direct. Lastly, consider taking courses or seminars on numerical methods and computational science, which frequently cover these types of problems and provide useful case studies.
