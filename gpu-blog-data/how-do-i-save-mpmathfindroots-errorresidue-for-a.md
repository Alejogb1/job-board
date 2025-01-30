---
title: "How do I save mpmath.findroot's error/residue for a specific initial guess?"
date: "2025-01-30"
id: "how-do-i-save-mpmathfindroots-errorresidue-for-a"
---
The `mpmath.findroot` function, while robust for finding roots of nonlinear equations, does not directly return the error or residue corresponding to the initial guess provided. Instead, it focuses on the final root found and the related metrics. To obtain the error associated with the *initial* guess, one needs to calculate the function's value at that point and use that as a proxy for the initial error, since `findroot` does not evaluate it. Here's a breakdown of why and how, based on my experience building numerical solvers for optical systems.

The challenge arises from `mpmath.findroot`'s iterative nature. It takes an initial guess, but its internal mechanisms, typically employing variants of Newton's method or bisection, modify this guess step-by-step. Only the final root is exposed. The initial guess is used internally but not saved for direct reporting. Therefore, directly extracting the associated error or residue is not a functionality natively built into `findroot`.

My practice has been to handle this by explicitly calculating the function value *before* calling `findroot`. This allows for a manual evaluation of the error at the initial guess. This approach requires one to possess knowledge of the function for which a root is sought. It also assumes the residue/error is the function's magnitude or a variant thereof that represents the distance from the actual zero.

Let’s illustrate this with concrete examples. Suppose we are trying to find a root of `f(x) = x^2 - 2`.

**Example 1: Simple Root Finding and Initial Error Calculation**

```python
from mpmath import mp, findroot

mp.dps = 50 # Increase precision

def f(x):
    return x**2 - 2

initial_guess = mp.mpf(1)

# Calculate the residue at the initial guess *before* calling findroot
initial_residue = abs(f(initial_guess))

root = findroot(f, initial_guess)

print(f"Initial guess: {initial_guess}")
print(f"Initial residue: {initial_residue}")
print(f"Found root: {root}")
print(f"Residue at found root: {abs(f(root))}") # Check residual at the found root
```
*Commentary:* In this basic example, we first define the function, set a high-precision level, and select an initial guess as a `mp.mpf` (arbitrary-precision floating-point number).  Crucially, we calculate `initial_residue` by evaluating `f` at the `initial_guess`, thus obtaining the value (magnitude) before the root-finding procedure begins. This gives us a sense of the function's "distance from zero" at that initial point. The value returned by findroot is the root found. I always check the residue at the root to verify that it is negligibly small, in terms of machine precision.

**Example 2: Exploring Different Initial Guesses**

```python
from mpmath import mp, findroot

mp.dps = 50

def f(x):
    return x**2 - 2

initial_guesses = [mp.mpf(1), mp.mpf(0), mp.mpf(-3)]

for guess in initial_guesses:
    initial_residue = abs(f(guess))
    root = findroot(f, guess)

    print(f"Initial guess: {guess}")
    print(f"Initial residue: {initial_residue}")
    print(f"Found root: {root}")
    print(f"Residue at found root: {abs(f(root))}\n")
```
*Commentary:* Here, I’ve iterated through several initial guesses, demonstrating the approach works equally well for diverse starting points. Note that for the initial guess 0, the root-finding method might take more steps and yield a root, in contrast to the other guesses. The initial residue, being the absolute value, always measures the 'size' of the function's output at the respective guess. This method is independent of whether the guess is "good" in the context of the root-finding.

**Example 3: Handling Functions with Complex Roots**

```python
from mpmath import mp, findroot

mp.dps = 50
mp.pretty = True # For better printing of complex numbers

def f(z):
    return z**2 + 1 # Function with imaginary roots

initial_guess = mp.mpc(1,1) # Initial guess as a complex number

initial_residue = abs(f(initial_guess))

root = findroot(f, initial_guess)

print(f"Initial guess: {initial_guess}")
print(f"Initial residue: {initial_residue}")
print(f"Found root: {root}")
print(f"Residue at found root: {abs(f(root))}\n")

initial_guess = mp.mpc(-1,1) # another initial guess

initial_residue = abs(f(initial_guess))

root = findroot(f, initial_guess)

print(f"Initial guess: {initial_guess}")
print(f"Initial residue: {initial_residue}")
print(f"Found root: {root}")
print(f"Residue at found root: {abs(f(root))}\n")
```
*Commentary:* This example showcases how to handle complex roots using `mp.mpc` for complex initial guesses. The function `f` now has complex roots (i and -i) and the residue at the initial complex guess is calculated as before, using the absolute value. The key is to define the function and initial guess using complex numbers. This allows findroot to explore the complex plane and return complex root values. When working in the complex domain it is also a good practice to check the residue at the root to verify the result.

In summary, the method for determining the 'initial' error/residue of `mpmath.findroot` involves manually evaluating the target function at the supplied initial guess. I consistently employ this method when building custom numerical tools, particularly when needing finer-grained control or needing to examine different starting points before finding solutions to complicated optical problems.  This workaround leverages the fact that the value of the function at the provided guess represents the proxy to the "error" associated with it before the iterative process of `findroot` modifies the starting guess.

For additional learning regarding numerical methods, I suggest exploring resources on: *Numerical Recipes* by Press et al., which provides a practical treatment of numerical algorithms; books specializing in *Numerical Analysis* that have theoretical foundations and proofs for root-finding methods; and Python libraries beyond `mpmath` such as `scipy.optimize`, as these offer additional methods which may explicitly return metrics related to the root-finding process and even the initial guess, although the initial residue still needs to be calculated manually for consistency. Further, investigating algorithms such as Brent's method or variations of Newton's method can be instructive, especially if the behavior of root-finding for edge cases is of interest.
