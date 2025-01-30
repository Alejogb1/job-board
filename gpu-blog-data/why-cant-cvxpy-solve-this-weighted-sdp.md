---
title: "Why can't cvxpy solve this weighted SDP?"
date: "2025-01-30"
id: "why-cant-cvxpy-solve-this-weighted-sdp"
---
The difficulty in solving weighted semidefinite programs (SDPs) with CVXPY often stems from the interaction between the weight matrix and the structure of the underlying SDP.  My experience working on robust control problems, specifically those involving uncertain systems with weighted performance objectives, has highlighted this issue repeatedly.  Simply adding weights doesn't guarantee solvability; the problem's inherent numerical properties, coupled with CVXPY's reliance on underlying solvers like SCS or Mosek, can lead to infeasibility or slow convergence, even when the unweighted problem is easily solvable. This often arises from ill-conditioning induced by the weight matrix, particularly when its condition number is very high.

**1. Explanation:**

CVXPY, a powerful Python-embedded modeling language for convex optimization problems, relies on external solvers to perform the actual optimization. These solvers, while robust, are not immune to numerical challenges.  Weighted SDPs introduce an additional layer of complexity.  Consider a standard SDP formulation:

Minimize:  `trace(C*X)`
Subject to: `trace(Aᵢ*X) = bᵢ`,  `X ⪰ 0`

where `X` is the positive semidefinite matrix variable, `C` and `Aᵢ` are given matrices, and `bᵢ` are given scalars.  Introducing weights modifies the objective function:

Minimize:  `trace(W*C*X)`

where `W` is a positive definite weight matrix.  While seemingly straightforward, this seemingly small change can significantly impact the problem's numerical properties.

The crucial issue is the condition number of `W*C`.  If `W` is ill-conditioned (i.e., has a large ratio of its largest to smallest eigenvalue), it can exacerbate any existing numerical instability in the original SDP problem `C`.  This leads to several potential issues:

* **Ill-conditioning:**  The solver may struggle to find an accurate solution due to numerical errors propagating during the iterative solution process.  Small changes in the input data (inherent in floating-point arithmetic) can lead to large changes in the solution.
* **Infeasibility:**  The solver might declare the problem infeasible, even if a slightly perturbed version is feasible, due to the amplified effect of numerical noise.
* **Slow Convergence:** The iterative algorithms used by the solvers may require significantly more iterations to converge to a solution within the desired tolerance.  This leads to increased computation time and potentially memory issues.


Furthermore, the choice of solver significantly impacts the outcome.  Solvers like SCS (Splitting Conic Solver), which is often the default in CVXPY, are first-order methods, generally more robust to ill-conditioning but slower than second-order methods.  Mosek, a commercial solver, typically employs interior-point methods, which can be faster for well-conditioned problems but more sensitive to ill-conditioning.


**2. Code Examples with Commentary:**

The following examples illustrate how weighting can affect SDP solvability using CVXPY.

**Example 1: A well-conditioned weighted SDP:**

```python
import cvxpy as cp
import numpy as np

# Generate a well-conditioned weight matrix
W = np.eye(3)

# Define the SDP problem (simplified for brevity)
X = cp.Variable((3, 3), symmetric=True)
C = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
objective = cp.Minimize(cp.trace(W @ C @ X))
constraints = [X >> 0] # X is positive semidefinite

problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal value:", problem.value)
print("Optimal X:\n", X.value)
```

In this example, `W` is the identity matrix, ensuring good conditioning. The problem is expected to solve efficiently.


**Example 2: An ill-conditioned weighted SDP:**

```python
import cvxpy as cp
import numpy as np

# Generate an ill-conditioned weight matrix
W = np.array([[1e6, 0, 0], [0, 1, 0], [0, 0, 1e-6]])

# Same SDP problem as Example 1
X = cp.Variable((3, 3), symmetric=True)
C = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
objective = cp.Minimize(cp.trace(W @ C @ X))
constraints = [X >> 0]

problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal value:", problem.value)
print("Optimal X:\n", X.value)

```

Here, `W` is deliberately ill-conditioned.  The solver might struggle or fail to find a solution, reporting infeasibility or returning an inaccurate result.  The extreme difference in eigenvalues of `W` causes numerical instability.


**Example 3:  Addressing Ill-Conditioning (Preconditioning):**

```python
import cvxpy as cp
import numpy as np

# Ill-conditioned weight matrix (same as Example 2)
W = np.array([[1e6, 0, 0], [0, 1, 0], [0, 0, 1e-6]])

# Precondition the problem
L = np.linalg.cholesky(W)
W_inv = np.linalg.inv(L)
C_preconditioned = W_inv @ C @ W_inv.T

X = cp.Variable((3, 3), symmetric=True)
objective = cp.Minimize(cp.trace(C_preconditioned @ X))
constraints = [X >> 0]

problem = cp.Problem(objective, constraints)
problem.solve()


print("Optimal value:", problem.value)
print("Optimal X:\n", X.value)
```

This example attempts to mitigate the ill-conditioning by preconditioning the problem. We decompose `W` using Cholesky decomposition and transform the problem to improve numerical stability.  However, this technique is not always guaranteed to work and might require careful analysis of the specific problem structure.


**3. Resource Recommendations:**

* **Boyd & Vandenberghe's "Convex Optimization":**  A comprehensive text covering the theoretical foundations of convex optimization, including SDPs.
* **Numerical Linear Algebra texts:**  Understanding the concepts of matrix condition numbers and their impact on numerical stability is vital.
* **Documentation for CVXPY and chosen solvers (SCS, Mosek, etc.):**  Familiarizing yourself with the solver's capabilities and limitations is essential for effective problem solving.  Understanding the parameters and tolerances available within these solvers can allow for tuning to address numerical issues.  Analyzing solver output messages is critical for debugging.


In conclusion, solving weighted SDPs with CVXPY requires careful consideration of the weight matrix's condition number.  Ill-conditioning can lead to various numerical difficulties, hindering the solver's ability to find an accurate and efficient solution.  Preconditioning techniques or alternative solver choices may be necessary to handle such problems effectively.  Thorough understanding of both the theoretical and practical aspects of convex optimization and numerical linear algebra is crucial for successful implementation.
