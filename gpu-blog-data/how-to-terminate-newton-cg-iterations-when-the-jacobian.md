---
title: "How to terminate Newton-CG iterations when the Jacobian is sufficiently small?"
date: "2025-01-30"
id: "how-to-terminate-newton-cg-iterations-when-the-jacobian"
---
The efficacy of Newton-CG methods hinges on a robust convergence criterion, particularly when dealing with ill-conditioned Jacobians.  Simply monitoring the residual norm isn't sufficient; a more nuanced approach is required to ensure termination when the Jacobian's influence becomes negligible, preventing unnecessary iterations and potential numerical instability.  My experience implementing large-scale optimization routines for inverse problems in seismic imaging highlighted this crucial aspect.  The key lies in monitoring a suitable norm of the Jacobian, not just the residual.

The Newton-CG method iteratively refines an initial guess,  `x₀`, to solve a system of nonlinear equations, `F(x) = 0`.  Each iteration involves solving a linear system using the conjugate gradient (CG) method, where the Jacobian matrix, `J(x)`, plays a central role.  The CG algorithm itself terminates when a predetermined convergence criterion is met, typically related to the residual of the linear system.  However, this alone does not guarantee that the Jacobian's contribution to further iterations is insignificant.  A small residual may still exist due to an ill-conditioned Jacobian, leading to slow convergence or divergence.

Therefore, a robust termination criterion should encompass both the residual norm and a measure of the Jacobian's magnitude or condition number.  My approach usually integrates a check on the Frobenius norm of the Jacobian, or a suitable approximation thereof, as a secondary stopping criterion. The Frobenius norm provides a readily computable measure of the overall size of the Jacobian.  It's computationally less expensive than calculating the condition number directly, particularly for large matrices.  We can combine this with the typical residual-based stopping criteria to create a more reliable termination condition.

**1.  Clear Explanation of the Termination Criterion**

The proposed termination criterion is a two-pronged approach:  first, checking the relative reduction in the residual norm; and second, independently checking the Frobenius norm of the Jacobian.  The algorithm terminates when *both* conditions are met.

* **Residual Convergence:** This is the standard Newton-CG termination criterion. We monitor the relative change in the residual norm, `||F(xₖ)||`, between successive iterations:

   `||F(xₖ)|| / ||F(xₖ₋₁)|| < ε₁`

   where `ε₁` is a small positive tolerance, typically in the range of 1e-6 to 1e-10, depending on the problem's sensitivity.  This condition ensures that the nonlinear system's solution is adequately approximated.

* **Jacobian Magnitude Check:** This is the crucial addition. We calculate the Frobenius norm of the Jacobian at each iteration:

   `||J(xₖ)||_F = sqrt( Σᵢ Σⱼ |Jᵢⱼ(xₖ)|²) `

   and check if it falls below a predefined threshold `ε₂`:

   `||J(xₖ)||_F < ε₂`

   where `ε₂` is another small positive tolerance, often chosen based on prior knowledge of the problem or through experimentation. This condition addresses the issue of ill-conditioning; a small Jacobian indicates that further iterations may not significantly improve the solution.

The algorithm terminates when *both* `||F(xₖ)|| / ||F(xₖ₋₁)|| < ε₁` and `||J(xₖ)||_F < ε₂` are true. This prevents premature termination due to a small residual resulting from a poorly conditioned Jacobian and ensures that the Jacobian's influence on the iterative process is negligible.


**2. Code Examples with Commentary**

The following examples demonstrate the implementation in Python using NumPy and SciPy.  These examples assume `F` and `J` are user-defined functions that compute the residual vector and the Jacobian matrix, respectively.  Error handling and robustness checks (like checking for singular Jacobians) are simplified for clarity.

**Example 1:  Basic Implementation**

```python
import numpy as np
from scipy.sparse.linalg import cg

def newton_cg(F, J, x0, epsilon1=1e-8, epsilon2=1e-4, max_iter=100):
    x = x0
    for i in range(max_iter):
        r = F(x)
        J_x = J(x)
        # Solve linear system using Conjugate Gradient
        dx, info = cg(J_x, -r)
        if info > 0:
            print("CG solver did not converge")
            break
        x_new = x + dx
        r_new = F(x_new)

        if np.linalg.norm(r_new) / np.linalg.norm(r) < epsilon1 and np.linalg.norm(J_x, ord='fro') < epsilon2:
            return x_new, i+1

        x = x_new
    return x, max_iter


# Example usage (replace with your actual F and J functions)
def F_example(x):
  return np.array([x[0]**2 + x[1] - 2, x[0] + x[1]**2 - 2])

def J_example(x):
  return np.array([[2*x[0], 1], [1, 2*x[1]]])

x0 = np.array([1.0, 1.0])
solution, iterations = newton_cg(F_example, J_example, x0)
print(f"Solution: {solution}, Iterations: {iterations}")
```

This example provides a basic implementation showcasing the integrated convergence criterion.  It utilizes SciPy's `cg` function for the linear system solve.

**Example 2:  Sparse Jacobian Handling**

For large-scale problems, the Jacobian is often sparse. This version adapts the previous example to handle sparse matrices efficiently.

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

# ... (F_example and J_example remain the same, potentially modified to return sparse matrices) ...

def newton_cg_sparse(F, J, x0, epsilon1=1e-8, epsilon2=1e-4, max_iter=100):
    x = x0
    for i in range(max_iter):
        r = F(x)
        J_x = J(x)  # Assumed to return a sparse matrix

        # Create LinearOperator for efficient matrix-vector multiplication
        A = LinearOperator(shape=J_x.shape, matvec=lambda v: J_x.dot(v))

        dx, info = cg(A, -r)
        if info > 0:
            print("CG solver did not converge")
            break
        x_new = x + dx
        r_new = F(x_new)

        # Frobenius norm calculation for sparse matrix
        frobenius_norm = np.linalg.norm(J_x.data, ord=2)

        if np.linalg.norm(r_new) / np.linalg.norm(r) < epsilon1 and frobenius_norm < epsilon2:
            return x_new, i + 1

        x = x_new
    return x, max_iter

```
This example uses `scipy.sparse.linalg.LinearOperator` to efficiently handle the matrix-vector products within the CG solver, improving performance for large sparse systems.  The Frobenius norm calculation is adapted for sparse matrices.

**Example 3:  Adaptive Tolerance**

In some cases, it might be beneficial to adapt the tolerances `ε₁` and `ε₂` based on the problem's behavior. This example demonstrates a simple adaptive strategy.


```python
import numpy as np
from scipy.sparse.linalg import cg

# ... (F_example and J_example remain the same) ...

def newton_cg_adaptive(F, J, x0, epsilon1_init=1e-6, epsilon2_init=1e-3, max_iter=100, reduction_factor=0.5):
  epsilon1 = epsilon1_init
  epsilon2 = epsilon2_init
  x = x0
  for i in range(max_iter):
    # ... (rest of the code is similar to Example 1, except for the tolerance updates) ...
    if i > 10 and np.linalg.norm(r_new) / np.linalg.norm(r) > 0.5: # Example condition for tolerance reduction
        epsilon1 *= reduction_factor
        epsilon2 *= reduction_factor

    if np.linalg.norm(r_new) / np.linalg.norm(r) < epsilon1 and np.linalg.norm(J_x, ord='fro') < epsilon2:
        return x_new, i + 1

    x = x_new
  return x, max_iter
```

This example reduces the tolerances if convergence is slow, potentially improving the algorithm's robustness in challenging scenarios. The specific logic for tolerance adaptation should be tailored to the problem at hand.


**3. Resource Recommendations**

For further understanding, I recommend consulting numerical optimization textbooks focusing on Newton methods and conjugate gradient techniques.  Explore literature on large-scale optimization and sparse matrix computations.  A comprehensive treatment of iterative methods and their convergence properties will be invaluable.  Also, consider referencing publications on specific applications of Newton-CG, especially in fields where ill-conditioned Jacobians are common.  This will provide practical insights and alternative approaches for managing convergence.
