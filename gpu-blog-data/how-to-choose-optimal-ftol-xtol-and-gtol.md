---
title: "How to choose optimal ftol, xtol, and gtol values for least_squares?"
date: "2025-01-30"
id: "how-to-choose-optimal-ftol-xtol-and-gtol"
---
The convergence criteria in least-squares minimization, specifically `ftol`, `xtol`, and `gtol`, are not universally optimal; their ideal values are heavily problem-dependent.  My experience optimizing large-scale non-linear models for material science simulations has taught me that a purely algorithmic approach to selecting these tolerances often fails.  Effective selection requires a deep understanding of the problem's inherent numerical characteristics and sensitivity to error.  This response will detail the role of each tolerance, provide guidance on selection, and illustrate practical applications through code examples.


**1. Understanding the Convergence Criteria**

Least-squares algorithms aim to minimize a residual function, often represented as the sum of squared differences between observed and predicted values.  The convergence criteria control when the algorithm terminates, preventing unnecessary iterations and ensuring a solution of sufficient accuracy.  Each tolerance monitors a different aspect of the convergence process:

* **`ftol` (Relative function tolerance):** This parameter monitors the relative change in the objective function (the sum of squares).  The algorithm terminates when the relative decrease in the objective function between iterations falls below `ftol`.  A smaller `ftol` implies a stricter requirement for the reduction in the objective function.

* **`xtol` (Relative parameter tolerance):**  This parameter monitors the relative change in the parameter vector (the variables being optimized).  Termination occurs when the relative change in the parameter vector between iterations falls below `xtol`.  A smaller `xtol` demands a more precise solution in terms of parameter values.

* **`gtol` (Gradient tolerance):** This parameter monitors the magnitude of the gradient of the objective function. The gradient indicates the direction of steepest ascent. The algorithm terminates when the norm of the gradient falls below `gtol`.  A smaller `gtol` implies a stricter requirement on the gradient magnitude, indicating a closer approximation to the minimum where the gradient is ideally zero.

**2. Choosing Optimal Values**

The optimal values for `ftol`, `xtol`, and `gtol` depend on several factors:

* **Problem Scale and Complexity:**  Large, complex models with many parameters often necessitate looser tolerances to avoid excessive computation time. Conversely, simpler models may benefit from stricter tolerances for higher accuracy.

* **Data Quality and Noise:** Noisy data may necessitate looser tolerances.  A strict tolerance might attempt to fit noise, leading to overfitting and a less-generalizable model.

* **Computational Resources:**  Computation time increases with stricter tolerances.  Balancing accuracy with computational cost is crucial.

* **Sensitivity Analysis:**  A sensitivity analysis helps determine the sensitivity of the solution to changes in the parameters. If the model is highly sensitive, stricter tolerances might be warranted.

A practical strategy involves a hierarchical approach: begin with relatively loose tolerances, then gradually tighten them, monitoring the changes in the solution and computational time.  This iterative process allows for a cost-effective approach to finding appropriate convergence criteria.


**3. Code Examples with Commentary**

The following examples use Python's `scipy.optimize.least_squares`, showcasing the effect of different tolerance values.  For simplicity, these illustrate a basic non-linear least-squares problem; in my real-world applications, this would often involve significantly larger and more complex models within a larger simulation framework.

**Example 1:  Loose Tolerances**

```python
import numpy as np
from scipy.optimize import least_squares

def model(x, t):
    return x[0] * np.exp(-x[1] * t)

def residuals(x, t, y):
    return y - model(x, t)

t = np.linspace(0, 10, 10)
y = model([1.5, 0.2], t) + 0.1 * np.random.randn(len(t))

res = least_squares(residuals, [1, 0.1], args=(t, y), ftol=1e-2, xtol=1e-2, gtol=1e-2)
print(res.x)
print(res.cost) #Sum of squared residuals
print(res.nfev) # Number of function evaluations
```

This example uses relatively loose tolerances. The solution will converge quickly, but the accuracy might be compromised.  The high values for `ftol`, `xtol`, and `gtol` will lead to early termination. This is appropriate when computational time is a major concern and high precision is less critical.


**Example 2:  Moderate Tolerances**

```python
import numpy as np
from scipy.optimize import least_squares

# ... (same model and residuals functions as Example 1) ...

t = np.linspace(0, 10, 10)
y = model([1.5, 0.2], t) + 0.1 * np.random.randn(len(t))

res = least_squares(residuals, [1, 0.1], args=(t, y), ftol=1e-4, xtol=1e-4, gtol=1e-4)
print(res.x)
print(res.cost)
print(res.nfev)
```

This example employs more stringent tolerances, leading to a potentially more accurate solution. The improvement in accuracy comes at the cost of increased computation time (indicated by `res.nfev`).  This setting provides a good balance between accuracy and computational efficiency for many problems.


**Example 3:  Strict Tolerances (Illustrative)**

```python
import numpy as np
from scipy.optimize import least_squares

# ... (same model and residuals functions as Example 1) ...

t = np.linspace(0, 10, 10)
y = model([1.5, 0.2], t) + 0.1 * np.random.randn(len(t))

res = least_squares(residuals, [1, 0.1], args=(t, y), ftol=1e-8, xtol=1e-8, gtol=1e-8)
print(res.x)
print(res.cost)
print(res.nfev)
```

This demonstrates extremely strict tolerances.  While achieving potentially the highest accuracy, this approach is computationally expensive and may be unnecessary. In many cases, the gain in accuracy may be negligible compared to the increased computation time. This is particularly true for noisy datasets where pursuing such high accuracy is unrealistic.



**4. Resource Recommendations**

For a deeper understanding of non-linear optimization algorithms and their convergence criteria, I recommend consulting numerical analysis textbooks focusing on optimization techniques and studying the documentation of numerical optimization libraries.  Specifically, exploring advanced topics in optimization theory, such as the impact of condition numbers and scaling of parameters on convergence, is very useful.  Understanding the underlying mathematical principles significantly aids in making informed decisions about tolerance selection.  Reviewing publications on the specific application domain will also provide invaluable insights into practical tolerance selection strategies.
