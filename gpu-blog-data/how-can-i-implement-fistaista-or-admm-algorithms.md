---
title: "How can I implement FISTA/ISTA or ADMM algorithms in PyTorch?"
date: "2025-01-30"
id: "how-can-i-implement-fistaista-or-admm-algorithms"
---
Iterative methods like FISTA (Fast Iterative Shrinkage-Thresholding Algorithm), ISTA (Iterative Shrinkage-Thresholding Algorithm), and ADMM (Alternating Direction Method of Multipliers) are not natively implemented as optimizers within PyTorch's `torch.optim` module. These algorithms are typically used for solving convex optimization problems, often involving sparsity or constraints, which diverge from the gradient-based optimization prevalent in deep learning. Thus, one must implement them directly, leveraging PyTorch’s tensor operations for computational efficiency.

ISTA provides a foundational approach to solving problems of the form:

```
min_x  f(x) + g(x)
```

where `f(x)` is a differentiable convex function, and `g(x)` is a potentially non-differentiable convex function with a readily available proximal operator. The algorithm iteratively updates `x` by taking a step in the negative gradient direction of `f(x)` and then applying the proximal operator associated with `g(x)`. FISTA, an accelerated version of ISTA, introduces a momentum term to speed up convergence. ADMM, on the other hand, focuses on problems that can be split into two or more blocks of variables, facilitating distributed optimization strategies. In essence, while gradient-based methods update based on the derivative of the loss, these algorithms leverage proximity, splitting, and penalty approaches.

My experience working on signal processing applications led me to develop efficient PyTorch implementations. Here's a breakdown of how one might approach this.

**Implementing ISTA**

The core operation in ISTA is the proximal operator of `g(x)`. When `g(x)` is the L1-norm (a common choice for sparsity), its proximal operator is the soft-thresholding operator. A general implementation in PyTorch, assuming a smooth function `f` with a gradient function `grad_f` and `g` as the L1 norm, would look like this:

```python
import torch

def soft_thresholding(x, lambda_val):
    """Proximal operator for L1 norm."""
    return torch.sign(x) * torch.max(torch.abs(x) - lambda_val, torch.zeros_like(x))


def ista_step(x, f, grad_f, lambda_val, learning_rate):
    """Performs a single step of ISTA."""
    grad = grad_f(x)
    x_update = x - learning_rate * grad
    x_updated = soft_thresholding(x_update, learning_rate * lambda_val)
    return x_updated


def ista(x_init, f, grad_f, lambda_val, learning_rate, iterations):
    """Performs iterations of ISTA."""
    x = x_init.clone()
    for _ in range(iterations):
        x = ista_step(x, f, grad_f, lambda_val, learning_rate)
    return x
```

In this code: `soft_thresholding` implements the L1 proximal operator. `ista_step` performs the gradient descent step on `f`, followed by the soft-thresholding, and `ista` loops this process for the desired number of iterations. In a practical situation, `f` would be, for example, the squared error of a linear system or neural network output, and `grad_f` its gradient with respect to the trainable parameters `x`.

**Implementing FISTA**

FISTA enhances ISTA by introducing momentum. This requires an additional variable and a modified update rule:

```python
def fista_step(x, x_prev, f, grad_f, lambda_val, learning_rate, t_prev):
    """Performs a single step of FISTA."""
    t = (1 + (1 + 4 * t_prev**2)**0.5) / 2
    y = x + ((t_prev - 1)/t) * (x - x_prev)

    grad = grad_f(y)
    x_update = y - learning_rate * grad
    x_updated = soft_thresholding(x_update, learning_rate * lambda_val)
    return x_updated, t, x


def fista(x_init, f, grad_f, lambda_val, learning_rate, iterations):
    """Performs iterations of FISTA."""
    x = x_init.clone()
    x_prev = x_init.clone()
    t_prev = 1
    for _ in range(iterations):
        x, t_prev, x_prev = fista_step(x, x_prev, f, grad_f, lambda_val, learning_rate, t_prev)
    return x
```

Here, `t` and the `x_prev` variable are used to introduce momentum. The updated `x` then also incorporates a proportion of `x-x_prev`. This structure often leads to faster convergence compared to ISTA. The implementation for `f` and `grad_f` can be re-used from the ISTA example.

**Implementing ADMM**

ADMM, in its simplest form, addresses problems of the form:

```
min_{x,z} f(x) + g(z)  subject to  Ax + Bz = c
```

The algorithm iteratively updates `x`, `z`, and a dual variable `u` using updates that involve proximal operators and augmented Lagrangian methods. Consider a setting where `g` is an L1 norm, and we have a system where the constraint `x - z = 0` is desirable (e.g. learning a sparse feature map with a regular feature map).

```python
def admm_step(x, z, u, f, grad_f, lambda_val, rho, A):
    """Performs a single step of ADMM."""
    # x-update
    x_update = x.clone()
    grad = grad_f(x_update)
    x_update = x_update - (rho * A.T @ (A @ x_update - z + u) )
    x_update = x_update - learning_rate * grad
    
    # z-update
    z_update = soft_thresholding(A@x_update + u , lambda_val/rho)
    
    #u-update
    u_update = u + A@x_update - z_update

    return x_update, z_update, u_update


def admm(x_init, z_init, f, grad_f, lambda_val, learning_rate, rho, iterations, A):
    """Performs iterations of ADMM."""
    x = x_init.clone()
    z = z_init.clone()
    u = torch.zeros_like(z).clone()

    for _ in range(iterations):
        x, z, u = admm_step(x, z, u, f, grad_f, lambda_val, rho, A)

    return x, z
```

In this example, `rho` is a penalty parameter, `A` is a matrix used in the constraint, and the updates for `x` and `z` are intertwined via the dual variable `u`. `grad_f` and `f` perform the same purpose as in the ISTA example, here for the problem component with `x`.

**Important Considerations:**

1.  **Gradient Implementation:** Ensure that the `grad_f` function accurately computes the gradient of your objective function `f` concerning the variables `x`. Autograd must be taken care of manually if gradients are not provided through existing torch functions.
2.  **Proximal Operators:** Implementing the appropriate proximal operator for `g(x)` is critical. The above examples use L1 norm, but others, like the indicator function of a constraint set, will require different operations.
3.  **Parameter Tuning:** Learning rate, `lambda_val`, and `rho` (in ADMM) are crucial for convergence. Experimentation and, in some cases, adaptive parameter selection, may be necessary.
4.  **Initialization:** The choice of initial points `x_init` significantly influences the convergence rate. Preconditioning or warm-starting can be helpful in practice.
5. **Convergence Criteria:** In practice, one needs to assess convergence at each iteration to determine if the algorithm has converged instead of running for a fixed number of iterations.

These implementations provide a framework for applying ISTA, FISTA, and ADMM within a PyTorch environment. They differ from PyTorch's built-in gradient descent-based methods and allow one to incorporate proximal operators and constraint handling directly into the optimization process.

For a deeper understanding of these algorithms, I recommend resources such as "Numerical Optimization" by Nocedal and Wright, and "Convex Optimization" by Boyd and Vandenberghe. Further exploration can be found in research papers specifically covering these algorithms in the signal processing and machine learning domains. Understanding proximal operators is also important and can be investigated using online resources and papers.
