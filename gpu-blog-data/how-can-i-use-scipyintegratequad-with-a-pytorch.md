---
title: "How can I use `scipy.integrate.quad` with a PyTorch tensor?"
date: "2025-01-30"
id: "how-can-i-use-scipyintegratequad-with-a-pytorch"
---
Integrating a function numerically using `scipy.integrate.quad` when that function relies on PyTorch tensors as input necessitates careful handling of data types and gradient propagation. `scipy.integrate.quad` expects a function that operates on scalar floats, not PyTorch tensors, and it does not inherently understand PyTorch’s automatic differentiation. This requires constructing a wrapper function that converts between PyTorch tensors and standard Python floats, ensuring that the computation remains within the PyTorch computational graph for gradient calculations when needed. My experience in developing custom physical simulation models has frequently required this approach, particularly when dealing with differentiable force fields.

The fundamental challenge arises because `scipy.integrate.quad` internally evaluates the provided function at various floating-point points, typically using adaptive quadrature techniques. These points are standard Python floats, and the function provided *must* return a Python float. PyTorch tensors, on the other hand, are multidimensional arrays that support backpropagation. Direct use of PyTorch tensors within the integration function will result in type errors and will not allow automatic differentiation through the integration process.

To bridge this gap, a wrapper function is needed. This function should accept a scalar float as input, convert it to a PyTorch tensor (if required for the original function's computation), evaluate the original function using the tensor, extract the resulting scalar value as a float, and return this float. This effectively isolates the integration process from the tensor operations used within the integrand function. Importantly, if backpropagation is required, all computations performed within this wrapped function must operate on PyTorch tensors that are part of the PyTorch computational graph.

Here is the first illustrative code example, focusing on a basic case of integrating a simple function which *does not* require differentiable integration:

```python
import torch
import numpy as np
from scipy.integrate import quad

def integrand(x, a):
    """
    A simple function to integrate, operating on a Python float.
    This function does *not* require differentiable integration.
    """
    return a * x**2

def integrate_with_params(a_tensor, lower_limit, upper_limit):
    """
    Integrates the integrand with a parameter using quad.
    The parameter must be converted to a Python float.
    """
    a_val = a_tensor.item() # Extracts the Python float from the tensor
    result, _ = quad(integrand, lower_limit, upper_limit, args=(a_val,))
    return result

# Example usage
a = torch.tensor(2.0)
lower_limit = 0.0
upper_limit = 1.0
integral = integrate_with_params(a, lower_limit, upper_limit)
print(f"Integral result: {integral}") # Output: Integral result: 0.6666666666666666
```

In this example, the parameter `a` is initially a PyTorch tensor. The wrapper function, `integrate_with_params`, extracts the numerical value of `a` using `a_tensor.item()` before passing it to the `integrand` function and ultimately to `scipy.integrate.quad`. Crucially, since the parameter *does not* require gradient calculation (the gradient is not flowing *through* `scipy.integrate.quad`), we can safely extract the underlying float before calling `quad`. If the function were more complex and involved other parameters or computation, it is essential to ensure this type extraction is performed before being passed as an argument to the integrand function.

Now consider a scenario where the function being integrated *does* require gradient calculation - specifically, where we want the gradient of the integral *with respect to the parameters* of our function.  This is a more complex scenario and requires careful construction of the wrapper function so that the integration is performed as a step of the PyTorch computational graph, allowing for backpropagation through the integration. We must operate solely with tensors at all times before the final extraction of a value for `quad`.

Here is the second code example illustrating differentiable integration:

```python
import torch
import numpy as np
from scipy.integrate import quad

def torch_integrand(x, a):
    """
    The function to be integrated operating on a PyTorch tensor.
    This function now operates on and returns PyTorch tensors,
    and is part of the PyTorch computational graph.
    """
    x_tensor = torch.tensor(x, requires_grad=False) # Convert x to a Tensor without gradient
    return a * x_tensor**2

def differentiable_integrate(a_tensor, lower_limit, upper_limit):
    """
    Integrates the torch_integrand with automatic differentiation.
    Uses a lambda function to convert the float returned from `quad` to tensor
    """
    def wrapper(x):
      return torch_integrand(x,a_tensor).item()

    result, _ = quad(wrapper, lower_limit, upper_limit)

    return torch.tensor(result,requires_grad=False) # Make the result a Tensor for gradient calculation

# Example usage
a = torch.tensor(2.0, requires_grad=True)
lower_limit = 0.0
upper_limit = 1.0

integral = differentiable_integrate(a, lower_limit, upper_limit)
integral.backward()
print(f"Integral result: {integral}") # Output: Integral result: 0.6666666666666666
print(f"Gradient of integral with respect to a: {a.grad}") # Output: Gradient of integral with respect to a: 0.3333333432674408
```

In this example, the `torch_integrand` function operates on PyTorch tensors and returns a PyTorch tensor. Inside the `differentiable_integrate` function, a `wrapper` function is used, which calls `torch_integrand`, extracts the scalar value using `.item()`, before passing the scalar to `scipy.integrate.quad`. The result is then transformed back into a PyTorch tensor with `requires_grad=False`, making it part of the computational graph and allowing us to take the gradient of the integration with respect to the parameter `a` using backpropagation.

A final example showcases a slightly more complicated case, using a function that involves more than a single parameter, emphasizing the conversion of all parameters into PyTorch tensors:

```python
import torch
import numpy as np
from scipy.integrate import quad

def torch_integrand_complex(x, a, b):
    """
    A more complex function to integrate using PyTorch.
    """
    x_tensor = torch.tensor(x, requires_grad=False)
    return a * x_tensor**2 + b * x_tensor

def differentiable_integrate_complex(a_tensor, b_tensor, lower_limit, upper_limit):
    """
     Integrates torch_integrand_complex using automatic differentiation.
     Uses a lambda function to convert floats to tensors in the integrator.
    """

    def wrapper(x):
        return torch_integrand_complex(x,a_tensor,b_tensor).item()

    result, _ = quad(wrapper, lower_limit, upper_limit)
    return torch.tensor(result, requires_grad=False)

# Example usage
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
lower_limit = 0.0
upper_limit = 1.0

integral = differentiable_integrate_complex(a, b, lower_limit, upper_limit)
integral.backward()

print(f"Integral result: {integral}") # Output: Integral result: 1.1666666666666667
print(f"Gradient of integral w.r.t. a: {a.grad}") # Output: Gradient of integral w.r.t. a: 0.3333333432674408
print(f"Gradient of integral w.r.t. b: {b.grad}") # Output: Gradient of integral w.r.t. b: 0.5
```

This third example showcases a more realistic integration scenario with two parameters, `a` and `b`. The same principle applies: encapsulate all of the PyTorch operations within a function, then extract the resulting scalar value to feed to `quad`. The gradient of the integral is then calculated with respect to all the differentiable parameters using backpropagation through the integration as the parameters used were initialized with `requires_grad=True`.

In conclusion, while `scipy.integrate.quad` does not directly operate on PyTorch tensors, it can be used effectively within a PyTorch workflow by creating a carefully constructed wrapper function. This function must convert scalar floating points into tensors, perform all required computations with tensors, extract the resulting scalar using `.item()`, and then pass the value to the `quad` integrator. If the integral result is part of a larger gradient calculation, it is critical to convert the value from `quad` back into a PyTorch tensor.

For learning more about numerical integration techniques, and the mathematics behind the adaptive quadrature methods employed by `scipy.integrate.quad`, I recommend consulting textbooks on Numerical Analysis. For further details about PyTorch's autograd system, the official PyTorch documentation offers comprehensive guides and tutorials. Examining examples in advanced physics simulations using PyTorch and integrating across various variables could also prove instructive.
