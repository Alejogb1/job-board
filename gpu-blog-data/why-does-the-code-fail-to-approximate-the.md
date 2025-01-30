---
title: "Why does the code fail to approximate the square function?"
date: "2025-01-30"
id: "why-does-the-code-fail-to-approximate-the"
---
The core issue lies in the choice of approximation method and its inherent limitations, specifically concerning the radius of convergence and the handling of edge cases.  My experience working on high-precision numerical algorithms for aerospace simulations highlights the critical need for rigorous error analysis when approximating non-linear functions.  A naive approach, such as relying on a low-order Taylor expansion without considering its limitations, will almost certainly result in significant inaccuracies, especially outside a small neighborhood of the expansion point.

Let's clarify this with a detailed explanation.  The square function, f(x) = x², is inherently simple, yet approximating it numerically introduces complexities.  Taylor expansion, a common approximation technique, represents a function as an infinite sum of terms involving its derivatives at a specific point.  A truncated Taylor series, however, provides only an approximation valid within a certain radius of convergence.  Beyond this radius, the approximation diverges rapidly, leading to large errors.  Furthermore, the accuracy of the Taylor approximation is directly related to the order of the expansion – higher-order expansions generally provide better accuracy within the radius of convergence but require significantly more computation.

Consider a first-order Taylor expansion of f(x) = x² around the point a:

f(x) ≈ f(a) + f'(a)(x - a) = a² + 2a(x - a)

This approximation is only accurate for values of x close to 'a'.  The further x is from 'a', the larger the error.  A second-order Taylor expansion improves accuracy, but the error still grows with distance from 'a':

f(x) ≈ f(a) + f'(a)(x - a) + (1/2)f''(a)(x - a)² = a² + 2a(x - a) + (x - a)²

Higher-order expansions follow the same pattern, diminishing error within a limited range but failing to provide a global accurate approximation.  This is why a low-order Taylor expansion, used naively, will produce a poor approximation of the square function over a wider range of input values.  This inherent limitation is compounded by the fact that the square function itself exhibits no problematic behavior—it's continuous and differentiable everywhere—which makes the approximation's failure purely a consequence of the method's inadequacies.


Let's illustrate this with three code examples in Python, highlighting the shortcomings of different approaches:

**Example 1: First-order Taylor expansion around a = 1:**

```python
import numpy as np
import matplotlib.pyplot as plt

def taylor_first_order(x, a=1):
    return a**2 + 2*a*(x - a)

x_values = np.linspace(-2, 3, 100)
y_values_approx = [taylor_first_order(x) for x in x_values]
y_values_true = [x**2 for x in x_values]

plt.plot(x_values, y_values_true, label='True Square Function')
plt.plot(x_values, y_values_approx, label='First-Order Taylor Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('First-Order Taylor Approximation of x^2 around a=1')
plt.grid(True)
plt.show()
```

This code demonstrates the significant deviation between the approximation and the true square function even for moderately distant values of x.  The accuracy is severely limited due to the low order of the expansion.


**Example 2: Second-order Taylor expansion around a = 1:**

```python
import numpy as np
import matplotlib.pyplot as plt

def taylor_second_order(x, a=1):
    return a**2 + 2*a*(x - a) + (x - a)**2

x_values = np.linspace(-2, 3, 100)
y_values_approx = [taylor_second_order(x) for x in x_values]
y_values_true = [x**2 for x in x_values]

plt.plot(x_values, y_values_true, label='True Square Function')
plt.plot(x_values, y_values_approx, label='Second-Order Taylor Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Second-Order Taylor Approximation of x^2 around a=1')
plt.grid(True)
plt.show()
```

While the second-order approximation is better than the first-order one, it still exhibits considerable error, particularly at the edges of the plotted range.  This underscores the limited radius of convergence even for higher-order approximations.


**Example 3:  A more robust, but still limited, approach using a piecewise linear approximation:**

```python
import numpy as np
import matplotlib.pyplot as plt

def piecewise_linear(x, breakpoints):
    if x < breakpoints[0]:
        return breakpoints[0]**2
    elif x > breakpoints[-1]:
        return breakpoints[-1]**2
    else:
        i = np.argmax(breakpoints >= x)
        x1, x2 = breakpoints[i - 1], breakpoints[i]
        y1, y2 = x1**2, x2**2
        return y1 + (y2 - y1) / (x2 - x1) * (x - x1)


breakpoints = np.linspace(-2, 3, 11)  # Define breakpoints for linear segments
x_values = np.linspace(-2, 3, 100)
y_values_approx = [piecewise_linear(x, breakpoints) for x in x_values]
y_values_true = [x**2 for x in x_values]

plt.plot(x_values, y_values_true, label='True Square Function')
plt.plot(x_values, y_values_approx, label='Piecewise Linear Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Piecewise Linear Approximation of x^2')
plt.grid(True)
plt.show()
```

This example demonstrates a different approach, employing piecewise linear interpolation. This method reduces error by dividing the approximation into smaller intervals. However, its accuracy is still limited by the density of breakpoints; more breakpoints lead to better accuracy, but at the cost of increased computational complexity.  A higher-order piecewise polynomial (e.g., spline interpolation) would provide further improvement.

In conclusion, the failure to accurately approximate the square function arises from limitations inherent in the chosen approximation method, specifically its radius of convergence.  Taylor expansions, while useful locally, are not suitable for global approximations without employing techniques like piecewise approximations or higher order methods. A thorough understanding of these limitations and the careful selection of appropriate techniques, considering computational cost and required accuracy, are crucial for reliable numerical approximation of functions.


**Resource Recommendations:**

* Numerical Analysis textbooks covering approximation theory and error analysis.
* Texts on interpolation and approximation methods.
* Advanced calculus resources covering Taylor series and their properties.  Understanding the remainder term is critical in assessing approximation errors.
