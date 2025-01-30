---
title: "How can complex-valued weights be backpropagated?"
date: "2025-01-30"
id: "how-can-complex-valued-weights-be-backpropagated"
---
The core of backpropagation with complex-valued weights relies on applying the chain rule, but adapted to handle complex numbers. Specifically, the challenge isn't in *whether* backpropagation works, but *how* gradients are computed and interpreted in the complex domain. This stems from the fact that complex functions are not differentiable in the same sense as real functions; instead, we work with Wirtinger derivatives.

I've encountered this frequently while working on signal processing models, especially when dealing with frequency-domain representations where complex numbers naturally arise for amplitude and phase information. The standard real-valued backpropagation, focusing on derivatives with respect to real-valued weights, simply isn't sufficient. The key difference centers on how we define the "derivative" of a complex function. Unlike real functions, which have a straightforward single derivative, complex functions can be differentiated with respect to the complex variable itself (z) or its conjugate (z̄). The Wirtinger calculus provides us with a framework to compute these derivatives independently, and we use these partial derivatives in the chain rule.

The practical implication is that, when updating complex weights, we must account for the partial derivative with respect to the weight's conjugate. Let's denote a complex weight by *w = a + bi*, where *a* is the real part and *b* is the imaginary part. In a real-valued scenario, we'd simply adjust the weight by the learning rate multiplied by the gradient. However, in the complex case, we update both the real and imaginary components, each influenced by the respective Wirtinger derivative.

Let’s elaborate with some pseudo-code and practical examples. Assume we have a loss function *L* that depends on the output of a complex-valued neural network, denoted by *ŷ*. And, ŷ depends on some previous complex operation with a complex weight *w*. We will propagate error to *w*. The crucial insight is that the gradient of *L* with respect to *w* is not simply dL/dw but rather dL/dw* + dL/dw̄ * . Where dL/dw and dL/dw̄ are Wirtinger derivatives, and the asterisk * denotes complex conjugate. These can be computed via chain rule in the forward pass using rules that mirror the real case, but are computed over a complex number system.

The computation of the partial derivatives requires understanding the relationships in a complex variable.
* **dl/dw:** the Wirtinger partial derivative with respect to w. This can be computed by using standard chain rules that apply to real numbers.
* **dl/dw̄:** the Wirtinger partial derivative with respect to w̄ (the conjugate of w). If we use dl/dw, we will only be backpropagating the error with respect to the real part and the imaginary parts; the Wirtinger derivative with respect to w̄ will ensure that we have information with respect to conjugate relationships that exist, and that we can appropriately perform gradient descent.

**Example 1: Linear Complex Operation**
Let's consider a simplified scenario where ŷ = w * x, where *x* and *w* are complex numbers. For example, x = 1+1j, and we can randomly initialise w.

```python
import numpy as np

# Forward Pass
def complex_linear_forward(x, w):
    return w * x

# Loss function (mean squared error, example)
def complex_mse_loss(y_true, y_pred):
    error = y_true - y_pred
    return np.mean(np.abs(error)**2)

# Backpropagation
def complex_linear_backward(x, w, y_true):
  y_pred = complex_linear_forward(x, w)
  dL_dy = 2 * (y_pred - y_true)
  # The Wirtinger derivatives
  dL_dw = dL_dy * x
  dL_dw_conj = 0 # because dy/dw_conj is 0, it's independent
  # Complex Gradient w/r w:
  dL_dw_total = dL_dw + np.conjugate(dL_dw_conj)

  # Update the weight
  learning_rate = 0.1
  updated_w = w - learning_rate * dL_dw_total.conjugate()

  return updated_w
# Example Use
x = 1 + 1j
w = 0.5 + 0.5j
y_true = 2 + 2j # ideal output.

updated_w = complex_linear_backward(x, w, y_true)
print(f"Updated w: {updated_w}")

```
Here, the `complex_linear_backward` function calculates the error gradient with respect to the weight `w`. In this example, the dL/dw_conj is zero because output (y_pred) is purely a function of w, not its conjugate w̄, in this case. Crucially, the weight is updated by subtracting the conjugate of dL_dw.total multiplied by the learning rate. It is crucial to note the conjugate of the total complex derivative, dL/dw.total is needed to obtain the steepest descent in complex space.

**Example 2: Simple Activation function**
The next example will involve a complex activation function, in this instance a sigmoid-type function. Note that the activation function will still need to be complex differentiatiable.

```python
import numpy as np
def complex_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def complex_sigmoid_derivative(z):
  sigmoid_z = complex_sigmoid(z)
  return sigmoid_z * (1-sigmoid_z)

def complex_activation_forward(w,x):
  z = x*w
  return complex_sigmoid(z)


def complex_activation_backward(x, w, y_true):
  y_pred = complex_activation_forward(w,x)
  dL_dy = 2 * (y_pred - y_true)
  # Compute the derivative of the sigmoid activation
  dz_dw = x
  dsigma_dz = complex_sigmoid_derivative(x * w)
  dL_dw = dL_dy * dsigma_dz * dz_dw
  dL_dw_conj = 0

  # Complex Gradient w/r w:
  dL_dw_total = dL_dw + np.conjugate(dL_dw_conj)

  # Update the weight
  learning_rate = 0.1
  updated_w = w - learning_rate * dL_dw_total.conjugate()

  return updated_w

# Example Use
x = 1 + 1j
w = 0.5 + 0.5j
y_true = 0.8 + 0.2j # ideal output.

updated_w = complex_activation_backward(x, w, y_true)
print(f"Updated w: {updated_w}")
```

In this case, the `complex_activation_backward` function computes the gradient with respect to the weight after application of the sigmoid function, and this gradient is used to update the complex weights.

**Example 3: Non-Zero Conjugate Derivative**

In the previous two examples, we had the conjugate derivative was zero. Here we will explicitly showcase what it means to have it non-zero. The example is contrived, to showcase what a non-zero conjugate derivative would imply, rather than a typical real world application. Assume we have an output of the form y_pred = (x * w) + (x*w).conjugate()

```python
import numpy as np
def complex_linear_plus_conjugate_forward(x,w):
  return x * w + (x*w).conjugate()

def complex_linear_plus_conjugate_backward(x,w,y_true):
  y_pred = complex_linear_plus_conjugate_forward(x, w)
  dL_dy = 2 * (y_pred - y_true)
  dL_dw = dL_dy * x
  dL_dw_conj = dL_dy * np.conjugate(x) # Because x * w.conjugate gives x.conjugate * w
  dL_dw_total = dL_dw + np.conjugate(dL_dw_conj)
  learning_rate = 0.1
  updated_w = w - learning_rate * dL_dw_total.conjugate()

  return updated_w

# Example Use
x = 1 + 1j
w = 0.5 + 0.5j
y_true = 2 + 0j # ideal output.

updated_w = complex_linear_plus_conjugate_backward(x, w, y_true)
print(f"Updated w: {updated_w}")
```
Here, `dL_dw_conj` is not zero because the output is explicitly a function of both *w* and its conjugate *w̄*. We compute the partial derivatives separately and then combine them correctly. The conjugate of the complex derivative is, again, needed for the update. This is crucial to note, the conjugate application when using total Wirtinger derivative.

In all examples, the update rule incorporates the conjugate of the *total* derivative, not just one component. The `conjugate()` operation is key for correctly adjusting the complex number to descend along the steepest gradient in complex space. If the total Wirtinger derivative is dL/dw_total, the update to w = w - learning_rate * np.conjugate(dL/dw_total).

**Resource Recommendations**

For a deeper understanding of complex-valued neural networks, I'd recommend exploring:

*   **Academic Papers:** Search for research papers on "complex-valued neural networks" or "complex backpropagation." These are generally found in IEEE or ACM-related databases. Papers often cover the mathematical derivations in detail, along with novel complex-valued architectures.

*   **Advanced Calculus Textbooks:** Look for textbooks that have sections on complex analysis. Wirtinger derivatives, Cauchy-Riemann equations, and complex differentiation are essential to understand.

*   **Numerical Computation Texts:** Textbooks focused on numerical methods might contain sections about optimization and complex-valued functions.
*   **Online Course Material:** Many online courses offer content on machine learning which cover advanced topics such as complex backpropagation, look for material that covers this specifically in a mathematical manner.

Understanding these resources will help build a strong mathematical foundation needed to handle the intricacies of complex numbers within the context of machine learning, going beyond what’s provided in generic or basic machine learning courses.
