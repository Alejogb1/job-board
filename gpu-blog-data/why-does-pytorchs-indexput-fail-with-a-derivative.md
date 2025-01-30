---
title: "Why does PyTorch's `index_put_` fail with a 'derivative not implemented' error for indices?"
date: "2025-01-30"
id: "why-does-pytorchs-indexput-fail-with-a-derivative"
---
Direct access manipulation of tensors in PyTorch, particularly using `index_put_`, while highly performant for in-place updates, introduces a significant constraint concerning backpropagation when the provided indices are the result of a computation within the computational graph. This limitation stems from the fact that standard gradient computation algorithms rely on the notion of differentiable operations. When indices are themselves tensors dependent on upstream computations, their derivatives with respect to the original input are generally not defined, making automatic differentiation infeasible.

Specifically, the error "derivative not implemented" arises when `index_put_` receives indices that are themselves outputs of other PyTorch operations, that are not constants or part of input tensors, and on which the computational graph depends for backpropagation. PyTorch’s autograd engine meticulously tracks differentiable operations to calculate gradients; however, the scattering operation implicit within `index_put_`, when applied with dynamic, variable-dependent indices, lacks a well-defined derivative that could be effectively implemented. In essence, the chain rule, the fundamental principle of backpropagation, breaks down as the derivative of the indexing function with respect to a tensor-valued index does not exist in a general and practical manner. Consequently, PyTorch's autograd system raises an exception to prevent silent errors and ill-defined computations.

To illustrate this, consider the core functionality of `index_put_`. This in-place operation takes a base tensor, a set of indices, and a source tensor (or a single value) as input. It modifies the base tensor at the locations specified by the indices, inserting the values from the source tensor. If these indices are pre-determined and static, meaning they do not result from PyTorch operations, then `index_put_` operates flawlessly within the backward pass. The backward pass then directly backpropagates the gradient at the updated locations with the help of stored positional information. However, when these indices become outputs of another computation within the graph, for example, by a `torch.argmax`, `torch.argsort` or any other tensor manipulation which might require backpropagation, the process is complicated, as the gradient at these locations must be traced back through the dynamic index generation, which introduces a discrete behavior that is not differentiable.

The crucial aspect to understand is that `index_put_` is not inherently a non-differentiable function. The lack of a derivative only occurs when the input indices themselves are a function of parameters that require gradient updates. When provided with constant indices, the operation is differentiable, as it performs a specific scatter operation, which is a linear operation in the locations where data is written. If no gradients are required on indices that have been computed within the computational graph then the tensors must be detached before being passed to `index_put_`, via `detach()`, to prevent the aforementioned error.

Let's explore this with concrete code examples.

**Example 1: Static Indices, Functional Backpropagation**

Here, the indices are fixed. `index_put_` functions within backpropagation without an issue.

```python
import torch

# Initialize tensor
x = torch.zeros(5, requires_grad=True)

# Define static indices
indices = torch.tensor([1, 3])

# Source values
src = torch.tensor([1.0, 2.0])

# Perform in-place update
x.index_put_((indices,), src)

# Loss function
loss = x.sum()

# Backpropagate
loss.backward()

print("Tensor after update:", x)
print("Gradient of x:", x.grad)
```

In this example, the `indices` are directly defined as a tensor, they are not the output of another computation. Thus, `index_put_` executes correctly, and backpropagation can occur as the operation is differentiable with respect to the source and the base tensor (x). The gradient with respect to `x` at indices 1 and 3 is 1, as all entries in `x` contribute to the scalar loss.

**Example 2: Dynamic Indices, Derivative Not Implemented**

In this scenario, the indices are computed via `torch.argmax`, introducing a non-differentiable discrete operation into the chain of computations.

```python
import torch

# Initialize tensor with requires_grad flag
x = torch.randn(5, requires_grad=True)

# Generate tensor to compute the index
y = torch.randn(5, requires_grad=True)

# Compute dynamic indices using argmax
indices = torch.argmax(y).reshape(1)

# Source values
src = torch.tensor([1.0])

try:
    # Attempt in-place update
    x.index_put_((indices,), src)

    # Loss function
    loss = x.sum()

    # Backpropagate
    loss.backward()
    
except RuntimeError as e:
    print("Error:", e)

print("Tensor after update:", x)
print("Gradient of x:", x.grad)
```

Here, `torch.argmax` provides the indices, it has parameters with gradients, and thus it becomes a part of the computational graph requiring differentiation. When PyTorch encounters `index_put_` with these calculated indices, it raises the "derivative not implemented" exception. The backpropagation cannot compute gradients through the indexing operation that depends on a non-differentiable discrete function, as the `argmax` outputs a single, fixed integer, and the chain rule for backpropagation fails. The error occurs during the backwards pass, and the tensor `x` will not have its gradient updated or computed, even if we were able to update it using the method.

**Example 3: Detaching Indices for Backpropagation**

This code shows that if the computed indices do not require gradients with respect to them, the tensor must be detached from the graph.

```python
import torch

# Initialize tensor with requires_grad flag
x = torch.randn(5, requires_grad=True)

# Generate tensor to compute the index
y = torch.randn(5, requires_grad=True)

# Compute dynamic indices using argmax, detach the gradient for backpropagation
indices = torch.argmax(y).detach().reshape(1)

# Source values
src = torch.tensor([1.0])

# Perform in-place update
x.index_put_((indices,), src)

# Loss function
loss = x.sum()

# Backpropagate
loss.backward()

print("Tensor after update:", x)
print("Gradient of x:", x.grad)
```

In this corrected example, `indices` are still computed via `torch.argmax(y)`, but `detach()` has been added before applying `reshape`. By calling detach the tensor `indices` no longer requires gradients and thus backpropagation does not need to trace the operation. The `index_put_` operation is no longer in an non-differentiable state, as it uses a constant and fixed location in `x` for writing the source.

Several alternative strategies exist for situations where dynamic indexing and backpropagation are both necessary. One approach is to avoid in-place modification and use more explicit scattering operations that are differentiable, such as `torch.scatter_add` or building your own, using operations that are supported by `autograd`. If one index is selected for updates it might be possible to implement that operation using more explicit multiplication operations of a one-hot vector to avoid index\_put\_. The precise strategy will vary based on the intended application and the structure of your computational graph.

For further exploration of PyTorch’s autograd system and tensor manipulation, I recommend reviewing the official PyTorch documentation, particularly sections concerning autograd mechanics and tensor operations. Additionally, research papers on differentiable programming and libraries specializing in differentiable data manipulation could provide valuable insights. Consider also studying open-source projects implementing complex neural network architectures, noting how these projects manage similar dynamic indexing situations with backpropagation.
