---
title: "Why does mixed precision training produce NaN loss?"
date: "2025-01-30"
id: "why-does-mixed-precision-training-produce-nan-loss"
---
Mixed precision training, while offering significant speed and memory benefits in deep learning, can frequently result in NaN (Not a Number) loss values. This issue typically stems from the interplay between reduced numerical precision and the operations within the training process, particularly when dealing with extremely small or large values during backpropagation. Specifically, the transition from higher precision floating-point formats (FP32) to lower precision formats (FP16 or BF16) introduces a greater chance of underflow or overflow, leading to unstable gradients and ultimately, NaN loss.

The core issue arises because FP16 and BF16 have a smaller range of representable numbers compared to FP32. During the forward pass, activations and weights can potentially fall outside this representable range, resulting in either zero values or infinity values if not properly handled. However, it's during the backward pass, specifically when computing gradients, that these numerical limitations become critically apparent. Gradients often accumulate over numerous layers and become extremely small, sometimes approaching zero. When these small gradients are stored in FP16/BF16, they can underflow to zero. A zero gradient at any point during backpropagation essentially halts learning for those parameters. Similarly, large gradients can overflow to infinity, and once any part of the computation involves infinity, further mathematical operations will propagate NaN values across the entire graph.

Moreover, operations like division and exponentiation, which are common within neural network architectures (e.g., normalization layers, activation functions), are particularly sensitive to these numerical issues. When small denominators are encountered during division or large arguments for exponentiation are present, the chance of underflow or overflow is drastically amplified in lower precision formats.

To address these challenges, several techniques are commonly implemented during mixed precision training. Loss scaling is one of the most crucial, where the loss is multiplied by a scale factor before backpropagation. This scaling shifts the magnitude of gradients upward before they are converted to the lower precision format. The backpropagated gradients are then divided by the same scale factor at the optimizer step. This process aims to avoid underflow during gradient calculation and prevent them from being lost entirely. Dynamic loss scaling is typically preferred, where the scale factor is adapted throughout the training based on the magnitude of gradients. Gradient clipping is another vital technique, which limits the maximum magnitude of gradients, mitigating the impact of overflow errors. These strategies address the specific numerical limitations of lower precision formats and provide stable training.

Here are a few code examples with commentary illustrating these problems and their solutions, based on my experience optimizing training routines:

**Example 1: Basic Underflow Issue**

```python
import torch

def simulate_underflow(precision=torch.float32):
    x = torch.tensor(1e-5, dtype=precision)
    grad = torch.tensor(1e-5, dtype=precision)

    print(f"Original x (dtype={x.dtype}): {x}")
    print(f"Original grad (dtype={grad.dtype}): {grad}")
    
    if precision==torch.float16:
      x_fp16 = x.to(torch.float16)
      grad_fp16 = grad.to(torch.float16)
      print(f"x in fp16: {x_fp16}")
      print(f"grad in fp16: {grad_fp16}")

      x_updated_fp16 = x_fp16 - grad_fp16 #Simulate updating weight during training
      print(f"Updated x in fp16 : {x_updated_fp16}")
    
    x_updated = x - grad
    print(f"Updated x in float32: {x_updated}")

simulate_underflow()
simulate_underflow(torch.float16)

```

*   **Commentary:** In this example, I simulate a simplified update during training, where `x` represents a weight and `grad` represents a very small gradient. With `float32`, the update works correctly. With `float16` the value is so small that after converting to `float16`, it becomes zero. The subsequent update yields the same zero value. This is a basic demonstration of how lower precision can lose essential information due to underflow. In actual training, this would happen through many backprop steps and render training impossible.

**Example 2: Loss Scaling Implementation**

```python
import torch
def scaled_backward_pass(loss, scale_factor=1024, precision = torch.float32):
    
  scaled_loss = loss * scale_factor
  print(f"loss before scaling: {loss}")
  print(f"loss after scaling : {scaled_loss}")
  scaled_loss.backward()
  
  
  for name, param in model.named_parameters():
        if param.grad is not None:
            
            print(f"Original Grad of {name} in {param.dtype}: {param.grad.abs().max()}")
            
            if precision == torch.float16:
               param.grad = (param.grad.to(torch.float16) / scale_factor).to(torch.float32)
            else:
               param.grad = param.grad / scale_factor #Unscale
            print(f"Unscaled Grad of {name} : {param.grad.abs().max()}")
            


torch.manual_seed(42)

model = torch.nn.Linear(10,1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.tensor(1e-7)


scaled_backward_pass(loss)


model = torch.nn.Linear(10,1).float()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.tensor(1e-7)

scaled_backward_pass(loss, precision=torch.float16)

```

*   **Commentary:** Here, `loss` is an extremely small value. I demonstrate the core idea behind loss scaling: the loss is multiplied by a `scale_factor` before backpropagation. Before scaling, the `loss` might produce gradients so small that they vanish to zero when converting to `float16`. Loss scaling amplifies the gradients to a representable magnitude during the backward pass. Following the backward step, we perform unscaling by dividing by the same factor. It should be noted that both the `unscaled gradients` have a small magnitude, but, crucially the unscaled gradient from the `float16` case is not zero.

**Example 3: Gradient Clipping Implementation**

```python
import torch

def clipped_backward_pass(loss, max_norm=10, precision = torch.float32):
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            
            print(f"Original Grad of {name} in {param.dtype}: {param.grad.abs().max()}")
            if precision == torch.float16:
                param.grad = param.grad.to(torch.float16).clip(-max_norm, max_norm).to(torch.float32)
            else:
                param.grad = param.grad.clip(-max_norm, max_norm)
            print(f"Clipped Grad of {name} : {param.grad.abs().max()}")

torch.manual_seed(42)
model = torch.nn.Linear(10,1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.tensor(1e10) #Simulate very large loss

clipped_backward_pass(loss)

torch.manual_seed(42)
model = torch.nn.Linear(10,1).float()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.tensor(1e10)

clipped_backward_pass(loss, precision=torch.float16)

```

*   **Commentary:** In this example, the initial loss is now very large.  I demonstrate gradient clipping. After performing the backward pass, the gradients are clipped using `clamp` to ensure they do not exceed a set maximum value (here `max_norm=10`). This avoids very large values that will overflow in low precision and become `NaN`. It should be noted that both clipped gradients are bounded by the `max_norm` value.

In summary, the occurrence of NaN loss during mixed precision training is primarily due to the limitations in representing small and large numerical values with lower precision floating-point formats. Techniques such as loss scaling and gradient clipping are essential for mitigating these issues. The choice of when to use such tools can be complex, and, as I have found through experience, often requires experimentation and a deep understanding of the numerical aspects of both neural networks and backpropagation.

For further learning, I would recommend consulting material covering numerical methods, particularly those related to floating-point arithmetic. Additionally, resources discussing best practices for mixed precision training with specific deep learning frameworks can provide invaluable insights. Technical documentation for common deep learning frameworks that feature API support for mixed precision training are also beneficial. Understanding these resources provides a more comprehensive understanding of the causes and solutions related to NaN loss when implementing mixed-precision training.
