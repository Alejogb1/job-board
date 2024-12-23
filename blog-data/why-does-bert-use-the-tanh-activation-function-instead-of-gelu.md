---
title: "Why does BERT use the tanh activation function instead of GELU?"
date: "2024-12-23"
id: "why-does-bert-use-the-tanh-activation-function-instead-of-gelu"
---

Let's tackle this head-on. I remember a particularly thorny project a few years back, involving a custom language model for medical text. We were experimenting with various activation functions, and the seemingly simple choice between tanh and GELU became surprisingly pivotal. The conventional wisdom at the time leaned heavily on ReLU and its variants for efficiency, but that project really highlighted how crucial it is to understand the nuances of activation functions within specific architectures like BERT.

The short answer to why BERT originally used tanh instead of GELU is rooted in historical context and architectural considerations, not necessarily a categorical superiority of tanh in all cases. BERT, released in 2018, predates the widespread adoption and empirical validation of GELU. While GELU is now the go-to for many modern transformers, including later iterations of some BERT-inspired models, its dominance is a more recent phenomenon.

At BERT's inception, tanh was a well-understood, widely-used activation function in recurrent neural networks and, crucially, within the pre-existing transformer model upon which BERT was built. The transformer architecture, initially designed for sequence-to-sequence tasks, relied heavily on the self-attention mechanism, which was initially paired with tanh or sigmoid activations within feed-forward networks inside the attention blocks. Moving to a different activation function required significant experimentation, and the initial BERT paper focused primarily on innovations in pre-training and the masking strategy, rather than a wholesale architectural redesign. Remember, the primary goal was a highly impactful model trained on massive data; changing too many variables upfront would have added significant complexity to the project.

Tanh, or the hyperbolic tangent function, outputs values between -1 and 1. This output range has specific properties useful in transformers. The zero-centered nature can help with gradient flow by preventing the accumulation of bias towards positive or negative values. It's also differentiable, a fundamental requirement for backpropagation, and while it saturates towards its extremes (leading to vanishing gradients if not used judiciously), the range remains useful. In comparison, ReLU has zero gradient for negative inputs which can kill the gradients flow in some networks, while its variants (like leaky ReLU) were not tested as extensively in deep transformers back then as tanh.

GELU, or Gaussian Error Linear Units, on the other hand, introduces a stochastic element that’s beneficial for larger and deeper networks by introducing noise to the activation behavior, thereby avoiding the dreaded dead neuron problem, while maintaining the non-linearity. It works by calculating the input multiplied by a cumulative normal distribution function. The output is non-linear and not bounded, which can lead to better performance than tanh in some scenarios, especially those requiring more flexible feature representation. However, this more complex behavior came at a computational cost and required a different tuning strategy for effective utilization.

To illustrate these differences, consider a simplified feed-forward network layer:

```python
import torch
import torch.nn as nn

# Simple example of a linear layer followed by tanh activation
class TanhLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.linear(x))

# Simple example of a linear layer followed by GELU activation
class GeluLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(x))

# Example usage:
input_size = 10
hidden_size = 20
batch_size = 5
random_input = torch.randn(batch_size, input_size)

tanh_layer = TanhLayer(input_size, hidden_size)
gelu_layer = GeluLayer(input_size, hidden_size)

output_tanh = tanh_layer(random_input)
output_gelu = gelu_layer(random_input)

print("Output using Tanh:")
print(output_tanh)
print("Output using GELU:")
print(output_gelu)

```

In this basic example, you can see that the tanh output is bounded within -1 and 1, while GELU exhibits a wider range. It's crucial to understand that the difference is amplified within a larger transformer network with multiple layers of feed forward networks and attention mechanisms.

Now, regarding the impact on gradient flow, let's inspect it with a more nuanced example:

```python
import torch
import torch.nn as nn

# Tanh activation layer with manual gradient calculation for a single input
class TanhLayerManualGrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        output = torch.tanh(x * self.weight + self.bias)
        return output

    def manual_backward(self, output_gradient, input_val):
        # derivative of tanh(x) is 1 - tanh(x)^2
        tanh_grad = 1 - torch.tanh(input_val* self.weight + self.bias)**2

        # derivative of linear component
        linear_grad = input_val
        weight_grad = output_gradient * tanh_grad * linear_grad
        bias_grad = output_gradient * tanh_grad
        return weight_grad, bias_grad

# GELU activation layer with manual gradient calculation for a single input
class GeluLayerManualGrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        output = 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.tensor(3.14159)) * (x* self.weight + self.bias + 0.044715 * (x* self.weight + self.bias)**3)))
        return output
    def manual_backward(self, output_gradient, input_val):

        # derivative of GELU using the approximation
        approx_gelu_derivative = 0.5 + 0.5* torch.tanh(torch.sqrt(2 / torch.tensor(3.14159)) * (input_val* self.weight + self.bias + 0.044715 * (input_val* self.weight + self.bias)**3)) + (torch.sqrt(2 / torch.tensor(3.14159)) / 2 * (1 - (torch.tanh(torch.sqrt(2 / torch.tensor(3.14159)) * (input_val * self.weight + self.bias + 0.044715 * (input_val * self.weight + self.bias)**3)) ** 2)) * (1+3* 0.044715*(input_val* self.weight + self.bias)**2))
        # derivative of the linear function
        linear_grad = input_val
        weight_grad = output_gradient * approx_gelu_derivative * linear_grad
        bias_grad = output_gradient * approx_gelu_derivative
        return weight_grad, bias_grad
# Example usage:
input_value = torch.randn(1)
gradient_value = torch.tensor([1.0])

tanh_layer = TanhLayerManualGrad()
gelu_layer = GeluLayerManualGrad()

output_tanh = tanh_layer(input_value)
output_gelu = gelu_layer(input_value)

weight_grad_tanh, bias_grad_tanh = tanh_layer.manual_backward(gradient_value, input_value)
weight_grad_gelu, bias_grad_gelu = gelu_layer.manual_backward(gradient_value, input_value)

print(f"Tanh Output: {output_tanh.item()}")
print(f"Tanh Weight Gradient: {weight_grad_tanh.item()}")
print(f"Tanh Bias Gradient: {bias_grad_tanh.item()}")

print(f"GELU Output: {output_gelu.item()}")
print(f"GELU Weight Gradient: {weight_grad_gelu.item()}")
print(f"GELU Bias Gradient: {bias_grad_gelu.item()}")
```

In this simplified layer setup we can analyze, even if crudely, how the gradients propagate using these activation functions. Note that for real-world purposes, pytorch’s autograd module handles these gradients efficiently, but this manual approach demonstrates how each function’s unique properties contribute to gradient behavior. Specifically notice that the gradients of `tanh` are bound between 0 and 1, while the gradients of `GELU` has a greater variability, allowing them to explore a wider parameter space.

Finally, to solidify this point, here’s a more complex snippet showing how to incorporate the activation functions within a standard transformer block. This is closer to how they would have been originally used in BERT.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, intermediate_dim, activation_function = "tanh"):
      super().__init__()
      self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
      self.linear1 = nn.Linear(embedding_dim, intermediate_dim)
      self.linear2 = nn.Linear(intermediate_dim, embedding_dim)
      self.norm1 = nn.LayerNorm(embedding_dim)
      self.norm2 = nn.LayerNorm(embedding_dim)
      if activation_function == "tanh":
         self.activation = nn.Tanh()
      elif activation_function =="gelu":
        self.activation = nn.GELU()
      else:
         raise ValueError("Invalid activation function, choose between tanh or gelu")
    def forward(self, x):
      # Attention mechanism
      attention_output, _ = self.attention(x,x,x)
      x = self.norm1(x+attention_output)

      # Feed forward network
      intermediate_output = self.linear1(x)
      activated_output = self.activation(intermediate_output)
      output_ffn = self.linear2(activated_output)

      x = self.norm2(x+output_ffn)
      return x

embedding_dim = 512
num_heads = 8
intermediate_dim = 2048
seq_len = 30
batch_size = 10

random_input = torch.randn(seq_len, batch_size, embedding_dim)

tanh_transformer = TransformerBlock(embedding_dim, num_heads, intermediate_dim, activation_function = "tanh")
gelu_transformer = TransformerBlock(embedding_dim, num_heads, intermediate_dim, activation_function = "gelu")

output_tanh = tanh_transformer(random_input)
output_gelu = gelu_transformer(random_input)

print("Output using Tanh")
print(output_tanh.shape)
print("Output using GELU")
print(output_gelu.shape)

```
Here we see two transformers blocks implemented with different activation functions, demonstrating how straightforward it is to incorporate these activation functions within existing architectures.

The takeaway here isn't that tanh is *better* or *worse* than GELU— but that each has different properties that affect the performance, training stability and overall architecture design. Early on, the transformer architecture leaned on what was familiar. Now, with broader research and the benefits of GELU becoming apparent, most modern variants of BERT are using it instead, or even other more recent alternatives. The choice isn't static and reflects an ongoing process of refinement in the field.

If you're looking to dive deeper into this, I'd recommend starting with the original transformer paper, "Attention is All You Need," by Vaswani et al. Then, for a good understanding of activation functions, the "Deep Learning" book by Goodfellow, Bengio, and Courville provides a comprehensive background. Finally, the original GELU paper by Hendrycks and Gimpel provides a detailed look at GELU. These resources will put the historical and theoretical context behind these activation functions. It will also highlight why models sometimes rely on one particular function over another in certain circumstances. It was a practical choice, not necessarily an inherent superiority.
