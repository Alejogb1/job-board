---
title: "How can a PyTorch loss function depend on the network's input gradient?"
date: "2025-01-30"
id: "how-can-a-pytorch-loss-function-depend-on"
---
Gradient-based loss function manipulation in PyTorch, while not a direct feature, can be achieved by employing autograd mechanisms and strategically crafting custom loss computations. Specifically, the gradient of a network’s input, rather than a weight or parameter gradient, is what we are after. This is less commonly used than parameter gradients for model learning, but it provides a unique avenue for incorporating input-specific properties into training. I’ve found this approach valuable for certain types of adversarial training and image manipulation tasks.

The core challenge lies in the fact that standard loss functions in PyTorch operate on network output and ground truth. To make the loss depend on the *input* gradient, we must first compute this gradient. This requires a carefully orchestrated sequence involving backpropagation and gradient retention, followed by the manipulation of that gradient data within our custom loss function. Crucially, the input tensor must have `requires_grad=True` enabled. Failing to set this flag means no gradient calculation will propagate to the input itself, rendering this technique ineffective.

The process generally unfolds as follows: first, the network forward pass is executed with the input tensor having `requires_grad=True`. Following this, we compute our output. A first backward pass is performed, but *not* on a traditional output-ground truth loss. Instead, we call `.backward()` on the *output* tensor itself, rather than on a scalar loss calculated against the ground truth; this will compute the input gradients. The `retain_graph=True` argument here is critical, because it retains the intermediate computations. We then extract the input gradients using `.grad` attribute on the input tensor. After extracting, we perform the standard loss computation based on network output and ground truth. Finally, we use the input gradient we extracted to adjust the loss we just computed. Backpropagation of this adjusted loss then updates model parameters as expected.

Let's consider some practical code examples to better demonstrate the approach.

**Example 1: Regularizing Input Variation using Norm of Gradient**

This first example demonstrates a case where we penalize large changes in the input space. Let’s imagine you want to constrain the network to map similar inputs to similar outputs. In that case, you may need the network's input to react slowly to minor changes. To enforce this, we can penalize large input gradients, effectively slowing down the speed at which the input affects the network output. This is essentially an L2 regularization applied to the input gradients, something that cannot be directly done within the loss layer since input gradients are not a part of model parameters.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize network, inputs, optimizer and dummy targets
model = SimpleNet()
input_tensor = torch.randn(1, 10, requires_grad=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
target_tensor = torch.tensor([1.5])

# Forward pass
output = model(input_tensor)

# Perform backward pass to calculate the input gradient
output.backward(retain_graph=True)
input_grad = input_tensor.grad.clone()
input_tensor.grad.zero_() # Clear the input gradient.

# Calculate output based loss, standard mean squared error loss.
output_loss = nn.MSELoss()(output, target_tensor)

# Calculate input gradient regularization term
input_grad_loss = torch.norm(input_grad)

# Combine the two losses
combined_loss = output_loss + 0.01 * input_grad_loss

# Backpropagate the total loss and update the model
optimizer.zero_grad()
combined_loss.backward()
optimizer.step()

print(f"Combined loss: {combined_loss.item():.4f}")
```
In this example, after the initial forward pass, the backpropagation is done on the *output* with `retain_graph=True`, computing the input gradient. We then zero it since we are going to do a second backpropagation later in the script. After computing the standard output loss, the L2 norm of the input gradient is calculated to form the input_grad_loss. This term is added to the output loss and we finally perform the second backward pass using the *combined_loss*. The `0.01` scaling factor here represents the strength of the regularization. A larger value would force the network to be more resistant to changes in the input, while small values would essentially ignore the input gradient.

**Example 2: Input Gradient Direction Matching**

The second example shows how one might use the input gradient to match the direction of the change in input. This is something I have used for image manipulation, where you want to guide the input towards a certain target direction based on how the network would change its output. This is akin to gradient ascent in image space, only now we are not directly changing the input, but training the network to be more receptive to our change direction.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize network, inputs, optimizer and dummy targets
model = SimpleNet()
input_tensor = torch.randn(1, 10, requires_grad=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
target_tensor = torch.tensor([1.5])

# Define a target gradient direction. In a real use case,
# this might come from a another processing step.
target_gradient_direction = torch.randn(1,10)

# Forward pass
output = model(input_tensor)

# Perform backward pass to calculate input gradient
output.backward(retain_graph=True)
input_grad = input_tensor.grad.clone()
input_tensor.grad.zero_()

# Calculate output based loss
output_loss = nn.MSELoss()(output, target_tensor)


# Calculate cosine similarity between input gradient and target direction
grad_similarity = torch.nn.functional.cosine_similarity(
    input_grad.flatten(),
    target_gradient_direction.flatten(),
    dim=0
)

# Modify the loss based on the cosine similarity.
input_grad_loss = 1 - grad_similarity

# Combine the two losses
combined_loss = output_loss +  0.05 * input_grad_loss

# Backpropagate and update the model
optimizer.zero_grad()
combined_loss.backward()
optimizer.step()

print(f"Combined loss: {combined_loss.item():.4f}")
```

Here, we introduce `target_gradient_direction`. The `input_grad_loss` is now computed based on the cosine similarity between the computed input gradient and this target direction, pushing the input gradient to align with our desired direction. Again, the 0.05 scaling factor controls the importance we give to the input gradient relative to the output loss. If the input gradient does not align with the direction we want, the similarity will decrease, and the input gradient loss term will increase.

**Example 3: Adversarial Input Attack**

This example will demonstrate how one might use input gradients to develop an adversarial input attack. Instead of a fixed direction, the idea here is to perturb the input such that the prediction is pushed away from the target. While it does not directly change the input, the network is trained to be susceptible to these changes, which can be used in a second-pass attack.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize network, inputs, optimizer and dummy targets
model = SimpleNet()
input_tensor = torch.randn(1, 10, requires_grad=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
target_tensor = torch.tensor([1.5])

# Forward pass
output = model(input_tensor)

# Perform backward pass to calculate input gradient
output.backward(retain_graph=True)
input_grad = input_tensor.grad.clone()
input_tensor.grad.zero_()

# Calculate output based loss
output_loss = nn.MSELoss()(output, target_tensor)


# Project the input gradient onto the target direction
perturbed_input = input_tensor - 0.1 * input_grad.sign()
perturbed_output = model(perturbed_input)
input_grad_loss = nn.MSELoss()(perturbed_output, target_tensor)


# Combine the two losses
combined_loss = output_loss + 0.1 * input_grad_loss

# Backpropagate the total loss and update the model
optimizer.zero_grad()
combined_loss.backward()
optimizer.step()

print(f"Combined loss: {combined_loss.item():.4f}")
```

In this case, we first compute the input gradient. Then we compute a small change in input by `input_tensor - 0.1 * input_grad.sign()`. We pass the perturbed input through the model and compute the loss against the target. By training on the `combined_loss` using a small scaling factor, this pushes the network to produce very different outcomes on this slightly perturbed input. While this does not constitute a complete adversarial attack, this is an example of how one could use the input gradient to make the network more vulnerable to such an attack. The `sign` function is used to have the perturbations either increase or decrease each dimension independently.

These examples demonstrate a fundamental pattern: compute the input gradient by propagating back through the *output* tensor, extract the gradient, use the gradient in a custom loss term along with a standard output-based loss, and finally update the network parameters by using the combined loss.

For a deeper understanding, I would recommend exploring literature on adversarial attacks and regularization techniques involving gradients. Review publications that delve into input-space gradient manipulation. Furthermore, a solid understanding of PyTorch’s autograd mechanism through the official PyTorch documentation is very helpful. The more you experiment, the more this technique becomes intuitive.
