---
title: "How does a decaying learning rate affect AdamW's weight decay?"
date: "2025-01-30"
id: "how-does-a-decaying-learning-rate-affect-adamws"
---
Weight decay, as implemented in the AdamW optimizer, is independent of the learning rate schedule. This separation is a crucial distinction between AdamW and its predecessor, Adam, and directly addresses the issue of weight decay interacting undesirably with adaptive learning rates.  My experience optimizing numerous deep learning models has repeatedly demonstrated the importance of this decoupling for stable and effective training, particularly in complex architectures.

Let's first clarify the mechanics of standard weight decay. In its traditional implementation (often found in optimizers like SGD with momentum), weight decay is applied *during* the parameter update step. The update rule looks conceptually like this:

```
w = w - lr * dw - lr * wd * w
```

Where `w` represents the model's weight, `lr` is the learning rate, `dw` is the gradient of the loss with respect to the weight, and `wd` is the weight decay coefficient. Notice how the weight decay term (`lr * wd * w`) is multiplied by the learning rate. This poses a problem when using adaptive learning rate methods like Adam, where the learning rate for each parameter can be significantly different and also fluctuate throughout training. Specifically, early in training, Adam often has larger effective learning rates which results in a much larger effect of weight decay than in later stages.  This dependence on the varying learning rate can make the weight decay an inconsistent regularizer, sometimes overly aggressive and at others too lenient, making it difficult to find suitable values for `wd`.  It's akin to attempting to control the speed of a car by changing the amount of weight on the accelerator rather than controlling the accelerator directly.

AdamW fundamentally changes how weight decay is applied, decoupling it from the learning rate. The update rule in AdamW is conceptually structured as:

```
w = w - lr * (dw)  # Standard Adam update
w = w * (1 - wd * lr)   # Weight decay applied after adam update
```

Here, the standard Adam update is performed first using the calculated gradient and the adaptive learning rate.  *After* this update, weight decay is applied directly to the parameters, using the current learning rate to control the *amount* of decay. This distinction, which may seem subtle, has a profound effect on the regularity of the regularization.

Now to directly address the question: how does a decaying learning rate affect AdamW's weight decay?

The key insight is that since the weight decay in AdamW is *multiplied by* the learning rate, a *decaying learning rate will directly reduce the magnitude of the weight decay applied to each weight*.  As the learning rate diminishes during training (according to a chosen schedule), the *effective* strength of the weight decay regularization decreases proportionally.  This is because in the second of the two conceptual steps, the weight is multiplied by `(1 - wd * lr)`. If `lr` is reduced, then `wd * lr` is reduced, and therefore the multiplier is closer to 1 (i.e., weight decay has less effect). In effect, weight decay is reduced as the optimizer converges to a solution. This is a different behaviour from Adam where weight decay can stay constant or even increase due to the adaptive learning rates.

Let's illustrate this with code. First, the standard Adam with weight decay that's dependent on the learning rate:

```python
import torch
import torch.optim as optim

# Hypothetical weights and gradients
w = torch.randn(10, requires_grad=True)
dw = torch.randn(10)

# Parameters for Adam
learning_rate = 0.01
weight_decay = 0.05

# Adam optimizer with weight decay in-line
optimizer_adam = optim.Adam([w], lr=learning_rate, weight_decay=weight_decay)

# Simulate an optimization step with an adaptive learning rate from the Adam optimizer
optimizer_adam.zero_grad()
w.grad = dw  # Assign computed gradients
optimizer_adam.step()

print("Adam Weight after update:", w)
print("Effective weight decay step with Adam:", learning_rate* weight_decay * w) # weight decay is multiplied by lr,
```

In this example, the `weight_decay` is baked into the `Adam` step. Now, let's show AdamW implementation with a similar process:

```python
import torch
import torch.optim as optim

# Hypothetical weights and gradients
w_adamw = torch.randn(10, requires_grad=True)
dw_adamw = torch.randn(10)

# Parameters for AdamW
learning_rate = 0.01
weight_decay = 0.05

# AdamW optimizer (requires separate weight decay application)
optimizer_adamw = optim.AdamW([w_adamw], lr=learning_rate, weight_decay=0)

# Simulate an AdamW optimization step
optimizer_adamw.zero_grad()
w_adamw.grad = dw_adamw
optimizer_adamw.step()

# Applying weight decay separately
with torch.no_grad():
  w_adamw.mul_(1 - weight_decay * learning_rate)  # Decoupled weight decay


print("AdamW Weight after update:", w_adamw)
print("Effective weight decay step with AdamW:", learning_rate* weight_decay * w_adamw) # weight decay is multiplied by lr,
```

In this second code example, you observe that the weight decay is applied *after* the Adam step, using the same learning rate. Now let’s use a learning rate scheduler to demonstrate how this will evolve when the learning rate decays:

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Initialize same weight again for comparison
w_adamw_decay = torch.randn(10, requires_grad=True)
dw_adamw_decay = torch.randn(10)

# Parameters
learning_rate = 0.01
weight_decay = 0.05
epochs = 5 # Number of training iterations

# AdamW optimizer
optimizer_adamw_decay = optim.AdamW([w_adamw_decay], lr=learning_rate, weight_decay=0)

# Learning rate scheduler with step decay
scheduler = StepLR(optimizer_adamw_decay, step_size=1, gamma=0.5) # Reduce learning rate by half every epoch.

print("Learning rates throughout training:")

for epoch in range(epochs):
  optimizer_adamw_decay.zero_grad()
  w_adamw_decay.grad = dw_adamw_decay
  optimizer_adamw_decay.step()

  with torch.no_grad():
      w_adamw_decay.mul_(1 - weight_decay * scheduler.get_last_lr()[0]) # Decoupled weight decay with current learning rate

  print("LR at epoch {}: {:.5f}".format(epoch+1, scheduler.get_last_lr()[0]))
  print("Weight Decay at epoch {}: {:.5f}".format(epoch+1, weight_decay * scheduler.get_last_lr()[0] ))

  scheduler.step()

print("AdamW Weight after multiple updates with decay:", w_adamw_decay)
```

This code illustrates how the learning rate is reduced over epochs, and, consequently, the effective weight decay is also reduced.  Notice how in each step, the `weight_decay` is multiplied by the *current* learning rate from the scheduler. The `get_last_lr` is used here to show the value of the learning rate at the point in time that weight decay is applied. As learning rate drops, so does the effective regularization strength applied by weight decay. The weight decay coefficient `wd` remains constant; only the *magnitude* of the decay diminishes.

The implications of this behaviour are significant.  A decaying learning rate applied in conjunction with AdamW effectively creates a dynamic regularization strategy. Initially, when the learning rate is high, the weight decay effect is pronounced, preventing the model's weights from becoming too large. As training progresses and the learning rate diminishes, the weight decay effect weakens, allowing the model to fine-tune its weights and reach a more precise solution. This behavior can be highly desirable in practice, providing a more controlled and less disruptive regularization of weights and allows the optimizer to converge properly towards a minima.

Recommendations for further investigation, without referring to specific links, include:

1.  **Study of optimization algorithms**: A rigorous understanding of different optimization techniques such as SGD, Adam, and their variants provides a fundamental basis for understanding these subtle differences. Focus on the mathematical formulations of the update rules.
2.  **Implementation details in libraries**: Deeply examine the source code of popular deep learning libraries (like PyTorch and TensorFlow) and compare how they implement each optimization algorithms. This provides important context for how the implementations effect the mathematical concepts.
3.  **Experimentation with different schedulers**: Explore the impact of various learning rate schedulers (e.g., step decay, cosine annealing, etc.) on the performance of AdamW. Try training the same model with different schedulers to understand which scheduler works best with the chosen optimizer.
4.  **Sensitivity analysis of hyperparameters**: Conduct extensive experiments to analyze how varying weight decay coefficients (`wd`) and the learning rate affects the training outcomes. By doing hyperparameter studies, we can fully understand the trade offs made when selecting hyper parameters.
5. **Research papers on AdamW:** Understanding the original motivations for the creation of AdamW provides key insights into the problem that it aims to address. The original paper describing the algorithm can provide important knowledge.
