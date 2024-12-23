---
title: "Why does momentum's loss history reset to zero?"
date: "2024-12-23"
id: "why-does-momentums-loss-history-reset-to-zero"
---

Let's tackle this concept from a somewhat different angle, perhaps reflecting on a past project where this issue became painfully evident. I recall working on a system for optimizing machine learning model training, specifically focusing on stochastic gradient descent (sgd) variants. We incorporated momentum, intending to accelerate convergence, and initially, the behavior seemed somewhat... baffling. We observed that, under certain conditions, especially after parameter updates involving large gradients or after a model reset, the momentum term seemed to vanish – its 'loss history' effectively reset to zero.

At its core, momentum in sgd isn't about storing a past *loss* but rather a weighted average of past *gradient* vectors. Think of it not as a memory of errors but as a velocity vector guiding parameter updates. This vector, typically denoted as 'v', is updated at each iteration by combining a fraction of the current gradient with a fraction of the previous velocity vector. When we say 'loss history reset to zero,' what’s actually resetting is this accumulated velocity, 'v', not any historical measure of the loss function itself. Loss is always calculated based on the current parameters and the data; it’s not a term in momentum.

The standard update rule for sgd with momentum typically looks like this:

v(t+1) = β * v(t) + (1- β) * ∇L(θ(t))

θ(t+1) = θ(t) - α * v(t+1)

Where:

*   v(t) is the velocity vector at time step t.
*   β is the momentum coefficient (typically a value between 0 and 1, often close to 0.9).
*   ∇L(θ(t)) is the gradient of the loss function with respect to the parameters at time step t.
*   θ(t) is the parameters at time step t.
*   α is the learning rate.

The crux of the issue arises when the 'v' vector is explicitly or implicitly reset to zero. This reset can occur in several common scenarios, and understanding these is vital:

1.  **Explicit Initialization:** In many frameworks and libraries, 'v' is initialized to zero when model training starts. This is a logical default; it’s difficult to pre-fill it with anything meaningful before training. Consequently, the effect of momentum builds up progressively over the training process.

2.  **Model Re-Initialization:** If you re-initialize your model weights due to changes in the network architecture, for instance, or if you're starting training from scratch after a significant adjustment, the optimizer is usually also re-initialized, which again sets ‘v’ back to zero. This causes the momentum to vanish, because the historical average of gradients is no longer being carried forward from the prior training run.

3.  **Custom Implementation Bugs:** In custom momentum implementations, it’s easy to inadvertently include a condition that resets ‘v’ under specific, unintended scenarios. I've seen cases where a flawed conditional statement inadvertently zeroed the velocity vector, leading to behaviors we were not expecting. This is where careful code review and rigorous testing become essential.

To illustrate these points, consider the following simplified examples:

**Example 1: Basic Momentum Update in Python (using numpy)**

```python
import numpy as np

def sgd_with_momentum(theta, grad, v, beta, learning_rate):
    v_new = beta * v + (1 - beta) * grad
    theta_new = theta - learning_rate * v_new
    return theta_new, v_new

# initial parameters and optimizer state
theta = np.array([1.0, 2.0])
v = np.array([0.0, 0.0]) # momentum is initialized to zero
beta = 0.9
learning_rate = 0.1
grad1 = np.array([0.5, -0.2]) # arbitrary gradients
grad2 = np.array([0.3, 0.1])


#update 1
theta, v = sgd_with_momentum(theta, grad1, v, beta, learning_rate)
print(f"theta after update 1: {theta}, v after update 1: {v}")

#update 2
theta, v = sgd_with_momentum(theta, grad2, v, beta, learning_rate)
print(f"theta after update 2: {theta}, v after update 2: {v}")


# demonstrate what would happen if the momentum was reinitialized
v= np.array([0.0,0.0])
#update 3, momentum reset.
theta, v = sgd_with_momentum(theta, grad2, v, beta, learning_rate)
print(f"theta after update 3, v reset: {theta}, v after update 3: {v}")
```

In this example, the `v` vector starts at zero and progressively accumulates information about past gradients, demonstrating how momentum gradually builds up. We also demonstrate how a reset of v influences the next step.

**Example 2: Model Re-initialization (conceptual)**

Imagine a scenario where, after a phase of training, we alter the network architecture by adding layers. The existing parameters are no longer suitable for the new architecture, and it makes sense to randomly initialize the model’s weights again. When the optimizer is used again with momentum, the 'v' vector is also typically reset to zero because the past gradient information is no longer applicable to this new weight configuration. There’s no direct code for this example since it’s a conceptual scenario. But this demonstrates why momentum will reset if you reload a model.

**Example 3: Flawed Momentum Implementation**

```python
import numpy as np

def sgd_momentum_bugged(theta, grad, v, beta, learning_rate,reset_flag):
    if reset_flag:
        v = np.array([0.0, 0.0])  # BUG: inadvertent reset
    v_new = beta * v + (1 - beta) * grad
    theta_new = theta - learning_rate * v_new
    return theta_new, v_new

# initial parameters and optimizer state
theta = np.array([1.0, 2.0])
v = np.array([0.0, 0.0])
beta = 0.9
learning_rate = 0.1
grad1 = np.array([0.5, -0.2])
grad2 = np.array([0.3, 0.1])
reset_trigger = False #normal behaviour.


#update 1
theta, v = sgd_momentum_bugged(theta, grad1, v, beta, learning_rate,reset_trigger)
print(f"theta after update 1: {theta}, v after update 1: {v}")

#update 2
theta, v = sgd_momentum_bugged(theta, grad2, v, beta, learning_rate,reset_trigger)
print(f"theta after update 2: {theta}, v after update 2: {v}")


reset_trigger = True # bugged behaviour.
theta, v = sgd_momentum_bugged(theta, grad2, v, beta, learning_rate,reset_trigger)
print(f"theta after update 3, with bug, v reset in function {theta}, v after update 3: {v}")
```

This illustrates how an error in implementation that may introduce a reset of v can disrupt momentum accumulation. Here, we've introduced the `reset_flag`, which causes the velocity vector to be zeroed each time `True` is set, regardless of the momentum state, this was a common issue we found when a previous programmer was still getting used to vectorization techniques.

To understand this topic in more depth, I'd highly recommend looking at the original paper by Rumelhart, Hinton, and Williams on backpropagation, "Learning representations by back-propagating errors," (1986) it lays the foundation. Additionally, "Deep Learning" by Goodfellow, Bengio, and Courville has a comprehensive explanation of optimization techniques, including sgd variants like sgd with momentum. Also, the book “Numerical Optimization” by Jorge Nocedal and Stephen J. Wright contains detailed mathematical treatments of optimization algorithms and their implementations. These are foundational resources that helped me understand the underlying mechanics much more thoroughly.
By focusing on understanding the update rules of momentum and being aware of potential reset points, whether they're at the start, through model re-initialization, or from a coding bug, you can better control the behavior of your optimizers, which was something I had to learn the hard way during the past.
