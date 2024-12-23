---
title: "How do gradient ascent and gradient descent differ?"
date: "2024-12-23"
id: "how-do-gradient-ascent-and-gradient-descent-differ"
---

Alright, let’s tackle the distinction between gradient ascent and gradient descent. This is a foundational concept in optimization, and while the names sound very similar, their applications and goals are actually quite different. I've spent a fair bit of time implementing both over the years, particularly during my work on adaptive learning models back at [fictional company name], so I've got a practical perspective on this.

Essentially, both gradient ascent and gradient descent are iterative optimization algorithms that rely on the gradient of a function to find either a maximum or minimum value of that function, respectively. They both leverage the concept of a derivative, which points in the direction of the steepest change of a function at any given point. The key difference lies in what we're trying to achieve. Gradient descent aims to minimize a function—think of finding the lowest point in a valley. Gradient ascent, conversely, seeks to maximize it—akin to finding the peak of a mountain.

When we talk about gradient descent, we’re almost always referring to minimizing a *cost* or *loss* function. The objective is to iteratively adjust parameters in our model to get closer and closer to the function's minimum, where performance is typically better. This is the workhorse of many machine learning algorithms, particularly in training neural networks, where we adjust weights and biases to reduce the error between predicted and actual values. We take the gradient, which is the vector of partial derivatives of the cost function with respect to the parameters, and move in the opposite direction of that gradient. This is why it’s called 'descent' – we're descending the slope towards the lowest point.

Gradient ascent, on the other hand, aims to maximize a function. This often shows up in reinforcement learning, particularly when maximizing a reward function, or in adversarial settings, where we might want to maximize an objective while minimizing another in a zero-sum game. In this case, instead of moving *against* the gradient, we move *along* it. We're ascending the function's surface toward the maximum.

Let's clarify with some conceptual code. First, gradient descent:

```python
import numpy as np

def cost_function(x):
    # Example cost function: a parabola
    return x**2 + 2*x + 1

def gradient_descent(initial_x, learning_rate, iterations):
    x = initial_x
    for i in range(iterations):
        gradient = 2*x + 2 # Derivative of the cost function
        x = x - learning_rate * gradient # Move against the gradient
    return x, cost_function(x)

initial_x = 10
learning_rate = 0.1
iterations = 100
final_x, final_cost = gradient_descent(initial_x, learning_rate, iterations)
print(f"Final x: {final_x}, Final cost: {final_cost}")
```

Here, we have a simple quadratic function as our cost. The `gradient_descent` function iteratively updates `x` by moving a fraction (`learning_rate`) of the negative gradient direction. After 100 iterations, it should converge near the minimum.

Now, let’s illustrate gradient ascent. Let's imagine a reward function for a simple agent:

```python
import numpy as np

def reward_function(x):
  # Example reward function: an inverted parabola
  return -x**2 + 2*x + 10

def gradient_ascent(initial_x, learning_rate, iterations):
  x = initial_x
  for i in range(iterations):
    gradient = -2*x + 2 # derivative of the reward function
    x = x + learning_rate * gradient # Move along the gradient
  return x, reward_function(x)

initial_x = -10
learning_rate = 0.1
iterations = 100
final_x, final_reward = gradient_ascent(initial_x, learning_rate, iterations)
print(f"Final x: {final_x}, Final reward: {final_reward}")
```

In this case, our `reward_function` is an inverted parabola, and `gradient_ascent` iteratively updates 'x' by moving *along* the direction of the gradient, maximizing the reward function. Notice the sign change when updating 'x'; this is the core difference in the update rule of gradient ascent compared to gradient descent.

A common mistake is using gradient descent when you need ascent or vice versa. I've seen cases where someone tried to train a reinforcement learning agent using gradient descent on the reward function, only to see the agent consistently making actions that minimize the reward. We had to completely restructure the objective function and employ gradient ascent to get the agent learning properly. It's not merely a matter of changing a sign in the code; it reflects a fundamental difference in what we aim to optimize.

Finally, let's consider an example where gradient descent is used with multiple parameters, which is typical for neural network training:

```python
import numpy as np

def cost_function_multi(params):
    # Cost function involving two parameters, x and y
    x, y = params
    return x**2 + y**2 - 2*x + 2*y + 2

def gradient_descent_multi(initial_params, learning_rate, iterations):
    params = np.array(initial_params, dtype=float)
    for i in range(iterations):
        x,y = params
        grad_x = 2*x - 2 # Partial derivative wrt to x
        grad_y = 2*y + 2 # Partial derivative wrt to y
        gradient = np.array([grad_x, grad_y])
        params = params - learning_rate * gradient # Update all params
    return params, cost_function_multi(params)


initial_params = [5, -5]
learning_rate = 0.05
iterations = 100
final_params, final_cost = gradient_descent_multi(initial_params, learning_rate, iterations)

print(f"Final parameters (x, y): {final_params}, Final cost: {final_cost}")
```

Here, we optimize a cost function that is dependent on two parameters using gradient descent. Note that the gradient is now a vector of partial derivatives for each parameter. We still use the rule of moving against the gradient, but now we move in the opposite direction of the multidimensional gradient.

For a deeper dive into the theoretical underpinnings of gradient-based optimization, I strongly recommend the book *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. It covers these topics in considerable mathematical detail, going way beyond the basics. Also, *Numerical Optimization* by Jorge Nocedal and Stephen Wright is an excellent reference for anyone looking into more advanced optimization techniques. Another useful paper is "An Overview of Gradient Descent Optimization Algorithms" by Sebastian Ruder. These resources will provide a more comprehensive and rigorous treatment of both gradient descent and gradient ascent.

Ultimately, understanding when to use gradient ascent versus gradient descent is crucial, and as you gain more experience, it will become second nature. It isn’t just about code; it's about comprehending the nature of your objective function and whether you're seeking to minimize a cost or maximize a reward. The devil, as they say, is in the direction of the gradient.
