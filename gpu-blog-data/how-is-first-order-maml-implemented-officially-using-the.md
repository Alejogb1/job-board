---
title: "How is first-order MAML implemented officially using the higher library in PyTorch?"
date: "2025-01-30"
id: "how-is-first-order-maml-implemented-officially-using-the"
---
First-order Model-Agnostic Meta-Learning (MAML) using the `higher` library in PyTorch focuses on simplifying the implementation of gradient-based meta-learning, specifically avoiding the manual creation of computational graphs for inner loop differentiation. The central challenge in MAML is calculating the second-order gradients needed for meta-optimization. `higher` addresses this by providing a functional interface that treats model parameters as regular tensors, allowing for direct gradient manipulation via PyTorch's automatic differentiation. This eliminates the need for manual graph tracking during the inner loop, drastically streamlining the implementation.

In essence, MAML aims to learn a good initialization for a model, such that fine-tuning it on a small number of examples from a new, but related, task leads to fast and accurate performance. The training procedure involves two loops: an inner loop for adapting the model to a specific task using task-specific data and an outer loop that updates the initial parameters based on the adapted model's performance on the same task, but a different dataset. The `higher` library excels in expressing this inner loop. Instead of creating new models for every update, which would break PyTorch's graph, `higher` provides a functional perspective. This means we create a copy of the original model parameters and treat them as regular tensors. We then perform the inner optimization steps on these parameters, which is directly differentiable and doesn't affect the original parameters.

My past work involved implementing variations of MAML for few-shot image classification and reinforcement learning tasks, where the complexities of manual second-order gradient calculations often became a major bottleneck. Using `higher` has significantly reduced implementation time and debugging effort.

Let’s consider a basic setup for illustrating first-order MAML with `higher`. We'll use a simple linear regression problem for clarity. Imagine we are tackling a meta-learning task where our goal is to learn an initialization of the model that can adapt to new linear functions after just one or few gradient updates.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from higher import innerloop_ctx

# 1. Define the Model
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# 2. Define Meta-Training Hyperparameters
input_size = 1
output_size = 1
meta_lr = 0.001 # Learning rate for the outer loop
task_lr = 0.01  # Learning rate for the inner loop
inner_steps = 1  # Number of inner loop updates

# 3. Instantiate the Meta-Model and Optimizer
model = LinearModel(input_size, output_size)
meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
loss_fn = nn.MSELoss()

# Dummy Data Generation
def generate_task(num_samples):
    true_weight = torch.randn(1, 1)
    true_bias = torch.randn(1)
    x = torch.randn(num_samples, 1)
    y = (x @ true_weight + true_bias)
    return x, y

# 4. Meta-Training Loop
num_meta_iterations = 1000

for meta_iter in range(num_meta_iterations):
    meta_optimizer.zero_grad()
    meta_loss = 0

    for task_idx in range(5): # Consider 5 tasks per meta-batch
        # Generate new task
        x_train, y_train = generate_task(10)
        x_test, y_test = generate_task(5)


        with innerloop_ctx(model, optim.SGD, lr=task_lr) as (fmodel, diffopt):
            # Inner loop adaptation
            for step in range(inner_steps):
                pred_train = fmodel(x_train)
                inner_loss = loss_fn(pred_train, y_train)
                diffopt.step(inner_loss)

            # Evaluate the adapted model
            pred_test = fmodel(x_test)
            task_loss = loss_fn(pred_test, y_test)
            meta_loss += task_loss


    # Outer loop update
    meta_loss /= 5 # Average meta-loss over tasks
    meta_loss.backward()
    meta_optimizer.step()

    if meta_iter % 100 == 0:
      print(f"Meta Iter {meta_iter}, Meta Loss: {meta_loss.item()}")
```

This first code example illustrates a simplified meta-training loop with a linear model. The core of using `higher` is in the `innerloop_ctx` context manager.  This context manager takes our base `model` as an argument, along with the inner optimization algorithm (SGD here) and learning rate. Crucially, it provides us with two objects: `fmodel`, which holds a functional copy of the model's parameters, and `diffopt`, a modified optimizer that works directly with the functional parameters. The inner loop performs the task-specific adaptation by stepping the functional optimizer against the inner loss. After this adaptation, we evaluate the adapted functional model on the test set, and the resulting `task_loss` is added to `meta_loss`.  Importantly, this entire process is differentiable, allowing `meta_loss` to be backpropagated through the computation graph, and updating the original model parameters through `meta_optimizer`.

Let's enhance this with a more complex model, a simple multi-layer perceptron (MLP), and a scenario with sinusoidal wave regression. Here, the meta-learning task involves learning a model that can adapt to new sine waves with varying amplitudes and frequencies.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from higher import innerloop_ctx
import math

# 1. Define the Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# 2. Define Meta-Training Hyperparameters
input_size = 1
hidden_size = 64
output_size = 1
meta_lr = 0.001
task_lr = 0.01
inner_steps = 5
num_meta_iterations = 1000

# 3. Instantiate the Meta-Model and Optimizer
model = MLP(input_size, hidden_size, output_size)
meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
loss_fn = nn.MSELoss()


# 4. Task generation for sinusoid regression
def generate_sin_task(num_samples):
  amplitude = torch.rand(1)*5 # random amplitude between 0 and 5
  frequency = torch.rand(1) * 2 + 0.1  # random frequency between 0.1 and 2.1
  x = torch.rand(num_samples, 1)* 2 * math.pi # input space [0, 2pi]
  y = amplitude * torch.sin(frequency*x)
  return x, y

for meta_iter in range(num_meta_iterations):
    meta_optimizer.zero_grad()
    meta_loss = 0

    for task_idx in range(5): # 5 tasks per meta-batch
        # Generate task specific data
        x_train, y_train = generate_sin_task(10)
        x_test, y_test = generate_sin_task(5)

        with innerloop_ctx(model, optim.SGD, lr=task_lr) as (fmodel, diffopt):
            # Inner loop adaptation
            for step in range(inner_steps):
                pred_train = fmodel(x_train)
                inner_loss = loss_fn(pred_train, y_train)
                diffopt.step(inner_loss)

            # Evaluate the adapted model
            pred_test = fmodel(x_test)
            task_loss = loss_fn(pred_test, y_test)
            meta_loss += task_loss

    # Outer loop update
    meta_loss /= 5
    meta_loss.backward()
    meta_optimizer.step()

    if meta_iter % 100 == 0:
        print(f"Meta Iter {meta_iter}, Meta Loss: {meta_loss.item()}")

```

This second example showcases how the first-order MAML concept can be applied with a slightly more complex model, an MLP, and a different type of task generation, sinusoid regression. The underlying logic remains the same, leveraging `innerloop_ctx` for its functional optimization. The `generate_sin_task` function now creates sinusoidal functions with varying amplitudes and frequencies. The MLP needs to learn how to adapt to these variations. The meta-loss is again aggregated over multiple tasks and backpropagated through `higher`'s functional operations.

Finally, let's demonstrate meta-batching, where multiple inner loops are processed in parallel by using `higher`'s functional parameter copies of the meta-model. This will allow us to better exploit parallel hardware.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from higher import innerloop_ctx
import math

# 1. Define the Model (Same as before)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# 2. Define Meta-Training Hyperparameters
input_size = 1
hidden_size = 64
output_size = 1
meta_lr = 0.001
task_lr = 0.01
inner_steps = 5
num_meta_iterations = 1000

# 3. Instantiate the Meta-Model and Optimizer
model = MLP(input_size, hidden_size, output_size)
meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
loss_fn = nn.MSELoss()

# 4. Task generation (same as before)
def generate_sin_task(num_samples):
  amplitude = torch.rand(1)*5
  frequency = torch.rand(1) * 2 + 0.1
  x = torch.rand(num_samples, 1)* 2 * math.pi
  y = amplitude * torch.sin(frequency*x)
  return x, y


for meta_iter in range(num_meta_iterations):
    meta_optimizer.zero_grad()
    meta_losses = []

    for task_idx in range(5): # 5 tasks per meta-batch

        # Generate new task specific data
        x_train, y_train = generate_sin_task(10)
        x_test, y_test = generate_sin_task(5)

        with innerloop_ctx(model, optim.SGD, lr=task_lr) as (fmodel, diffopt):
            # Inner loop adaptation
            for step in range(inner_steps):
                pred_train = fmodel(x_train)
                inner_loss = loss_fn(pred_train, y_train)
                diffopt.step(inner_loss)

            # Evaluate the adapted model
            pred_test = fmodel(x_test)
            task_loss = loss_fn(pred_test, y_test)
            meta_losses.append(task_loss)


    # Outer loop update (Now averaged directly)
    meta_loss = torch.stack(meta_losses).mean()
    meta_loss.backward()
    meta_optimizer.step()

    if meta_iter % 100 == 0:
      print(f"Meta Iter {meta_iter}, Meta Loss: {meta_loss.item()}")
```

In the final example, I made a minor modification to accumulate all the task losses in a list and calculate the mean loss before the backward pass. The primary advantage here is that `meta_losses` is now a list of tensors calculated in parallel, so that by stacking them together we are taking advantage of vectorized computation. The logic remains identical to the previous example, but the change highlights how we might scale up the meta-training by meta-batching the inner loops.

For further exploration into MAML and the `higher` library, I recommend consulting the official PyTorch documentation. Research papers covering MAML and other meta-learning algorithms, and source code implementations released by the original authors, are also invaluable learning resources. Specifically, material focusing on gradient-based meta-learning approaches will likely prove beneficial. Additionally, resources that explore optimization techniques within PyTorch will enhance understanding.
