---
title: "Can reinforcement learning be done with supervised datasets?"
date: "2024-12-23"
id: "can-reinforcement-learning-be-done-with-supervised-datasets"
---

Alright,  It's a question that often comes up, and while the direct answer isn't a simple yes or no, it's definitely nuanced and worth exploring. The short version is that you *can* leverage supervised data to bootstrap or influence a reinforcement learning (rl) process, but it’s not rl in the purest sense. It's more about using supervised learning as a pre-training method or as a guiding force within the rl framework. Let's break that down.

From my experience, particularly during that project involving robotics path planning back in '18, we were faced with a similar dilemma. We had a wealth of human-demonstrated paths, which essentially represented supervised learning data (input: initial state; output: action or sequence of actions). However, we needed the robot to learn to navigate even *beyond* those demonstrations, handle unforeseen circumstances, and optimize its path based on some reward function (like time taken or energy used). So, a direct supervised approach wouldn't cut it, but completely ignoring the demonstrations would have been wasteful.

The core distinction lies in the underlying mechanism. Supervised learning aims to map inputs to outputs based on given labeled data. There’s no concept of an agent interacting with an environment to maximize a cumulative reward. Reinforcement learning, in contrast, focuses entirely on this iterative interaction. An agent takes actions in an environment, receives feedback (rewards or penalties), and adjusts its behavior to accumulate higher future rewards.

So, where does the supervised data come into play? It mainly comes down to two strategies: initialization and imitation learning. Let's dive into each.

**Initialization:**

The first strategy is to use supervised learning to initialize the policy or value function of an rl agent. This is especially helpful when the agent needs to begin exploration in a complex, large state space. Imagine trying to teach a robot arm complex manipulation tasks by pure trial and error—it would be incredibly inefficient and likely result in early failures. Instead, we can pre-train the network (that will form the basis of the rl policy) using the human demonstrations. By pre-training, we're essentially giving the agent a decent initial policy that's already in the ballpark of the desired behavior. The rl algorithm can then fine-tune this initialized policy to maximize its reward.

Here’s a simple code snippet using a basic neural network for policy initialization:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulated supervised data (states and actions)
states = torch.randn(100, 10) # 100 samples, each state is 10 dimensional
actions = torch.randint(0, 4, (100,)) # 4 possible actions

# Define a simple neural network (policy)
class SimplePolicy(nn.Module):
  def __init__(self, input_size, output_size):
    super(SimplePolicy, self).__init__()
    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, output_size)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

policy_net = SimplePolicy(10, 4) # 10 input dimensions, 4 possible actions
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Supervised training (pre-training)
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = policy_net(states)
    loss = criterion(outputs, actions)
    loss.backward()
    optimizer.step()

print("Pre-training complete")

# Now use this policy_net in a reinforcement learning loop.
```

In this snippet, we train the `SimplePolicy` network using the supervised data (states and actions) via cross-entropy loss. After this pre-training, the `policy_net` would be used as the initial policy for the rl agent.

**Imitation Learning:**

The second strategy involves imitation learning (often also called behavioral cloning), where the supervised data is used to directly train a policy to imitate the demonstrations. Think of it as trying to replicate expert behavior from observed examples. Essentially, we are training a function to mimic the actions the demonstrator would take in a particular state.

However, while this method helps the agent learn to perform well in observed conditions, it struggles with situations not present in the demonstration data. The agent hasn’t learned any underlying principles of optimal decision-making in the environment, but only to mimic the actions observed. To counter this, techniques such as DAgger (Dataset Aggregation) can be used, where the rl agent is trained iteratively, with new demonstrations gathered from the agent's exploration attempts.

Here's a simplified imitation learning example:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulate dataset of states and actions from demonstrations
states = torch.randn(200, 10) # 200 samples, 10 dimensional states
actions = torch.randint(0, 3, (200,)) # 3 possible discrete actions

# Define a policy network
class ImitationPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImitationPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate and configure training elements
imitation_policy = ImitationPolicy(10, 3) # 10 input features, 3 output actions
optimizer = optim.Adam(imitation_policy.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Imitation learning training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = imitation_policy(states)
    loss = criterion(outputs, actions)
    loss.backward()
    optimizer.step()

print("Imitation learning training complete")

# Now the 'imitation_policy' can be used to predict actions,
# although it won't generalize beyond the training data well.
```
This script creates and trains `ImitationPolicy` to mimic actions based on the given `states` and `actions`. It's a standard supervised learning setup, and hence suffers from its typical limitations.

**Combining Supervised and Reinforcement Learning:**

Ultimately, a common approach combines both methods, pre-training with supervised data and then fine-tuning using a reinforcement learning algorithm. This balances the benefits of expert-driven knowledge with the ability of rl to optimize for rewards.

For illustration, a more complex example might incorporate both. Consider a situation where we are training a system to navigate through a grid world using A2C (Advantage Actor-Critic), and also leverage human-demonstrated paths:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define a simple grid world for the sake of this example
GRID_SIZE = 10
START_STATE = (0,0)
GOAL_STATE = (9,9)

def get_state_representation(x, y):
  # Representing the grid using one-hot encoding
  representation = np.zeros(GRID_SIZE * GRID_SIZE)
  representation[x * GRID_SIZE + y] = 1
  return representation

def is_valid(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def get_next_state(state, action):
  x, y = state
  if action == 0: # up
    x -= 1
  elif action == 1: # down
    x += 1
  elif action == 2: # left
    y -= 1
  elif action == 3: # right
    y += 1
  if is_valid(x, y):
    return (x,y)
  return state # stay in same position if invalid

# Sample expert demonstration data
expert_demonstrations = []
current_state = START_STATE
while current_state != GOAL_STATE:
    x, y = current_state
    if x < GOAL_STATE[0]:
        next_state = (x+1,y)
        expert_demonstrations.append((get_state_representation(x,y), 1)) # action down
        current_state = next_state
    elif y < GOAL_STATE[1]:
        next_state = (x,y+1)
        expert_demonstrations.append((get_state_representation(x,y), 3)) # action right
        current_state = next_state

# Neural network for policy and value function (Actor-Critic)
class ActorCritic(nn.Module):
  def __init__(self, input_size, action_size):
    super(ActorCritic, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.actor = nn.Linear(128, action_size)
    self.critic = nn.Linear(128, 1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    policy_logits = self.actor(x) # for action probabilities
    state_value = self.critic(x) # for state value approximation
    return policy_logits, state_value

ac_net = ActorCritic(GRID_SIZE*GRID_SIZE, 4) # one-hot input, 4 actions
actor_optimizer = optim.Adam(ac_net.actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(ac_net.critic.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
value_criterion = nn.MSELoss()

# Pre-train using demonstrations
for state, action in expert_demonstrations:
    actor_optimizer.zero_grad()
    state_tensor = torch.tensor(state, dtype=torch.float)
    action_tensor = torch.tensor([action],dtype=torch.long)
    policy_logits, _ = ac_net(state_tensor)
    loss = criterion(policy_logits.unsqueeze(0), action_tensor)
    loss.backward()
    actor_optimizer.step()
print('Pre-training complete, ready for rl.')

# A2C RL loop for fine-tuning
GAMMA = 0.99  # Discount factor
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    state = START_STATE
    states = []
    rewards = []
    actions = []
    while state != GOAL_STATE:
        states.append(state)
        state_rep = get_state_representation(state[0],state[1])
        state_tensor = torch.tensor(state_rep, dtype=torch.float)
        policy_logits, state_value = ac_net(state_tensor)
        action_probs = torch.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        next_state = get_next_state(state, action)
        if next_state == GOAL_STATE:
          reward = 100
        else:
          reward = -1 # small negative reward for taking a step

        rewards.append(reward)
        actions.append(action)
        state = next_state

    # calculate returns
    returns = []
    cumulative_return = 0
    for reward in reversed(rewards):
      cumulative_return = reward + GAMMA * cumulative_return
      returns.insert(0,cumulative_return)
    returns = torch.tensor(returns, dtype=torch.float)

    # actor/critic update
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    for state, action, ret in zip(states, actions, returns):
      state_rep = get_state_representation(state[0], state[1])
      state_tensor = torch.tensor(state_rep,dtype=torch.float)
      policy_logits, state_value = ac_net(state_tensor)

      action_tensor = torch.tensor([action], dtype=torch.long)
      action_prob_log = torch.log_softmax(policy_logits, dim=-1)
      advantage = ret - state_value.squeeze(0)

      actor_loss = -action_prob_log[action]*advantage
      critic_loss = value_criterion(state_value.squeeze(0), ret)

      actor_loss.backward(retain_graph=True)
      critic_loss.backward()


    actor_optimizer.step()
    critic_optimizer.step()

    if episode % 100 == 0:
      print(f'Episode: {episode}, Reward: {sum(rewards)}')
```
This combined example demonstrates how supervised pre-training can improve the initial policy in an rl setting, while also allowing the agent to continue learning and improve performance further through environment interactions.

**Further Exploration:**

For deeper insights, I highly recommend exploring the following resources:

*   **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto**: This book is a cornerstone of reinforcement learning literature, providing a thorough theoretical and practical understanding.
*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**: While not solely focused on rl, this book delves into the foundational deep learning techniques, critical for many rl algorithms used today.
*   **Research papers on imitation learning and reinforcement learning from top conferences such as NeurIPS, ICML, and ICLR**: Reading the latest research provides cutting-edge perspectives and techniques.

In essence, while you can't directly "do" pure reinforcement learning with solely supervised datasets, supervised techniques play a pivotal role in accelerating and guiding the learning process, often resulting in improved overall performance and practical solutions. It's about leveraging the best of both worlds, combining the efficiency of expert knowledge with the adaptability of reinforcement learning. The key takeaway is that supervised data provides a valuable starting point, but ultimately, interaction with the environment is fundamental for true rl.
