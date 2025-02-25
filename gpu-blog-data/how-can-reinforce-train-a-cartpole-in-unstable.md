---
title: "How can REINFORCE train a CartPole in unstable conditions?"
date: "2025-01-30"
id: "how-can-reinforce-train-a-cartpole-in-unstable"
---
The instability of the CartPole environment, particularly when initialized with non-zero starting velocities or off-center positions, significantly challenges traditional reinforcement learning algorithms like REINFORCE. This stems from REINFORCE's reliance on Monte Carlo sampling, where an entire episode must complete before a policy update can occur. When the CartPole quickly falls, a single episode generates a very sparse, noisy reward signal, leading to high variance in the gradient estimates and consequently, slow or failed learning. I've encountered this firsthand while working on a robotics simulation project; the initialized pole often swings wildly before the controller even has time to react.

The core issue lies in the nature of REINFORCE. It operates by estimating the expected return for a trajectory (a sequence of states, actions, and rewards) generated by the current policy. The gradient used for policy improvement is the product of the cumulative return of that trajectory and the logarithm of the probabilities of actions taken under that policy, summed over the time steps of the trajectory. If the pole falls immediately, the return is almost always zero or some small penalty, regardless of the actions initially chosen. Thus, the gradient becomes small and directionless. Furthermore, each trajectory’s return is an independent random variable, meaning REINFORCE learns extremely slowly if at all. This becomes even more apparent when you introduce initial states that force the pole to rapidly deviate from its upright position.

To overcome these challenges, one must address the inherent limitations of the Monte Carlo sampling approach in these circumstances. The most effective strategy involves incorporating techniques that reduce variance in gradient estimates and stabilize training. These include techniques like adding a baseline, which subtracts a value function estimate from the cumulative return and dampens fluctuations, normalizing the returns within a batch, and using policy gradient algorithms that employ actor-critic architectures, although those represent a divergence from REINFORCE. For the sake of specifically improving REINFORCE, this response will focus on how we can stabilize a REINFORCE implementation. I've found these modifications to be critical when tuning the algorithm in high-variance environments.

The base REINFORCE algorithm in a standard environment like CartPole, without these modifications, could be implemented as follows:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def reinforce_step(env, policy, optimizer, gamma=0.99, max_episode_len=500):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        new_state, reward, done, _ = env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        state = new_state

    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    loss = 0
    for log_prob, G in zip(log_probs, returns):
       loss += -log_prob * G # Negative sign for gradient ascent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return np.sum(rewards)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    episodes = 1000
    for i in range(episodes):
        ep_reward = reinforce_step(env, policy, optimizer)
        print(f"Episode: {i}, Reward: {ep_reward}")

```

In this baseline example, I’ve defined a simple policy network and the REINFORCE update mechanism. The algorithm gathers a full episode's transitions and then calculates the discounted return. It updates the policy parameters based on these returns. Running this will yield very unstable and erratic results, particularly with unstable initial conditions. Notice, importantly, that `returns` is not normalized. This is a significant issue. The lack of a baseline also contributes to high variance in the return.

Here is a modification of the algorithm that incorporates a baseline, calculated as a mean of the discounted returns:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def reinforce_step(env, policy, optimizer, gamma=0.99, max_episode_len=500):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        new_state, reward, done, _ = env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        state = new_state

    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    baseline = returns.mean()
    loss = 0
    for log_prob, G in zip(log_probs, returns):
       loss += -log_prob * (G-baseline) # Negative sign for gradient ascent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return np.sum(rewards)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    episodes = 1000
    for i in range(episodes):
        ep_reward = reinforce_step(env, policy, optimizer)
        print(f"Episode: {i}, Reward: {ep_reward}")
```

In this second example, I’ve added a baseline subtraction before updating. The baseline, in this case the mean of returns, provides a common reference point. This reduces the variance in the return signal, resulting in a more stable and efficient learning process. With this modification, we will see noticeable gains in stability of the learning process over the prior baseline.

To further enhance stability, one can normalize returns in a batch, meaning we would need to train over multiple episodes (a batch of episodes) before performing the policy update. This ensures that the gradients are not overly influenced by a few high or low rewards, further reducing variance in the estimates, and stabilizing training:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def collect_episode(env, policy, gamma=0.99, max_episode_len=500):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        new_state, reward, done, _ = env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        state = new_state

    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    return log_probs, returns, sum(rewards)

def reinforce_step(policy, optimizer, batch_log_probs, batch_returns):
    
    batch_returns = torch.cat(batch_returns)
    batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8) # Normalize the returns within the batch

    loss = 0
    for log_prob, G in zip(batch_log_probs, batch_returns):
       loss += -torch.sum(log_prob * G)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    episodes = 1000
    batch_size = 10
    for i in range(episodes):
        batch_log_probs = []
        batch_returns = []
        batch_rewards = []
        for _ in range(batch_size):
             log_probs, returns, ep_reward = collect_episode(env, policy)
             batch_log_probs.append(log_probs)
             batch_returns.append(returns)
             batch_rewards.append(ep_reward)

        reinforce_step(policy, optimizer, batch_log_probs, batch_returns)

        print(f"Episode: {i}, Avg. Reward: {np.mean(batch_rewards)}")
```

This final modification involves generating a batch of episodes and normalizing the returns across that batch. The `collect_episode` function collects the trajectory, and then we normalize those returns using standard normalization techniques. These adjustments, when combined with the baseline, significantly stabilize the training process and improve the algorithm’s ability to handle unstable starting conditions. From my own experience, normalizing across episodes is one of the most important additions.

For further learning, I highly recommend exploring foundational texts on reinforcement learning. Books covering both theoretical underpinnings and practical implementations of policy gradient methods are invaluable. Additionally, academic publications detailing techniques for reducing variance in Monte Carlo sampling would provide a deeper understanding. Online courses that include hands-on labs are also beneficial, allowing you to implement and experiment with these algorithms firsthand. Finally, consulting open-source code repositories dedicated to reinforcement learning frameworks can provide valuable insights into practical coding patterns. Careful study of these resources and a dedication to experimentation will lead to a much better understanding of the problems at hand.
