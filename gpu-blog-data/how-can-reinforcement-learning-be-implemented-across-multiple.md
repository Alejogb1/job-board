---
title: "How can reinforcement learning be implemented across multiple agents or devices?"
date: "2025-01-30"
id: "how-can-reinforcement-learning-be-implemented-across-multiple"
---
Reinforcement learning (RL) implementation across multiple agents presents a challenge due to the increased complexity in coordinating exploration, learning, and policy application compared to single-agent scenarios. I’ve encountered this while developing a simulated traffic management system, where individual vehicles act as independent agents needing to learn optimal driving behaviors collectively. The core issue arises from the need to manage a distributed state space and action space, which often introduces non-stationarity from the perspective of any single agent.

Implementing multi-agent reinforcement learning (MARL) generally involves a shift from the traditional Markov Decision Process (MDP) framework to a more complex setting such as a Markov Game or a Stochastic Game. These frameworks explicitly model the interactions between multiple agents. A common hurdle I faced early on was determining the appropriate level of information sharing between agents, and selecting suitable algorithms designed for multi-agent scenarios. Ignoring this often resulted in unstable learning and unpredictable system behavior, essentially negating the potential benefits of collaborative learning.

The transition from single-agent RL to MARL requires a re-evaluation of several key concepts. In single-agent RL, an agent interacts with a stationary environment, attempting to learn an optimal policy that maps states to actions. In MARL, the environment includes the other agents, whose policies are also changing during training. This leads to a non-stationary environment from the perspective of each agent, making standard RL algorithms less reliable. The actions of other agents now directly influence the state transitions of each individual agent. This is where the specific algorithms employed become crucial. Approaches like Independent Q-Learning (IQL), Counterfactual Multi-Agent Policy Gradients (COMA), and Multi-Agent Deep Deterministic Policy Gradient (MADDPG) are designed to address this.

IQL is one of the simplest approaches. It treats each agent as an independent learner, ignoring the presence of other agents and using a standard RL algorithm, such as Q-learning, to learn their own policies based only on local observations and actions. Although straightforward to implement, IQL suffers from the problem of non-stationarity; the policy changes of other agents effectively change the environment dynamically, rendering the convergence guarantees of standard Q-learning unreliable.

```python
# Example IQL implementation using a simple Q-table
import numpy as np

class IndependentQLearner:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state,:])

    def update_q_table(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (target - predict)

    def decay_exploration(self):
       self.exploration_rate *= self.exploration_decay

# Example usage, each agent has its own independent learner
num_agents = 2
num_states = 10
num_actions = 4

agents = [IndependentQLearner(num_states, num_actions) for _ in range(num_agents)]

for episode in range(1000):
  for agent_id in range(num_agents):
      # Agent interacts with its environment based on local state observation
      state = np.random.randint(num_states) # Simplified state for demonstration
      action = agents[agent_id].choose_action(state)
      next_state = np.random.randint(num_states)
      reward = np.random.rand() #Simplified reward for demonstration

      agents[agent_id].update_q_table(state, action, reward, next_state)
      agents[agent_id].decay_exploration()

```
This Python example demonstrates how to create and use several independent Q-learners. Each agent maintains its own Q-table and does not directly consider the actions or policies of other agents when updating its table or choosing actions. This simplicity is convenient for initial experimentation, but it is prone to instability in more complex interactions. The code initializes individual `IndependentQLearner` objects, each having its own Q-table, learning rate, discount factor, and exploration parameters. During the training loop, each agent takes actions, receives rewards, and updates their Q-table separately. This illustrates how IQL can be applied with minimal modification to a standard Q-learning implementation.

A more sophisticated approach is the use of centralized training with decentralized execution. An example of this is MADDPG, which uses a centralized critic during training to leverage global information about the state and actions of all agents, while during execution each agent selects actions based only on its local observations. This addresses the non-stationarity issue more effectively than IQL. The centralized critic helps each agent understand the influence of other agents’ actions on the system.

```python
# Simplified MADDPG example using PyTorch, not fully operational
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Simplified Training setup (Illustrative)
num_agents = 2
state_dim = 10
action_dim = 2

actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]

actor_optimizers = [optim.Adam(actor.parameters(), lr=0.001) for actor in actors]
critic_optimizers = [optim.Adam(critic.parameters(), lr=0.001) for critic in critics]

# Simplified episode simulation
def simulate_episode():
    states = [torch.randn(1, state_dim) for _ in range(num_agents)]
    actions = []
    for agent_id in range(num_agents):
       with torch.no_grad():
           actions.append(actors[agent_id](states[agent_id]))

    #Simplified reward calculation and next states, incorporating interactions.
    rewards = [torch.randn(1,1) for _ in range(num_agents)] # Simplified reward
    next_states = [torch.randn(1,state_dim) for _ in range(num_agents)] # Simplified next state

    return states, actions, rewards, next_states

def update_networks(states, actions, rewards, next_states):
    for agent_id in range(num_agents):
      # Centralized critic update
      critic_optimizer = critic_optimizers[agent_id]
      critic = critics[agent_id]
      critic_optimizer.zero_grad()

      combined_state = torch.cat(states, dim=1)
      combined_actions = torch.cat(actions, dim =1)

      current_q = critic(combined_state, combined_actions)
      with torch.no_grad():
           next_actions = [actors[i](next_states[i]) for i in range(num_agents)]
           next_combined_actions = torch.cat(next_actions, dim =1)
           next_combined_state = torch.cat(next_states, dim=1)
           target_q = rewards[agent_id] + 0.99 * critic(next_combined_state, next_combined_actions)

      critic_loss = nn.MSELoss()(current_q, target_q)
      critic_loss.backward()
      critic_optimizer.step()

      # Actor update
      actor_optimizer = actor_optimizers[agent_id]
      actor_optimizer.zero_grad()

      new_actions = [actors[i](states[i]) if i==agent_id else actions[i] for i in range(num_agents)]
      new_combined_actions = torch.cat(new_actions, dim=1)
      actor_loss = -critic(combined_state, new_combined_actions).mean()
      actor_loss.backward()
      actor_optimizer.step()

# Training loop (simplified)
for episode in range(100):
    states, actions, rewards, next_states = simulate_episode()
    update_networks(states, actions, rewards, next_states)
```
The PyTorch example demonstrates a highly simplified and not fully runnable implementation of the MADDPG algorithm, focusing primarily on network structure and the basic update process. It initializes actors and critics for multiple agents. Critically, it showcases the central idea that the critic receives global information (concatenated states and actions of all agents). During the update process, the critic is updated based on the joint state-action, and the actor for each agent is updated through a policy gradient computed via the centralized critic. Note that some specifics like replay buffer and target networks would be needed to make a fully functioning MADDPG implementation, but this example showcases the core architecture.

Finally, another key area I've seen great results in for more coordinated interaction is communication-based MARL. In cases where agents need explicit mechanisms to collaborate, techniques where agents communicate their intention to other agents are beneficial. This introduces another layer of complexity but can resolve some cooperation problems that implicit coordination cannot, for example when complex strategies need to be adopted. These algorithms typically employ additional network components specifically for communication. I found that the overhead introduced through communication is frequently offset by the improved convergence and performance in cooperative settings.

```python
# Example using simplified comms with a shared memory object
import numpy as np

class CommunicatingAgent:
    def __init__(self, id, num_states, num_actions):
        self.id = id
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.99
        self.communication_buffer = None # Shared buffer reference

    def set_comms_buffer(self, buffer):
       self.communication_buffer = buffer;

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state,:])

    def update_q_table(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (target - predict)

    def decay_exploration(self):
       self.exploration_rate *= self.exploration_decay

    def send_message(self, message):
       self.communication_buffer[self.id] = message

    def receive_messages(self):
        return [msg for idx, msg in enumerate(self.communication_buffer) if idx != self.id]


# Example usage
num_agents = 2
num_states = 10
num_actions = 4
communication_buffer = [None for _ in range(num_agents)] #Shared buffer object

agents = [CommunicatingAgent(i, num_states, num_actions) for i in range(num_agents)]
for agent in agents:
   agent.set_comms_buffer(communication_buffer)

for episode in range(1000):
  for agent_id in range(num_agents):
      state = np.random.randint(num_states)
      action = agents[agent_id].choose_action(state)
      next_state = np.random.randint(num_states)
      reward = np.random.rand()

      agents[agent_id].update_q_table(state, action, reward, next_state)
      agents[agent_id].decay_exploration()
      agents[agent_id].send_message(f"Agent {agent_id} acted at state {state}")

  for agent_id in range(num_agents): #Print messages for demo
        messages = agents[agent_id].receive_messages()
        print(f"Agent {agent_id} received messages: {messages}")
```

In this last Python snippet, a simplified communication scheme is introduced on top of the independent learning framework. Here, each agent has a shared communication buffer to send messages to the others. While a full communication protocol and usage of messages in the decision-making process is omitted for simplicity, this illustrates the fundamental concept of shared information between agents. In more complex scenarios, the message content would inform actions, allowing agents to actively coordinate.

For further exploration, I recommend consulting "Multiagent Systems: Algorithmic, Game-Theoretic and Logical Foundations" by Yoav Shoham and Kevin Leyton-Brown for a formal treatment of multi-agent theory. For implementation specifics, “Deep Reinforcement Learning Hands-On" by Maxim Lapan provides clear and concise explanations of numerous MARL algorithms, accompanied by practical code examples. Finally, “Reinforcement Learning: An Introduction” by Richard S. Sutton and Andrew G. Barto, serves as the foundational text and is crucial for a deeper comprehension of the underlying principles.
