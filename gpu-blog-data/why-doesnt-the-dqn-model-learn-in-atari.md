---
title: "Why doesn't the DQN model learn in Atari Pong (NoFrameskip)?"
date: "2025-01-30"
id: "why-doesnt-the-dqn-model-learn-in-atari"
---
The Deep Q-Network (DQN) struggles to learn effectively in the Atari Pong (NoFrameskip) environment due to a confluence of factors, primarily related to the inherent challenges of reinforcement learning coupled with the specific characteristics of the game. My experience across multiple attempts at implementing DQN, varying hyperparameters and network structures, has consistently revealed this difficulty and its causes.

The core issue lies in the instability of Q-learning, especially when combined with the complex state-action space of raw pixel data characteristic of games like Pong. DQN aims to approximate the optimal action-value function, represented by Q(s, a), which estimates the expected cumulative reward for taking action ‘a’ in state ‘s’. However, this function is constantly being updated using a target network and mini-batch updates, a process that inherently involves noise and can lead to unstable learning, especially when using function approximation (such as neural networks).

In Pong, the environment provides a sparse reward structure. A reward of +1 is only given when the agent successfully hits the ball past the opponent, and -1 when the opponent does the same. Many frames go by without any reward signal whatsoever. This creates what is often called a “credit assignment problem.” The network needs to learn which of its actions, even multiple steps back in time, contributed to the eventual reward. Since the agent mostly encounters zero rewards, and the positive or negative rewards happen only at the end of relatively long sequences of actions, learning is difficult and slow. The network may fail to associate the necessary sequences of actions, such as moving the paddle up or down appropriately, with the ultimate outcome of winning or losing the game. This is particularly challenging when training from pixel input, as the agent first needs to construct an understanding of game dynamics from raw visual data, adding another layer of complexity.

Further compounding this challenge are the temporal correlations between subsequent states. The frames are not independent but follow one another sequentially. Because of this sequential dependency, the mini-batches sampled from experience replay buffer, a technique used to address non-iid data, can still contain highly correlated transitions, thus potentially amplifying instability. As the network is updated based on potentially correlated data, the learning becomes highly erratic and can diverge, making it difficult for the model to learn a reliable Q-function. The agent might over- or under-estimate the value of certain states or actions, and this can propagate through the training process.

Moreover, without using NoFrameskip, the agent has to manage every frame individually, increasing the number of potential states and also adding to the complexity of the temporal dependencies. This is because consecutive frames are nearly identical without any skipping. This means that the network is trying to learn from highly similar samples, which do not contribute a lot to the training, and can potentially cause convergence issues. The redundancy does not help but only increases computational load.

Finally, the default exploration strategies used with DQN, such as ε-greedy policies, can be inadequate for environments that require a long sequence of specific actions to obtain a reward. The agent may be randomly exploring a vast space of actions and states without discovering the critical actions that lead to success. This is the core of the problem; random exploration is not enough to lead to the desired outcome in a complex environment like Pong, even when the rules of the game itself are very simple.

Now, let us consider code examples showcasing the issue. I will present three simplified fragments focusing on problematic areas within typical DQN training loop, along with commentary.

```python
# Example 1: Basic DQN training loop with focus on reward sparsity
import random
import numpy as np

class SimpleDQNAgent:
    def __init__(self, state_size, action_size):
        # Simplified model initialization (no details for clarity)
        self.state_size = state_size
        self.action_size = action_size
        self.model = lambda x : np.random.rand(action_size) # Place holder, replace with an actual NN.

    def get_action(self, state, epsilon):
       if random.random() < epsilon:
          return random.randrange(self.action_size) # Exploration
       else:
          return np.argmax(self.model(state)) # Exploitation

    def train(self, experience, batch_size, gamma, learning_rate):
        if len(experience) < batch_size:
            return
        batch = random.sample(experience, batch_size)
        for state, action, reward, next_state, done in batch:
            # Here the problem happens, most of the time reward is zero, especially at begining.
            target = reward
            if not done:
                target = reward + gamma * np.max(self.model(next_state))
            # Simplified update (place holder, should be backprop of actual NN)
            target_vector = self.model(state).copy()
            target_vector[action] = target
            self.model = lambda x: target_vector

# Example of usage (highly simplified)
state_size = 8 # arbitrary size
action_size = 3 # Example with three possible actions for Pong
agent = SimpleDQNAgent(state_size, action_size)
experience = [] # Experience replay
gamma = 0.99
batch_size = 32
learning_rate = 0.001
epsilon = 0.9
epsilon_decay = 0.0001
min_epsilon = 0.1

for episode in range(1000):
   state = np.random.rand(state_size) # Start state (randomized)
   done = False
   episode_rewards = 0
   while not done:
      action = agent.get_action(state, epsilon)
      # Get the next state and reward based on the selected action.
      next_state = np.random.rand(state_size) # Mocking transition (randomized)
      reward = 0 # Reward is zero most of the time unless done is reached
      if random.random() < 0.01: # Mocking rare event
         reward = 1 if random.random() < 0.5 else -1
         done = True
      experience.append((state, action, reward, next_state, done))
      agent.train(experience, batch_size, gamma, learning_rate)
      state = next_state
      episode_rewards += reward
   epsilon = max(min_epsilon, epsilon- epsilon_decay)
   print(f"Episode: {episode}, Reward: {episode_rewards}, Epsilon: {epsilon}")

```
This first example illustrates the stark reward sparsity problem. Most actions do not lead to a reward, causing the Q-function to regress back to 0 on most iterations. The target Q values are also affected by noise propagated through each update. The network fails to converge and no learning happens.

```python
# Example 2: Instability from correlated experiences
import collections
import random
import numpy as np

class SimpleExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
       if len(self.buffer) < batch_size:
            return None
       return random.sample(self.buffer, batch_size)

    def __len__(self):
       return len(self.buffer)

class SimplifiedDQN:
   def __init__(self, state_size, action_size):
       self.model = lambda x : np.random.rand(action_size) # Place holder, replace with an actual NN
   def get_action(self, state):
       return np.argmax(self.model(state))

   def train(self, batch, gamma):
       for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                 target = reward + gamma * np.max(self.model(next_state))
            target_vector = self.model(state).copy()
            target_vector[action] = target
            self.model = lambda x: target_vector


# Example of Usage:
state_size = 8
action_size = 3
agent = SimplifiedDQN(state_size, action_size)
experience_replay = SimpleExperienceReplay(1000)
gamma = 0.99
batch_size = 32

# Simulating a sequence of correlated experiences
state = np.random.rand(state_size)
for i in range(100):
  for j in range(5): # Each sequence creates highly correlated transitions
     action = agent.get_action(state)
     next_state = np.random.rand(state_size) + np.random.rand(state_size)*0.1 # Very close next states
     reward = 0
     done = False
     if random.random() < 0.01:
      reward = 1
      done = True
     experience_replay.add((state, action, reward, next_state, done))
     state = next_state
  # Training after every 5 correlated experience steps
  batch = experience_replay.sample(batch_size)
  if batch:
      agent.train(batch, gamma)

print("End of simulation with correlation")
```

This second code block emphasizes the impact of correlated samples. Consecutive experience tuples in the experience replay buffer are very similar (and with very low reward), making it hard to escape local minima, thus the learning is unstable and slow. It is hard to estimate the best q-values as the updates are very noisy.

```python
# Example 3: Insufficient exploration
import random
import numpy as np
class ExplorationLimitedAgent:
   def __init__(self, state_size, action_size):
      self.state_size = state_size
      self.action_size = action_size
      self.model = lambda x: np.random.rand(action_size) # Place holder NN
   def get_action(self, state, epsilon):
      if random.random() < epsilon:
           # Limited exploration is done mostly at the beginning.
           # No long exploration sequences are implemented.
           return random.randrange(self.action_size)
      else:
          return np.argmax(self.model(state))

   def train(self, experience, batch_size, gamma):
       if len(experience) < batch_size:
           return
       batch = random.sample(experience, batch_size)
       for state, action, reward, next_state, done in batch:
           target = reward
           if not done:
                target = reward + gamma * np.max(self.model(next_state))
           target_vector = self.model(state).copy()
           target_vector[action] = target
           self.model = lambda x: target_vector

# Example Usage
state_size = 8
action_size = 3
agent = ExplorationLimitedAgent(state_size, action_size)
experience = []
gamma = 0.99
batch_size = 32
epsilon = 0.9
min_epsilon = 0.1
epsilon_decay = 0.0001

for episode in range(1000):
   state = np.random.rand(state_size)
   done = False
   episode_reward = 0
   while not done:
      action = agent.get_action(state,epsilon)
      next_state = np.random.rand(state_size)
      reward = 0
      if random.random() < 0.01:
          reward = 1
          done = True

      experience.append((state,action,reward,next_state,done))
      agent.train(experience,batch_size,gamma)
      state = next_state
      episode_reward += reward
   epsilon = max(min_epsilon, epsilon - epsilon_decay)
   print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}")
```

This final example emphasizes the problem of insufficient exploration. The epsilon-greedy strategy, while standard, is not enough to find a successful sequence of actions within a complex environment where rewards are sparse. The agent fails to explore enough, and thus fails to learn useful policies.

To improve learning in Pong (NoFrameskip), I would suggest exploring techniques like Prioritized Experience Replay, Double DQN, and using more advanced exploration methods. For state representation, techniques that reduce the state dimensionality, such as frame stacking and convolutional layers, are very important to manage the raw pixel input. Additionally, reward shaping can assist in giving the agent more signal during the learning process. Finally, longer training periods and the right choice of hyperparameters are key for convergence.

For further study, I recommend consulting the original Deep Q-Network paper and studying resources that detail reinforcement learning algorithms in general. It would be beneficial to explore materials on the various practical aspects of training neural networks for reinforcement learning tasks, with a particular focus on the challenges associated with sparse reward environments. Researching different experience replay methods is beneficial as well. Understanding the nuances of temporal difference learning and function approximation is also key for fully grasping this complex topic.
