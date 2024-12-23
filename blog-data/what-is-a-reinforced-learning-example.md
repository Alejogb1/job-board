---
title: "What is a reinforced learning example?"
date: "2024-12-23"
id: "what-is-a-reinforced-learning-example"
---

Alright, let's talk reinforcement learning (rl). I've seen quite a few implementations of it over the years, some beautifully elegant, others... less so. I've debugged systems where the agent stubbornly refused to learn, and also witnessed incredible emergent behavior. So, I'm speaking from experience when I say, understanding an rl example thoroughly can really illuminate the whole field.

Essentially, at its core, reinforcement learning is about teaching an agent to make decisions by interacting with an environment and learning from the feedback it receives. Unlike supervised learning, where we have labeled data telling us what is 'correct', rl agents learn from trial and error, maximizing a reward signal. It's a bit like training a dog: you don't give the dog a labelled dataset of "sit" and "don't sit". Instead, you reward it when it sits correctly and give it a negative feedback, like a gentle correction, when it doesn't.

A typical rl setup involves these key components:

*   **The Agent:** This is the learner, the entity that makes decisions. Think of it as the algorithm making decisions on a strategy.
*   **The Environment:** The world the agent interacts with. It can be a simulation, a game, or even a real-world robotic system.
*   **The State:** The current situation of the environment that the agent observes. It provides the context for the agent to make an informed decision.
*   **The Action:** The decision the agent makes, influencing the environment and thus the next state.
*   **The Reward:** A scalar value that provides feedback to the agent about the consequences of its action. A positive reward is good; a negative one, obviously, not so good.

Now, let’s look at a specific, simplified example I encountered in a project where I was developing a rudimentary autonomous navigation system for a simulated robot. It needed to move around a small maze, and the goal was to reach the target location. This is where reinforcement learning came in.

The environment was a simple 2D grid. Each cell could be empty, an obstacle, or the goal. The robot, our agent, could move up, down, left, or right. The state was the robot's (x,y) coordinates on the grid. The actions were those four directional movements. The reward was +1 for reaching the goal, -0.1 for each move (to encourage efficiency), and -1 for hitting an obstacle, which was not directly a termination signal.

To manage this problem, I utilized a tabular q-learning approach, a fundamental method in rl, to represent the state-action values. Here's a python snippet that embodies this simplified version:

```python
import numpy as np
import random

class MazeEnvironment:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.current_position = start
        self.rows = len(grid)
        self.cols = len(grid[0])
    def reset(self):
        self.current_position = self.start
        return self.current_position

    def step(self, action):
        row, col = self.current_position
        if action == 0: # Up
            new_row = max(0, row - 1)
            new_col = col
        elif action == 1: # Down
            new_row = min(self.rows - 1, row + 1)
            new_col = col
        elif action == 2: # Left
            new_row = row
            new_col = max(0, col - 1)
        elif action == 3: # Right
             new_row = row
             new_col = min(self.cols-1, col+1)

        if self.grid[new_row][new_col] == 1:
             reward = -1 # obstacle
             new_row, new_col = row,col
        elif (new_row,new_col) == self.goal:
             reward = 1
             self.current_position = (new_row, new_col)
             return (new_row, new_col), reward, True
        else:
             reward = -0.1
        self.current_position = (new_row, new_col)
        return (new_row, new_col), reward, False

def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = {}
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if str(state) not in q_table:
                q_table[str(state)] = [0,0,0,0] #initialize to 0
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[str(state)])

            next_state, reward, done = env.step(action)
            if str(next_state) not in q_table:
               q_table[str(next_state)] = [0,0,0,0]
            best_next_q = np.max(q_table[str(next_state)])
            q_table[str(state)][action] += alpha * (reward + gamma * best_next_q - q_table[str(state)][action])
            state = next_state

    return q_table

grid = [[0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 2]]

start = (0,0)
goal = (4,4)
env = MazeEnvironment(grid, start, goal)
q_table = q_learning(env)
print ("q table after training is:", q_table)

```

In this code, the `MazeEnvironment` class manages the simulation, handling the robot's movements and reward assignments, and the `q_learning` function implements the learning logic. The `q_table` maps each state to a set of q-values for each action, representing how beneficial it is to take an action in that state. Initially, the q-values are set to 0, but they're updated through iterative experience with the environment. Each episode involves the agent starting at its initial location, exploring the maze, selecting the best action by maximizing Q table values (or taking a random action for exploration), and using this to learn the optimal path to the goal.

It's worth noting this is a basic implementation. Real-world navigation systems involve a significantly more complex state space, incorporating sensor data, and require more sophisticated algorithms. This was, however, a useful starting point for us.

Now, let's say you're looking at a slightly different scenario: instead of a simple grid, consider a more continuous domain, such as controlling a robotic arm. In such situations, tabular q-learning becomes impractical due to the dimensionality of the state-action space. Instead, we often turn to function approximation techniques like neural networks. Consider the following basic code to approximate q values using a simple feedforward neural network:

```python
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers

class ContinuousEnvironment:
    def __init__(self):
        self.state = 0 # represents some continuous variable, could be an angle
        self.max_state = 10
        self.min_state = -10

    def reset(self):
        self.state = random.uniform(self.min_state, self.max_state)
        return self.state

    def step(self, action):
        # here action is a scalar, think of a force applied to arm
        self.state += action
        self.state = max(self.min_state, min(self.max_state, self.state)) # Keep inside bounds
        reward = - abs(self.state)  # closer to 0 more reward
        done = False
        if abs(self.state) < 0.1:
            done = True
            reward = 10

        return self.state, reward, done

def build_model(input_size, output_size):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def dqn_learning(env, state_size=1, action_size=1, episodes=1000, batch_size = 32, gamma = 0.99, epsilon = 0.1):
     model = build_model(state_size, action_size)
     memory = [] # replay buffer
     for _ in range(episodes):
          state = env.reset()
          done = False
          while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.uniform(-1, 1)
            else:
                 action_values = model.predict(np.array([state]).reshape(1,-1), verbose = 0)
                 action = action_values[0][0] # chose the best action
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            if len(memory) > batch_size: # experience replay
                 batch = random.sample(memory, batch_size)
                 states = np.array([item[0] for item in batch])
                 actions = np.array([[item[1]] for item in batch])
                 rewards = np.array([[item[2]] for item in batch])
                 next_states = np.array([item[3] for item in batch])
                 dones = np.array([[1 if item[4] else 0] for item in batch])
                 q_values = model.predict(states, verbose = 0)
                 next_q_values = model.predict(next_states, verbose = 0)
                 targets = rewards + gamma * np.max(next_q_values, axis = 1, keepdims = True)*(1-dones)
                 targets = np.array([targets[i][0] if dones[i] ==0 else targets[i] for i in range(batch_size)])
                 mask = np.eye(action_size, dtype = int) # get mask from number of possible actions, its 1 in the place where correct action happened
                 targets = np.array([x for x in targets])
                 targets = np.array([np.array(targets[i]* mask[0] if targets[i] is not None else 0) for i in range(len(targets))])
                 loss = model.train_on_batch(states, targets)
     return model


env = ContinuousEnvironment()
model = dqn_learning(env)

test_state = 2
print("value from network:", model.predict(np.array([test_state]).reshape(1,-1)))


```
Here, we use a neural network to approximate the q-function using deep q learning. The `ContinuousEnvironment` simulates a simplified control task, with state being continuous. The neural network takes the current state as input and outputs the value function, representing how good is to be at that state. The experience replay is implemented and used to train the network. The learning method is based on the deep q learning method, which is an improvement on the traditional tabular q-learning.

Finally, let’s illustrate one last type of rl scenario, where policy gradients can be useful, especially when dealing with actions that are continuous. Imagine you want to train a robot to make its joints move smoothly. We can leverage techniques like policy gradient. Here's a simplified version:

```python
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class ContinuousActionEnv:
      def __init__(self):
          self.state = 0 # Continuous value
          self.max_state = 5
          self.min_state = -5
      def reset(self):
         self.state = 0 # random.uniform(self.min_state, self.max_state)
         return self.state
      def step(self, action):
           self.state += action
           self.state = max(self.min_state, min(self.max_state,self.state))
           reward = - abs(self.state)
           done = False
           if abs(self.state) < 0.1:
                 done = True
                 reward = 10
           return self.state, reward, done

def build_actor_model(state_size, action_size):
    inputs = tf.keras.Input(shape=(state_size,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(action_size, activation='tanh')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_critic_model(state_size):
    inputs = tf.keras.Input(shape=(state_size,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

def policy_gradient(env, state_size = 1, action_size = 1, episodes = 1000, batch_size = 32, gamma = 0.99):
     actor_model = build_actor_model(state_size, action_size)
     critic_model = build_critic_model(state_size)
     actor_optimizer = optimizers.Adam(learning_rate=0.001)
     critic_optimizer = optimizers.Adam(learning_rate=0.001)
     for _ in range(episodes):
          state = env.reset()
          done = False
          states, actions, rewards, next_states = [],[],[],[]
          while not done:
             action = actor_model(np.array([state]).reshape(1,-1)).numpy()
             next_state, reward, done = env.step(action[0])
             states.append(state)
             actions.append(action[0])
             rewards.append(reward)
             next_states.append(next_state)
             state = next_state
          returns = []
          G = 0
          for r in reversed(rewards):
             G = r + gamma*G
             returns.insert(0, G) # reversed order
          returns = np.array(returns)
          returns = (returns - np.mean(returns))/(np.std(returns) + 1e-8) # normalize
          with tf.GradientTape() as tape:
              action_probs = actor_model(np.array(states))
              log_probs = tf.math.log(tf.keras.activations.sigmoid(action_probs)) # convert to probabilities, calculate log
              advantages = returns - critic_model(np.array(states)).numpy().flatten()
              actor_loss = -tf.reduce_mean(log_probs * advantages) # policy loss
          actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
          actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

          with tf.GradientTape() as tape:
                critic_values = critic_model(np.array(states))
                critic_loss = tf.reduce_mean(tf.square(returns - critic_values)) # value function loss
          critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
          critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))
     return actor_model

env = ContinuousActionEnv()
model = policy_gradient(env)
test_state = 2
print("output of model:", model(np.array([test_state]).reshape(1,-1)))
```

In this version, we employ both an actor network and a critic network, which is known as an actor-critic method. The actor decides on the actions, whereas the critic evaluates them. The learning is based on the policy gradients approach. The code provides a basic implementation of it which is helpful to learn the key concepts of policy gradient method.

These are just three basic examples, and each has layers of detail that could be further explored. To dig deeper, I'd suggest picking up "Reinforcement Learning: An Introduction" by Sutton and Barto. It’s a standard reference and provides solid theoretical and practical knowledge on the field, and also for the general understanding of the concepts. If you want something more specific to deep reinforcement learning, I would highly recommend the paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al, which demonstrated the power of combining deep learning with reinforcement learning.

I hope this gives you a solid foundation in understanding reinforcement learning through these examples. It's a fascinating field with numerous possibilities. I wish you the best of luck as you explore its complexities.
