---
title: "How can a tic-tac-toe AI be implemented using a neural network?"
date: "2025-01-26"
id: "how-can-a-tic-tac-toe-ai-be-implemented-using-a-neural-network"
---

Implementing a tic-tac-toe AI using a neural network, while seemingly overkill for a game with such a small state space, offers a valuable pedagogical exercise in understanding fundamental concepts within machine learning and reinforcement learning. The core challenge lies in teaching the network to make decisions that lead to optimal outcomes without explicitly programming a comprehensive strategy. Instead, the network learns via experience, improving its performance iteratively.

Essentially, this problem can be approached using a reinforcement learning technique, particularly Q-learning, which is suitable given the discrete nature of tic-tac-toe actions and states. The neural network serves as a function approximator for the Q-function, mapping state-action pairs to expected rewards. In a traditional Q-learning approach, one would use a lookup table, but the number of possible board states grows rapidly in more complex games. The neural network provides the necessary generalization capacity to handle such expansion.

Here's a breakdown of the process I’ve successfully used on similar projects:

1. **State Representation:** We must convert the tic-tac-toe board into a numerical format suitable for the neural network. I typically use a flattened vector representation of the 3x3 grid. Each cell can be represented by one of three values: -1 for an 'O', 1 for an 'X', and 0 for an empty cell. This results in a 9-dimensional vector representing the game state.

2. **Network Architecture:** A simple multi-layer perceptron (MLP) will suffice for this task. My favored architecture consists of:
   *   An input layer of 9 neurons, corresponding to the flattened board state.
    *  One or two hidden layers with, perhaps, 32 or 64 neurons each, using a rectified linear unit (ReLU) activation function.
   *  An output layer of 9 neurons, each representing the Q-value of taking an action (placing an 'X' or 'O') in a particular cell. The output activation is typically linear since Q-values can be arbitrary.

3. **Q-learning:** The core of the learning algorithm revolves around updating the Q-values based on observed experiences. This process consists of:
    *  **Exploration vs. Exploitation:** Initially, the agent explores the game space by choosing random actions. As the network learns, the agent begins to exploit its knowledge, selecting actions that yield higher Q-values. An epsilon-greedy policy is commonly used, where with probability ε the agent takes a random action, and with probability 1-ε the agent takes the action with the highest Q-value for the current state.
    *  **Experience Collection:** The agent plays games, storing transitions that consist of a tuple (current state, action, reward, next state). The reward is typically +1 for winning, -1 for losing, and 0 for a draw or an intermediate state.
   *   **Q-Value Update:** The Q-value for the action taken in the current state is updated using the Bellman equation: `Q(s, a) = Q(s, a) + α * (reward + γ * max_a' Q(s', a') - Q(s, a))`, where α is the learning rate, γ is the discount factor, and s' is the next state. This update is performed using backpropagation through the neural network.

4. **Training:** The network learns by repeatedly presenting the collected experiences and updating its weights via backpropagation. The training process continues for a sufficiently large number of episodes (games), typically tens of thousands, until the network's performance converges.

Here are three Python code examples that demonstrate key aspects of the implementation using TensorFlow:

**Example 1: Defining the Neural Network**

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_q_network():
  model = tf.keras.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='linear')
  ])
  return model

q_network = build_q_network()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

*Commentary:* This code snippet defines a simple multi-layer perceptron (MLP) using Keras. It constructs the network architecture mentioned earlier. The input layer takes a 9-dimensional vector, representing the flattened board. The hidden layers utilize ReLU activation, while the output layer uses a linear activation because the Q-values can be any real number. An Adam optimizer with a learning rate of 0.001 is initialized, which will be used to train the model.

**Example 2: Epsilon-Greedy Action Selection**

```python
import numpy as np

def choose_action(state, epsilon):
  if np.random.rand() < epsilon:
    available_actions = [i for i, val in enumerate(state) if val == 0]
    if available_actions:
        return np.random.choice(available_actions)
    else: return -1 # No available actions
  else:
    q_values = q_network.predict(np.array([state]))[0]
    available_actions = [i for i, val in enumerate(state) if val == 0]
    if available_actions:
      valid_q_values = q_values[available_actions]
      return available_actions[np.argmax(valid_q_values)]
    else: return -1 # No available actions
```

*Commentary:* This function implements the epsilon-greedy strategy for action selection. Given a current state and the epsilon value, it either chooses a random available action (exploring) or selects the action that has the highest predicted Q-value (exploiting).  The network prediction is based on converting the current state array into a batch of 1, to be compatible with the Keras model’s .predict() method. An edge case is also handled when no available action exists on the board which returns a -1 value.

**Example 3: Q-Value Update**

```python
def update_q_values(state, action, reward, next_state, gamma):
  with tf.GradientTape() as tape:
    q_values = q_network(np.array([state]))[0]
    next_q_values = q_network(np.array([next_state]))[0] if next_state is not None else tf.zeros(9)

    max_next_q = tf.reduce_max(next_q_values) if next_state is not None else 0

    target_q = q_values.numpy()
    target_q[action] = reward + gamma * max_next_q
    loss = tf.keras.losses.MSE(target_q, q_values)

  grads = tape.gradient(loss, q_network.trainable_variables)
  optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
```

*Commentary:* This function calculates the loss using the Bellman equation for a given experience tuple. A `GradientTape` is used to track all operations for backpropagation. The `next_state` being None signifies the end of the game, therefore maximum Q-value is set to 0. The mean squared error between the current Q-values and the updated target Q-values is computed and then used for updating the neural network’s weights. The use of a gradient tape enables automatic calculation of the gradient.

In summary, training a tic-tac-toe AI using a neural network and Q-learning is a feasible, albeit somewhat resource-intensive, approach. It’s crucial to carefully design the state representation, network architecture, and training process to achieve optimal performance. This problem is a good example of a reinforcement learning environment where an agent learns to make decisions by interacting with its surroundings and receiving feedback, enabling the agent to perform optimally on the board over time.

For further study, I recommend investigating books and courses on reinforcement learning. Resources focusing on deep reinforcement learning, specifically Q-learning and its variations like Deep Q-Networks (DQNs), would be particularly beneficial. Texts and tutorials on TensorFlow and Keras will also be necessary for practical implementation. Specifically, exploring concepts like experience replay, target networks, and various optimization algorithms will enrich understanding and facilitate the development of more robust AI agents.
