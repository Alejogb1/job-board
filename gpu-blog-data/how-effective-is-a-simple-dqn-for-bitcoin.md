---
title: "How effective is a simple DQN for Bitcoin trading?"
date: "2025-01-30"
id: "how-effective-is-a-simple-dqn-for-bitcoin"
---
Deep Reinforcement Learning, specifically a Deep Q-Network (DQN), applied to Bitcoin trading presents a significant challenge due to the stochastic and highly non-stationary nature of cryptocurrency markets. While conceptually straightforward to implement, a simple DQN's effectiveness tends to be severely limited in achieving consistent, profitable trading outcomes when deployed in realistic, live market conditions.

**Understanding the Limitations of a Simple DQN**

The core idea behind a DQN is to approximate the optimal action-value function (Q-function) that predicts the expected cumulative reward for taking a particular action in a given state. In the context of trading, the state typically represents the market conditions (e.g., price, volume, indicators), actions are buying, selling, or holding, and the reward is often the profit or loss from the trade. The network learns through trial and error, adjusting its weights to minimize the difference between its Q-value prediction and the actual observed reward.

A simple DQN implementation typically utilizes a single neural network, with a fixed architecture (e.g., fully connected layers or basic recurrent layers), and employs experience replay and a target network for stability. However, this basic setup struggles significantly in Bitcoin trading for several key reasons:

1.  **Non-Stationarity:** Bitcoin markets exhibit extreme volatility and frequently change their underlying statistical properties. A DQN trained on one period might become entirely ineffective in a subsequent period due to shifts in market behavior. The network struggles to adapt quickly enough to these changes, leading to poor predictions.
2.  **Limited Feature Engineering:** The efficacy of a DQN is profoundly influenced by the input features provided. A simple implementation often relies on rudimentary market indicators without adequate feature engineering or domain-specific knowledge. These inadequate representations fail to capture the intricate dynamics of the market, leading to suboptimal trading decisions.
3.  **Sparse Rewards:** In trading, the reward is often only realized after a transaction takes place. This leads to a sparse reward structure where many actions contribute negligibly to the reward and the network struggles to connect actions to eventual outcomes. This makes training slow and often ineffective.
4.  **Exploration vs. Exploitation:** DQN needs a mechanism to explore new actions in the environment while simultaneously exploiting the best actions that it has learnt so far. Balancing this can be tricky with a simple DQN, often leading to an ineffective training cycle that either gets stuck in a local optimum or fails to converge.
5. **Overfitting:** In a volatile environment, like Bitcoin trading, there's a high risk that the model overfits to the training data. This overfitting happens as the model will remember the patterns in the training data, but fail to generalize to new, unseen data which is a characteristic of real-world live trading.
6.  **Transaction Costs and Slippage:** A basic DQN often neglects critical aspects of real-world trading such as transaction fees and market slippage. These factors significantly reduce actual profitability and can invalidate decisions made based on idealized conditions used in training.
7.  **Hyperparameter Optimization:** A simple DQN relies on a single set of hyperparameters, which may not be appropriate for the complexities of the Bitcoin market. Manually setting parameters like learning rate, discount factor, and neural network size without extensive optimization can result in slow convergence or unstable training.

**Code Examples and Analysis**

To illustrate these points, consider three simplified Python code examples, using TensorFlow/Keras for the DQN:

**Example 1: Basic DQN with Simple Features**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class SimpleDQN(Model):
    def __init__(self, state_size, action_size):
        super(SimpleDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        inputs = Input(shape=(state_size,))
        dense1 = Dense(24, activation='relu')(inputs)
        dense2 = Dense(24, activation='relu')(dense1)
        outputs = Dense(action_size, activation='linear')(dense2)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.optimizer = Adam(lr=0.001)

    def call(self, states):
        return self.model(states)

    def train_step(self, states, actions, targets):
       with tf.GradientTape() as tape:
            q_values = self(states)
            action_masks = tf.one_hot(actions, self.action_size)
            predicted_q_values = tf.reduce_sum(action_masks * q_values, axis=1)
            loss = tf.keras.losses.MeanSquaredError()(targets, predicted_q_values)

       grads = tape.gradient(loss, self.trainable_variables)
       self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
       return loss

# Setup example environment (simplified price, no actual trading)
state_size = 3 # Price, Simple Moving Average (SMA), volume
action_size = 3 # Buy, Sell, Hold
dqn = SimpleDQN(state_size, action_size)
dummy_states = np.random.rand(64, state_size)
dummy_actions = np.random.randint(0, action_size, size=(64,))
dummy_targets = np.random.rand(64, )
loss = dqn.train_step(dummy_states, dummy_actions, dummy_targets)
print("Loss", loss)

```

This code demonstrates a basic DQN with fully connected layers, trained on random dummy data. The state consists of price, SMA, and volume. The simplicity of both the state representation and network architecture are limitations of a basic setup. In my experience, this type of architecture does not generalize well to the complexities of the Bitcoin market.

**Example 2: DQN with Experience Replay and Target Network**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)

    def size(self):
        return len(self.buffer)


class DQN_with_replay(Model):
    def __init__(self, state_size, action_size):
        super(DQN_with_replay, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        inputs = Input(shape=(state_size,))
        dense1 = Dense(24, activation='relu')(inputs)
        dense2 = Dense(24, activation='relu')(dense1)
        outputs = Dense(action_size, activation='linear')(dense2)

        self.model = Model(inputs=inputs, outputs=outputs)

        target_inputs = Input(shape=(state_size,))
        target_dense1 = Dense(24, activation='relu')(target_inputs)
        target_dense2 = Dense(24, activation='relu')(target_dense1)
        target_outputs = Dense(action_size, activation='linear')(target_dense2)

        self.target_model = Model(inputs=target_inputs, outputs=target_outputs)

        self.optimizer = Adam(lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=1000)

    def call(self, states):
        return self.model(states)

    def train_step(self, states, actions, targets):
       with tf.GradientTape() as tape:
            q_values = self(states)
            action_masks = tf.one_hot(actions, self.action_size)
            predicted_q_values = tf.reduce_sum(action_masks * q_values, axis=1)
            loss = tf.keras.losses.MeanSquaredError()(targets, predicted_q_values)

       grads = tape.gradient(loss, self.trainable_variables)
       self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
       return loss

    def update_target_model(self):
         self.target_model.set_weights(self.model.get_weights())

    def get_q_values(self, states):
        return self.model(states)

    def get_target_q_values(self, states):
        return self.target_model(states)



# Setup example environment (simplified price, no actual trading)
state_size = 3 # Price, SMA, volume
action_size = 3 # Buy, Sell, Hold
dqn = DQN_with_replay(state_size, action_size)
for _ in range(100):
    dummy_states = np.random.rand(state_size)
    dummy_next_states = np.random.rand(state_size)
    dummy_actions = np.random.randint(0, action_size, size=1)
    dummy_reward = np.random.rand()
    dummy_done = False
    dqn.replay_buffer.add(dummy_states, dummy_actions, dummy_reward, dummy_next_states, dummy_done)

# Training
if dqn.replay_buffer.size() > 64:
        states, actions, rewards, next_states, dones = dqn.replay_buffer.sample(64)

        next_q_values = dqn.get_target_q_values(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)

        targets = rewards + 0.99*max_next_q_values*(1-dones)
        loss = dqn.train_step(states, actions, targets)
        print("Loss", loss)

        dqn.update_target_model()
```

This example introduces experience replay and a target network, addressing some of the instability issues of the basic DQN. While an improvement, in my experience, it still falls short of handling the dynamic nature of Bitcoin.  The target network is a common improvement but it struggles to generalize to unseen market data in a practical context.

**Example 3: DQN with a simple Exploration strategy**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)

    def size(self):
        return len(self.buffer)


class DQN_with_replay(Model):
    def __init__(self, state_size, action_size):
        super(DQN_with_replay, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        inputs = Input(shape=(state_size,))
        dense1 = Dense(24, activation='relu')(inputs)
        dense2 = Dense(24, activation='relu')(dense1)
        outputs = Dense(action_size, activation='linear')(dense2)

        self.model = Model(inputs=inputs, outputs=outputs)

        target_inputs = Input(shape=(state_size,))
        target_dense1 = Dense(24, activation='relu')(target_inputs)
        target_dense2 = Dense(24, activation='relu')(target_dense1)
        target_outputs = Dense(action_size, activation='linear')(target_dense2)

        self.target_model = Model(inputs=target_inputs, outputs=target_outputs)

        self.optimizer = Adam(lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def call(self, states):
        return self.model(states)

    def train_step(self, states, actions, targets):
       with tf.GradientTape() as tape:
            q_values = self(states)
            action_masks = tf.one_hot(actions, self.action_size)
            predicted_q_values = tf.reduce_sum(action_masks * q_values, axis=1)
            loss = tf.keras.losses.MeanSquaredError()(targets, predicted_q_values)

       grads = tape.gradient(loss, self.trainable_variables)
       self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
       return loss

    def update_target_model(self):
         self.target_model.set_weights(self.model.get_weights())

    def get_q_values(self, states):
        return self.model(states)

    def get_target_q_values(self, states):
        return self.target_model(states)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            q_values = self.get_q_values(np.expand_dims(state,axis=0))
            action = np.argmax(q_values)
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return action



# Setup example environment (simplified price, no actual trading)
state_size = 3 # Price, SMA, volume
action_size = 3 # Buy, Sell, Hold
dqn = DQN_with_replay(state_size, action_size)
for _ in range(100):
    dummy_states = np.random.rand(state_size)
    dummy_next_states = np.random.rand(state_size)
    dummy_actions = dqn.act(dummy_states)
    dummy_reward = np.random.rand()
    dummy_done = False
    dqn.replay_buffer.add(dummy_states, dummy_actions, dummy_reward, dummy_next_states, dummy_done)

# Training
if dqn.replay_buffer.size() > 64:
        states, actions, rewards, next_states, dones = dqn.replay_buffer.sample(64)

        next_q_values = dqn.get_target_q_values(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)

        targets = rewards + 0.99*max_next_q_values*(1-dones)
        loss = dqn.train_step(states, actions, targets)
        print("Loss", loss)

        dqn.update_target_model()
```

This example adds a basic epsilon-greedy exploration strategy. While it does introduce a rudimentary exploration method, in practice, more sophisticated approaches are usually needed, that can deal with the complexities of time-series data and market changes.

**Recommendations**

For improved performance, I would strongly suggest researching advancements over simple DQNs.  This would include exploring alternative reinforcement learning algorithms, like Proximal Policy Optimization (PPO) or Actor-Critic methods, as these are often less sensitive to hyperparameter tuning and are more adept at learning in unstable environments.

Feature engineering beyond basic indicators is also critical. Include more relevant features related to market sentiment, volatility indices, order book data, and on-chain data. You should also look into using time-series specific models such as LSTMs or attention mechanisms, to better capture temporal dependencies in the data.

To deal with the non-stationarity of the markets, methods like continual learning, meta-learning, or robust training strategies should be explored. This is especially important in Bitcoin and crypto trading. Parameter tuning needs to be done carefully, for which techniques like Bayesian optimization can be implemented. Also, consider adding transaction costs to the reward function, and introduce slippage into the environment. Lastly, do proper evaluation using metrics like the Sharpe ratio and drawdown.

Implementing a DQN for Bitcoin trading is not a trivial task, and relying on a simple implementation is unlikely to yield meaningful results. It requires a deep understanding of financial markets and advanced techniques in machine learning to overcome the inherent challenges.
