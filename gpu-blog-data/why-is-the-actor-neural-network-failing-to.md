---
title: "Why is the actor neural network failing to learn?"
date: "2025-01-30"
id: "why-is-the-actor-neural-network-failing-to"
---
A frequent cause for an actor neural network's failure to learn, particularly in reinforcement learning scenarios, is the instability arising from high variance in the estimated gradients. My experience on projects involving complex robotics control, specifically an autonomous manipulation task, highlighted this issue. We observed our actor, tasked with generating optimal joint torques, oscillating wildly between poor and mediocre policies despite seemingly well-tuned hyperparameters. This instability stemmed from the interconnectedness of the actor’s actions, the environment's state, and the critic's evaluation, creating a feedback loop prone to divergence.

The actor network’s learning process relies on gradients calculated from the critic’s evaluation. Ideally, these gradients nudge the actor towards actions that yield higher cumulative rewards, as predicted by the critic. However, when the critic’s estimations are noisy or inaccurate – which often occurs early in training or during exploration – the actor receives erratic update signals. This erraticism leads to inconsistent policy adjustments, effectively pushing the actor into a cycle of overshooting or underperforming, thus impeding convergence towards an optimal policy. This problem is significantly exacerbated in environments with delayed rewards, or when the action space is high dimensional, increasing the complexity of the relationship between actions and their outcomes, and thereby making the gradient estimation even more prone to noise.

The following examples demonstrate common scenarios where the actor's learning can be hindered, and offer practical techniques to mitigate such problems:

**Example 1: Inadequate Exploration & Vanishing Gradients**

```python
import tensorflow as tf
import numpy as np

# Assume a simplified actor network with a single output action
class SimpleActor(tf.keras.Model):
  def __init__(self, action_dim=1):
    super(SimpleActor, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dense2 = tf.keras.layers.Dense(action_dim, activation='tanh') # Scaled action between -1 and 1

  def call(self, state):
    x = self.dense1(state)
    return self.dense2(x)


actor = SimpleActor()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def train_step(state, critic_grad):
    with tf.GradientTape() as tape:
        action = actor(state)
        #The actor wants to take actions in the direction that increases critic performance. Hence actor loss is negative critic grad
        actor_loss = -tf.reduce_mean(tf.multiply(action, critic_grad))
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    return actor_loss


#In the initial iterations, the random initialisations of the neural network may lead to tiny actions from our tanh layer
#In that case, the negative critic grad will multiply with a small value and hence the loss becomes negligibly small, causing almost no gradient and hence almost no update.
state = np.random.rand(1, 10) # 1 observation of 10 dimensions
critic_grad = np.random.rand(1, 1) # 1 critic estimation of 1 dimension
for i in range(10):
    loss = train_step(state,critic_grad)
    print(f"Loss during training iteration {i} is {loss}")

```

In this basic setup, the `tanh` activation function of the output layer scales the action values between -1 and 1. Initial network weights, in the absence of targeted exploration, often produce actions near zero. Since the actor loss depends directly on the *product* of the predicted action and the critic's gradient, small action values lead to minuscule losses, resulting in vanishing gradients. The actor struggles to learn because its parameter updates are excessively tiny. This lack of meaningful changes leads to inadequate exploration of the state and action space, keeping the actor stuck in its initial sub-optimal performance range.

A solution is to incorporate an exploration strategy. Epsilon-greedy approach or using an Ornstein-Uhlenbeck process for adding exploration noise in actions is good place to start.

**Example 2: High Critic Variance & Unstable Updates**

```python
import tensorflow as tf
import numpy as np

class SimpleActor(tf.keras.Model):
  def __init__(self, action_dim=1):
    super(SimpleActor, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dense2 = tf.keras.layers.Dense(action_dim) # No activation

  def call(self, state):
    x = self.dense1(state)
    return self.dense2(x)

actor = SimpleActor()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def train_step(state, critic_grad):
    with tf.GradientTape() as tape:
        action = actor(state)
        actor_loss = -tf.reduce_mean(tf.multiply(action, critic_grad)) # negative as the actor wants to maximize the rewards
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    return actor_loss


state = np.random.rand(1, 10)
# Introduce high variance in critic's estimation
for i in range(10):
    critic_grad = np.random.normal(0, 100, (1,1))
    loss = train_step(state, critic_grad)
    print(f"Loss during training iteration {i} is {loss}")

```

Here, we illustrate the impact of high variance in the critic's gradient estimation. The critic’s value estimations are often noisy, especially early in training or in complex environments. As you can see in the above example, the randomness in the critic gradient cause the actor loss to take drastic jumps or even change signs (positive values indicate the critic is sending the actor updates which will eventually reduce overall reward) which will cause the actor policy to vary too much causing it to never converge. This leads to unstable updates, where the actor receives contradictory signals across different training steps. This results in the network fluctuating significantly in performance and fails to learn any meaningful policy.

To stabilize the learning, we often adopt various techniques. Clipping gradients is a helpful method that limits the maximum magnitude of updates, which prevents large and damaging updates. Further, using more advanced critic architectures (like using ensembles) or using reward scaling and batch normalization to reduce noise can help. Also incorporating target network for the critic can also help in better stability.

**Example 3: Sparse Rewards & Lack of Credit Assignment**

```python
import tensorflow as tf
import numpy as np

class SimpleActor(tf.keras.Model):
  def __init__(self, action_dim=1):
    super(SimpleActor, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dense2 = tf.keras.layers.Dense(action_dim, activation='tanh') # Scaled action between -1 and 1

  def call(self, state):
    x = self.dense1(state)
    return self.dense2(x)

actor = SimpleActor()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def train_step(state, reward, action, prev_action_value):
    with tf.GradientTape() as tape:
        #Compute the log probabilities of the action using the actor network
        action_value = actor(state)
        action_probability = action_value
        advantage = reward - prev_action_value
        actor_loss = -tf.reduce_mean(tf.multiply(tf.math.log(action_probability), advantage))
        # The actor loss will be small if the reward is sparse as the advantage will be close to zero, causing very low update
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    return actor_loss


# Scenario: sparse reward signal only at end of episode
state = np.random.rand(1, 10)
action = np.random.rand(1, 1)
prev_action_value = np.random.rand(1, 1)
for i in range(10):
  if i == 9: # Reward received only at end of sequence
      reward = 10 # large reward at end
  else:
      reward = 0
  loss = train_step(state, reward, action, prev_action_value)
  print(f"Loss during training iteration {i} is {loss}")
```

In scenarios with sparse rewards, like our manipulation task where a robot might only receive a positive reward after successful completion of a complex series of actions, the issue becomes one of credit assignment. The actor needs to infer which specific actions within the sequence were crucial for achieving the final outcome. The issue here is that most of the states do not generate any reward and hence only the last state will contribute significantly to the loss, causing significant lag in the policy update. This example uses policy gradients, which estimate the gradients of the loss by using the advantage function, defined as the difference between the reward at a given time step and the baseline value function. If the reward is sparse, the advantage is zero for all time steps where the reward is not available and hence the loss is close to zero causing little to no update for the policy. Without intermediary reward signals, the actor may struggle to learn the causal relationships between its actions and their eventual results. This makes it challenging for the actor to correctly adjust its policy, often resulting in no learning or extremely slow convergence.

Techniques like reward shaping, where intermediate rewards are designed to encourage specific behavior, or using more advanced exploration strategies that enable the agent to explore long-term consequences of their actions can help alleviate this problem. Experience replay buffers with prioritization for rewarding episodes can also improve learning speed. Further, use of actor critic algorithms that employ temporal difference learning and use value function to bootstrap can speed up learning in such scenario.

In summary, the failure of an actor network to learn often stems from a confluence of factors. Gradient instability, particularly due to high critic variance, can severely impede policy updates. Inadequate exploration exacerbates the problem by limiting exposure to beneficial actions, and the issue is intensified by vanishing gradients, where the updates are ineffective because of poor function initialisations. Sparse reward signals create a credit assignment issue, making it difficult for the actor to identify the connection between its actions and their long-term effects. Addressing these issues through techniques such as gradient clipping, more robust critic networks, targeted exploration, and the implementation of shaped or more dense reward functions is often necessary to achieve effective policy learning. These are areas I would explore while troubleshooting.

For further information on these topics, I would recommend the following resources:

*   Reinforcement Learning: An Introduction by Sutton and Barto for theoretical background.
*   Deep Learning by Goodfellow, Bengio, and Courville for fundamental understanding of neural networks.
*   OpenAI Spinning Up documentation for practical insights and code examples in reinforcement learning.
*   Papers and resources on Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG) algorithms for more advanced understanding of specific solutions.
