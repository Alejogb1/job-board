---
title: "How can reinforcement learning control a cart pole using only screen pixel input?"
date: "2024-12-23"
id: "how-can-reinforcement-learning-control-a-cart-pole-using-only-screen-pixel-input"
---

Alright, let's delve into this. I've actually tackled cart-pole balancing from pixel input in a past project involving a simulated robotic arm, and it's definitely a problem with several layers of complexity. We're moving away from clean, state-based representations and into the realm of raw sensory input, which requires a very different approach.

Essentially, the crux of the issue lies in bridging the gap between pixel data—which is essentially a matrix of color values—and the actions we want to take on the cart, namely, pushing it left or right. We can't feed pixels directly into a standard reinforcement learning algorithm like Q-learning or policy gradients. We need an intermediary stage, a representation that the RL algorithm can actually work with.

This is where convolutional neural networks (CNNs) come into play. CNNs are exceptional at learning spatial hierarchies, which makes them perfect for image processing. Think of them as a series of filters, each looking for specific features in the image. Early layers might detect edges and corners, while later layers combine these features to recognize more complex objects, in our case, the cart and pole. This learned representation is the input we'll ultimately use for our RL agent.

My go-to approach, and what I used in my simulated robotic arm project, is to combine CNNs with a deep Q-network (DQN). The CNN acts as a feature extractor, and the DQN uses these extracted features to learn the optimal action-value function. This means, it learns to predict how good a particular action would be in a particular state.

Here's how it usually breaks down in practice:

1.  **Pixel Input:** We start with the raw screen capture, let's assume it's grayscale to simplify our initial implementation.
2.  **Preprocessing:** The pixel data undergoes a bit of preprocessing like resizing and possibly normalization to a range like \[0, 1] to improve training.
3.  **CNN Feature Extraction:** The preprocessed images are fed through a CNN. Typically, we'd use a structure with multiple convolutional layers, pooling layers, and finally, some fully connected layers. The output of the final fully connected layer is our feature vector, a compressed representation of the visual scene.
4.  **DQN Agent:** This feature vector is passed to the DQN agent. The DQN has a network that approximates the action-value function (Q-function).
5.  **Action Selection:** Based on the Q-values predicted by the DQN, our agent chooses an action (push left, push right). Initially, this will probably involve a large degree of random exploration (often using an epsilon-greedy strategy), but gradually, the agent will exploit its knowledge of the best action given its learned Q function.
6.  **Reward and Training:** The agent then interacts with the environment, observes the new state (new pixels), and receives a reward (based on balancing the pole). These transitions (state, action, reward, next state) are used to update the parameters of the DQN via backpropagation using the Bellman equation.

Let me illustrate with a couple of simplified code examples using TensorFlow/Keras. I will use python for the examples.
Firstly, here is the CNN architecture:

```python
import tensorflow as tf
from tensorflow.keras import layers

def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu')
    ])
    return model
```

This `create_cnn_model` function defines a simple convolutional network that processes images of a specified shape. This model will output a flattened vector that represents features extracted from the image, which will then feed into the DQN.

Now, let's look at how we might construct the DQN architecture, which utilizes the output from the CNN:

```python
def create_dqn_model(cnn_output_size, num_actions):
  model = tf.keras.Sequential([
      layers.Dense(cnn_output_size, activation='relu', input_shape=(512,)), # assumes previous CNN out of 512
      layers.Dense(num_actions)
    ])
  return model
```

This `create_dqn_model` function sets up a fully connected neural network. The input layer's size is determined by the output size of our CNN, and the output size is equal to the number of actions our agent can take (e.g. 2 for push left or right).

Finally, here is some pseudo code on how we might train:

```python
import numpy as np
import random

# Parameters:
image_shape = (80, 80, 1) # Height, Width, Channels for grayscale
num_actions = 2
epsilon = 1.0
epsilon_decay_rate = 0.995
min_epsilon = 0.01
gamma = 0.99
batch_size = 32
memory_size = 10000

cnn = create_cnn_model(image_shape)
dqn = create_dqn_model(512, num_actions) # CNN output is 512 based on the architecture defined above
dqn_target = create_dqn_model(512, num_actions)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
memory = []

def preprocess_image(image):
  resized_image = tf.image.resize(image, (80, 80))
  grayscale_image = tf.image.rgb_to_grayscale(resized_image)
  normalized_image = tf.cast(grayscale_image, tf.float32)/255.0
  return normalized_image.numpy()

def epsilon_greedy_policy(state, epsilon):
    if random.random() > epsilon:
      feature_vector = cnn.predict(np.expand_dims(state, axis=0))
      q_values = dqn.predict(feature_vector)
      action = np.argmax(q_values)
    else:
        action = random.randint(0, num_actions-1)
    return action

def train_dqn(memory, optimizer, batch_size):
  if len(memory) < batch_size:
      return
  batch = random.sample(memory, batch_size)
  states = np.array([transition[0] for transition in batch])
  actions = np.array([transition[1] for transition in batch])
  rewards = np.array([transition[2] for transition in batch])
  next_states = np.array([transition[3] for transition in batch])
  dones = np.array([transition[4] for transition in batch])

  feature_vectors = cnn.predict(states)
  next_feature_vectors = cnn.predict(next_states)

  with tf.GradientTape() as tape:
    q_values = dqn(feature_vectors)
    next_q_values = dqn_target(next_feature_vectors)
    max_next_q = np.max(next_q_values, axis=1)
    targets = rewards + (gamma * max_next_q * (1 - dones))
    one_hot_actions = tf.one_hot(actions, num_actions)
    chosen_q_values = tf.reduce_sum(one_hot_actions * q_values, axis=1)
    loss = tf.keras.losses.MeanSquaredError()(targets, chosen_q_values)

  gradients = tape.gradient(loss, dqn.trainable_variables)
  optimizer.apply_gradients(zip(gradients, dqn.trainable_variables))


def update_target_network(dqn, dqn_target):
    dqn_target.set_weights(dqn.get_weights())

# Training loop:
for episode in range(500): # arbitrary number of episodes
  state = env.reset()
  state = preprocess_image(state)
  done = False
  episode_reward = 0

  while not done:
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, _ = env.step(action)
    next_state = preprocess_image(next_state)
    memory.append((state, action, reward, next_state, done))
    if len(memory) > memory_size:
          memory.pop(0)
    train_dqn(memory, optimizer, batch_size)
    state = next_state
    episode_reward += reward
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay_rate

    if episode % 10 == 0:
          update_target_network(dqn, dqn_target)
  print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}")
```
This pseudo-code provides a basic outline of how the CNN and DQN can be combined for training, using a buffer, experience replay, and target network updates. Note that this is a simplified version and may require fine tuning and improvements such as adding noise to actions or using double DQN for better performance and stability, but its a reasonable starting point.

For deeper dives into this subject, I would strongly recommend looking into "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville; it provides a solid foundational understanding of deep learning architectures. For specifically reinforcement learning, "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto is the definitive text. Also, research papers on topics such as the "Playing Atari with Deep Reinforcement Learning" by Volodymyr Mnih et al. and its extensions, is critical for understanding how these networks are trained using pixel inputs.

In practice, you'll find the training process is quite sensitive to hyperparameters like learning rate, batch size, network architecture, and exploration strategy. Experimentation is definitely key. You might also consider adding techniques such as frame stacking which helps the network infer motion and adding a replay buffer for better training stability.

This is a complex problem, but very rewarding when you see it come together. Using a CNN to extract meaningful features from pixel data and feeding this representation to a reinforcement learning agent like DQN allows us to build agents that can interact with environments directly from their sensory experiences. It's a fascinating area, and with a little practice and patience, you can definitely achieve great results with this methodology.
