---
title: "How can reinforcement learning control a cart pole using only screen pixel data?"
date: "2025-01-30"
id: "how-can-reinforcement-learning-control-a-cart-pole"
---
The core challenge in controlling a cart pole using only screen pixel data lies in bridging the gap between high-dimensional raw visual input and the low-dimensional action space required to balance the pole. This effectively necessitates learning a suitable *representation* of the visual input, a crucial step before applying reinforcement learning techniques. My experience working on robotics projects, particularly those involving visual servoing, has highlighted the importance of this initial representation learning stage. Directly applying RL algorithms to raw pixel data often leads to instability and excessively long training times due to the curse of dimensionality. Therefore, this problem is less about RL itself and more about the architecture used to preprocess pixel data into a manageable state representation.

The fundamental problem is that the raw pixel data contains far more information than is necessary for balancing the pole. Consider a single image frame: it contains colors, lighting conditions, background details, and various other pixel values. These features are either irrelevant or redundant for determining the pole's angle and cart’s position. We need to extract features that correlate to these critical parameters, and then use those features to make decisions. Deep learning architectures, specifically Convolutional Neural Networks (CNNs), are commonly employed for this purpose. CNNs are particularly well-suited to learn spatially invariant features, meaning they can recognize relevant patterns within an image regardless of their precise location, a key requirement for our task.

The architecture would therefore typically comprise a CNN as a feature extractor, followed by a fully-connected layer or layers to map these extracted features into action probabilities or Q-values, depending on the reinforcement learning algorithm being used. The CNN effectively compresses the raw pixel data into a lower-dimensional representation, which is then used by the RL agent. The agent, in turn, outputs action probabilities or Q-values, which are then used to select actions.

Here's a simplified example architecture using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CartPoleCNN(nn.Module):
    def __init__(self, num_actions):
        super(CartPoleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)  # Assuming RGB input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(32 * 15 * 15, 128)  # Adjusted based on typical image size
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
if __name__ == '__main__':
    model = CartPoleCNN(num_actions=2)  # 2 actions for left/right
    dummy_input = torch.randn(1, 3, 64, 64) # Batch size 1, RGB image, 64x64 size
    output = model(dummy_input)
    print(output.shape) # Should print torch.Size([1, 2])
```
This code defines a basic CNN which takes in a 64x64 pixel RGB image and outputs two values, which could be used as action probabilities in an actor-critic method or Q-values in a Q-learning context. The choice of kernel sizes and stride parameters is deliberately simple, and should be tuned for a particular input resolution. The flatten layer in `forward` takes the 3D feature map (batch x height x width) from the second convolutional layer and transforms it into a 2D vector, which can be used by fully connected layers. The number of outputs is equal to the number of possible actions (left or right in this case).

Now, consider using the output of the CNN in an RL algorithm like Deep Q-Learning. Here's how you would use the previous `CartPoleCNN` within a basic Q-learning framework, also using PyTorch:

```python
import torch.optim as optim
import random
import numpy as np

def epsilon_greedy_action(q_values, epsilon):
  if random.random() < epsilon:
    return random.randint(0, 1) # Random action
  else:
    return torch.argmax(q_values).item()

def train(model, optimizer, memory, gamma, batch_size):
  if len(memory) < batch_size:
    return
  batch = random.sample(memory, batch_size)
  states, actions, next_states, rewards, dones = zip(*batch)
  states = torch.stack(states).float()
  next_states = torch.stack(next_states).float()
  actions = torch.tensor(actions).long()
  rewards = torch.tensor(rewards).float()
  dones = torch.tensor(dones).float()

  q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
  next_q_values = model(next_states).max(1)[0]
  expected_q_values = rewards + gamma * next_q_values * (1 - dones)

  loss = F.mse_loss(q_values, expected_q_values)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

if __name__ == '__main__':
    model = CartPoleCNN(num_actions=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99
    memory = [] # Experience replay memory
    batch_size = 32
    episodes = 1000

    for episode in range(episodes):
        # Assume get_initial_state_from_pixels, take_action_get_next_state_reward functions.
        state = get_initial_state_from_pixels()  # Mocked function to get the initial state
        done = False
        while not done:
          q_values = model(state.unsqueeze(0))
          action = epsilon_greedy_action(q_values, epsilon)
          next_state, reward, done = take_action_get_next_state_reward(action) # Mocked. Returns tensor, reward, done
          memory.append((state, action, next_state, reward, done)) # For experience replay
          state = next_state

          train(model, optimizer, memory, gamma, batch_size)

        epsilon = max(0.01, epsilon*epsilon_decay)

        if episode % 50 == 0:
            print(f"Episode: {episode}, Epsilon: {epsilon}")
```

This example outlines a basic Q-learning loop, utilizing an epsilon-greedy policy for action selection. The `train` function pulls a batch of memories from the experience replay buffer. Note that I am assuming the existence of functions `get_initial_state_from_pixels` and `take_action_get_next_state_reward`, these would interface with the simulation environment to acquire pixel data and execute actions and are not provided due to environment dependencies. The code uses MSE loss and standard backpropagation to update network weights. The epsilon decay provides a gradual shift from exploration to exploitation.

Finally, to further illustrate the interplay between the CNN and the RL algorithm, let's consider how a custom loss function can be used to encourage smoothness of the learned Q-function, and also integrate a moving average target network to increase stability:

```python
def train_with_target(model, target_model, optimizer, memory, gamma, batch_size, tau):
  if len(memory) < batch_size:
      return
  batch = random.sample(memory, batch_size)
  states, actions, next_states, rewards, dones = zip(*batch)
  states = torch.stack(states).float()
  next_states = torch.stack(next_states).float()
  actions = torch.tensor(actions).long()
  rewards = torch.tensor(rewards).float()
  dones = torch.tensor(dones).float()

  q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
  next_q_values = target_model(next_states).max(1)[0] # Target network used here
  expected_q_values = rewards + gamma * next_q_values * (1 - dones)

  loss = F.mse_loss(q_values, expected_q_values)

  # Optional smoothness term (encourages similar q-values for similar states)
  # This is a placeholder and requires definition of an embedding comparison method
  # smoothness_loss = some_embedding_comparison_func(states)

  # combined_loss = loss + smoothness_loss # if smoothing term is included

  optimizer.zero_grad()
  loss.backward() #combined_loss.backward()
  optimizer.step()

  # Soft update target network parameters
  for target_param, param in zip(target_model.parameters(), model.parameters()):
      target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
if __name__ == '__main__':
    model = CartPoleCNN(num_actions=2)
    target_model = CartPoleCNN(num_actions=2)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    gamma = 0.99
    tau = 0.005 # Soft update parameter
    epsilon = 1.0
    epsilon_decay = 0.99
    memory = [] # Experience replay memory
    batch_size = 32
    episodes = 1000
    for episode in range(episodes):
      state = get_initial_state_from_pixels() # Mock
      done = False
      while not done:
        q_values = model(state.unsqueeze(0))
        action = epsilon_greedy_action(q_values, epsilon)
        next_state, reward, done = take_action_get_next_state_reward(action) # Mock
        memory.append((state, action, next_state, reward, done))
        state = next_state

        train_with_target(model, target_model, optimizer, memory, gamma, batch_size, tau)
      epsilon = max(0.01, epsilon*epsilon_decay)
      if episode % 50 == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon}")
```

This modification shows how to introduce a target network, used to provide more stable training by reducing the correlation between the current Q-values and the target Q-values. It also illustrates a space for custom loss terms that can be introduced to regularize the learning process, although I left it out for brevity.

For further exploration of this subject, I recommend reviewing materials on deep reinforcement learning, specifically the use of CNNs for visual feature extraction and techniques like DQN, DDPG, or PPO. Books such as "Reinforcement Learning: An Introduction" by Sutton and Barto are excellent resources for the underlying theory. For a practical implementation perspective, consider researching tutorials and documentation on libraries like PyTorch, TensorFlow, and OpenAI Gym. Specific papers on the use of convolutional networks for RL control tasks will also prove helpful.
