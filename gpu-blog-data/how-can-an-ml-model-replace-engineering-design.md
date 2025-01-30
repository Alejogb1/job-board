---
title: "How can an ML model replace engineering design software?"
date: "2025-01-30"
id: "how-can-an-ml-model-replace-engineering-design"
---
The core challenge in directly substituting engineering design software with a machine learning (ML) model lies not in its computational power, but in the inherent nature of design itself – the iterative, often human-guided process of balancing constraints, performance targets, and manufacturability considerations. Engineering design involves both optimization and exploration; current ML excels at the former within well-defined spaces but struggles with the latter where novelty and unstructured problems often reside.

As a former lead engineer in an aerospace firm, I've seen firsthand how CAD systems like SolidWorks and finite element analysis (FEA) tools like ANSYS are employed. They provide robust environments for parametrically defining designs, simulating their behavior, and iteratively refining them based on calculated or observed performance. These workflows are deeply entrenched and depend on a combination of physics-based modeling, design heuristics, and the engineer's domain expertise. An ML model, in its current state, cannot simply replicate this entire pipeline. It lacks the inherent understanding of physical laws and the capability to generate designs from the ground up based solely on abstract performance criteria. Instead, where ML excels, and where its integration into the design process is most promising, lies in specific, targeted applications within the existing design framework.

One area where ML is already demonstrably useful is in optimization. Imagine a scenario where we need to minimize the weight of a bracket while maintaining a specific stiffness. With traditional methods, this might involve an engineer manually modifying dimensions, running simulations, and iterating over multiple designs. An ML model, specifically a reinforcement learning (RL) agent, can be trained to explore the design space much more efficiently. Here is a conceptual example, implemented using Python and PyTorch for illustrative purposes:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple function simulating the bracket's performance
def simulate_bracket(params):
  # Simplified model: params[0] is thickness, params[1] is width
  thickness = params[0]
  width = params[1]
  weight = thickness * width  # Simulate weight
  stiffness = 1000 * thickness * width**2 # Simplified stiffness model
  return weight, stiffness

class SimpleAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) # Actions are within [-1,1]
        return x

def train_agent(agent, num_episodes=1000):
  optimizer = optim.Adam(agent.parameters(), lr=0.01)
  gamma = 0.99
  for episode in range(num_episodes):
    state = torch.tensor([0.5], dtype=torch.float32) # Initial state, assuming one input of state
    total_reward = 0

    for step in range(10): # Simulate 10 design modification steps
      action = agent(state)
      action_np = action.detach().numpy()
      params = [max(0.1, min(1, state.item() + action_np[0])), max(0.1, min(1,state.item()+action_np[0]))] # Example: Adjust params based on action
      weight, stiffness = simulate_bracket(params)
      reward = -weight if stiffness> 500 else -100 # Weight minimization subject to a stiffness requirement
      total_reward += reward
      next_state = torch.tensor([params[0]], dtype=torch.float32) #Update state
      optimizer.zero_grad()
      loss = -reward # Gradient descent will favor actions with larger rewards
      loss.backward()
      optimizer.step()

      state = next_state
    if episode % 100 ==0:
      print(f"Episode: {episode}, Total Reward: {total_reward}")

agent = SimpleAgent(state_size=1, action_size=1)
train_agent(agent)
```
This code demonstrates a basic RL agent attempting to adjust parameters (thickness and width) to minimize weight, while enforcing a stiffness requirement using a simplified simulation model. The agent uses a neural network to determine the next action (parameter adjustment). Note the simplicity of this illustrative example; a practical application would require significantly more complex networks and simulation models. This example shows how ML could augment design workflows by quickly exploring the design space, allowing engineers to focus on high-level design decisions.

Another promising area is surrogate modeling. Detailed FEA simulations can be computationally expensive, often taking hours or even days to complete. An ML model, specifically a regression model, can be trained on a set of design parameters and corresponding FEA results. Once trained, this model, often called a surrogate model, can predict the performance of new designs virtually instantaneously, significantly speeding up the design iteration process. Consider the following example using Scikit-learn:
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate dummy training data
num_samples = 1000
thickness_range = (0.1, 1.0)
width_range = (0.5, 2.0)

thicknesses = np.random.uniform(thickness_range[0], thickness_range[1], num_samples)
widths = np.random.uniform(width_range[0], width_range[1], num_samples)

# Simulate output of a expensive simulation
stresses = 1000 * thicknesses * widths**2 + np.random.normal(0, 100, num_samples)  # Add noise
X = np.array(list(zip(thicknesses, widths)))
y = stresses

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction on a new design
new_design = np.array([[0.6, 1.2]])
predicted_stress = model.predict(new_design)
print(f"Predicted stress: {predicted_stress}")
```
This example shows a Random Forest regressor trained to predict stress based on thickness and width inputs. While this a simplified scenario, in practice this approach can be applied to much more complicated design parameters and FEA results. This type of surrogate model greatly reduces the time spent running computationally intensive simulations. The trade-off is the need to first generate sufficient data from real simulations to effectively train the model, which itself will take time and computational resources.

Finally, ML models can aid in generative design. Traditional design often starts with an engineer's initial concept, refined through iterative loops. A generative model, trained on a large dataset of existing designs, can generate new, potentially novel designs that satisfy certain design constraints. For example, consider a simple variational autoencoder (VAE) trained on data from a 2D shape model, represented as an array of points.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2_mean = nn.Linear(256, latent_size)
        self.fc2_logvar = nn.Linear(256, latent_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, latent_size)
        self.decoder = Decoder(latent_size, input_size)

    def reparameterize(self, mean, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

def loss_function(recon_x, x, mean, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kld

def train_vae(vae, data, num_epochs=1000, batch_size=64, lr=0.001):
  optimizer = optim.Adam(vae.parameters(), lr = lr)
  data_tensor = torch.tensor(data, dtype = torch.float32)
  data_loader = torch.utils.data.DataLoader(data_tensor, batch_size= batch_size, shuffle = True)

  for epoch in range(num_epochs):
      for batch in data_loader:
        recon_batch, mean, logvar = vae(batch)
        loss = loss_function(recon_batch, batch, mean, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      if epoch % 100 ==0:
         print(f"Epoch: {epoch}, Loss: {loss.item()}")


#Generate dummy 2D shape data
num_samples = 1000
num_points = 20
shapes = np.random.rand(num_samples, num_points*2)

input_size = num_points * 2
latent_size = 20
vae = VAE(input_size, latent_size)
train_vae(vae, shapes)

# Generate new shape
with torch.no_grad():
  z = torch.randn(latent_size)
  new_shape = vae.decoder(z).numpy()
  new_shape = new_shape.reshape(-1,2)
  plt.scatter(new_shape[:,0], new_shape[:,1])
  plt.show()
```

The code initializes and trains a VAE model on the generated shape data. The VAE attempts to learn latent space representation of the input shapes. After training it can generate new shapes by sampling from the latent space. While the shapes generated in the simplified example are not meaningful, in practical use the model could be trained to learn shapes representative of components of an engineering design allowing to generate novel, albeit rough, design proposals.

In summary, while fully replacing engineering design software with ML is not currently feasible, the integration of ML techniques into the design process is already transforming the field. For example, optimization techniques using reinforcement learning can dramatically reduce time needed to optimize design parameters. Surrogate models can bypass expensive simulations using data driven approximation techniques. Finally, generative approaches are opening a way for exploring new design possibilities. These applications do not aim to replace engineers, but rather to empower them with more efficient and creative design tools.

To further explore these topics, resources on reinforcement learning like Sutton and Barto's textbook, documentation for machine learning libraries such as TensorFlow and PyTorch, and publications on surrogate modeling and generative design techniques would be valuable.
