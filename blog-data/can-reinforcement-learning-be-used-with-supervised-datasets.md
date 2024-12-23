---
title: "Can reinforcement learning be used with supervised datasets?"
date: "2024-12-16"
id: "can-reinforcement-learning-be-used-with-supervised-datasets"
---

, let's tackle this one. I recall a project several years ago, involving a complex financial simulation where we initially had a large, labeled dataset from historical market data. Our first attempt, naturally, was a purely supervised model. It performed… adequately. But we yearned for more adaptable behavior, something that could make decisions beyond the strict confines of the training set, especially during times of unprecedented market volatility. That's when we started seriously considering whether reinforcement learning (rl) could be integrated, despite the fact that we *already* possessed a seemingly perfect supervised dataset.

The short answer is, yes, reinforcement learning can indeed leverage supervised datasets, but not in the direct, drop-in replacement sense that many might initially expect. It’s less about “replacing” supervised learning and more about augmentation. The core challenge, and the beauty of it, lies in reframing the supervised problem into an rl framework, effectively using the labeled data as a *guide* for an rl agent rather than its sole source of truth.

Here's the crux: supervised datasets provide us with examples of *what actions were taken* and, crucially, the associated *outcomes*. In the context of rl, these become potential demonstrations or, more accurately, *hints* towards desirable behavior. We are not merely trying to imitate actions as with pure imitation learning, but instead utilizing this information to train an agent that will explore and potentially find optimal strategies that go beyond the existing labels. The labeled data gives us a good *initial policy*, a starting point that is generally better than starting from scratch.

Let me break this down into practical terms using specific approaches:

**1. Pre-training the Policy Network with Supervised Learning:**

One of the most common techniques involves using your labeled dataset to pre-train the policy network of your rl agent. In a typical rl setup, we might initialize the policy network randomly. However, with supervised data, we can train this network to mimic the actions in the dataset. This has the effect of pushing the agent towards actions known to be, at least historically, beneficial. This phase treats the problem as a classification or regression task, mapping states from the dataset to the appropriate actions.

Here’s a simplified python snippet using pytorch to illustrate this:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume `states` and `actions` are tensors representing your labeled data.
# `actions` are encoded as integers for classification.
# `num_states` and `num_actions` are your input and output space dimensions

class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x # returns the logit

# Example placeholders, replace with your dataset parameters:
num_states = 10
num_actions = 3
states = torch.randn(1000, num_states) # 1000 states
actions = torch.randint(0, num_actions, (1000,)) # actions as class IDs


policy_net = PolicyNetwork(num_states, num_actions)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    logits = policy_net(states) # raw output before softmax
    loss = criterion(logits, actions) # cross-entropy loss
    loss.backward()
    optimizer.step()

print("Pre-training complete")

# Now `policy_net` has a pre-trained starting point for rl
```

**2. Using Behavioral Cloning as a Starting Point:**

Behavioral cloning, a specific sub-type of imitation learning, directly leverages the supervised dataset. This approach essentially trains a policy function by treating the expert's demonstrations in your labeled data as the desired behavior. We’re directly learning a mapping from observations (states in rl terms) to actions. The learned policy becomes the initial policy for our rl algorithm. However, this approach *alone* has the limitations of any supervised model: it will likely only perform as well as the training data and will struggle to generalize outside of it. This is where the rl fine-tuning comes into play.

Consider this python snippet:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#Assume data is already loaded in NumPy arrays
states_np = np.random.rand(1000, 5)  # 1000 states with 5 features
actions_np = np.random.randint(0, 4, 1000)  #1000 actions [0-3]

#Convert to Tensors
states = torch.tensor(states_np, dtype=torch.float32)
actions = torch.tensor(actions_np, dtype=torch.long)


class BehaviourCloningNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
      super(BehaviourCloningNetwork,self).__init__()
      self.fc1=nn.Linear(input_dim,64)
      self.fc2=nn.Linear(64,output_dim)

    def forward(self,x):
      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x

#Parameters setup
input_dim= states.shape[1]
output_dim = len(set(actions_np))

bc_policy = BehaviourCloningNetwork(input_dim,output_dim)
optimizer= optim.Adam(bc_policy.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

epochs=100
for epoch in range(epochs):
  optimizer.zero_grad()
  action_logits= bc_policy(states)
  loss = loss_function(action_logits, actions)
  loss.backward()
  optimizer.step()

print("Behavioral cloning complete. The policy network is ready for RL fine-tuning")
```

**3. Using the Supervised Data for Reward Shaping:**

Sometimes, we don’t directly use the supervised actions as a target, but rather the *outcomes* associated with those actions. These outcomes can be used to shape the reward function in our rl environment, biasing the agent toward learning patterns from the historical data. For example, if the supervised data shows a sequence of actions leading to a particularly good outcome, we can give higher rewards when the rl agent performs similar action sequences. Here, we're not dictating the *actions* but nudging the agent towards exploring *areas* of the state space which have historically led to better results.

This can be a bit abstract, so let’s use a simplified example of a custom reward function:

```python
import numpy as np
import random

# Assume we have historical data consisting of state sequences and their associated outcomes.
# Let's simulate the data with a sequence length of 10, 5 state features, and a reward.
# `historical_data` is a dictionary: keys are states, values are [sequence of states], and rewards
historical_data = {
  "seq1" : {
      'states': np.random.rand(10,5),
      'reward' : 0.8,
      'actions': np.random.randint(0,2, size = 10)
  },
    "seq2" : {
      'states': np.random.rand(10,5),
      'reward' : 0.2,
      'actions': np.random.randint(0,2, size = 10)
  },
  "seq3" : {
       'states': np.random.rand(10,5),
      'reward' : 0.95,
       'actions': np.random.randint(0,2, size = 10)
  }
}


def custom_reward_function(current_state, action, historical_data):

    # A basic implementation of 'similarity' comparison:
    for key, value in historical_data.items():
        if np.allclose(value["states"][0], current_state, atol=0.1): #check for close state start
            #if found, check if action taken is similar
            for i, hist_action in enumerate(value["actions"]):
              if hist_action == action:
                #give reward proportional to how good the historical outcome was
                  return value["reward"]
    return -0.05  # small penalty for not being close

#This function then serves as your environment reward function for a RL algorithm
#example with a random action and state
current_state=np.random.rand(5)
random_action = random.choice([0,1])
reward = custom_reward_function(current_state, random_action, historical_data)
print (f"reward = {reward}")
```

In essence, the reward function is no longer a fixed equation or formula but is informed by the outcomes of past actions, as seen in the supervised dataset. Note that this is extremely simplified; real-world reward shaping based on historical data can involve far more sophisticated similarity metrics, sequence matching, and potentially temporal difference methods.

**A note on resources:**

For a deep dive, I’d highly recommend starting with Sutton and Barto's "Reinforcement Learning: An Introduction", specifically its sections discussing imitation learning and reward shaping. “Deep Learning” by Goodfellow, Bengio, and Courville will cover the foundational concepts behind neural networks, which are often used as function approximators in rl. For a paper specific to combining supervised and reinforcement learning, try searching for research that focuses on ‘imitation learning with reinforcement learning’ or ‘guided policy search.’ Also, explore works that focus on 'reward shaping' in specific domains, as that will provide specific examples of how these ideas are applied practically.

In closing, the judicious use of supervised datasets in reinforcement learning unlocks the potential for creating more robust and effective agents. It's not a magic bullet, but a powerful technique when applied correctly, blending the strengths of both approaches. Remember, the key is the strategic *re-interpretation* of supervised data to fit an rl paradigm, focusing on learning optimal behavior instead of merely mimicking past performance.
