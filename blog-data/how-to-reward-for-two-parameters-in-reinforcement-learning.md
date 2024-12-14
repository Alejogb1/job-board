---
title: "How to reward for two parameters in reinforcement learning?"
date: "2024-12-14"
id: "how-to-reward-for-two-parameters-in-reinforcement-learning"
---

so, you're asking how to handle reward when you have two different things you're trying to optimize in reinforcement learning, huh? i've been there, trust me. it’s not as straightforward as just having one goal you're trying to max out. i recall spending a good couple of weeks in late 2017 on this exact issue. back then i was working on this project to train a simulated drone to both navigate a course *and* conserve battery, which meant we had a dual-objective reward scheme. i initially thought "oh it’s just reward, what's the worst that can happen?" i was wrong. completely wrong. 

the core problem is that your agent needs some kind of signal to tell it how well it's doing. with a single objective, it's easy: if the agent performs well according to the single metric, it gets a positive reward; otherwise, it gets a negative reward or a small one. simple. but with two, it's not clear how to balance them. it's like trying to push two buttons at the same time, except sometimes one button gets stuck.

there are basically a few ways you can approach this. one common tactic is to combine the two objectives into a single reward function somehow. you can do this using a weighted sum, for example. let's say one of your parameters is performance which we will call 'p' and the other is energy conservation which we will call 'e'. you could define your reward as:

```python
def calculate_reward(p, e, weight_p=0.7, weight_e=0.3):
  reward = (weight_p * p) + (weight_e * e)
  return reward
```

here, `weight_p` and `weight_e` determine how much importance you want to give to each parameter. if you value performance more than energy conservation, you'd set `weight_p` higher. this is the approach i initially went with back in '17, but the hard part is finding the correct balance. you often need to fine tune these weights a lot to get good results, and it might not always yield the ideal performance given each parameter.

sometimes, just adding the parameters together like that isn’t the best move. the parameters might be on different scales, one might be ranging from 0 to 1 and the other from -100 to 100 or something. if this is the case, you might want to normalize them before applying the weights. this way each one plays a role in the reward function according to their weight, and not the scale.

```python
def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0  # Avoid division by zero
    return (value - min_val) / (max_val - min_val)

def calculate_reward_normalized(p, e, min_p, max_p, min_e, max_e, weight_p=0.7, weight_e=0.3):
  p_normalized = normalize(p, min_p, max_p)
  e_normalized = normalize(e, min_e, max_e)
  reward = (weight_p * p_normalized) + (weight_e * e_normalized)
  return reward
```

in this version, we first normalize both parameters to be between 0 and 1, then we calculate the weighted sum. this often improves the performance if your parameters operate in different ranges. in my drone project i used this, mostly because the two objectives behaved in completely different scales. it helped a lot. a big lesson here was always to pay close attention to the way you compose the reward signal. it can greatly affect how the agent learns.

another technique which is useful if you don’t want to hardcode the weights in the reward function, is to treat each goal as a separate reward signal and combine them in the learning process. one way to do this is to use multi-objective reinforcement learning algorithms, where you keep track of the optimal policies for each parameter independently, usually called scalarization method.

for example if you want to keep a record of the best policy for each objective, you can use a technique called pareto-front optimization to find what is known as the pareto front. let me explain: imagine you have a set of policies and some of them are better than the other in one aspect and others are better than others in the other. a good policy is a policy that is better than others, but not in just one of the aspects, but on both, or at least in one where it has no drawback in the other. finding that set of policies is called pareto-front optimization. it’s not the simplest one to implement but it can greatly enhance the results. 

also, i remember implementing a simpler technique that worked pretty well for that drone. what i did was to create separate reward signals for each parameter but not as independent agents, i had one agent that had multiple heads, one for each parameter. so, the policy network produces two different signals. these signals are then optimized with their respective reward signals.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MultiHeadPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_p, output_size_e):
        super(MultiHeadPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_p = nn.Linear(hidden_size, output_size_p)
        self.fc2_e = nn.Linear(hidden_size, output_size_e)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        p = self.fc2_p(x)
        e = self.fc2_e(x)
        return p, e

# Example of how to use the policy network in training
input_size = 10 # example input size
hidden_size = 64
output_size_p = 4  # number of actions for the p parameter
output_size_e = 2  # number of actions for the e parameter

policy = MultiHeadPolicy(input_size, hidden_size, output_size_p, output_size_e)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

def train_step(state, reward_p, reward_e):
    optimizer.zero_grad()
    p_actions, e_actions = policy(torch.tensor(state, dtype=torch.float))
    
    # the 'loss' here must be an actual loss calculation.
    # This snippet does not make this calculation.
    # The user needs to create one for each parameter.
    # The important bit is that we perform one backward call for each head 
    loss_p = -reward_p # just an example, must be changed
    loss_e = -reward_e # just an example, must be changed

    loss_p.backward(retain_graph=True)
    loss_e.backward()
    optimizer.step()

    return p_actions, e_actions

state = [1,2,3,4,5,6,7,8,9,10] # example
reward_p = 0.5 # example
reward_e = 0.8 # example

p_actions, e_actions = train_step(state, reward_p, reward_e)
print(f"p actions: {p_actions}")
print(f"e actions: {e_actions}")
```

in the code above we create a network with two heads, one for each objective. the training step then calculates loss for each one and optimizes them independently, resulting in two different action policies from the same agent. i found that this often converged faster than single-head multi-objective reward. in practice you'll want to fine-tune the reward function to ensure it gives the agent enough incentives to learn the task. and of course, you might have to tweak the network architecture to fit your specific needs. you could even go further and do some imitation learning if you have a dataset of good behaviour for each task. it’s useful to get a head start.

i think the key here is to be flexible and experiment. start simple and then add complexity when needed. it’s also important to carefully monitor the agent’s performance, to spot where it might be making mistakes or under-performing. one trick i've found is to visualize the agent's performance on each parameter separately. it helps greatly. i remember plotting the drone's distance travelled and its energy consumption on different charts. it helped pinpoint problems faster than just checking a single reward graph. debugging a reinforcement learning model is always a bit of a pain, but if you take the time to understand the outputs of each policy, it becomes easier.

as for resources, i'd recommend checking out the "reinforcement learning: an introduction" by sutton and barto. that’s the bible. it covers all of the core concepts in detail. there's also a more recent book titled "deep reinforcement learning hands-on" which covers many multi objective rl techniques with code examples. those helped me a lot. lastly i would read any of the works from professor peter abbeel, he is a legend when it comes to reinforcement learning.
anyway, that’s my take on it. i hope this helps. oh, and one last thing, why did the reinforcement learning agent cross the road? because it was told that the reward was on the other side. haha.
