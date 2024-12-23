---
title: "How can I create custom multi-agent environments for Deep RL and MADDPG?"
date: "2024-12-23"
id: "how-can-i-create-custom-multi-agent-environments-for-deep-rl-and-maddpg"
---

Alright, let's tackle this one. The creation of custom multi-agent environments for deep reinforcement learning, specifically with MADDPG (Multi-Agent Deep Deterministic Policy Gradient) in mind, isn't a walk in the park, but it's certainly a solvable challenge. I've spent quite some time down this rabbit hole, having once been tasked with simulating collaborative autonomous drones for a disaster response scenario. It taught me a lot about the nuances involved.

First, we need to move beyond thinking of environments as mere static backdrops. In multi-agent RL, the environment *is* the dynamic stage upon which the agents interact, and their combined actions fundamentally alter its state. Therefore, designing a useful environment requires careful consideration of its structure, dynamics, and observability, especially when dealing with MADDPG, which inherently operates within a partially-observable setting for each individual agent.

At its core, defining a multi-agent environment involves these key steps: defining the state space, action space, transition function, and reward function. Let's break this down further in the context of MADDPG:

1. **State Space Definition:** This is arguably the trickiest part. In a single-agent setting, it's usually straightforward—like the position and velocity of a robot. But with multiple agents, we have to consider how much information each agent has about the others. Do they have full observability (i.e., they know the states of all other agents), or partial observability (i.e., they only see what's in their immediate vicinity)? For MADDPG, partial observability is key as each agent has its own critic conditioned on the observations. The state space, therefore, should ideally represent the local observation for *each* agent, concatenated with any relevant global information that the environment makes available. This could be done using, for example, a dictionary where each key represents agent id and value is an array or tuple representing local state. We must also be very clear about the type of data this will contain (e.g., floats for position, integers for agent ids, etc) and ensure this is consistent throughout the environment.

2. **Action Space Definition:** Similar to state space, the action space for a multi-agent environment must be explicitly defined for each agent. You need to specify the set of actions an agent can take at each step, whether discrete (e.g., move north, south, east, west) or continuous (e.g., a steering angle). For MADDPG, continuous actions are more common, and it requires the action space to be the same type for each agent. Often, we define a bounded continuous space in multi-agent systems. For example, a steering action space could be a float between -1 and 1 where -1 means full left and 1 means full right.

3. **Transition Function:** This defines how the environment's state changes in response to the agents’ actions. This function will need to update not just the position of each agent but also simulate any interactions between the agents. These interactions are key to developing useful multi-agent behaviour. It should take the current state, the actions of each agent, and return the next state and a reward associated with the combination of agents' actions.

4. **Reward Function:** Here, it is important to carefully consider what we want the agents to learn. With MADDPG, where there is a centralized critic, it can be common to define a global reward to incentivize agents to work together rather than against each other. The reward function must be shaped in a manner that encourages cooperation and the desired overall system behavior. For example, in the disaster scenario I worked on, we gave agents rewards for successfully identifying affected people and reduced the reward when they competed with each other for the same goal. This was combined with a global reward for all people rescued by all agents to encourage collaboration.

Now, let me illustrate this with some hypothetical examples.

**Example 1: Cooperative Navigation**

Imagine a simple 2D grid world where two agents need to reach target locations. Each agent observes its own position and the relative positions of the target and the other agent. The state space is a tuple for each agent, representing `(x_position, y_position, target_x, target_y, relative_agent_x, relative_agent_y)`. The action space is continuous: `(delta_x, delta_y)`, representing movement in x and y directions respectively. A simplified version might look like this:

```python
import numpy as np

class CoopNavEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.agents_pos = np.random.randint(0, grid_size, size=(2, 2))
        self.targets = np.random.randint(0, grid_size, size=(2, 2))

    def step(self, actions):
        next_states = []
        rewards = []
        for i, action in enumerate(actions):
            # Ensure the agent stays within grid
            new_x = max(0, min(self.grid_size - 1, int(self.agents_pos[i][0] + action[0])))
            new_y = max(0, min(self.grid_size - 1, int(self.agents_pos[i][1] + action[1])))
            self.agents_pos[i] = np.array([new_x, new_y])
            
            relative_x = self.targets[i][0] - new_x
            relative_y = self.targets[i][1] - new_y

            next_state = (new_x, new_y, self.targets[i][0], self.targets[i][1], 
                         self.agents_pos[1-i][0]- new_x, self.agents_pos[1-i][1] - new_y ) if i == 0 else \
                         (new_x, new_y, self.targets[i][0], self.targets[i][1],
                         self.agents_pos[0][0] - new_x, self.agents_pos[0][1]- new_y)
            
            next_states.append(next_state)
            reward = -1 # Default penalty for every step
            if np.array_equal(self.agents_pos[i], self.targets[i]):
                reward = 100 # Bonus for reaching the target
            rewards.append(reward)
        
        return next_states, rewards, False # Terminate is not defined here

    def reset(self):
        self.agents_pos = np.random.randint(0, self.grid_size, size=(2, 2))
        self.targets = np.random.randint(0, self.grid_size, size=(2, 2))

        return [
           (self.agents_pos[0][0], self.agents_pos[0][1], self.targets[0][0], self.targets[0][1], self.agents_pos[1][0] - self.agents_pos[0][0], self.agents_pos[1][1]-self.agents_pos[0][1]),
           (self.agents_pos[1][0], self.agents_pos[1][1], self.targets[1][0], self.targets[1][1], self.agents_pos[0][0] - self.agents_pos[1][0], self.agents_pos[0][1]-self.agents_pos[1][1])
         ]
```

**Example 2: Competitive Resource Gathering**

Now let's consider two agents competing for resources in a shared environment. The state space could be, for each agent, its current location and the locations of a finite set of resources and other agents. The action space would be discrete - choosing between moving in cardinal directions and gathering a resource.

```python
import numpy as np

class ResourceGatheringEnv:
    def __init__(self, grid_size=10, num_resources=3):
        self.grid_size = grid_size
        self.agents_pos = np.random.randint(0, grid_size, size=(2, 2))
        self.resources = np.random.randint(0, grid_size, size=(num_resources, 2))
        self.resources_available = np.ones(num_resources, dtype=bool)

    def step(self, actions):
        next_states = []
        rewards = []
        
        for i, action in enumerate(actions):
          
            new_x, new_y = self.agents_pos[i]
           
            if action == 0: # move up
                 new_y = max(0, self.agents_pos[i][1]-1)
            elif action == 1: # move down
                 new_y = min(self.grid_size - 1, self.agents_pos[i][1]+1)
            elif action == 2: # move left
                 new_x = max(0, self.agents_pos[i][0]-1)
            elif action == 3: # move right
                 new_x = min(self.grid_size - 1, self.agents_pos[i][0]+1)

            self.agents_pos[i] = np.array([new_x, new_y])
           
            
            reward = -0.1 # Penality for each step

            resource_collected = False
            for j, resource_pos in enumerate(self.resources):
                if self.resources_available[j] and np.array_equal(self.agents_pos[i], resource_pos) and action == 4:
                    reward = 1 # bonus for collecting a resource
                    resource_collected = True
                    self.resources_available[j] = False
                    break
            
            next_state = (new_x, new_y, *self.resources.flatten(), *self.resources_available, 
             self.agents_pos[1-i][0], self.agents_pos[1-i][1])  if i == 0 else \
                         (new_x, new_y, *self.resources.flatten(), *self.resources_available,
             self.agents_pos[0][0], self.agents_pos[0][1])


            next_states.append(next_state)
            rewards.append(reward)


        
        return next_states, rewards, not np.any(self.resources_available)
    
    def reset(self):
        self.agents_pos = np.random.randint(0, self.grid_size, size=(2, 2))
        self.resources = np.random.randint(0, self.grid_size, size=(3, 2))
        self.resources_available = np.ones(3, dtype=bool)

        return [
           (self.agents_pos[0][0], self.agents_pos[0][1], *self.resources.flatten(), *self.resources_available, self.agents_pos[1][0], self.agents_pos[1][1]  ),
           (self.agents_pos[1][0], self.agents_pos[1][1], *self.resources.flatten(), *self.resources_available, self.agents_pos[0][0], self.agents_pos[0][1]  )
        ]
```

**Example 3: A modified grid world where agents have a visibility range**

This adds a bit more complexity to the state space. In this example, consider two agents moving within a 2d grid. They can each see the map within a specific radius. This state space is more aligned with MADDPG in the sense that the agents only see their localized environment and there is no global information available.

```python
import numpy as np

class VisibilityEnv:
    def __init__(self, grid_size=10, visibility_range=3):
        self.grid_size = grid_size
        self.visibility_range = visibility_range
        self.agents_pos = np.random.randint(0, grid_size, size=(2, 2))

    def step(self, actions):
        next_states = []
        rewards = []

        for i, action in enumerate(actions):
            new_x, new_y = self.agents_pos[i]
            if action == 0:  # move up
                new_y = max(0, self.agents_pos[i][1]-1)
            elif action == 1:  # move down
                new_y = min(self.grid_size - 1, self.agents_pos[i][1]+1)
            elif action == 2:  # move left
                new_x = max(0, self.agents_pos[i][0]-1)
            elif action == 3:  # move right
                new_x = min(self.grid_size - 1, self.agents_pos[i][0]+1)
            
            self.agents_pos[i] = np.array([new_x, new_y])
            
            
            local_map = np.zeros((2*self.visibility_range + 1, 2 * self.visibility_range + 1))

            for y in range(-self.visibility_range, self.visibility_range+1):
                for x in range(-self.visibility_range, self.visibility_range+1):
                    current_y = new_y+ y
                    current_x = new_x + x
                    
                    if 0<= current_y < self.grid_size and 0 <= current_x < self.grid_size:
                        local_map[y+self.visibility_range][x+self.visibility_range] = 1

                        for j, agent_pos in enumerate(self.agents_pos):
                            if (agent_pos[0] == current_x and agent_pos[1] == current_y) and j != i:
                                local_map[y+self.visibility_range][x+self.visibility_range] = 2

            next_states.append(local_map)
            rewards.append(-0.1)

        return next_states, rewards, False

    def reset(self):
        self.agents_pos = np.random.randint(0, self.grid_size, size=(2, 2))
        local_maps = []
        for i in range(2):
          local_map = np.zeros((2*self.visibility_range + 1, 2*self.visibility_range + 1))
          for y in range(-self.visibility_range, self.visibility_range+1):
              for x in range(-self.visibility_range, self.visibility_range+1):
                  current_y = self.agents_pos[i][1] + y
                  current_x = self.agents_pos[i][0] + x

                  if 0<= current_y < self.grid_size and 0 <= current_x < self.grid_size:
                    local_map[y+self.visibility_range][x+self.visibility_range] = 1

                    for j, agent_pos in enumerate(self.agents_pos):
                        if (agent_pos[0] == current_x and agent_pos[1] == current_y) and j != i:
                              local_map[y+self.visibility_range][x+self.visibility_range] = 2

          local_maps.append(local_map)
        return local_maps
```

**Important Considerations and Further Reading:**

*   **Vectorization:** For performance, especially with many agents, ensure your environment's `step` function is vectorized using libraries like `numpy` to minimize for loops.
*   **Observation Spaces:** The observation space for each agent needs to be compatible with MADDPG training. It's very likely that you may need to utilize some form of feature extraction via a neural network prior to inputting the observation into your policy network. This can be especially true with environments like the VisibilityEnv.
*   **Complexity:** Don't begin with a very complex environment. It's best to test your MADDPG setup on something simple first and build from there. This prevents chasing after many bugs, and allows you to focus on one area of concern at a time.
*   **Frameworks:** If you do not want to write this all from scratch, consider using existing multi-agent RL frameworks such as the rllib package within the ray framework, or petting zoo which are designed to facilitate the creation of multi-agent environments.

For deep dives, I recommend checking out the following resources:

*   **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto:** This is the canonical text on RL and provides essential background knowledge.
*   **"Multi-Agent Systems: Algorithmic, Game-Theoretic, and Logical Foundations" by Yoav Shoham and Kevin Leyton-Brown:** This is a good text to gain a more theoretical understanding of multi-agent systems.
*   **"Trust Region Policy Optimization" by John Schulman et al.** This paper introduces a very common RL algorithm that uses the same policy gradient method as MADDPG. This is important to understand to get to grips with the underpinnings of MADDPG.
*   **"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" by Ryan Lowe et al.:** This is the original MADDPG paper. It provides details on the algorithm and rationale for its design.

Building custom environments is often iterative. The environment design choices directly impact the learned behavior of your agents, and it's a balancing act between modeling realistic situations and creating environments that are learnable. These examples are starting points, and they can be tailored based on your specific needs and the research direction you're aiming for. It is not an easy task, but having a clear understanding of these concepts should set you up well for success in this area. Good luck!
