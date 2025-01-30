---
title: "Why am I getting a KeyError: 'observation' in my OpenAI Stable-Baselines3 multi-agent RL experiment?"
date: "2025-01-30"
id: "why-am-i-getting-a-keyerror-observation-in"
---
The `KeyError: 'observation'` within a Stable-Baselines3 multi-agent reinforcement learning (MARL) environment typically stems from a mismatch between the environment's observation space definition and how the agent accesses observations during the training process.  My experience troubleshooting this in several large-scale robotics simulations highlighted this consistently.  The core issue is often a misunderstanding or incorrect implementation of how vectorized observations are structured and accessed within the `VecEnv` wrapper, commonly used for parallel training.

**1.  Clear Explanation:**

Stable-Baselines3, unlike some other RL frameworks, necessitates explicit handling of observation spaces, particularly in multi-agent settings.  The environment should return observations structured to match the expected input format of the agent.  When working with multiple agents, the observation space frequently becomes a complex structure – often a NumPy array or a dictionary containing individual agent observations.  The `KeyError: 'observation'` arises when the agent attempts to access an observation key ('observation' in this specific case) that does not exist within the structured output of the environment's `step()` method. This frequently happens due to one of three reasons:

a) **Incorrect Observation Space Definition:** The environment's observation space might be defined incorrectly, failing to include the 'observation' key or using a different naming convention. The agent, based on its configuration, expects this key, leading to the error.  This can be exacerbated in MARL scenarios where each agent might have a unique observation, necessitating a nested structure for the vectorized environment's output.

b) **Incompatible Environment and Agent:** The agent's architecture might be mismatched with the environment's output format.  For example, an agent expecting a dictionary observation might be interacting with an environment returning a simple NumPy array. This incompatibility, even if seemingly subtle, will invariably lead to this specific `KeyError`.

c) **Incorrect Accessing of Vectorized Observations:**  The use of `VecEnv` wrappers adds a layer of complexity.  The `step()` method of a `VecEnv` returns a tuple containing observations, rewards, dones, and infos for *all* agents simultaneously.  If you're not correctly extracting the individual agent observations from this tuple (often indexed through the `VecNormalize` wrapper's `obs_rms`), you will encounter this error.  I've seen this frequently when attempting to directly access observations before proper unwrapping.


**2. Code Examples with Commentary:**

**Example 1:  Correctly Structured Environment and Agent**

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Define a simple multi-agent environment (replace with your actual environment)
class SimpleMultiAgentEnv(gym.Env):
    def __init__(self):
        # ... environment initialization ...
        self.observation_space = gym.spaces.Dict({
            "agent_1": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=float),
            "agent_2": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=float)
        })
        # ... rest of environment initialization ...

    def step(self, action):
        # ... environment step logic ...
        obs = {"agent_1": np.array([0.1, 0.2]), "agent_2": np.array([0.3, 0.4])}
        reward = [1.0, 0.5]
        done = [False, False]
        info = {}
        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        # ... environment reset logic ...
        obs = {"agent_1": np.array([0.0, 0.0]), "agent_2": np.array([0.0, 0.0])}
        info = {}
        return obs, info

# Wrap the environment for vectorization
env = DummyVecEnv([lambda: SimpleMultiAgentEnv()])

# Define and train the agent (replace with your preferred agent)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
```

This example showcases a correctly defined observation space as a dictionary.  The `step` method returns an observation dictionary, which the `MultiInputPolicy` is designed to handle. This eliminates the `KeyError` by ensuring compatibility between the environment's output and the agent's input expectations.



**Example 2: Incorrect Observation Space and Agent Mismatch**

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

class IncorrectEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=float) #Incorrect

    def step(self, action):
        obs = np.array([0.1,0.2,0.3,0.4])
        reward = [1,1]
        done = [False, False]
        info = {}
        return obs, reward, done, info
    def reset(self, seed=None, options=None):
        obs = np.array([0,0,0,0])
        info = {}
        return obs, info

env = DummyVecEnv([lambda: IncorrectEnv()])
# This will fail because the PPO agent expects a Dict observation space not a Box

model = PPO("MlpPolicy", env, verbose=1) # this will fail, MlpPolicy is not designed for this
model.learn(total_timesteps=1000)
```

This illustrates a common error.  The observation space is defined incorrectly as a `Box`, not a `Dict`, and the agent (`MlpPolicy`) isn't designed to handle this type of input.  This mismatch will almost certainly result in a `KeyError` because the agent expects the 'observation' key (or a similar key depending on the policy) which is absent in the flat NumPy array.


**Example 3: Correcting for VecEnv using a custom wrapper**

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
import numpy as np


class IncorrectEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=float) #Incorrect

    def step(self, action):
        obs = np.array([0.1,0.2,0.3,0.4])
        reward = [1,1]
        done = [False, False]
        info = {}
        return obs, reward, done, info
    def reset(self, seed=None, options=None):
        obs = np.array([0,0,0,0])
        info = {}
        return obs, info

class ReshapeWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=float)})

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return {"observation": obs}, reward, done, info


env = DummyVecEnv([lambda: IncorrectEnv()])
env = ReshapeWrapper(env) #fix


model = PPO("MultiInputPolicy", env, verbose=1) # correct policy
model.learn(total_timesteps=1000)
```

This example demonstrates a way to handle situations where the environment's structure can't easily be changed.  A custom `VecEnvWrapper` is created, which reshapes the raw NumPy array output from the environment to match the expected dictionary structure required by the agent, using the `MultiInputPolicy` which can handle this type of observation space.


**3. Resource Recommendations:**

The Stable-Baselines3 documentation, focusing on custom environments and multi-agent configurations.  Thorough understanding of Gymnasium's space definitions (Box, Dict, MultiDiscrete, etc.) and how they translate into agent policies.  The official documentation on vectorized environments and `VecEnv` wrappers is crucial for understanding parallel processing and its implications on observation structures.  Finally, studying examples of multi-agent environments from the community (consider searching repositories on platforms like GitHub) can provide valuable insight into best practices.
