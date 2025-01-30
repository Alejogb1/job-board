---
title: "Are CustomEnv rollout summary statistics being monitored correctly using Stable-Baselines3?"
date: "2025-01-30"
id: "are-customenv-rollout-summary-statistics-being-monitored-correctly"
---
Monitoring rollout statistics in Stable-Baselines3, particularly within a custom environment context, requires meticulous attention to detail.  My experience debugging similar issues in reinforcement learning projects highlighted a crucial oversight: the default logging mechanisms within Stable-Baselines3 might not automatically capture all custom environment-specific metrics.  This often leads to incomplete or misleading rollout summaries, hindering effective model evaluation and hyperparameter tuning.  Therefore, directly accessing and logging relevant statistics from within the custom environment itself is paramount.

**1. Clear Explanation:**

Stable-Baselines3 provides robust tools for training reinforcement learning agents. However, its default logging functionalities primarily focus on standard metrics like episode rewards and lengths. When incorporating a custom environment, which often introduces unique performance indicators (e.g., success rate in a robotic manipulation task, collision frequency in a navigation task, or specific resource consumption metrics in a resource management scenario), these custom metrics are not intrinsically tracked.

The core issue stems from the separation between the agent's interaction with the environment and the logging mechanisms of the training algorithm.  The agent interacts with the environment via the `step()` method, receiving observations, rewards, and done signals. The training algorithm, in turn, processes these signals and logs the standard metrics.  Custom metrics generated within the `step()` method of a custom environment are not automatically passed to the training loop's logging functionality.

Therefore, the solution involves explicitly logging these custom metrics within the custom environment's `step()` method and then incorporating these logs into a suitable visualization or data storage mechanism. This ensures that these crucial statistics are captured alongside the default Stable-Baselines3 metrics, providing a complete picture of the agent's performance.  Failure to do so will result in incomplete rollout summaries, potentially leading to erroneous conclusions about the agent’s training progress and effectiveness.

**2. Code Examples with Commentary:**

**Example 1: Basic Custom Metric Logging**

This example demonstrates adding a simple custom metric (success rate) to a custom environment and logging it during each timestep.


```python
import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    # ... (Environment definition: observation_space, action_space, reset method) ...

    def step(self, action):
        observation, reward, done, info = self._step(action) #Internal step logic
        success = self._check_success(observation) #Custom success check
        info['success'] = success #Adding custom metric to info dictionary
        return observation, reward, done, info

    def _check_success(self, observation):
        #Implementation for checking success based on observation
        return np.random.rand() > 0.5 #Example: 50% success rate

#Training loop
env = CustomEnv()
model = ... #Your Stable-Baselines3 model
model.learn(total_timesteps=10000, callback=CustomCallback()) #Custom callback for handling logging

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        success_rate = self.locals['info'].get('success', 0)  #Access from the info dict
        self.logger.record("rollout/success_rate", success_rate) #Record custom metric
        return True
```


This example adds a 'success' key to the `info` dictionary returned by the `step()` method. A custom callback then accesses this key and logs it using the `self.logger.record()` method.  Note the reliance on the `info` dictionary for communication between the environment and the logging mechanism. This is crucial because the callback has no direct access to the internal state of the environment.


**Example 2:  Averaging Custom Metrics over Episodes**

The previous example logs the success at every step.  For better analysis, we may want to average the success rate over each episode.

```python
import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CustomEnv(gym.Env):
    # ... (Environment definition) ...
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_successes = []

    def step(self, action):
        observation, reward, done, info = self._step(action)
        success = self._check_success(observation)
        self.episode_successes.append(success)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_successes = []
        return observation

class CustomCallback(BaseCallback):
    def _on_episode_end(self):
        env = self.training_env
        episode_success_rate = np.mean(env.episode_successes)
        self.logger.record("rollout/episode_success_rate", episode_success_rate)
        return True
```

Here, the `episode_successes` list is maintained within the environment, accumulating success indicators throughout an episode. The `_on_episode_end` callback method calculates and logs the average success rate for each episode. This provides a clearer picture of overall performance than step-by-step logging.


**Example 3: Multiple Custom Metrics and Tensorboard Integration**

This example expands on the previous examples, adding multiple custom metrics and illustrating integration with Tensorboard for visualization.

```python
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import tensorflow as tf
import os

# Assume CustomEnv and _check_success are defined as before.  Add energy_consumed metric.
class CustomEnv(gym.Env):
    # ... (Environment definition) ...
    def step(self, action):
      observation, reward, done, info = self._step(action)
      success = self._check_success(observation)
      energy_consumed = self._calculate_energy(action) #New metric
      info['success'] = success
      info['energy_consumed'] = energy_consumed
      return observation, reward, done, info

    def _calculate_energy(self, action):
        return np.sum(np.abs(action)) # Example energy consumption

class TensorboardCallback(BaseCallback):
  def __init__(self, log_dir):
    super().__init__()
    self.log_dir = log_dir

  def _on_step(self) -> bool:
    success_rate = self.locals['info'].get('success', 0)
    energy_consumed = self.locals['info'].get('energy_consumed', 0)
    with tf.summary.create_file_writer(os.path.join(self.log_dir, 'train')).as_default():
      tf.summary.scalar('rollout/success_rate', success_rate, step=self.num_timesteps)
      tf.summary.scalar('rollout/energy_consumed', energy_consumed, step=self.num_timesteps)
    return True

#Training
env = DummyVecEnv([lambda: CustomEnv()])
log_dir = 'runs/experiment_1'
model = ... #Your model
model.learn(total_timesteps=10000, callback=[TensorboardCallback(log_dir)])
```

This sophisticated example introduces a second custom metric (`energy_consumed`) and leverages Tensorboard for visualization, providing a more comprehensive analysis of the training process. The custom callback now utilizes TensorFlow's `summary` functionalities to write these metrics to TensorBoard logs for interactive analysis.


**3. Resource Recommendations:**

For a deeper understanding of Stable-Baselines3, consult the official documentation.  Explore the `BaseCallback` class and its methods for custom logging functionalities.  Understanding vectorized environments and their impact on logging is crucial.  Finally, review the TensorFlow documentation regarding the usage of `tf.summary` for visualization, particularly if using Tensorboard.  Familiarity with NumPy for array manipulation is also essential for effectively processing and aggregating custom metrics.
