---
title: "How can I create an observation space with multilevel features in OpenAI Gym's SIMPLE package?"
date: "2024-12-23"
id: "how-can-i-create-an-observation-space-with-multilevel-features-in-openai-gyms-simple-package"
---

Alright, let's tackle this. I remember wrestling with a similar issue back in my days working on multi-agent simulations. The challenge, as you’ve pointed out, is effectively representing complex environments where the observations aren't just a single, flat vector, but instead have nested, hierarchical structures. OpenAI Gym’s `simple` package, while wonderfully straightforward for many tasks, does require a little bit of finesse to handle these more elaborate observation spaces.

The core problem isn't inherently with gym itself, but rather with how we define the `observation_space` attribute of our custom environment. With simple environments, you'd typically use `gym.spaces.Box` or `gym.spaces.Discrete` to represent things like sensor readings or a position on a grid. But when your observation is itself composed of different types of data, perhaps one part being numerical and another categorical, these single-type space definitions become inadequate. We need a composite space, specifically a `gym.spaces.Dict`.

Think of `gym.spaces.Dict` as a way to combine multiple independent observation spaces into one. Each key in the dictionary represents a distinct aspect of the observation, and the associated value is the corresponding space object (i.e., `Box`, `Discrete`, or even another `Dict`). This approach allows us to maintain the type and shape information of each component while treating the entire observation as a single unit.

Now, let's say we are dealing with a hypothetical environment where an agent has visual perception via three distinct camera feeds (each a grayscale image), and also receives some system status information consisting of two scalar readings: battery level (normalized between 0 and 1) and a current speed (which is continuous, possibly negative). Here is how we might define such a space:

```python
import gym
from gym import spaces
import numpy as np

class MultiLevelObservationEnv(gym.Env):
    def __init__(self):
        super(MultiLevelObservationEnv, self).__init__()

        self.observation_space = spaces.Dict({
            "camera_1": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
            "camera_2": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
            "camera_3": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
            "system_status": spaces.Dict({
               "battery": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
               "speed": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
           })
        })

        self.action_space = spaces.Discrete(4) # Example
        # ... other initial setup

    def reset(self, seed=None, options=None):
        # generate a sample observation adhering to space
        observation = {
            "camera_1": np.random.randint(0, 256, size=(64, 64, 1), dtype=np.uint8),
            "camera_2": np.random.randint(0, 256, size=(64, 64, 1), dtype=np.uint8),
            "camera_3": np.random.randint(0, 256, size=(64, 64, 1), dtype=np.uint8),
            "system_status": {
              "battery": np.array([np.random.uniform(0, 1)], dtype=np.float32),
              "speed": np.array([np.random.uniform(-10, 10)], dtype=np.float32)
             }
         }
        return observation, {}


    def step(self, action):
        #generate a sample observation according to the defined observation space for every step.
        observation = {
            "camera_1": np.random.randint(0, 256, size=(64, 64, 1), dtype=np.uint8),
            "camera_2": np.random.randint(0, 256, size=(64, 64, 1), dtype=np.uint8),
            "camera_3": np.random.randint(0, 256, size=(64, 64, 1), dtype=np.uint8),
             "system_status": {
              "battery": np.array([np.random.uniform(0, 1)], dtype=np.float32),
              "speed": np.array([np.random.uniform(-10, 10)], dtype=np.float32)
             }
        }

        reward = 1
        done = False
        info = {}
        return observation, reward, done, False, info
```

In this snippet, you'll see that we've defined `self.observation_space` as a `spaces.Dict`, where the keys are "camera_1", "camera_2", "camera_3", and "system_status". The cameras each have a `spaces.Box` to capture their image data, while `system_status` is another `spaces.Dict`, which itself contains `"battery"` and `"speed"`, both using `spaces.Box`.

The important part is adhering to the structure you've defined in your `observation_space`. Whenever you return observations, you need to return a dictionary that matches this structure. In the `reset` and `step` methods of the `MultiLevelObservationEnv`, I've provided examples of how to generate these observations for each part of the observation space, ensuring it matches our `Dict` structure.

This nesting can, of course, go deeper. Imagine a scenario where, within system status, you also have a set of flags, and perhaps even a sub-dictionary describing the condition of each component. In such a case, you can extend the logic as shown above by nesting more `spaces.Dict` objects.

Now, let’s explore another use-case, where we have a robot that is aware of its own state and the state of its surrounding environment. The environment could be represented as a grid with different types of obstacles, and the robot’s internal state includes its position, energy level, and current task. This could look like this:

```python
import gym
from gym import spaces
import numpy as np

class RobotGridEnvironment(gym.Env):
    def __init__(self):
        super(RobotGridEnvironment, self).__init__()

        self.grid_size = (10, 10)  # Define the size of the grid
        self.num_obstacle_types = 3 # Types of obstacle
        self.num_tasks = 2

        self.observation_space = spaces.Dict({
            "robot_state": spaces.Dict({
                "position": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]]),
                "energy": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                "current_task": spaces.Discrete(self.num_tasks)
            }),
             "grid": spaces.MultiDiscrete(np.full(self.grid_size, self.num_obstacle_types))
        })

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        #... initial setup here

    def reset(self, seed=None, options=None):
        # Generate initial observation
        observation = {
            "robot_state": {
               "position": np.array([np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]),
               "energy": np.array([np.random.randint(0, 100)], dtype=np.int32),
               "current_task": np.random.randint(0, self.num_tasks)
            },
            "grid": np.random.randint(0, self.num_obstacle_types, size=self.grid_size)
        }
        return observation, {}


    def step(self, action):
        # Generate observation for the current state
        observation = {
            "robot_state": {
               "position": np.array([np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]),
               "energy": np.array([np.random.randint(0, 100)], dtype=np.int32),
                "current_task": np.random.randint(0, self.num_tasks)
            },
            "grid": np.random.randint(0, self.num_obstacle_types, size=self.grid_size)
        }

        reward = 1
        done = False
        info = {}
        return observation, reward, done, False, info
```

Here we use `spaces.MultiDiscrete` to capture the position and the grid since they are discrete values and we want to specify limits on their values. `spaces.Discrete` is used for the `current_task` which is an index. Note that, yet again, the structure of observations from `reset` and `step` matches that of `observation_space`.

Finally, it’s also worth considering the case where an environment has time-dependent observations. For this, let’s consider an environment where the agent observes several time series of data with various lengths, perhaps from sensors or logs. We can define a time-series aware observation space like so:

```python
import gym
from gym import spaces
import numpy as np

class TimeSeriesEnv(gym.Env):
    def __init__(self):
        super(TimeSeriesEnv, self).__init__()

        self.max_time_steps = 5
        self.num_sensors = 3

        self.observation_space = spaces.Dict({
            "sensor_data": spaces.Dict({
                f"sensor_{i}": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_time_steps,), dtype=np.float32)
                for i in range(self.num_sensors)
            }),
            "time_step": spaces.Discrete(self.max_time_steps)
        })

        self.action_space = spaces.Discrete(4)
        # ...initial setup

    def reset(self, seed=None, options=None):
        observation = {
           "sensor_data": {
               f"sensor_{i}": np.random.randn(self.max_time_steps).astype(np.float32) for i in range(self.num_sensors)
           },
            "time_step": 0
        }
        return observation, {}

    def step(self, action):
        observation = {
           "sensor_data": {
               f"sensor_{i}": np.random.randn(self.max_time_steps).astype(np.float32) for i in range(self.num_sensors)
           },
            "time_step": np.random.randint(0, self.max_time_steps)
        }

        reward = 1
        done = False
        info = {}
        return observation, reward, done, False, info
```
In this example, each sensor provides a time series, and the time step is also included as part of the observation. In `reset` and `step` we generate sample observations, adhering to the defined observation space which allows for each time series to be a separate entry.

In terms of further study, I’d highly recommend checking out the OpenAI’s documentation on `gym.spaces`. Additionally, Richard S. Sutton and Andrew G. Barto’s book, “Reinforcement Learning: An Introduction,” provides excellent insights into how different kinds of observation spaces influence the learning process. Finally, if your data is structured, like images or time series, familiarizing yourself with the basics of computer vision and time series analysis is useful for handling the observations effectively.

By using `spaces.Dict` and understanding how the space definitions need to mirror your actual returned observation data, you should be well-equipped to handle complex multilevel features in your gym environments. Remember to meticulously check data types and dimensions to ensure a smooth integration with your reinforcement learning algorithms.
