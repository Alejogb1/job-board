---
title: "How to read real data in RL with a custom environment?"
date: "2024-12-14"
id: "how-to-read-real-data-in-rl-with-a-custom-environment"
---

well, this is something i've definitely banged my head against a few times. getting real-world data to play nice with a reinforcement learning environment, especially when it's a custom one, it can be tricky. it's not always the textbook perfect scenario they give you in the tutorials. i've seen many folks new to the field struggle with this, and frankly, it’s a common spot where things go from "hello world" to "what's going on".

the core of it, as i see it, is this: your rl agent expects a certain structure. it needs observations, rewards, and an indication of when an episode ends. real-world data, more often than not, is not directly in that format. there's usually a pre-processing and re-shaping step involved. so, let's break down how i've approached this in the past, covering some typical pain points and how i've worked through them.

first thing's first, data ingestion. i've had projects where the data was coming in as a csv, sometimes it was a json, and one memorable time it was a weird binary format from some old sensor array. regardless of the format, the goal is the same: load the raw information and transform it into something usable by your environment.

here's a basic example using pandas, which is my go-to for tabular data. say, you have a csv that represents a sensor reading over time with columns like timestamp, sensor_a, sensor_b, and a target label:

```python
import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath):
  """
  loads csv, cleans nans, normalizes sensor data
  returns numpy array of features and labels
  """
  df = pd.read_csv(filepath)
  df = df.dropna() #handle missing data, just remove
  sensor_cols = ['sensor_a', 'sensor_b']
  df[sensor_cols] = (df[sensor_cols] - df[sensor_cols].mean()) / df[sensor_cols].std() #mean 0, std 1
  features = df[sensor_cols].values
  labels = df['target_label'].values
  return features, labels
```

notice a few things: i'm cleaning the data with `.dropna()` – dealing with missing values it’s quite important. i'm also doing standardisation, calculating mean and standard deviation to scale sensor data to have mean zero and standard deviation of one, a good practice for lots of ml algorithms. the data needs to be fed to the environment in numerical format, and these are the first steps.

now, let's assume your custom environment is set up. the critical piece is to connect the processed data to the environment's step function. this function needs to take an action from the agent, update its state based on the real data, give back a reward and the next state, and say if the episode is done.

here's a rough idea of what that might look like, considering i already loaded my data using the prior function. i will make it simple and assume i have actions in the form of 0 and 1 only for a simple example.

```python
class CustomEnv:
    def __init__(self, features, labels, window_size=10):
      self.features = features
      self.labels = labels
      self.window_size = window_size
      self.current_step = 0
      self.current_episode_start = 0

    def reset(self):
      self.current_episode_start = np.random.randint(0, len(self.features) - self.window_size -1)
      self.current_step = self.current_episode_start
      return self._get_observation()

    def _get_observation(self):
      return self.features[self.current_step: self.current_step+self.window_size].flatten()

    def step(self, action):
      self.current_step += 1
      obs = self._get_observation()
      done = self.current_step >= (self.current_episode_start + len(self.features) - self.window_size)
      reward = 1 if (action == self.labels[self.current_step -1] and not done) else 0
      return obs, reward, done, {}
```

the important thing is the `step` function: based on `action` from the agent, it moves forward one timestamp. `done` is based on if the `current_step` is already at the end of the current episode. `reward` is a simple indicator if the agent's action was correct based on the real label. you can make this reward more complex depending on your task. the `reset` function just resets the state at the beginning of the next episode by randomly choosing a starting index.

this 'random episode start' strategy ensures you don't just iterate through the data deterministically. the agent is forced to learn patterns. this approach assumes you have some sort of labelled data. if your data isn’t labelled or the task is an unsupervised task, you need to change the reward function based on your problem definition and data.

now, for the tricky part i encountered once when dealing with time series data: episodes are not always neatly defined. sometimes, you need to create artificial episodes by using sliding windows. think of having a single very long time series, and slicing this long series into small sub-sequences. in the prior example, that `window_size` variable is the sliding window. another case i remember well was with a production system where you want to optimise control of some equipment. the data would only make sense if evaluated within each production run. you have to slice the data into these production runs, which is usually easy because you have some identifier for the end and beginning of the runs.

the challenge is often how to handle episode termination. what if your data doesn't explicitly tell you when an episode ends? a common solution is to have a time limit on episodes or to introduce a concept of a termination event based on the data itself. for example, in the code i just showed the end of the episode is reached if the last point of the window has reached the limit of the training sequence.

here's a third example that incorporates a termination condition based on change in a variable (this is a silly example that doesn't mean anything but to show the point):

```python
class CustomEnvTerminate:
    def __init__(self, features, labels, window_size=10, threshold = 0.5):
      self.features = features
      self.labels = labels
      self.window_size = window_size
      self.current_step = 0
      self.current_episode_start = 0
      self.threshold = threshold

    def reset(self):
      self.current_episode_start = np.random.randint(0, len(self.features) - self.window_size -1)
      self.current_step = self.current_episode_start
      return self._get_observation()

    def _get_observation(self):
      return self.features[self.current_step: self.current_step+self.window_size].flatten()

    def step(self, action):
      self.current_step += 1
      obs = self._get_observation()
      done = self.current_step >= (self.current_episode_start + len(self.features) - self.window_size)
      if not done:
          delta = self.features[self.current_step,0] - self.features[self.current_step-1,0]
          if abs(delta) > self.threshold: # check if variable changed too much in the last step
            done = True
      reward = 1 if (action == self.labels[self.current_step -1] and not done) else 0
      return obs, reward, done, {}
```

in the last example, i've added a rule that if the first feature changes more than the threshold, the episode terminates, as an example of a termination criteria based on the data, this was needed to make some control system work, because certain state transitions could mean that the episode was no longer valid.

a funny thing happened to me once when i missed some data type conversion. my agent was making absolutely no progress for 2 days, until i realised the features were being read as strings. it was a quite a facepalm moment after that debugging experience.

when it comes to further resources, instead of pointing to specific websites, i would recommend books and papers. for a solid grounding in reinforcement learning, "reinforcement learning: an introduction" by sutton and barto is the bible. for more on data handling, look into papers on time series analysis if this is your domain, depending on the specifics of your data there could be a lot of things to consider. there is a lot of literature on temporal data and that could be useful for the data preparation needed in many RL problems.

to sum up, hooking real-world data into a custom rl environment requires careful consideration of data loading, preprocessing, episode definition, and the reward function. getting it working smoothly often involves iterative testing, debugging and data analysis, which can be frustrating but also very rewarding. the key is to start simple, and gradually increase the complexity.
