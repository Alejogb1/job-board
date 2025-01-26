---
title: "How do I load a TensorFlow Agents checkpoint?"
date: "2025-01-26"
id: "how-do-i-load-a-tensorflow-agents-checkpoint"
---

TensorFlow Agents checkpoints, unlike simple variable saves, encapsulate not just model weights but also policy and environment state, requiring a specific approach for restoration. I’ve frequently encountered developers stumbling with this nuanced process, often misinterpreting how the agent's learning components are structured within these checkpoints. The key distinction is that you are not simply reloading model weights; you are reloading the entire agent including its actor, critic, and replay buffer.

A TensorFlow Agents checkpoint typically holds serialized data representing the agent's policy (which dictates action selection), any learned value functions, and potentially the state of any training components such as replay buffers. Incorrect handling during restoration can lead to unexpected behavior, most commonly performance degradation or outright failure to reproduce previous learning. The loading process needs to be explicit about the various agent components. Specifically, the checkpoint is not a monolithic entity directly consumable into a model; instead, one must restore individual components of the agent using a `tf.train.Checkpoint` object, configuring this object to track the agent's sub-modules.

The recommended procedure begins by defining a `tf.train.Checkpoint` object to manage the restoration process. This object acts as a container holding references to the agent’s policy, any Q-networks (in the case of Q-learning), and potentially the replay buffer. When loading a checkpoint, we are not directly loading files; instead, we’re restoring the variables associated with these objects to the saved values within the checkpoint file. Therefore, our first task involves setting up this object.

Here's a basic code example illustrating the proper approach, assuming a Deep Q-Network (DQN) agent:

```python
import tensorflow as tf
import tf_agents.agents.dqn.dqn_agent as dqn_agent
import tf_agents.environments.tf_environment as tf_env
import tf_agents.networks.q_network as q_network
import numpy as np

# Define dummy environment (replace with your actual environment)
class DummyEnv(tf_env.TFEnvironment):
    def __init__(self):
        self._action_spec = tf.TensorSpec(shape=(), dtype=tf.int32, name="action")
        self._observation_spec = tf.TensorSpec(shape=(4,), dtype=tf.float32, name="observation")
        self._time_step_spec = tf_agents.specs.TimeStep(
            step_type=tf.TensorSpec(dtype=tf.int32, shape=()),
            reward=tf.TensorSpec(dtype=tf.float32, shape=()),
            discount=tf.TensorSpec(dtype=tf.float32, shape=()),
            observation=self._observation_spec
        )
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    def time_step_spec(self):
       return self._time_step_spec
    def _reset(self):
        return tf_agents.trajectories.time_step.restart(tf.random.normal((4,)))
    def _step(self, action):
        reward = tf.random.normal(())
        return tf_agents.trajectories.time_step.transition(tf.random.normal((4,)), reward, discount=0.99)

env = DummyEnv()

# 1. Define Q-Network
q_net = q_network.QNetwork(
    input_tensor_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    fc_layer_params=(100, 50))

# 2. Create DQN Agent
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
agent = dqn_agent.DqnAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=tf.keras.losses.Huber(),
    train_step_counter=tf.Variable(0))

agent.initialize()

# 3. Set up Checkpoint object
checkpoint_dir = 'path_to_your_checkpoint' # Replace with actual directory
checkpoint = tf.train.Checkpoint(agent=agent)

# 4. Restore from Checkpoint (if a checkpoint exists)
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
if checkpoint_path:
    checkpoint.restore(checkpoint_path)
    print(f"Checkpoint loaded from {checkpoint_path}")
else:
   print("No checkpoint found. Training will start from scratch.")

# Example action after potential restoration
policy_state = agent.policy.get_initial_state(batch_size=1)
time_step = env.reset()
action_step = agent.policy.action(time_step, policy_state)
action = action_step.action
print("Example action:", action)
```

In this example, we first define a dummy environment, Q-Network, and then a DQN agent. Crucially, a `tf.train.Checkpoint` object is created, mapping the string `agent` to the defined agent instance. The `tf.train.latest_checkpoint` function checks the specified directory for the newest checkpoint. If a checkpoint is present, the `checkpoint.restore()` method recovers the agent's parameters. If none is found, a message is printed, and it’s understood the agent will operate with its default initialization. Finally, a simple policy step demonstrates how to interact with the potentially restored agent. The comment `replace with actual directory` highlights a necessary user-provided setting, as checkpoint paths depend entirely on user storage strategies.

Now consider a more complex situation involving a replay buffer. Loading a saved replay buffer requires an adjustment. The replay buffer is not implicitly part of the agent; it needs explicit tracking by the `tf.train.Checkpoint` object.

```python
import tensorflow as tf
import tf_agents.agents.dqn.dqn_agent as dqn_agent
import tf_agents.environments.tf_environment as tf_env
import tf_agents.networks.q_network as q_network
import tf_agents.replay_buffers.tf_uniform_replay_buffer as tf_uniform_replay_buffer
import numpy as np

# Define dummy environment (same as before)
class DummyEnv(tf_env.TFEnvironment):
    def __init__(self):
        self._action_spec = tf.TensorSpec(shape=(), dtype=tf.int32, name="action")
        self._observation_spec = tf.TensorSpec(shape=(4,), dtype=tf.float32, name="observation")
        self._time_step_spec = tf_agents.specs.TimeStep(
            step_type=tf.TensorSpec(dtype=tf.int32, shape=()),
            reward=tf.TensorSpec(dtype=tf.float32, shape=()),
            discount=tf.TensorSpec(dtype=tf.float32, shape=()),
            observation=self._observation_spec
        )
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    def time_step_spec(self):
       return self._time_step_spec
    def _reset(self):
        return tf_agents.trajectories.time_step.restart(tf.random.normal((4,)))
    def _step(self, action):
        reward = tf.random.normal(())
        return tf_agents.trajectories.time_step.transition(tf.random.normal((4,)), reward, discount=0.99)

env = DummyEnv()

# 1. Define Q-Network
q_net = q_network.QNetwork(
    input_tensor_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    fc_layer_params=(100, 50))

# 2. Create DQN Agent
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0) # Keep track of training steps
agent = dqn_agent.DqnAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=tf.keras.losses.Huber(),
    train_step_counter=train_step_counter)

agent.initialize()

# 3. Set up Replay Buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,
    max_length=1000)

# 4. Set up Checkpoint object INCLUDING the Replay Buffer
checkpoint_dir = 'path_to_your_checkpoint'  # Replace with actual directory
checkpoint = tf.train.Checkpoint(agent=agent, replay_buffer=replay_buffer, train_step_counter = train_step_counter)


# 5. Restore from Checkpoint (if a checkpoint exists)
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
if checkpoint_path:
    checkpoint.restore(checkpoint_path)
    print(f"Checkpoint loaded from {checkpoint_path}")
else:
   print("No checkpoint found. Training will start from scratch.")

# Example interaction demonstrating potential restoration
time_step = env.reset()
policy_state = agent.policy.get_initial_state(batch_size=1)
action_step = agent.policy.action(time_step, policy_state)
action = action_step.action

print("Example action:", action)
print(f"Current step: {agent.train_step_counter.numpy()}")
```

Here, I’ve included a replay buffer (`tf_uniform_replay_buffer`). The `tf.train.Checkpoint` now has an added `replay_buffer=replay_buffer` entry.  Crucially, the train_step_counter was also added to the checkpoint to restore training at the correct step.  During restoration, the replay buffer’s data is loaded along with the agent parameters. This is paramount for continuing training without loss of experience data. If I were to resume training without the replay buffer restoration, it would start again as a fresh agent, ignoring prior experience. Notice how now the `train_step_counter` variable is loaded which ensures training can start from the correct step.

Finally, beyond restoring the agent and the replay buffer, it's crucial to recognize that some environments or agents may have other state to manage. For instance, consider a custom environment that requires restoring an internal state. Here, you would need to ensure that that environment's state variables are also added to the `tf.train.Checkpoint`.

```python
import tensorflow as tf
import tf_agents.agents.dqn.dqn_agent as dqn_agent
import tf_agents.environments.tf_environment as tf_env
import tf_agents.networks.q_network as q_network
import numpy as np

# Define dummy environment (extended with state)
class StatefulDummyEnv(tf_env.TFEnvironment):
    def __init__(self):
        self._action_spec = tf.TensorSpec(shape=(), dtype=tf.int32, name="action")
        self._observation_spec = tf.TensorSpec(shape=(4,), dtype=tf.float32, name="observation")
        self._time_step_spec = tf_agents.specs.TimeStep(
            step_type=tf.TensorSpec(dtype=tf.int32, shape=()),
            reward=tf.TensorSpec(dtype=tf.float32, shape=()),
            discount=tf.TensorSpec(dtype=tf.float32, shape=()),
            observation=self._observation_spec
        )
        self.internal_state = tf.Variable(0, dtype=tf.int32)
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    def time_step_spec(self):
       return self._time_step_spec
    def _reset(self):
        self.internal_state.assign(0) # Reset the internal state
        return tf_agents.trajectories.time_step.restart(tf.random.normal((4,)))
    def _step(self, action):
        self.internal_state.assign_add(1)
        reward = tf.random.normal(())
        return tf_agents.trajectories.time_step.transition(tf.random.normal((4,)), reward, discount=0.99)

env = StatefulDummyEnv()

# 1. Define Q-Network
q_net = q_network.QNetwork(
    input_tensor_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    fc_layer_params=(100, 50))

# 2. Create DQN Agent
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
agent = dqn_agent.DqnAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=tf.keras.losses.Huber(),
    train_step_counter=tf.Variable(0))

agent.initialize()

# 3. Set up Checkpoint object including the environment state
checkpoint_dir = 'path_to_your_checkpoint' # Replace with actual directory
checkpoint = tf.train.Checkpoint(agent=agent, environment_state=env.internal_state)

# 4. Restore from Checkpoint (if a checkpoint exists)
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
if checkpoint_path:
    checkpoint.restore(checkpoint_path)
    print(f"Checkpoint loaded from {checkpoint_path}")
else:
    print("No checkpoint found. Training will start from scratch.")

# Example interaction demonstrating potential restoration
time_step = env.reset()
policy_state = agent.policy.get_initial_state(batch_size=1)
action_step = agent.policy.action(time_step, policy_state)
action = action_step.action
print("Example action:", action)
print(f"Environment state after reset: {env.internal_state.numpy()}")

time_step = env.step(action)
print(f"Environment state after step: {env.internal_state.numpy()}")
```

In this extension, a `StatefulDummyEnv` is created with an `internal_state` variable. This variable is then explicitly included in the `tf.train.Checkpoint`, and subsequently its state is restored alongside the agent’s parameters. This illustrates the general principle: any state crucial for correct agent behavior needs to be serialized and restored using a `tf.train.Checkpoint`.

In summary, to properly load a TensorFlow Agents checkpoint, you must use `tf.train.Checkpoint`, include all the necessary components (agent, replay buffer, environment state), and utilize `tf.train.latest_checkpoint` to locate the most recent save, followed by calling the checkpoint object's `restore` method.  For resources on this topic, examine the TensorFlow Agents documentation, particularly the sections on checkpointing and saving training state, the TensorFlow Core documentation for `tf.train.Checkpoint`, and examples provided within the TensorFlow Agents codebase itself.
