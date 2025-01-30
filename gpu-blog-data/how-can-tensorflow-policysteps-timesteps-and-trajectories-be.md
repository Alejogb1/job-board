---
title: "How can TensorFlow PolicySteps, TimeSteps, and Trajectories be flattened?"
date: "2025-01-30"
id: "how-can-tensorflow-policysteps-timesteps-and-trajectories-be"
---
TensorFlow agents often represent sequential data used in reinforcement learning through the structures `PolicyStep`, `TimeStep`, and `Trajectory`. Direct manipulation of these nested structures can become unwieldy, making it necessary to flatten them for easier processing within models, loss calculations, or data pre-processing. These structures are essentially namedtuples, which can be recursively flattened into single tensors or dictionaries of tensors.

`TimeStep` objects, defined within the TF-Agents library, hold information about a single step in an environment, specifically the `step_type`, `reward`, `discount`, and `observation`. They do not always have dimensions which we can directly flattern, due to the fact that the observation may be a nested dictionary or tuple. A `PolicyStep` represents the output of a policy network, usually including the `action` and `log_probability` of that action. These may also have a nested structure. A `Trajectory` combines one or more `TimeStep` and `PolicyStep` instances within a batch of sequential data. Specifically, trajectories contain the elements: `observation`, `action`, `policy_info`, `next_observation`, `reward`, and `discount`.

The goal of flattening is to convert these structures to a format where each individual tensor or component of a nested tensor is directly addressable and can be combined with other tensors. This facilitates efficient batch processing and vector operations, avoiding complex indexing and recursion when implementing gradient descent or other operations. The underlying data might be images, text embeddings, or some other complex representation.

The key idea to flatten these structures involves using helper functions. These functions recursively unpack the structure and append the tensor components into either a single list of tensors or a dictionary with string keys indicating which field each tensor originated from. I have consistently used recursive flattening strategies in custom agents which enabled me to incorporate complex state observation spaces or action policies without the need to rewrite my core training loop.

The specific approach involves checking the type of each element within these structures. If it's a tensor, it’s directly appended to a list, or added to a dictionary. If it’s another nested structure, such as a tuple or dictionary, we call the same function recursively. This ensures that any level of nesting is handled.

The following code illustrates flattening a TimeStep.

```python
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec
from collections import OrderedDict

def flatten_timestep(timestep):
    flattened_tensors = []
    def recursive_flatten(element):
        if isinstance(element, tf.Tensor):
            flattened_tensors.append(element)
        elif isinstance(element, (tuple, list)):
            for item in element:
                recursive_flatten(item)
        elif isinstance(element, dict):
             for _,value in element.items():
                recursive_flatten(value)
    recursive_flatten(timestep)
    return flattened_tensors

#Create example timestep with nested obs
obs_spec = tensor_spec.TensorSpec((2,2), dtype=tf.float32, name='obs')
nested_obs_spec = tensor_spec.BoundedTensorSpec((2,3), dtype=tf.float32, name='obs2', minimum=0, maximum=1)

time_step_spec = ts.time_step_spec(observation_spec= {"obs1": obs_spec, "obs2": nested_obs_spec})
time_step = time_step_spec.sample()

flattened = flatten_timestep(time_step)
print("Flattened TimeStep Shape:", [tensor.shape for tensor in flattened])

```
This code first defines the `flatten_timestep` function, which recursively inspects the `TimeStep` object. Note that the `TimeStep` might contain nested observation spaces, so these must also be flattened. The example creates a nested observation spec and then constructs a sample TimeStep object and displays the shape of the flattened output.

Next, we demonstrate a `PolicyStep` flattening implementation.

```python
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import policy_step
from tf_agents.specs import tensor_spec
from collections import OrderedDict


def flatten_policystep(policystep):
   flattened_dict = {}
   def recursive_flatten(element, prefix=None):
      if isinstance(element, tf.Tensor):
          if prefix:
            flattened_dict[prefix] = element
          else:
            flattened_dict['tensor'] = element
      elif isinstance(element, (tuple, list)):
           for i,item in enumerate(element):
            recursive_flatten(item, f'{prefix}_{i}' if prefix else str(i))
      elif isinstance(element, dict):
           for key,value in element.items():
               recursive_flatten(value, f'{prefix}_{key}' if prefix else key)

   recursive_flatten(policystep)
   return flattened_dict


#Example policy step
action_spec = tensor_spec.BoundedTensorSpec((1,), dtype=tf.int64, name="action", minimum=0, maximum=3)
policy_info_spec = {"log_probability": tensor_spec.TensorSpec((1,), dtype=tf.float32)}
policy_step_spec = policy_step.policy_step_spec(action_spec=action_spec, policy_info_spec=policy_info_spec)
example_policy_step = policy_step_spec.sample()


flattened = flatten_policystep(example_policy_step)
print("Flattened PolicyStep:")
for k,v in flattened.items():
  print(f"  Key: {k}, Shape: {v.shape}")

```
Here we use a dictionary as output and we utilize a prefix parameter, in order to capture the hierarchy within the `PolicyStep` structure. The example `PolicyStep` uses a bounded integer action and also includes log probabilities. We see the resultant flattened output and the shape of each tensor with its key.

Finally, we show how to flatten a `Trajectory`.

```python
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import policy_step


def flatten_trajectory(trajectory):
  flattened_dict = {}
  def recursive_flatten(element, prefix=None):
      if isinstance(element, tf.Tensor):
         if prefix:
            flattened_dict[prefix] = element
         else:
             flattened_dict['tensor'] = element
      elif isinstance(element, (tuple, list)):
         for i,item in enumerate(element):
           recursive_flatten(item, f'{prefix}_{i}' if prefix else str(i))
      elif isinstance(element, dict):
          for key,value in element.items():
            recursive_flatten(value, f'{prefix}_{key}' if prefix else key)
  recursive_flatten(trajectory)
  return flattened_dict

#Create specs
obs_spec = tensor_spec.TensorSpec((2,2), dtype=tf.float32, name='obs')
action_spec = tensor_spec.BoundedTensorSpec((1,), dtype=tf.int64, name="action", minimum=0, maximum=3)
policy_info_spec = {"log_probability": tensor_spec.TensorSpec((1,), dtype=tf.float32)}
nested_obs_spec = tensor_spec.BoundedTensorSpec((2,3), dtype=tf.float32, name='obs2', minimum=0, maximum=1)


time_step_spec = ts.time_step_spec(observation_spec= {"obs1": obs_spec, "obs2": nested_obs_spec})
policy_step_spec = policy_step.policy_step_spec(action_spec=action_spec, policy_info_spec=policy_info_spec)


#Create sample trajectory
sample_time_step = time_step_spec.sample()
sample_next_time_step = time_step_spec.sample()
sample_policy_step = policy_step_spec.sample()
example_trajectory = trajectory.Trajectory(
    observation=sample_time_step.observation,
    action=sample_policy_step.action,
    policy_info=sample_policy_step.info,
    next_observation=sample_next_time_step.observation,
    reward=sample_time_step.reward,
    discount=sample_time_step.discount,
)

flattened = flatten_trajectory(example_trajectory)
print("Flattened Trajectory:")
for k, v in flattened.items():
  print(f"  Key: {k}, Shape: {v.shape}")

```
This code demonstrates a similar dictionary output as the previous example, but this time operates on a full Trajectory object. We can see each of the tensors, including those from the `TimeStep`, and `PolicyStep` object within the trajectory flattened into the dictionary with appropriate prefixes.

When implementing flattening functions, consider how to handle variable-length sequences if present in your data. While these examples show flattening of a single instance of each data structure, real-world data is often batched and can vary in length due to padding. When flattening batch data, you should take extra care to account for padding and masking during computation and ensure that these are consistent within your loss function calculation. It is generally recommended to maintain batch dimensions while flattening.

For more in-depth information, refer to the TensorFlow documentation for `tf.nest`. The TF-Agents library includes multiple examples of using `tf.nest` and also has tutorials that cover agent development with more complex observation and action spaces. Reviewing the implementation of the base classes in the TF-Agents package can also be beneficial to understand how this flattening is commonly used. Consider exploring the source code for TF-Agents’ common agent implementations, and the utilities that are used to process trajectories and other data samples. In particular, pay attention to the data pre-processing steps of the agents, and any utilities provided for flattening or combining multiple fields within a trajectory.
