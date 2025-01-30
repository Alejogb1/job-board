---
title: "How do custom policies work in stable-baselines3?"
date: "2025-01-30"
id: "how-do-custom-policies-work-in-stable-baselines3"
---
Stable-baselines3, a reinforcement learning library, provides flexibility by allowing the implementation of custom neural network policies beyond its built-in architectures. The standard policies, like `MlpPolicy` or `CnnPolicy`, are often sufficient, but complex environments or specific task requirements might necessitate modifications to the network structure or the actor-critic architecture itself. Understanding how custom policies function within the framework is paramount for advanced use cases.

At the core, a stable-baselines3 policy object is responsible for two primary functions: (1) feature extraction from the environment’s observation, and (2) predicting actions based on those extracted features. These functions are performed through neural networks, the exact structure of which is what you define in a custom policy. The framework abstracts away much of the backpropagation and learning mechanics, letting you concentrate on the design of the neural network itself, as long as it adheres to the required interface.

Creating a custom policy involves inheriting from the base `BasePolicy` class provided in stable-baselines3, typically in tandem with another more specific base policy like `ActorCriticPolicy` if your algorithm utilizes both an actor (policy) and a critic (value function). Within this class, you implement the necessary methods, most importantly `__init__`, `forward`, and in many cases `_predict`, and `evaluate_actions`. This inheritance structure leverages pre-built components, such as input processing for image-based or vector-based observations, and simplifies training procedures. The constructor (`__init__`) typically defines the network’s architecture through PyTorch modules. The `forward` method takes the environment observation as input and outputs the actions (and optionally, the value function estimation). The `_predict` method translates the output of `forward` into usable actions given the current policy distribution. Finally, `evaluate_actions` calculates log probabilities and entropies of given actions based on the current policy.

This separation allows stable-baselines3 to internally handle data batching, gradient computation, and optimization loops, while using your custom architecture to generate the necessary gradients and parameters updates. Consequently, a custom policy does not typically interact directly with the training process but rather acts as a trainable computation block within the larger reinforcement learning algorithm.

Here are three examples demonstrating custom policy construction:

**Example 1: A Simple MLP Policy with a Custom Layer Configuration**

```python
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch, **kwargs):
        super(CustomMLPPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.net_arch = net_arch

        self.features_extractor = nn.Flatten()  # Flatten the input if needed
        input_dim = self.observation_space.shape[0] if len(self.observation_space.shape) == 1 else th.prod(th.tensor(self.observation_space.shape)).item()


        layers = []
        prev_dim = input_dim
        for dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        self.mlp = nn.Sequential(*layers)
        self.policy_net = nn.Linear(prev_dim, self.action_space.n) if hasattr(self.action_space, 'n') else nn.Linear(prev_dim, self.action_space.shape[0])
        self.value_net = nn.Linear(prev_dim, 1)


    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)
        features = self.mlp(features)
        values = self.value_net(features)
        policy_dist = self._get_distribution(features)
        return policy_dist, values

    def _predict(self, observation, deterministic=False):
      features = self.features_extractor(observation)
      features = self.mlp(features)
      policy_dist = self._get_distribution(features)
      return policy_dist.get_actions(deterministic=deterministic)

```

This example demonstrates a custom multi-layer perceptron (MLP) policy. The constructor takes a list `net_arch` that defines the hidden layer sizes. Within `__init__`, the input is flattened, and an MLP with the specified architecture is constructed. The `forward` method feeds the observation through the network to extract features used to derive both policy and value. The `_predict` method uses those features to predict actions. In this example, `_get_distribution` would be either a gaussian if it is a continuous action space, or categorical if it is discrete. This class should be passed to an algorithm that uses an Actor Critic policy.

**Example 2: A Custom CNN Policy with Shared Convolutional Layers**

```python
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomCNNPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.features_extractor = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
          dummy_input = th.zeros((1,) + observation_space.shape, dtype=th.float32)
          output_dim = self.features_extractor(dummy_input).shape[-1]


        self.policy_net = nn.Linear(output_dim, self.action_space.n) if hasattr(self.action_space, 'n') else nn.Linear(output_dim, self.action_space.shape[0])
        self.value_net = nn.Linear(output_dim, 1)

    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)
        values = self.value_net(features)
        policy_dist = self._get_distribution(features)
        return policy_dist, values

    def _predict(self, observation, deterministic=False):
        features = self.features_extractor(observation)
        policy_dist = self._get_distribution(features)
        return policy_dist.get_actions(deterministic=deterministic)
```

Here, we construct a convolutional neural network (CNN) policy for processing image-based observations. Shared convolutional layers extract spatial features, followed by fully connected layers for the policy and value heads. Notably, a dummy input is processed to infer the output dimension of the convolutional stack, used to instantiate the linear layers. The `forward` and `_predict` methods are similar to the previous example. This policy is designed for environments with image-based observations, demonstrating the versatility of custom policy definitions.

**Example 3: A Policy with an LSTM Layer for Temporal Dependencies**

```python
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomLSTMPolicy(ActorCriticPolicy):
  def __init__(self, observation_space, action_space, lr_schedule, lstm_hidden_size=128, **kwargs):
    super(CustomLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
    input_dim = self.observation_space.shape[0] if len(self.observation_space.shape) == 1 else th.prod(th.tensor(self.observation_space.shape)).item()
    self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)
    self.policy_net = nn.Linear(lstm_hidden_size, self.action_space.n) if hasattr(self.action_space, 'n') else nn.Linear(lstm_hidden_size, self.action_space.shape[0])
    self.value_net = nn.Linear(lstm_hidden_size, 1)
    self.lstm_hidden_size = lstm_hidden_size


  def forward(self, obs, deterministic=False, lstm_states = None):
    batch_size = obs.shape[0]
    sequence_len = obs.shape[1]
    obs = obs.reshape(batch_size * sequence_len, -1)
    obs = obs.unsqueeze(1)
    out, lstm_states = self.lstm(obs, lstm_states)
    out = out.squeeze(1)
    values = self.value_net(out)
    policy_dist = self._get_distribution(out)
    return policy_dist, values, lstm_states

  def _predict(self, observation, deterministic=False, lstm_states = None):
        batch_size = observation.shape[0]
        sequence_len = observation.shape[1]
        obs = observation.reshape(batch_size * sequence_len, -1)
        obs = obs.unsqueeze(1)
        out, lstm_states = self.lstm(obs, lstm_states)
        out = out.squeeze(1)
        policy_dist = self._get_distribution(out)
        return policy_dist.get_actions(deterministic=deterministic), lstm_states

  def reset_noise(self, batch_size):
    return  (th.zeros(1, batch_size, self.lstm_hidden_size), th.zeros(1, batch_size, self.lstm_hidden_size))
```

This example introduces a policy with an LSTM layer for handling time-series data. The `forward` method processes the observation through the LSTM. Crucially, this implementation assumes a temporal input, which is a batch of timesteps, resulting in a 3D tensor `(batch, seq_len, obs_dim)`. Thus, before feeding to the lstm, it must be reshaped into `(batch * seq_len, 1, obs_dim)`. The state of the LSTM is also returned from both `forward` and `_predict`, allowing it to preserve information about time-dependent contexts. The `reset_noise` function initializes a new lstm state to be used when a new episode begins.

In summary, custom policies in stable-baselines3 provide a flexible mechanism to extend the library's capabilities. By inheriting from base policy classes and implementing methods like `__init__`, `forward`, and `_predict`, researchers can tailor neural network architectures to specific reinforcement learning problems.

For deeper understanding, consult PyTorch's documentation regarding `nn.Module` and layers, focusing on constructing various neural network layers. Additionally, review the stable-baselines3 documentation, particularly the section detailing how custom policies integrate with the library’s reinforcement learning algorithms. Further investigation into policy gradient methods can deepen insight into how these components work together. Open source policy implementation for other reinforcement learning tasks can also be an invaluable resource to learn the conventions and common practices within the domain. Examining the implementations of the default policies available in stable-baselines3 can also be beneficial.
