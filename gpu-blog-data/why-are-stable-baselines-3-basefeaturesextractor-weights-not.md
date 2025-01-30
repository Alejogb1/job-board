---
title: "Why are Stable Baselines 3 BaseFeaturesExtractor weights not updating?"
date: "2025-01-30"
id: "why-are-stable-baselines-3-basefeaturesextractor-weights-not"
---
The core issue with Stable Baselines3's `BaseFeaturesExtractor` weights not updating stems from a misunderstanding of how the feature extraction component interacts with the policy network within the overall reinforcement learning (RL) architecture.  My experience debugging similar problems in large-scale robotics simulations highlighted the importance of correctly integrating the feature extractor into the learning process; specifically, ensuring its parameters are included in the optimization loop.  The `BaseFeaturesExtractor` itself doesn't inherently possess an update mechanism; its weights are updated indirectly through the policy network's gradient descent.

**1. Clear Explanation:**

Stable Baselines3 uses a modular design. The `BaseFeaturesExtractor` acts as a preprocessing step, transforming raw observations into a feature vector suitable for the policy network. The policy network, in turn, uses these features to determine actions. The crucial element often overlooked is that the gradient descent algorithm, typically Adam or RMSprop, updates the *entire* network's parameters, encompassing both the policy network and the feature extractor's weights. If the feature extractor's weights are not being updated, it implies a disconnect between the extractor and the optimization process. This disconnect can manifest in several ways:

* **Incorrect Model Construction:** The `BaseFeaturesExtractor` might not be correctly integrated into the policy network's architecture.  If it's merely used to pre-process data but not directly incorporated within the computational graph, its weights will remain untouched during training.  This is a common mistake, particularly when utilizing custom feature extractors.

* **Detached Gradients:** The gradients calculated during backpropagation may not flow back to the feature extractor's parameters. This can occur if the feature extractor is implemented in a way that prevents automatic differentiation, such as through the use of external libraries or custom operations that are not differentiable.  In my experience working with custom CNN architectures, I've encountered this specifically when using layers not readily supported by PyTorch's automatic differentiation engine.

* **Gradient Clipping or Regularization Interference:** Aggressive gradient clipping or regularization techniques (e.g., strong L1/L2 penalties) applied to the entire network can effectively suppress updates to the feature extractor's weights, especially if these weights have small initial gradients or are already near their optimal values.  I've seen this in high-dimensional observation spaces where the initial feature representations are already relatively informative.


**2. Code Examples with Commentary:**

**Example 1: Correct Integration:**

This example demonstrates the proper integration of a convolutional neural network (CNN) as a `BaseFeaturesExtractor` within a PPO algorithm.

```python
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(3, 32, 8, 4),  # Example architecture; adjust as needed
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, 4, 2),
            th.nn.ReLU(),
            th.nn.Flatten()
        )
        self.linear = th.nn.Linear(self.compute_features_dim(observation_space), features_dim)

    def forward(self, observations):
        x = self.cnn(observations)
        return self.linear(x)

    def compute_features_dim(self, observation_space):
        # Calculate output dimension for flattening
        o = th.zeros((1, *observation_space.shape))
        o = self.cnn(o)
        return o.shape[1]

# ... (Environment setup) ...
policy_kwargs = dict(net_arch=dict(pi=[128,128], vf=[128,128]), features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=64))
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=10000)
```

**Commentary:** This code explicitly defines a custom `BaseFeaturesExtractor` inheriting from `BaseFeaturesExtractor` and overriding `forward` and `compute_features_dim`. This ensures that the CNN is properly incorporated into the policy network. The `features_dim` parameter controls the dimensionality of the extracted features.

**Example 2:  Incorrect Usage (Detached Gradients):**

This demonstrates a scenario where the gradients might not propagate:

```python
import numpy as np
from stable_baselines3 import PPO
# ... (Environment setup) ...

def my_feature_extractor(obs):
    # Some non-differentiable operation (e.g., using NumPy only)
    return np.mean(obs, axis=0)

# Incorrect usage: this will not be integrated into gradient calculation
policy_kwargs = dict(net_arch=dict(pi=[128,128], vf=[128,128]), features_extractor_fn=my_feature_extractor)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=10000)

```

**Commentary:** Here, `my_feature_extractor` uses NumPy functions which are not tracked by PyTorch's automatic differentiation. Consequently, the gradients won't flow back to the policy network, rendering the weights unaffected.


**Example 3:  Overly Aggressive Gradient Clipping:**

This illustrates how overly strong gradient clipping can hinder learning.

```python
import torch as th
from stable_baselines3 import PPO
# ... (Environment setup) ...

model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=dict(pi=[128,128], vf=[128,128])), verbose=1, clip_range=0.001) # Very small clip range
model.learn(total_timesteps=10000)
```

**Commentary:** Setting a very small `clip_range` severely limits the magnitude of gradient updates, potentially preventing the feature extractor's weights from changing significantly. This extreme case demonstrates the effect;  in practice, less extreme but still problematic clipping might be used.

**3. Resource Recommendations:**

*   Stable Baselines3 documentation: The official documentation provides detailed explanations of model architecture and training procedures.  Pay close attention to the sections on custom network architectures and policy customization.
*   PyTorch documentation:  A thorough understanding of PyTorch's automatic differentiation system is essential for debugging issues related to gradient propagation.
*   Reinforcement Learning textbooks:  Standard RL textbooks cover the theoretical foundations of policy gradient methods and backpropagation.  These provide a deeper understanding of the underlying principles.


In conclusion, ensuring that the `BaseFeaturesExtractor` is correctly integrated into the policy network and that gradients propagate effectively to its parameters is crucial for successful training.  Carefully examine the construction of your custom feature extractor, the absence of non-differentiable operations, and the impact of any hyperparameters influencing gradient updates.  These points, based on my extensive experience, offer a systematic approach to diagnosing and resolving the issue of stagnant `BaseFeaturesExtractor` weights.
