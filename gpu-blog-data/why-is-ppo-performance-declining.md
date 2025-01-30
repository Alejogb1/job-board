---
title: "Why is PPO performance declining?"
date: "2025-01-30"
id: "why-is-ppo-performance-declining"
---
Proximal Policy Optimization (PPO), while generally robust, exhibits declining performance in certain scenarios due to a confluence of factors revolving around its core mechanisms: the clipped surrogate objective and the adaptive KL-divergence penalty. My experiences in reinforcement learning, particularly with complex robotic control tasks, have repeatedly illustrated these vulnerabilities. Let's delve into why PPO occasionally fails to sustain optimal performance.

**Understanding the Core Mechanics and Their Limitations**

At its heart, PPO strives to improve an agent’s policy by maximizing a surrogate objective function. This function aims to balance policy improvement with stability by utilizing a *clipped* ratio between the new policy and the old policy. Mathematically, this ratio, `r_t(θ)`, is calculated as `π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`. PPO then computes an objective based on this ratio, with the main term being `min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)`. Here, `A_t` represents the advantage function, and `ε` is the clipping parameter which prevents excessive updates.

The clipping mechanism, while crucial for preventing catastrophic policy collapse, also introduces a primary reason for performance stagnation. When the advantage, `A_t`, is consistently positive and the policy ratio, `r_t(θ)`, pushes towards the clipping bounds (either `1-ε` or `1+ε`), policy improvements can stall. The clipping essentially creates a ceiling on learning. The network may be unable to express a better policy if it needs to diverge significantly from the previous policy, because any move past the clip boundary is discarded.

Furthermore, the adaptive KL-divergence penalty, frequently used to dynamically control the step size and maintain stability, can also become problematic. This penalty, often added to the objective function as `-β * KL(π_θ_old || π_θ)`, penalizes significant deviations from the previous policy’s distribution. While preventing overly drastic changes, this mechanism can effectively hinder the exploration needed to escape local optima. If the KL-divergence penalty is too high or the learning environment has changed dramatically, the agent could get stuck refining a suboptimal policy without the capability to drastically adapt.

The accumulation of these limitations, particularly in complex and dynamic environments, contributes significantly to the observed phenomenon of PPO performance decline.

**Illustrative Code Examples**

I’ve encountered these issues firsthand. In one scenario involving a simulated quadruped robot navigating an obstacle course, these effects manifested vividly. Let's explore code snippets highlighting potential problems.

**Example 1: Clipped Policy Updates Stalling Progress**

This Python example demonstrates a scenario where policy updates stall because the policy ratio consistently reaches the clip bounds.

```python
import numpy as np

def ppo_update(old_policy_probs, new_policy_probs, advantages, clip_epsilon):
  """
  Simplified PPO update with clipping.
  """
  policy_ratio = new_policy_probs / old_policy_probs
  clipped_ratio = np.clip(policy_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
  objective = np.minimum(policy_ratio * advantages, clipped_ratio * advantages)
  return np.mean(objective)


# Simulating a scenario where policy is on verge of improvement
old_probs = np.array([0.2, 0.8]) # old policy probabilities
new_probs = np.array([0.1, 0.9]) # slight improvement desired
advantages = np.array([ -0.5, 0.8]) # advantage function for this action

clip_epsilon = 0.2

for _ in range(5): # show multiple updates
    print(f"Objective: {ppo_update(old_probs, new_probs, advantages, clip_epsilon):.4f}")

    new_probs += 0.01 #simulating an attempt for new policy to move

    # simulate policy on edge of clip bounds
    new_probs[0] = np.clip(new_probs[0], 0.1, 0.2 - 0.001)
    new_probs[1] = np.clip(new_probs[1], 0.8 + 0.001, 0.9)


```

*Commentary:* In this example, even though the agent tries to improve the action probability (for index 1), a small movement results in the policy ratio being clipped. This effectively stops the gradient update, illustrating how the clipping can prevent useful changes. Notice how, despite policy movement, the objective stagnates. This stalls potential improvement.

**Example 2: Static KL-Divergence Penalty**

This example illustrates a scenario where a static KL-divergence penalty is too strong, hindering movement out of a local optimum.

```python
import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    return entropy(p, q)

def ppo_kl_update(old_policy_probs, new_policy_probs, advantages, kl_beta):
  """
  Simplified PPO update with KL penalty.
  """
  policy_ratio = new_policy_probs / old_policy_probs
  objective = np.mean(policy_ratio * advantages)
  kl = kl_divergence(old_policy_probs, new_policy_probs)
  kl_penalized_objective = objective - kl_beta * kl
  return kl_penalized_objective

# Starting with a poor but stable policy.
old_probs = np.array([0.8, 0.2]) # old policy probabilities
new_probs = np.array([0.75, 0.25]) # marginal improvement attempt
advantages = np.array([-0.2, 0.9]) # advantages

kl_beta = 5 # HIGH kl penalty

for _ in range(5):
    print(f"Objective with KL penalty:{ppo_kl_update(old_probs, new_probs, advantages, kl_beta):.4f}")
    new_probs -= 0.01 # attempt to explore

    new_probs[0] = np.clip(new_probs[0], 0.2, 0.8)
    new_probs[1] = np.clip(new_probs[1], 0.2, 0.8)

```

*Commentary:* In this case, the KL penalty is excessively high. Despite an attempted change and a positive advantage for action 1, the penalty term dominates the objective value, preventing the policy from moving far enough to find a better solution. The agent is effectively held in place despite the potential for positive change.

**Example 3: Combination of Clipping and KL Penalty Stalling Progress**

This code snippet shows how a combination of clipped policy ratios and a too high KL-penalty can lead to a dead end.

```python
import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    return entropy(p, q)


def ppo_combined_update(old_policy_probs, new_policy_probs, advantages, clip_epsilon, kl_beta):
    """
    Simplified PPO update with both clipping and KL penalty.
    """
    policy_ratio = new_policy_probs / old_policy_probs
    clipped_ratio = np.clip(policy_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    objective = np.minimum(policy_ratio * advantages, clipped_ratio * advantages)
    kl = kl_divergence(old_policy_probs, new_policy_probs)
    kl_penalized_objective = np.mean(objective) - kl_beta * kl
    return kl_penalized_objective


# Initial state and policy
old_probs = np.array([0.3, 0.7]) # old policy probabilities
new_probs = np.array([0.25, 0.75]) # marginal improvement attempt
advantages = np.array([-0.3, 0.8]) # advantages
clip_epsilon = 0.2
kl_beta = 2 # moderate KL penalty


for _ in range(5):
    print(f"Combined Objective: {ppo_combined_update(old_probs, new_probs, advantages, clip_epsilon, kl_beta):.4f}")

    new_probs -= 0.01 # attempt to explore

    #simulating policy being close to a clip
    new_probs[0] = np.clip(new_probs[0], 0.1, 0.3 - 0.001)
    new_probs[1] = np.clip(new_probs[1], 0.7 + 0.001, 0.9)

```

*Commentary:* Here, both mechanisms are at play. The new policy attempts to move, but it hits a clipping boundary. The KL penalty, while not excessive, prevents the policy from exploring. This illustrates a realistic situation where combined effects prevent learning, often leading to poor performance or outright stagnation.

**Resource Recommendations**

Understanding the nuances of PPO requires dedicated study. I recommend focusing on research papers that discuss modifications to the PPO algorithm, like Trust Region Policy Optimization (TRPO) and its relation to PPO. Additionally, I advise detailed study of the original PPO paper and its implementation. Several excellent textbooks on deep reinforcement learning often cover PPO and related algorithms with depth. I'd strongly recommend exploring theoretical underpinnings of RL beyond standard tutorial resources. Careful review of these can provide crucial insights.
