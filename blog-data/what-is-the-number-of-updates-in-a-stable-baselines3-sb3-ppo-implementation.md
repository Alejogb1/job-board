---
title: "What is the Number of updates in a Stable baselines3 (SB3) PPO implementation?"
date: "2024-12-15"
id: "what-is-the-number-of-updates-in-a-stable-baselines3-sb3-ppo-implementation"
---

alright, so you're asking about the number of updates in a stable baselines3 ppo implementation. it's a good question, and one that often trips people up when they're starting with reinforcement learning (rl). i remember when i first started, the difference between an update, a rollout, and an epoch was all a bit of a jumble. i think i even accidentally trained a model for way too long and ran out of gpu credits on a google cloud instance once (that was a fun bill to explain).

let's break it down. in sb3's ppo, “updates” refer to how frequently the policy network and the value network are updated based on the collected experience. it’s not just a single step, and it's definitely not the same as an environment step.

the core of ppo involves these steps:

1.  **rollout:** the agent interacts with the environment for a certain number of steps (`n_steps`), gathering trajectories of states, actions, rewards, and next states. this whole process is what we call a rollout.

2.  **data storage:** the collected data from the rollout is stored in a buffer. this buffer holds these interaction experiences and is used to calculate loss and update the models.

3.  **optimization:** the data in the buffer is used to compute the loss function and update the policy and value networks (the two important parts of a ppo agent). this is where the 'update' happens, and it typically happens over multiple mini-batches of that stored data from the buffer for a specific number of epochs.

so, the number of updates isn’t directly specified as a parameter like `n_steps`. it’s derived from a combination of other parameters: `n_steps`, `batch_size`, and `n_epochs`.

let’s look at how these parameters relate:

*   `n_steps`: this determines the size of the rollout buffer, which is the amount of experience gathered before updating the agent.
*   `batch_size`: this is the size of the mini-batches used when updating the networks during optimization. the rollout data is divided into mini-batches.
*   `n_epochs`: this determines how many times the optimizer loops through the mini-batches of data from the buffer to update the networks in a single update phase.

the number of updates is then influenced by how many times you iterate through that rollout buffer in mini-batches during training.

here's a breakdown of how you can calculate the number of updates:

1.  **number of batches:** the total number of batches in your rollout data is calculated by dividing `n_steps` by `batch_size`. for example, if `n_steps` is 2048 and `batch_size` is 64, there are 2048 / 64 = 32 batches.
2.  **updates per rollout:** since each update phase utilizes the data in mini-batches from the rollout buffer `n_epochs` times, the number of model parameter updates per rollout is `n_epochs`. so if we set `n_epochs=10`, then we update the parameters of the policy and value network using data from the rollout buffer 10 times (iterating over the mini-batches of that buffer).
3.  **total updates over training:** during training we typically execute `total_timesteps` of environmental interactions. the number of rollouts is `total_timesteps` divided by `n_steps`. then, to get the total number of updates, you have to multiply number of rollouts, number of batches and `n_epochs`, but since the updates happen during each rollout, you would multiply the number of rollouts by `n_epochs` which is the number of parameter updates per rollout buffer. and the number of rollouts is calculated by the `total_timesteps` divided by `n_steps`. so the total updates is `(total_timesteps / n_steps) * n_epochs`.

here’s a snippet of how you might specify those parameters when setting up a ppo model with sb3:

```python
from stable_baselines3 import ppo
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=4)

model = ppo("MlpPolicy", env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            verbose=1)

model.learn(total_timesteps=100000)

print(f"number of updates: {100000/2048 * 10}")
```

in this snippet, `n_steps=2048`, `batch_size=64`, and `n_epochs=10`. the model will take 2048 steps in the environment, store those interactions into a buffer, which is then divided into 32 mini-batches to update its parameters during each rollout. it uses that mini-batch data for 10 epochs each rollout before repeating again another 2048 environment steps. during the `learn()` phase, if you run for `total_timesteps=100000`, then you will do approximately 488 updates. `100000/2048` is about 48.8, and multiplying it by `n_epochs` (10) is about 488.

another common question is related to gradient updates. a single update in sb3 ppo involves several gradient update steps, due to the mini-batch approach. let’s say you have a batch size of 64 and a `n_steps` of 2048 and an `n_epochs` of 10. in this case, during one rollout update phase, the gradients are computed and applied 320 times (32 mini-batches times 10 epochs). you then repeat another rollout by interacting with the environment, this process is repeated until the `total_timesteps` is reached. if this sounds confusing, don't feel bad, i once got stuck thinking that the number of training steps was the `n_steps`, and i was really wondering why my model did not learn at all.

here's another example with different parameters:

```python
from stable_baselines3 import ppo
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=4)

model = ppo("MlpPolicy", env,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            verbose=1)

model.learn(total_timesteps=50000)

print(f"number of updates: {50000/1024 * 5}")
```

here, with `n_steps=1024`, `batch_size=32`, and `n_epochs=5` and training for 50000 environment steps, the model would perform approximately 244 updates. `50000/1024` which is roughly 48.8 and then multiplied by `n_epochs=5` gets about 244.14. it’s a little less updating than the previous example, given the reduced parameters.

you can also have a case with very low `batch_size`, something that i tried once by mistake. it took me a while to realize my training process was very slow, but also it ended up with a model that was a bit more general and robust:

```python
from stable_baselines3 import ppo
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=4)

model = ppo("MlpPolicy", env,
            n_steps=2048,
            batch_size=8,
            n_epochs=10,
            verbose=1)

model.learn(total_timesteps=100000)

print(f"number of updates: {100000/2048 * 10}")
```

notice here that `n_steps=2048`, `batch_size=8`, and `n_epochs=10`, so the total number of updates is still approximately 488, because the `n_steps`, `n_epochs` and `total_timesteps` remained the same. what changed is the number of mini-batches per rollout, which went from 32 in the first example to 256 in this example. the number of mini-batch gradient updates are thus 2560, higher than the 320 of the first example. so, while the overall number of updates did not change, the number of gradient updates during each update phase increased.

to really grasp the inner workings of ppo, i'd recommend diving into the original ppo paper by schulman et al. from 2017. it's surprisingly readable, despite its technical nature. “proximal policy optimization algorithms” is its full title. another book that has helped me a lot in understanding the subject is “reinforcement learning: an introduction” by sutton and barto. it gives a strong foundation in rl concepts.

finally, to tie it all together, a higher number of updates isn't always better. it’s a balancing act. sometimes, too many updates can lead to overfitting to the current data batch and make generalization worse. it's all about finding the right balance for your specific problem. it really reminds me of the time i was told that the computer scientists are just glorified gardeners (i laughed a lot but i still remember that).

hope that explanation clears things up. let me know if you have any more questions!
