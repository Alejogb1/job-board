---
title: "How is `tfp.experimental.mcmc.particle_filter` used?"
date: "2025-01-30"
id: "how-is-tfpexperimentalmcmcparticlefilter-used"
---
Particle filters, specifically `tfp.experimental.mcmc.particle_filter` within TensorFlow Probability, provide a powerful method for approximating the posterior distribution of latent states in a hidden Markov model (HMM) or a more general state-space model. Unlike methods like the Kalman filter, which are restricted to linear-Gaussian systems, particle filters can handle arbitrary non-linearities and non-Gaussian noise distributions. My experience, developing a tracking system for robotic manipulators that incorporated sensor noise characterized by a mixture distribution, directly demonstrated the utility of this flexibility. I found it indispensable for accurately estimating the manipulator's pose.

`tfp.experimental.mcmc.particle_filter` implements a sequential Monte Carlo (SMC) algorithm. This algorithm maintains a set of 'particles,' each representing a potential trajectory of the system's hidden state. The algorithm proceeds in steps, iteratively updating the distribution of these particles by: (1) propagating each particle through the system’s dynamics according to the state transition model; (2) weighting each particle based on its likelihood under the current observation model; and (3) resampling particles to focus computational effort on the most likely trajectories. The resulting set of weighted particles provides an approximation to the true posterior distribution of the hidden state sequence given the observations.

The key benefit of `tfp.experimental.mcmc.particle_filter` is that, under suitable conditions, the approximation converges to the true posterior as the number of particles increases. However, the computational cost also increases linearly with the number of particles, presenting a trade-off between accuracy and efficiency. The efficiency can be partially mitigated by careful choices in the transition and observation models and through optimized TensorFlow implementations, which the TFP library often provides. Another crucial consideration is the "degeneracy" problem, where particle weights become concentrated on a small subset of particles, resulting in a poor approximation. Resampling strategies are designed to mitigate this, distributing the computational load more evenly.

Let's delve into how this functions in practice with code examples.

**Example 1: A Simple 1D Location Tracking Model**

Consider a scenario where we are tracking the one-dimensional location of a moving object. The object moves with some inherent variability and we receive noisy measurements of its location. The true location can be represented as the hidden state, and the noisy measurements are our observations.

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

def particle_filter_1d_location(initial_state_prior,
                                 transition_model,
                                 observation_model,
                                 observations,
                                 num_particles,
                                 num_steps):

  initial_state = initial_state_prior.sample(num_particles)

  def state_transition_fn(particle_state, step):
      transition_dist = transition_model(particle_state)
      return transition_dist.sample()

  def observation_fn(particle_state, step):
      observation_dist = observation_model(particle_state)
      return observation_dist

  particle_filter_results = tfp.experimental.mcmc.particle_filter(
      initial_state=initial_state,
      transition_fn=state_transition_fn,
      observation_fn=observation_fn,
      observations=observations,
      num_steps=num_steps
  )

  return particle_filter_results

# Define specific models:
num_particles = 1000
num_steps = 100

initial_state_prior = tfd.Normal(loc=0., scale=1.)
def transition_model(previous_state):
  return tfd.Normal(loc=previous_state, scale=0.5) # Random Walk
def observation_model(state):
  return tfd.Normal(loc=state, scale=0.8)

# Simulate some observations:
true_states = tf.cumsum(tf.random.normal([num_steps], scale = 0.5))
observations = tfd.Normal(loc=true_states, scale = 0.8).sample()


# Run the particle filter
results = particle_filter_1d_location(
  initial_state_prior,
  transition_model,
  observation_model,
  observations,
  num_particles,
  num_steps)

estimated_states = tf.reduce_mean(results.final_particles, axis=0)

print('Estimated state average (First 10):', estimated_states[:10].numpy())
print('True states (First 10):', true_states[:10].numpy())

```

In this example, the `particle_filter_1d_location` function encapsulates all the necessary components for running a particle filter. First, we initialize the particles with the `initial_state_prior`. The `state_transition_fn` propagates the particles according to the defined transition dynamics (in this case, a simple random walk). The `observation_fn` defines the distribution of the observations given a particular particle state. We then use `tfp.experimental.mcmc.particle_filter` to perform the filtering process with the set of generated simulated `observations`. The mean of the final particle locations is then extracted as our estimate of the true states.

**Example 2: A 2D Robot Localization Model**

Extending to two dimensions, imagine tracking the position of a robot moving in a plane, where we receive noisy GPS readings.

```python
def particle_filter_2d_robot(initial_state_prior,
                              transition_model,
                              observation_model,
                              observations,
                              num_particles,
                              num_steps):

    initial_state = initial_state_prior.sample(num_particles)
    def state_transition_fn(particle_state, step):
      transition_dist = transition_model(particle_state)
      return transition_dist.sample()

    def observation_fn(particle_state, step):
        observation_dist = observation_model(particle_state)
        return observation_dist
    particle_filter_results = tfp.experimental.mcmc.particle_filter(
        initial_state=initial_state,
        transition_fn=state_transition_fn,
        observation_fn=observation_fn,
        observations=observations,
        num_steps=num_steps
    )

    return particle_filter_results

# Define specific models for 2D:
num_particles = 1000
num_steps = 100

initial_state_prior = tfd.MultivariateNormalDiag(loc=tf.zeros(2), scale_diag=tf.ones(2))

def transition_model(previous_state):
   return tfd.MultivariateNormalDiag(loc=previous_state + tf.random.normal(shape=[2], scale=0.1), scale_diag=tf.fill(tf.shape(previous_state),0.4))

def observation_model(state):
  return tfd.MultivariateNormalDiag(loc=state, scale_diag=tf.fill(tf.shape(state), 0.5))

# Simulate observations for 2D movement:
true_states = tf.cumsum(tf.random.normal([num_steps, 2], scale = 0.1), axis=0)
observations = tfd.MultivariateNormalDiag(loc=true_states, scale_diag=tf.fill(tf.shape(true_states), 0.5)).sample()

# Run the particle filter:
results = particle_filter_2d_robot(
  initial_state_prior,
  transition_model,
  observation_model,
  observations,
  num_particles,
  num_steps
)

estimated_states = tf.reduce_mean(results.final_particles, axis=0)
print("Estimated states average (first 5):\n", estimated_states[:5].numpy())
print("True states (first 5):\n", true_states[:5].numpy())
```

This code extends the previous example by using `tfd.MultivariateNormalDiag` for the initial state prior, transition and observation models. The state is now a 2D vector, but the structure of the `particle_filter_2d_robot` function remains the same. The observations and transition model are also sampled in two dimensions.

**Example 3: Handling Non-Linear Observations**

Finally, let us examine a scenario with a non-linear observation function. Suppose our sensor provides observations of the squared location. This introduces a significant challenge for linear-Gaussian filters, but particle filters handle such non-linearities effectively.

```python
def particle_filter_nonlinear(initial_state_prior,
                                 transition_model,
                                 observation_model,
                                 observations,
                                 num_particles,
                                 num_steps):

  initial_state = initial_state_prior.sample(num_particles)
  def state_transition_fn(particle_state, step):
    transition_dist = transition_model(particle_state)
    return transition_dist.sample()

  def observation_fn(particle_state, step):
      observation_dist = observation_model(particle_state)
      return observation_dist

  particle_filter_results = tfp.experimental.mcmc.particle_filter(
      initial_state=initial_state,
      transition_fn=state_transition_fn,
      observation_fn=observation_fn,
      observations=observations,
      num_steps=num_steps
  )

  return particle_filter_results

# Define non-linear observation models:
num_particles = 1000
num_steps = 100

initial_state_prior = tfd.Normal(loc=0., scale=1.)
def transition_model(previous_state):
  return tfd.Normal(loc=previous_state, scale=0.5)

# Non-linear Observation Model
def observation_model(state):
   return tfd.Normal(loc=tf.square(state), scale = 0.5)

# Generate Observations
true_states = tf.cumsum(tf.random.normal([num_steps], scale = 0.5))
observations = tfd.Normal(loc = tf.square(true_states), scale = 0.5).sample()

# Run the particle filter with nonlinear observations:
results = particle_filter_nonlinear(
  initial_state_prior,
  transition_model,
  observation_model,
  observations,
  num_particles,
  num_steps
)


estimated_states = tf.reduce_mean(results.final_particles, axis=0)
print('Estimated States (First 10):\n', estimated_states[:10].numpy())
print('True States (First 10):\n', true_states[:10].numpy())
```

Here, the `observation_model` maps each state to its square, introducing a non-linearity. This demonstrates the versatility of `tfp.experimental.mcmc.particle_filter` in handling complex, non-linear state-space models, which, in my work with robotic vision, proved essential for dealing with non-linearities introduced by camera projection.

For deeper understanding, I recommend consulting academic texts on sequential Monte Carlo methods. These texts provide a rigorous treatment of particle filter algorithms and their convergence properties. Additionally, exploring the TensorFlow Probability documentation and its example notebooks is beneficial for learning specific implementation details and exploring other related MCMC methods. The 'Probabilistic Programming and Bayesian Methods for Hackers' text presents the concepts of Bayesian inference in a practical and accessible way that can help solidify your understanding of the underlying principles, while specialized textbooks on state-space models provide a more theoretical perspective. These resources combined offer both theoretical and practical insights into the effective application of `tfp.experimental.mcmc.particle_filter` for a wide range of problems.
