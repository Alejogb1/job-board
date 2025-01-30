---
title: "How can a simple MCMC method be applied to an SIR infection model?"
date: "2025-01-30"
id: "how-can-a-simple-mcmc-method-be-applied"
---
The efficacy of Markov Chain Monte Carlo (MCMC) methods in Bayesian inference, particularly within the context of epidemiological models like SIR, hinges on their ability to efficiently sample from complex, high-dimensional posterior distributions.  My experience working on disease outbreak simulations for the CDC highlighted the challenges in estimating parameters –  specifically the infection rate (β) and recovery rate (γ) –  given noisy observational data.  MCMC methods provided a robust solution, enabling a principled approach to uncertainty quantification.

**1.  Explanation of MCMC Application to SIR Models:**

The SIR model, a compartmental model, describes the progression of an infection through a population divided into susceptible (S), infected (I), and recovered (R) individuals.  Its dynamics are governed by the following system of ordinary differential equations:

dS/dt = -βSI/N
dI/dt = βSI/N - γI
dR/dt = γI

where:

* β is the infection rate (probability of transmission per contact per unit time)
* γ is the recovery rate (inverse of the average infectious period)
* N is the total population size.

Estimating β and γ directly from observed infection counts is complicated by the inherent stochasticity of the disease transmission process and the limited observational data.  A Bayesian approach, coupled with MCMC, provides a framework to address these complexities.  We specify prior distributions for β and γ, reflecting our prior knowledge or beliefs about their values.  Then, using observed data (e.g., daily infection counts), we construct the likelihood function, expressing the probability of observing the data given specific values of β and γ.  The posterior distribution, which represents our updated belief about β and γ after observing the data, is proportional to the product of the prior and the likelihood:

Posterior ∝ Likelihood × Prior

Unfortunately, we rarely can derive the posterior distribution analytically; this is where MCMC methods become essential.  MCMC algorithms, such as the Metropolis-Hastings algorithm or Gibbs sampling, generate a Markov chain whose stationary distribution is the target posterior distribution.  By running the chain for a sufficient number of iterations, we obtain a sample from the posterior, allowing us to estimate the parameters and quantify their uncertainty through credible intervals.

**2. Code Examples with Commentary:**

The following examples demonstrate a basic Metropolis-Hastings implementation.  Note that these are simplified illustrations and optimized implementations would require more sophisticated techniques, such as adaptive proposals or Hamiltonian Monte Carlo for higher dimensionality.

**Example 1:  Basic Metropolis-Hastings in Python:**

```python
import numpy as np

# Prior distributions (uniform for simplicity)
prior_beta = lambda x: 1 if 0 < x < 1 else 0  #Example prior
prior_gamma = lambda x: 1 if 0 < x < 1 else 0 #Example prior

# Likelihood function (assuming Poisson distribution for daily cases)
def likelihood(beta, gamma, observed_cases, simulated_cases):
    return np.prod(np.exp(-simulated_cases) * (simulated_cases)**observed_cases / np.math.factorial(observed_cases))

# SIR simulation (simplified Euler method)
def sir_simulate(beta, gamma, initial_conditions, days):
    #Implementation of a basic Euler method for the SIR model
    # ... (Implementation details omitted for brevity, but would involve iterative solution of the ODEs) ...
    return simulated_cases #Returns the simulated number of cases.

# Metropolis-Hastings algorithm
def metropolis_hastings(observed_cases, n_iterations, initial_beta, initial_gamma, proposal_sd):
    beta = initial_beta
    gamma = initial_gamma
    chain = [(beta, gamma)]
    acceptance_rate = 0
    for i in range(n_iterations):
        proposed_beta = np.random.normal(beta, proposal_sd)
        proposed_gamma = np.random.normal(gamma, proposal_sd)
        #Run the SIR Model.
        simulated_cases = sir_simulate(proposed_beta, proposed_gamma, initial_conditions, days)
        acceptance_prob = min(1, (likelihood(proposed_beta, proposed_gamma, observed_cases, simulated_cases) * prior_beta(proposed_beta) * prior_gamma(proposed_gamma)) / (likelihood(beta, gamma, observed_cases, simulated_cases) * prior_beta(beta) * prior_gamma(gamma)) )

        if np.random.rand() < acceptance_prob:
            beta = proposed_beta
            gamma = proposed_gamma
            acceptance_rate +=1
        chain.append((beta, gamma))
    return np.array(chain), acceptance_rate / n_iterations

# Example usage
observed_cases = np.array([10, 20, 30, 25, 15]) #Example observed cases
initial_conditions = [990,10,0] # Example initial conditions for S,I,R
days = len(observed_cases)
chain, acceptance_rate = metropolis_hastings(observed_cases, 10000, 0.5, 0.2, 0.1)
print("Acceptance rate:", acceptance_rate)
```

This example illustrates a basic Metropolis-Hastings implementation. The crucial aspect is the acceptance probability calculation, which governs the movement of the chain through the parameter space.  The `sir_simulate` function (implementation omitted for brevity) would contain a numerical solver for the SIR ODEs.  The choice of proposal distribution (here a normal distribution) significantly impacts the efficiency of the sampler.

**Example 2:  Gibbs Sampling (Illustrative):**

Gibbs sampling, while not always directly applicable to the joint posterior of β and γ without transformations, could be used if we could condition on one parameter to easily sample the other.  This requires carefully constructed conditional distributions.  This example is purely illustrative and might not always be feasible.

```python
# ... (Prior and likelihood functions as before) ...

# Assume conditional posteriors are easily sampled from (this is often not the case)
def sample_beta_given_gamma(gamma, observed_cases):
    # This would require advanced techniques or approximations, often unavailable.
    # ... (Implementation omitted due to complexity) ...
    return beta_sample

def sample_gamma_given_beta(beta, observed_cases):
    # This would require advanced techniques or approximations, often unavailable.
    # ... (Implementation omitted due to complexity) ...
    return gamma_sample

def gibbs_sampling(observed_cases, n_iterations, initial_beta, initial_gamma):
    beta = initial_beta
    gamma = initial_gamma
    chain = [(beta, gamma)]
    for i in range(n_iterations):
        beta = sample_beta_given_gamma(gamma, observed_cases)
        gamma = sample_gamma_given_beta(beta, observed_cases)
        chain.append((beta, gamma))
    return np.array(chain)

# ... (Example usage similar to Metropolis-Hastings) ...
```

This example highlights the potential of Gibbs sampling, but its applicability depends on the tractability of the conditional posterior distributions.

**Example 3:  Incorporating Stan (Conceptual):**

For more complex models or more sophisticated MCMC algorithms, using a probabilistic programming language like Stan is highly recommended. Stan handles the complexities of the algorithm internally, allowing the user to focus on model specification.

```R
# Stan code (conceptual)
data {
  int<lower=0> N; // Population size
  int<lower=0> T; // Number of time points
  int<lower=0> cases[T]; // Observed cases
}

parameters {
  real<lower=0> beta;
  real<lower=0> gamma;
}

model {
  // Prior distributions (example: uniform)
  beta ~ uniform(0, 1);
  gamma ~ uniform(0, 1);

  // Likelihood (assuming Poisson distribution for daily cases, simplified)
  for (t in 1:T) {
    cases[t] ~ poisson(expected_cases[t]);
    }
}
```

This R code snippet uses the `rstan` package to interface with Stan. The Stan code defines the model (priors and likelihood), allowing the Stan engine to automatically implement an efficient MCMC algorithm (often Hamiltonian Monte Carlo).  The `expected_cases` vector would need to be calculated based on the SIR model’s output.

**3. Resource Recommendations:**

"Bayesian Data Analysis" by Gelman et al.
"Introducing Monte Carlo Methods with R" by Robert and Casella
"Statistical Rethinking" by McElreath


These resources provide a comprehensive understanding of Bayesian inference and MCMC methods, essential for effectively applying these techniques to epidemiological modeling.  Remember to adapt these examples to your specific dataset and model assumptions, paying close attention to the prior distributions and the likelihood function to ensure the accuracy and efficiency of your inference.  Always check convergence diagnostics to ensure the reliability of the MCMC samples.
