---
title: "How can pymoo and NSGA-II be used for multi-objective portfolio optimization in Python?"
date: "2025-01-26"
id: "how-can-pymoo-and-nsga-ii-be-used-for-multi-objective-portfolio-optimization-in-python"
---

I've spent a significant amount of time developing algorithmic trading strategies, and one of the recurring challenges has been optimizing portfolios across multiple competing objectives, such as maximizing returns while simultaneously minimizing risk. The Non-dominated Sorting Genetic Algorithm II (NSGA-II), implemented in Python via the `pymoo` library, has proven to be a highly effective approach for addressing these kinds of multi-objective optimization problems. Portfolio construction inherently presents a trade-off between these competing goals, making NSGA-II well-suited for finding a set of optimal solutions (a Pareto front) rather than a single best solution.

Let's delve into how this works. `pymoo` is a powerful framework for evolutionary optimization, providing a modular and extensible environment for implementing genetic algorithms, including NSGA-II. The core idea of NSGA-II is to maintain a diverse population of potential solutions (in this context, portfolios), evaluate their performance based on multiple objectives, and apply genetic operators (selection, crossover, mutation) to iteratively improve the population. The key differentiator from single-objective optimization lies in its ability to handle trade-offs. It doesn't converge to a single best solution but rather to a set of non-dominated solutions. A solution is considered non-dominated if no other solution in the population is better in all objectives. The outcome is a set of portfolios representing various compromises between return and risk, allowing the decision-maker to choose the preferred balance.

The implementation process generally involves these steps: defining the problem, specifying the objectives, defining the constraints, setting up the genetic algorithm, and visualizing the results.

First, defining the problem requires formulating the portfolio optimization as a mathematical optimization problem. This involves identifying the assets (e.g., stocks, bonds), gathering their historical data (e.g., returns, volatilities, correlations), and formalizing the portfolio objective(s). For instance, one objective may be to maximize the portfolio return, and another to minimize its volatility.

Here's how you might define a simple portfolio optimization problem with two objectives using a synthetic dataset:

```python
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

class PortfolioProblem(Problem):

    def __init__(self, n_assets=10, num_periods=252, **kwargs):
        self.n_assets = n_assets
        self.num_periods = num_periods
        super().__init__(n_var=self.n_assets, n_obj=2, n_constr=1, xl=0, xu=1, **kwargs)
        # Generate random returns and covariance matrix for illustration
        self.returns = np.random.normal(0.0005, 0.01, size=(self.num_periods, self.n_assets))
        self.cov_matrix = np.cov(self.returns, rowvar=False)

    def _evaluate(self, x, out, *args, **kwargs):
       # x: array of portfolio weights (sum to 1)
        portfolio_returns = np.mean(np.dot(self.returns, x.T), axis=0)
        portfolio_volatility = np.sqrt(np.dot(x.T, np.dot(self.cov_matrix, x)))
        
        out["F"] = np.column_stack([-portfolio_returns, portfolio_volatility]) #negate return for maximization
        out["G"] = 1 - np.sum(x, axis=1) #constraint: weights sum to 1
```

In this `PortfolioProblem` class, I define the number of assets, a basic simulation of historical returns, and the covariance matrix. Crucially, I encapsulate the evaluation logic in the `_evaluate` method. It takes portfolio weights (x) as input, calculates portfolio returns (mean return across historical periods) and volatility, and then returns these two values (volatility is already a value to be minimized, so return is negated to ensure both objectives are to be minimized.) It returns the objectives, F, as well as constraint values, G. In my implementation I've added the constraint to enforce that weights must sum to 1.

Next, you need to set up the NSGA-II algorithm and run it against this defined problem. This involves choosing the optimization algorithm and setting relevant parameters such as population size, number of generations, and crossover and mutation rates. The `pymoo` library provides a dedicated class for NSGA-II and associated configurations:

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

#Define algorithm with relevant parameters
algorithm = NSGA2(pop_size=100,
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(eta=20),
                  eliminate_duplicates=True
                  )
termination = get_termination("n_gen", 50)
problem = PortfolioProblem()
res = minimize(problem,
                 algorithm,
                 termination,
                 verbose=False)
```

This code snippet defines the NSGA-II algorithm using a population size of 100, SBX crossover, and PM mutation operators. I've configured a termination criterion of 50 generations and a verbosity flag to suppress unnecessary output. The optimization is then executed using the `minimize` function, resulting in an object `res` containing the final solution.

Finally, it's important to analyze and visualize the results. `res` from the optimizer contains the population of non-dominated solutions (the Pareto front). You would typically plot these solutions to see the trade-offs between objectives:

```python
import matplotlib.pyplot as plt

#Extract Pareto front and objective values
pareto_front = res.F
returns = -pareto_front[:,0]
volatilities = pareto_front[:,1]

#visualise results
plt.figure(figsize=(8, 6))
plt.scatter(volatilities, returns,  s=20)
plt.xlabel("Portfolio Volatility", fontsize=12)
plt.ylabel("Portfolio Returns", fontsize=12)
plt.title("Pareto Front for Multi-Objective Portfolio Optimization", fontsize=14)
plt.grid(True)
plt.show()
```
Here, I extract the objective values and then use matplotlib to generate a scatter plot. The x-axis represents volatility, and the y-axis represents returns. Each point in the scatter represents a specific portfolio on the Pareto front, allowing for a visual understanding of available trade-offs between the objectives.

For more advanced analysis, one could consider techniques like calculating the Sharpe ratio of the solutions and adding this as additional performance metric or exploring specific portfolios of interest, based on investor risk tolerance or preferences. One should also add additional constraints like position limits or concentration constraints to reflect real-world needs. Additionally, to make this more robust, testing the optimization over a range of different asset classes and different data frequencies would be very worthwhile.

`pymoo` offers much more functionality than I have described. For further learning and more complex problems, I recommend exploring the `pymoo` documentation. Specifically, the section on constraint handling and objective scaling would be useful. Furthermore, consider academic papers discussing the NSGA-II algorithm to further your understanding.  Books on portfolio management often include sections on multi-objective optimization techniques and may be valuable for a holistic view. Finally, financial engineering textbooks can provide further context on specific financial applications.
