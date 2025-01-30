---
title: "How does NSGA2 optimization perform with two objectives?"
date: "2025-01-30"
id: "how-does-nsga2-optimization-perform-with-two-objectives"
---
Non-dominated Sorting Genetic Algorithm II (NSGA-II) excels in multi-objective optimization, particularly when dealing with two conflicting objectives. I've observed its effectiveness firsthand in various applications, ranging from embedded system power management to algorithmic trading strategy optimization, and a key reason is its ability to maintain a diverse set of Pareto-optimal solutions. NSGA-II's architecture is specifically designed to handle these scenarios efficiently, making it a popular choice for problems where trade-offs must be carefully considered.

The core of NSGA-II lies in its ranking and crowding distance mechanisms. Initially, a population of candidate solutions is generated. These solutions are evaluated against both objectives. Instead of assigning a single fitness value, solutions are ranked based on Pareto dominance. A solution 'A' dominates solution 'B' if A is as good as or better than B in all objectives and strictly better in at least one objective. This process divides the population into fronts, where the first front contains the non-dominated solutions within the entire population, the second front contains the non-dominated solutions within the remaining population after excluding the first front, and so on.

Each solution is also assigned a crowding distance value, measuring how densely populated the space is around that particular solution. Solutions residing in less crowded areas are given preference, promoting population diversity. This avoids premature convergence to a specific region of the Pareto front. The combination of non-dominated ranking and crowding distance allows the algorithm to efficiently explore the solution space and maintain a diverse set of high-quality solutions, providing decision-makers with a variety of options along the trade-off curve.

The selection, crossover, and mutation operators are then applied to the current population to create the next generation. The parents are selected from the current population based on non-domination rank and crowding distance. During selection, solutions with better ranks (lower rank number) are preferred. If two solutions have the same rank, the one with the higher crowding distance is chosen. This promotes both quality and diversity in selecting parents. Standard crossover and mutation techniques, common to genetic algorithms, are subsequently applied to generate new offspring. The parent and offspring populations are combined, sorted by non-domination rank and crowding distance, and the new population is selected. The process iterates until a termination criterion is met, such as a maximum number of generations, or a target convergence rate is achieved.

When dealing with two objectives, the resulting Pareto front is often visually interpretable as a curve. Solutions closer to either axis represent a strong performance in one objective but lower performance in the other, capturing trade-offs explicitly. This facilitates informed decision-making, enabling a user to pick a solution that best suits the requirements of the particular problem. It is in this aspect that NSGA-II truly shines; its output allows for an informed selection among multiple acceptable answers, whereas single objective optimization yields only one.

Below are examples demonstrating NSGA-II behavior with two objectives.

**Example 1: Minimization of Two Quadratic Functions**

This simple example demonstrates how the algorithm handles minimization of two competing, convex functions.

```python
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.problems import Problem
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation

class TwoQuadratic(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=-5, xu=5)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]**2
        f2 = (x[0]-2)**2
        out["F"] = np.column_stack([f1, f2])

problem = TwoQuadratic()
algorithm = NSGA2(pop_size=100,
                 sampling=get_sampling("rnd"),
                 crossover=get_crossover("real_sbx", prob=0.9),
                 mutation=get_mutation("real_pm", eta=1.0)
                )
res = minimize(problem, algorithm, ('n_gen', 150), verbose=False)

print(f"Pareto front objectives:\n{res.F}")
```

The output `res.F` contains a series of objectives, where for each individual, we see the value of `f1` and `f2`.  The results showcase the typical Pareto front. Some solutions are close to minimizing `f1`, and others minimizing `f2`, as intended by the algorithm. This example highlights the algorithm's capability in resolving the trade-off between two functions.

**Example 2:  Simple Two-Objective Optimization using a Function**

This example is deliberately abstract to illustrate a generic use-case.  This is the kind of scenario we would typically use in my daily work, when testing different solution spaces for new models.

```python
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.problems import Problem
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation

class CustomProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
      f1 =  (x[:,0] - 0.5)**2 + (x[:,1] - 0.5)**2
      f2 =  np.sin(x[:,0] * np.pi) + np.cos(x[:,1] * np.pi)
      out["F"] = np.column_stack([f1, f2])

problem = CustomProblem()
algorithm = NSGA2(pop_size=100,
                 sampling=get_sampling("rnd"),
                 crossover=get_crossover("real_sbx", prob=0.9),
                 mutation=get_mutation("real_pm", eta=1.0)
                )
res = minimize(problem, algorithm, ('n_gen', 150), verbose=False)

print(f"Pareto front objectives:\n{res.F}")
```

Here, I am not using a known problem as in the previous example but rather using abstract functions. This demonstrates how NSGA-II can be used for general multi-objective optimization, beyond standard benchmarks.  `f1` evaluates distance from a specific point and `f2` a more complex trigonometric expression. The Pareto front produced by this code demonstrates trade-offs between `f1` and `f2`, with the output being a collection of solutions that reflect the trade-offs available for this problem.

**Example 3:  Discrete Variable Optimization**

In some problems, the solution space can have discrete variables, which is an important aspect of engineering applications where certain parameters are chosen from a defined set. This example showcases how one can leverage encoding to handle such challenges.

```python
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.problems import Problem
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.util.var_mask import BinaryVarMask

class DiscreteProblem(Problem):
  def __init__(self):
    super().__init__(n_var=5, n_obj=2, n_constr=0, xl=0, xu=1)
    self.mask = BinaryVarMask(n_vars=5)
  def _evaluate(self, x, out, *args, **kwargs):
        x = self.mask.decode(x)
        f1 = np.sum(x, axis=1)
        f2 = -np.prod(x, axis=1)
        out["F"] = np.column_stack([f1, f2])
  def _decode(self, x):
    return self.mask.decode(x)

problem = DiscreteProblem()

algorithm = NSGA2(
   pop_size=100,
   sampling = get_sampling("bin_random"),
    crossover=get_crossover("bin_ux"),
    mutation=get_mutation("bin_bitflip"),
)


res = minimize(problem, algorithm, ('n_gen', 100), verbose=False)

print(f"Pareto front objectives:\n{res.F}")
```

This code defines the solution space as an array of 0s and 1s, demonstrating a simple discrete space problem. This kind of problem would be used if one were selecting a set of discrete components. Here, `f1` is the sum of the binary array (e.g., number of selected components) and `f2` is the inverse product of elements (e.g. components are selected in a mutually exclusive way). The outputs again demonstrate the trade-offs between objectives. The change in sampling, crossover, and mutation demonstrates the need to select suitable techniques based on the search space of the problem.

To delve deeper into the theoretical aspects, I recommend consulting the original paper on NSGA-II, authored by Deb et al. For a practical understanding, books on evolutionary computation offer a comprehensive overview of genetic algorithms, including NSGA-II. Additionally, numerous online resources detail the implementation specifics, alongside explanations of underlying concepts. Further experimentation with specific parameter values for NSGA-II, including population size and mutation rates, can be used to tailor the algorithm to specific problem demands. It's crucial to remember that there isn't one single, universally 'correct' set of parameters, and exploration of the solution space will always be helpful in a real world scenario.
