---
title: "How do I interpret the n_nds, eps, and indicator output columns in a Pymoo/NSGA-II multi-objective optimization?"
date: "2025-01-26"
id: "how-do-i-interpret-the-nnds-eps-and-indicator-output-columns-in-a-pymoonsga-ii-multi-objective-optimization"
---

The core of interpreting `n_nds`, `eps`, and `indicator` columns in Pymoo's NSGA-II output lies in understanding their roles within the non-dominated sorting and crowding distance calculation – the very mechanisms that drive the algorithm's search for diverse, Pareto-optimal solutions. These values are not inherently objective function outputs; instead, they reflect the solution’s ranking and density relative to other solutions within the population *at a given generation*. Their significance is thus tightly coupled to the evolutionary process.

Specifically, `n_nds` indicates the non-domination rank of a solution. NSGA-II utilizes a hierarchical ranking system. The 'best' solutions—those that are not dominated by any other solution in the population—belong to the first non-dominated front, and they receive `n_nds=1`. Solutions dominated only by those in the first front are placed in the second front and assigned `n_nds=2`, and so forth. A lower `n_nds` value indicates a superior solution in terms of non-domination, meaning that it is closer to the Pareto frontier for the current population. The term "non-dominated" means that there exists no other solution in the current population that is better across all objective functions.  A solution dominated by another means that at least one solution has all of the same or better objective values than it.

The `eps` value represents the crowding distance of a solution within its non-dominated front. Once solutions are ranked into fronts, the algorithm aims to preserve diversity among solutions within each front. This is achieved through the crowding distance calculation, an estimate of the space "surrounding" a particular solution.  Conceptually, it measures the density of solutions in the immediate vicinity of each particular solution, calculated as the average distance to the two nearest neighbors along each objective dimension.  Solutions on the “edges” of the Pareto front will tend to have very large distances, while solution tightly grouped will have very small distances. A higher `eps` value generally indicates a more isolated solution within its front, which is favored as it helps to ensure the search is not concentrated within small areas of the objective space. The distance is calculated independently per front; that is, each front has its own values.

Finally, the `indicator` column provides a generic placeholder for different performance metrics that can be used to track the optimization process. By default, in many Pymoo examples and configurations, this will likely be empty. However, it is often populated using various *indicators*, such as the Hypervolume. The Hypervolume is a metric commonly used in multi-objective optimization which represents the volume of the objective space dominated by the Pareto front. It is a way to measure the quality and spread of the solutions found, where larger volumes are better. Other indicators like generational distance, inverted generational distance, or the spacing metric, could also be tracked using this column.  The specific contents of this column, thus, depend on the specific needs of the optimization and requires careful inspection when working with a given optimization problem.

```python
# Example 1: Basic Non-dominated Sorting Visualization

import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt1") # ZDT1 is a simple bi-objective test problem
algorithm = NSGA2(pop_size=100)
res = minimize(problem, algorithm, ('n_gen', 10), seed=42) # Run for a few generations

print("Example 1: n_nds values in the last generation")
print(res.pop.get("n_nds"))
```
This first example demonstrates a fundamental property of non-dominated sorting. In this snippet, after only 10 generations of the NSGA-II algorithm the solutions are ranked into non-dominated fronts. By printing `res.pop.get("n_nds")`, I retrieve the `n_nds` value for each solution in the final generation's population. In practice, I would expect many of these values to be 1 (i.e. belonging to the non-dominated front), but there will likely be other values as well, showing how solutions have been pushed towards the front over time. The higher ranked solutions may have moved closer to the Pareto front, but they do not actually dominate other solutions as defined within this population.

```python
# Example 2: Crowding Distance Calculation and its impact

import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt2")  # ZDT2 has a convex Pareto front
algorithm = NSGA2(pop_size=20) # Reduced population size to emphasize diversity
res = minimize(problem, algorithm, ('n_gen', 20), seed=42)

print("\nExample 2: eps values in the last generation")
print(res.pop.get("eps"))
```
In this example, I use ZDT2 which has a convex Pareto front. The critical change is the reduced `pop_size` to 20. This will lead to a sparser solution set in the objective space. After running the optimization, I print the `eps` values.  I will expect a greater variation in `eps` values, some quite large and some that are small. This illustrates that solutions on the periphery of the non-dominated front tend to be favored, as the crowding distance penalizes tightly grouped solutions, even though they may both belong to the same non-dominated front.

```python
# Example 3: Hypervolume Indicator Tracking

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
import numpy as np


problem = get_problem("dtlz1") # DTLZ1 is a three objective problem
ref_point = np.array([1.1, 1.1, 1.1]) # a reference point that should dominate all solutions
hv = Hypervolume(ref_point=ref_point)
algorithm = NSGA2(pop_size=100) # set population size

history = []
for gen in range(20):
  res = minimize(problem, algorithm, termination=('n_gen', 1))
  hv_value = hv.do(res.F) # get the hypervolume of this generation
  history.append((res.pop.get("n_nds"), res.pop.get("eps"), hv_value)) # collect info about the solutions

print("\nExample 3: Hypervolume Indicator Evolution")

for n_nds, eps, hv_value in history:
    print(f"n_nds, eps, hypervolume:{n_nds}, {eps}, {hv_value}")
```
In this final example, I focus on the `indicator` column’s potential by tracking hypervolume through the optimization. Here, I use the DTLZ1 problem which provides a 3D objective space. I define a reference point and then use Pymoo's `Hypervolume` indicator. Within the main loop, I run the algorithm for a single generation each time, collect information including the hypervolume, and store it in the `history` list. Finally, I print the collected data. The printed data shows how the hypervolume increases as the solutions improve and fill out more of the objective space. This highlights the dynamic nature of these metrics. Here, the `indicator` column is replaced with the computed hypervolume value.  This provides useful feedback on the algorithm's performance across the run.

In summary, the `n_nds`, `eps`, and `indicator` columns are not merely passive outputs; they are reflections of the underlying mechanisms of NSGA-II. Understanding `n_nds` is critical for assessing the non-domination ranking of solutions, `eps` reveals the density and diversity within non-dominated fronts, and `indicator` is a user-defined slot to monitor key performance metrics, such as hypervolume, during the optimization process. When reviewing these values, it is critical to consider the population at a given generation, rather than thinking of these metrics as absolute values across the solution space.

For further exploration, I suggest reviewing academic literature on multi-objective optimization and evolutionary algorithms. Research papers focusing on NSGA-II and non-dominated sorting will prove invaluable. Specifically, papers introducing the NSGA-II algorithm itself will help the most. Finally, exploring various examples and configurations in the Pymoo documentation will assist in gaining practical experience with the algorithm. Focusing on the concepts of non-dominated sorting, crowding distance, and performance indicators will allow one to better interpret the output of an NSGA-II optimization run.
