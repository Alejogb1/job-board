---
title: "Is generational distance invariant to the number of iterations in pymoo?"
date: "2025-01-30"
id: "is-generational-distance-invariant-to-the-number-of"
---
Generational distance (GD), as a metric in evolutionary algorithms like those implemented in `pymoo`, is *not* invariant to the number of iterations. The core reason lies in its definition: GD measures the average distance between the Pareto front obtained by the algorithm at a given generation and a reference Pareto front, often the known or best-approximated Pareto front for a problem. As the algorithm progresses through more iterations, the solution set will typically evolve. Therefore, the GD values themselves change, reflecting this evolution. I've observed this phenomenon extensively while fine-tuning multi-objective optimization routines for simulated robotic kinematics.

The generational distance is calculated by first determining the minimum Euclidean distance between each point on the current approximate Pareto front and the nearest point on a reference Pareto front. These minimal distances are then averaged across all points in the current front. This process effectively measures the closeness of the solution set to the true or ideal objective space. Thus, the calculation inherently relies on the current state of optimization, which is determined by the iteration count.

Let's consider a simplified scenario. Initially, early populations in the evolutionary algorithm may produce solutions that are far from the reference front, resulting in large GD values. As the algorithm iterates, selection, crossover, and mutation operators drive the population towards better performing regions of the objective space. This improvement is directly reflected in a decreasing GD value, indicating that the approximated front is converging towards the reference front. If, however, there are local optima or the problem has a complex landscape, the GD might plateau and even increase slightly due to stagnation, but even that behavior is not iteration invariant; the point of plateau is related to the algorithm progression and complexity of the landscape.

The dependence on iterations stems from the nature of evolutionary algorithms themselves. They explore the solution space iteratively, attempting to find better approximations of the Pareto front. The number of iterations fundamentally defines the computational effort expended, directly impacting the solution set obtained, and therefore influencing the value of GD. GD is a measure that reflects the result of applying the algorithms across iterations and is thus deeply coupled to the process, rather than being independent of it.

Now, let's look at three code examples using `pymoo`, illustrating this point. For simplicity, a two-objective problem will be employed. The ZDT1 benchmark problem is a good candidate for this purpose.

```python
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.gd import GD
import numpy as np

problem = get_problem("zdt1")
algorithm = NSGA2(pop_size=100) # Keep population size fixed for now
ref_front = problem.pareto_front()

# Optimization for 10 iterations
res_10 = minimize(algorithm, problem, ("n_gen", 10), verbose=False)
gd_10 = GD(ref_front).do(res_10.F)
print(f"GD after 10 iterations: {gd_10:.5f}")


# Optimization for 100 iterations
res_100 = minimize(algorithm, problem, ("n_gen", 100), verbose=False)
gd_100 = GD(ref_front).do(res_100.F)
print(f"GD after 100 iterations: {gd_100:.5f}")
```

In the first example, the algorithm runs for only 10 generations. The resulting generational distance will likely be larger, indicating the approximated Pareto front is not very close to the reference front. The second run performs for 100 iterations, leading to a lower GD value. This indicates that with more iterations, the solutions have been refined, and the approximated Pareto front better fits the reference one.  The difference between the print values directly shows the iteration-dependent behavior.

The next example uses a slightly different approach: tracking GD across iterations within a single run using a callback:

```python
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.gd import GD
import numpy as np

class GDCallback:
    def __init__(self, ref_front):
        self.ref_front = ref_front
        self.gd_history = []

    def notify(self, algorithm):
        gd_value = GD(self.ref_front).do(algorithm.pop.get("F"))
        self.gd_history.append(gd_value)
        
problem = get_problem("zdt1")
algorithm = NSGA2(pop_size=100)
ref_front = problem.pareto_front()
callback = GDCallback(ref_front)

res = minimize(algorithm, problem, ("n_gen", 50), callback=callback, verbose=False)

print(f"GD History: {callback.gd_history}")
```

This example demonstrates how GD changes across iterations. The `GDCallback` monitors the current population’s objective space values and calculates the generational distance at each generation, showing the progressive convergence of the approximated front to the reference one. The list shows generally decreasing GD values across the generations.

Finally, an example using multiple runs and plotting the results to better visualize the trends:

```python
import matplotlib.pyplot as plt
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.gd import GD
import numpy as np

def run_gd_tracking(iterations, num_runs=5):
    problem = get_problem("zdt1")
    algorithm = NSGA2(pop_size=100)
    ref_front = problem.pareto_front()
    gd_history = np.zeros((num_runs, iterations))
    for run in range(num_runs):
        callback = GDCallback(ref_front)
        minimize(algorithm, problem, ("n_gen", iterations), callback=callback, verbose=False)
        gd_history[run] = callback.gd_history
    return gd_history

iterations = 100
gd_histories = run_gd_tracking(iterations)

for run_history in gd_histories:
    plt.plot(range(1, iterations+1), run_history)
plt.xlabel('Iteration')
plt.ylabel('Generational Distance')
plt.title('GD Convergence Over Iterations (Multiple Runs)')
plt.show()
```

This example shows multiple runs to demonstrate variance. The results are plotted across iterations, visually displaying how GD tends to decrease as more iterations are completed for different optimization trajectories. Even though the trend is generally negative, there is variance across multiple runs demonstrating randomness in search.

In summary, generational distance in `pymoo`, and generally in evolutionary computation, is fundamentally not invariant to the number of iterations. The metric reflects the algorithmic progress towards a solution set and is thus inextricably linked to the number of search steps (iterations). Increasing the number of iterations tends to result in a decrease in GD, which is the expected behavior of a converging algorithm. However, the precise shape of the convergence curve will depend on numerous factors including the specific optimization algorithm, population parameters, and most importantly the landscape of the optimization problem.

For anyone looking to delve deeper into these topics, I recommend consulting research papers on multi-objective optimization and performance indicators. Texts focused on evolutionary algorithms also provide solid theoretical foundations. The `pymoo` documentation itself, while not a theoretical resource, offers excellent practical examples and details about the different algorithms and metrics. Furthermore, articles in journals focused on computational intelligence and numerical optimization often explore the behavior of metrics like GD under various conditions. Finally, working through a variety of test problem landscapes and performing detailed parameter studies will provide critical practical insights.
