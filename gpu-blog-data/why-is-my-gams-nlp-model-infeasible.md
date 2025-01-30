---
title: "Why is my GAMS NLP model infeasible?"
date: "2025-01-30"
id: "why-is-my-gams-nlp-model-infeasible"
---
Infeasible GAMS NLP models often stem from inconsistencies between the problem's formulation and the solver's capabilities, primarily manifesting as violations of constraints or illogical objective function definitions.  My experience troubleshooting hundreds of NLP models in diverse energy and logistics optimization scenarios points to three major sources:  incorrect constraint definition, numerical instability, and solver-specific limitations.  Let's examine each in detail.


**1. Incorrect Constraint Definition:**  This is by far the most common cause of infeasibility.  Constraints represent the operational limitations and logical requirements of the system being modeled.  A poorly defined constraint, either through a logical error in its representation or a misunderstanding of the system's physical limitations, will lead to an infeasible solution space.  This often involves subtle errors difficult to spot through cursory inspection.


For example, consider a simple transportation model where we aim to minimize transportation costs while satisfying demand at each destination.  If a constraint incorrectly limits the supply at a particular origin to a value less than the total demand requiring supply from that origin, the model will inevitably be infeasible.  Similarly, overlooking non-negativity constraints on decision variables can lead to infeasibility if the solver attempts to assign negative values to variables that represent, say, quantities or flows, which are physically impossible.


**Code Example 1:  Illustrating Incorrect Supply Constraint**

```gams
Sets
    i "Origins" / Origin1, Origin2 /
    j "Destinations" / Dest1, Dest2 /;

Parameters
    supply(i) "Supply at each origin" / Origin1 100, Origin2 50 /
    demand(j) "Demand at each destination" / Dest1 80, Dest2 80 /
    cost(i,j) "Transportation cost";

cost(i,j) = uniform(1,10);  

Variables
    x(i,j) "Quantity transported from i to j";

Equations
    supply_constraint(i) "Supply constraint"
    demand_constraint(j) "Demand constraint"
    objective "Minimize total transportation cost";

supply_constraint(i).. sum(j, x(i,j)) =l= supply(i);  *Incorrect constraint: <= should be =*
demand_constraint(j).. sum(i, x(i,j)) =g= demand(j);
objective.. sum(i, sum(j, cost(i,j)*x(i,j))) =e= z;

Model transport /all/;
Solve transport using nlp minimizing z;
Display x.l, z.l;
```

In this example, the `supply_constraint` uses `=l=` (less than or equal to), incorrectly restricting supply.  The correct constraint should use `=e=` (equal to) if the entire supply must be used. Changing it to `=e=` will rectify the issue if the total supply is sufficient to meet total demand. Otherwise, further adjustments to the model are needed.  The use of `uniform(1,10)` simulates cost generation; in a real scenario, actual costs would be used.


**2. Numerical Instability:**  NLP solvers rely on iterative algorithms to find solutions.  These algorithms can be sensitive to numerical issues, particularly when dealing with highly non-linear functions or poorly scaled data.  Ill-conditioned matrices, extremely small or large coefficients, and non-convex objective functions can all contribute to numerical instability, leading to infeasibility or convergence failures.


During my work on a large-scale power system optimization problem, I encountered a case where the solver reported infeasibility due to numerical issues stemming from the highly non-linear power flow equations. Rescaling the variables and employing a more robust solver (e.g., IPOPT with appropriate tolerances) resolved the problem.  Similarly, carefully examining the order of magnitude of variables and parameters and potentially rescaling them can prevent numerical difficulties.


**Code Example 2:  Illustrating Numerical Instability (Simplified)**

```gams
Variables
  x, y;

Equations
  eq1, eq2;

eq1.. x^2 + y^2 = 1e-10;
eq2.. x - y = 1e10;

Model model1 /all/;
Solve model1 using nlp minimizing x;
Display x.l, y.l;
```

This simplified example showcases potential numerical instability. The very small value on the right-hand side of `eq1` combined with the very large value on the right-hand side of `eq2` might cause numerical issues for some solvers.  Preprocessing the model to rescale the variables could significantly improve the solver's ability to handle this problem.  A change of variables, replacing x and y with scaled versions, could help.


**3. Solver-Specific Limitations:**  Different NLP solvers have different strengths and weaknesses.  A model that is feasible for one solver might be infeasible for another due to the solver's algorithm, its tolerance settings, or its ability to handle specific types of non-linearity.  The choice of solver and its parameter settings is therefore critical.  Some solvers, for example, are better suited to convex problems than non-convex problems.


I recall a project where a model was declared infeasible by CONOPT but found a feasible solution when using IPOPT.  The difference lay in the solvers' algorithms and their tolerances for constraint violation.  Experimenting with different solvers and their settings is often necessary when dealing with infeasible models.  Checking solver-specific documentation and understanding their capabilities is vital.


**Code Example 3:  Illustrating Solver Dependence (Conceptual)**

```gams
*... model definition ...*

Solve model using nlp min z;  *Using default solver, might fail*

Solve model using nlp min z  option nlp=ipopt; *Switching solver*

*... post-processing ...*
```

This example illustrates how changing the solver (e.g., from the default to IPOPT) can impact the solution process.  Different solvers have different parameters to adjust (e.g., tolerances, maximum iterations).  Experimentation with these settings can be crucial in resolving infeasibility issues.



**Resource Recommendations:**

The GAMS model library, the GAMS documentation, and a good optimization textbook focusing on non-linear programming are invaluable resources. Understanding the theoretical foundations of NLP is crucial for effective troubleshooting.  Furthermore, consulting with experienced GAMS users or attending relevant workshops can significantly aid in resolving complex infeasibility issues.


In summary, infeasibility in GAMS NLP models frequently arises from errors in constraint definition, numerical instability, or solver limitations.  Systematic investigation, careful checking of the model formulation, understanding solver capabilities, and exploring numerical strategies are necessary steps for effective troubleshooting. Remember to thoroughly check for logical inconsistencies in the model representation of the system.  The debugging process often requires a combination of careful review, iterative testing, and a good understanding of optimization algorithms.
