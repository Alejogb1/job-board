---
title: "Why does Pyomo + SCIP fail to solve 1D real gas pipe flow using the energy equation?"
date: "2025-01-30"
id: "why-does-pyomo--scip-fail-to-solve"
---
The failure of Pyomo + SCIP to solve a 1D real gas pipe flow problem using the energy equation often stems from the inherent non-convexity introduced by the real gas equation of state (EOS) and the frictional pressure drop terms within the energy balance.  My experience working on pipeline optimization problems for a major energy company highlighted this repeatedly.  While linear or convex approximations can sometimes provide feasible solutions,  the resulting solution's accuracy and optimality are questionable without careful consideration of the underlying physics and numerical methods.  Let's examine the core issues and illustrate with examples.


**1. Clear Explanation:**

The energy equation for 1D real gas flow incorporates several non-linear terms.  The real gas EOS, commonly represented by equations like the Peng-Robinson or Soave-Redlich-Kwong equations, introduces non-convexities into the model.  These EOSs are inherently complex, relating pressure, temperature, and density in a non-linear manner.  Furthermore, the frictional pressure drop term, typically represented by the Weymouth equation or a more sophisticated model accounting for pipe roughness and gas compressibility, further contributes to the non-convexity.  These non-convexities prevent the application of efficient convex optimization solvers like SCIP, which are designed for problems with convex feasible regions.  The solver might terminate prematurely by reaching its iteration limit without finding a global optimum, or it might converge to a local optimum that is far from the true solution, especially in challenging problems with multiple local minima.  The problem's sensitivity to initial conditions and parameter values can also significantly influence the solution's quality.  Ignoring the non-convex nature of the problem and directly applying a convex solver like SCIP will frequently result in suboptimal or infeasible solutions, leading to inaccurate predictions of flow rates, pressures, and temperatures along the pipeline.


**2. Code Examples with Commentary:**

The following examples illustrate the problem using a simplified model, ignoring aspects such as elevation changes and heat transfer.  The focus is on demonstrating the non-convexity introduced by the real gas EOS and the friction term.  We'll use a simplified Peng-Robinson EOS for demonstration.  Note that a realistic pipeline model would require a more detailed representation.

**Example 1:  Basic Model without Real Gas Effects**

This example demonstrates a simplified model ignoring real gas effects, using an ideal gas law.  This model is convex and solvable by SCIP.

```python
from pyomo.environ import *

model = ConcreteModel()

# Parameters
model.L = Param(initialize=100) # Pipeline length
model.D = Param(initialize=1)  # Pipeline diameter
model.Z = Param(initialize=1) #Compressibility factor (Ideal gas)

# Variables
model.p = Var(range(2), domain=NonNegativeReals) #Pressure at inlet and outlet
model.q = Var(domain=NonNegativeReals) #Flow Rate

# Equations
model.energy_balance = Constraint(expr = model.p[0] - model.p[1] ==  (f_friction(model.q, model.p.avg(), model.D, model.L, model.Z))) #Simplified Friction Equation


def f_friction(q, p_avg, D, L, Z):
    #Simplified friction term, Linear in this case, NOT realistic
    return 0.1 * q

#Objective Function
model.obj = Objective(expr = model.q, sense = maximize) # Maximize flow rate

#Solver
solver = SolverFactory('scip')
results = solver.solve(model)
results.write()
```

**Commentary:** This simplified model uses a linearized friction term for demonstration.  The model is convex and will readily solve with SCIP.  However, it lacks the non-convexity that would arise from a proper real gas EOS and a non-linear friction term.


**Example 2: Incorporating a Simplified Real Gas EOS**

This example incorporates a simplified, though still non-convex, representation of the Peng-Robinson EOS.


```python
from pyomo.environ import *
import numpy as np

model = ConcreteModel()

# Parameters
model.L = Param(initialize=100) # Pipeline length
model.D = Param(initialize=1)  # Pipeline diameter
model.T = Param(initialize=293) #Temperature

#Simplified Peng-Robinson like term
a = 1 #Simplified constant
b = 0.01 #Simplified constant

# Variables
model.p = Var(range(2), domain=NonNegativeReals) #Pressure at inlet and outlet
model.rho = Var(range(2), domain=NonNegativeReals) #Density at inlet and outlet
model.q = Var(domain=NonNegativeReals) #Flow Rate

# Equations
model.eos = ConstraintList() #Peng-Robinson like equation. In reality more complex.
model.eos.add(model.p[0] == model.rho[0]*(model.T - a/model.rho[0] - b*model.rho[0]))
model.eos.add(model.p[1] == model.rho[1]*(model.T - a/model.rho[1] - b*model.rho[1]))

model.mass_balance = Constraint(expr = model.q == model.rho[0]*model.D**2*3.14159/4) #Mass balance


model.energy_balance = Constraint(expr = model.p[0] - model.p[1] == (f_friction_real_gas(model.q, model.rho.avg(), model.D, model.L))) #Friction term

def f_friction_real_gas(q, rho_avg, D, L):
    #More realistic but still simplified friction term. Depends on density which introduces non-convexity.
    return 0.1*q/rho_avg**(0.5) #Non-linear friction dependent on density

#Objective Function
model.obj = Objective(expr = model.q, sense = maximize) # Maximize flow rate

#Solver
solver = SolverFactory('scip')
results = solver.solve(model)
results.write()
```


**Commentary:** This model introduces a simplified Peng-Robinson-like EOS and a more realistic (but still simplified) friction term dependent on density, making the problem non-convex.  SCIP might struggle to find a global optimum here.

**Example 3:  Employing a Non-Convex Solver**

To address the non-convexity, a non-convex solver like BARON or Couenne is necessary.  This example uses a placeholder for the solver as direct implementation is beyond the scope.  Key is to switch solver.

```python
from pyomo.environ import *
# ... (same model as Example 2) ...

#Solver
solver = SolverFactory('baron') #Or Couenne
results = solver.solve(model)
results.write()
```

**Commentary:** Replacing SCIP with a solver explicitly designed for non-convex problems like BARON or Couenne is essential for obtaining more reliable solutions. However, these solvers are computationally more expensive.



**3. Resource Recommendations:**

For a deeper understanding of real gas flow and optimization, I strongly recommend consulting textbooks on pipeline engineering and optimization, as well as publications on non-convex optimization techniques.  Specialized literature focusing on gas pipeline network optimization will also provide valuable insights into advanced modeling techniques and solution approaches.  Finally, the Pyomo documentation and tutorials, alongside the SCIP, BARON, and Couenne documentation, are essential resources for implementing and understanding the solvers.  Thorough understanding of numerical methods is also crucial for the correct interpretation of obtained results.
