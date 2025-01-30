---
title: "How do I identify infeasible constraints in a Gurobi model using AMPL?"
date: "2025-01-30"
id: "how-do-i-identify-infeasible-constraints-in-a"
---
The core difficulty in debugging optimization models often stems from infeasible constraint sets, where no solution can simultaneously satisfy all imposed conditions. This is particularly challenging in large models built with AMPL and solved using Gurobi, where pinpointing the source of infeasibility can be like searching for a needle in a haystack. After years spent building and debugging linear programming and mixed-integer programming models, I've found a systematic approach to be essential.

The initial step isn't to blindly re-examine every constraint; rather, it's to leverage Gurobi's sophisticated IIS (Irreducible Infeasible Set) functionality via AMPL. An IIS is a minimal set of constraints that, when combined, cause the model to be infeasible. Removing any single constraint from this set renders the remaining constraints feasible. This process provides a focused path toward identifying modeling errors rather than aimless debugging.

My experience has shown that the standard Gurobi error codes from an infeasible model provide little actionable information. The error report usually indicates that the model has no feasible solution; however, it doesn't illuminate *why* this is the case. This is where the IIS capability, invoked through AMPL’s `solve` command, becomes invaluable.

The first action I take is to modify the `solve` statement in the AMPL model. Instead of a simple `solve;`, I incorporate the `option iisfind 1;` statement *prior* to solving. This tells AMPL and, by extension, Gurobi, to perform an IIS analysis if the solver detects an infeasible model. Subsequently, if the solution is infeasible, AMPL will report on the IIS constraints.

Here's an initial code snippet demonstrating this, assuming a basic model loaded into AMPL:

```ampl
# Assume a model named 'myModel.mod' has already been loaded

option iisfind 1;
solve;

if solve_result = 'infeasible' then {
    printf "The model is infeasible. The following constraints are in the IIS:\n";
    for {c in _con} {
      if _con.iis[c] = 1 then
        printf "%s\n", _con.name[c];
    }
} else if solve_result = 'optimal' then {
   printf "Model solved to optimality.\n";
} else {
  printf "Solver returned status: %s\n", solve_result;
}
```

This example illustrates a fundamental technique. I first set the `iisfind` option to 1. Then, after the solve command is executed, I evaluate the `solve_result`. If it is infeasible, I iterate through all constraints (`_con`) using the `.iis` property, which is set to 1 for constraints included in the irreducible infeasible set. The constraint name, derived from the `.name` property, is then printed. Note that this approach assumes the constraints have user-defined names in the AMPL model, which is a recommended practice. This avoids referencing constraints by automatically generated names, which can be unclear.

In practice, infeasibility frequently arises from inadvertently contradictory constraints or logical errors in the formulation. For instance, suppose you've set a production quota that cannot be met given the resource constraints in your model. This will manifest as an infeasible IIS. Another scenario involves incorrect upper and lower bounds on variables or constraints; for example, setting an upper bound to a value lower than a required minimum.

Consider this code example, illustrative of a resource allocation problem with an intentional infeasibility. In this scenario, we are producing two types of widgets using two resources: labor and material:

```ampl
# Infeasibility via a resource constraint

set PRODUCTS;
set RESOURCES;
param resource_avail{RESOURCES} >= 0;
param resource_usage{PRODUCTS, RESOURCES} >=0;
param demand_min {PRODUCTS} >= 0;
var production{PRODUCTS} >= 0;

minimize cost: sum{p in PRODUCTS} production[p]; # Minimize the total production amount for this example

subject to demand_meet{p in PRODUCTS}: production[p] >= demand_min[p];
subject to resource_limit{r in RESOURCES}: sum{p in PRODUCTS} resource_usage[p, r] * production[p] <= resource_avail[r];

data;

set PRODUCTS := widget1 widget2;
set RESOURCES := labor material;

param demand_min :=
widget1 100
widget2 200;

param resource_avail :=
labor 500
material 500;

param resource_usage: labor material :=
widget1 2 1
widget2 1 3;

option iisfind 1;
solve;

if solve_result = 'infeasible' then {
    printf "The model is infeasible. The following constraints are in the IIS:\n";
    for {c in _con} {
      if _con.iis[c] = 1 then
        printf "%s\n", _con.name[c];
    }
} else if solve_result = 'optimal' then {
   printf "Model solved to optimality.\n";
} else {
  printf "Solver returned status: %s\n", solve_result;
}
```

Here, we've intentionally created infeasibility by setting high demand minimums coupled with resource constraints. In this instance, the IIS will most likely include the `demand_meet` and `resource_limit` constraints. The IIS output would indicate which specific `demand_meet` constraints, perhaps just for `widget1` and `widget2`, and which resource constraints, likely involving both `labor` and `material`, contributed to the infeasibility. By examining the specific instances within the infeasible set, I am better positioned to locate and correct my modeling errors.

Beyond these straightforward cases, more complex scenarios can involve logical interdependencies within the model. For example, a chain of constraints might only become infeasible under specific conditions within a larger conditional block. In those situations, the IIS output might identify constraints which, when considered in isolation, are not obviously problematic, but in the context of the model they cause infeasibility.

Consider this example, which utilizes conditional logic and an if-then-else statement to determine demand targets based on a boolean parameter. In this slightly modified model, the infeasibility arises because the conditional logic produces demand targets that exceed resource limits only under specific circumstances.

```ampl
# Infeasibility via Conditional Logic

set PRODUCTS;
set RESOURCES;
param resource_avail{RESOURCES} >= 0;
param resource_usage{PRODUCTS, RESOURCES} >=0;
param demand_high {PRODUCTS} >= 0;
param demand_low {PRODUCTS} >= 0;
param high_demand_flag binary;
var production{PRODUCTS} >= 0;

minimize cost: sum{p in PRODUCTS} production[p]; # Minimize the total production amount for this example

subject to demand_meet{p in PRODUCTS}: 
  if high_demand_flag = 1 then production[p] >= demand_high[p] 
    else production[p] >= demand_low[p];

subject to resource_limit{r in RESOURCES}: sum{p in PRODUCTS} resource_usage[p, r] * production[p] <= resource_avail[r];

data;

set PRODUCTS := widget1 widget2;
set RESOURCES := labor material;

param demand_high :=
widget1 200
widget2 300;

param demand_low :=
widget1 50
widget2 100;

param resource_avail :=
labor 500
material 500;

param resource_usage: labor material :=
widget1 2 1
widget2 1 3;

param high_demand_flag := 1;

option iisfind 1;
solve;

if solve_result = 'infeasible' then {
    printf "The model is infeasible. The following constraints are in the IIS:\n";
    for {c in _con} {
      if _con.iis[c] = 1 then
        printf "%s\n", _con.name[c];
    }
} else if solve_result = 'optimal' then {
   printf "Model solved to optimality.\n";
} else {
  printf "Solver returned status: %s\n", solve_result;
}
```
In this instance, the value of the `high_demand_flag` parameter set to 1 results in the `demand_high` values being used as targets, resulting in an infeasible model. If `high_demand_flag` is set to zero, the `demand_low` values result in a feasible solution. The IIS would again indicate the relevant `demand_meet` constraints alongside the `resource_limit` constraints, leading me to investigate the implications of the conditional statement and, in this specific case, prompting me to adjust parameter values or constraints to achieve a feasible solution.

Based on my experience, the IIS feature within AMPL and Gurobi is invaluable for diagnosing infeasibility. However, the output requires interpretation within the context of the complete model. I recommend consulting the Gurobi documentation directly for a detailed explanation of its IIS implementation; additionally, the AMPL documentation provides guidance on how AMPL interfaces with Gurobi, which is also highly valuable. Lastly, optimization theory texts can provide a foundation for recognizing common modeling pitfalls that lead to infeasibility.
