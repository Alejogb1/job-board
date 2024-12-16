---
title: "How do I use AddBoolOr and AddImplication in cp_sat solver?"
date: "2024-12-16"
id: "how-do-i-use-addboolor-and-addimplication-in-cpsat-solver"
---

Okay, let’s tackle this. The usage of `AddBoolOr` and `AddImplication` within the `cp_sat` solver, particularly when crafting constraint satisfaction problems, can appear deceptively straightforward initially. However, it’s in the nuanced understanding of their behavior and how they interact with other constraints that true mastery resides. I’ve spent quite a few late nights debugging optimization models where these two played pivotal roles, often leading to frustratingly subtle errors if not handled carefully. I recall one specific project, an automated warehouse routing system, where a poorly constructed conditional relationship built with these tools ended up sending a robotic arm straight into a wall – thankfully, only virtually! That experience certainly drove home the importance of precise constraint definition. Let’s get into it.

Firstly, `AddBoolOr`. At its core, this function allows you to express a disjunction – a logical *or* – between a series of boolean variables or their negations. Think of it like this: at least one of the conditions you’re listing *must* be true for the constraint to be satisfied. The `cp_sat` solver internally represents boolean variables as propositional variables, so the logic flows quite naturally. However, a common pitfall is thinking that `AddBoolOr` necessarily restricts variables to only be *one* of the options specified. The constraint merely ensures *at least one* condition is true. In practical scenarios, this typically manifests in situations where you want to model several mutually exclusive possibilities, but you must enforce one among them. That’s where additional constraints often become necessary, which we'll see later.

Now, let's explore `AddImplication`. This constraint enforces a directional relationship between two boolean variables. In essence, if the antecedent (the 'if' part) is true, then the consequent (the 'then' part) *must also* be true. If the antecedent is false, the truth value of the consequent becomes irrelevant to the constraint's satisfaction; it can be either true or false without violating the rule. I've found `AddImplication` particularly useful in expressing conditional dependencies: triggering a specific action (or in a model, a specific set of variable assignments) only when certain preconditions are met.

The interplay between these two constraints is where things get truly powerful. In many cases, you need not merely declare individual relationships but compose them, layering several constraints to construct a more robust solution. For example, you might use `AddBoolOr` to express that one of several possible actions *must* happen, and then use `AddImplication` to establish which specific actions are available under a given circumstance.

Let’s move on to some concrete examples in Python with the `ortools` library, the preferred tool for `cp_sat` solver tasks. I’ll show you how they work with working code snippets.

**Example 1: Simple Disjunction with `AddBoolOr`**

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Boolean variables
var_a = model.NewBoolVar('var_a')
var_b = model.NewBoolVar('var_b')
var_c = model.NewBoolVar('var_c')

# Ensure at least one of var_a, var_b, or var_c is true.
model.AddBoolOr([var_a, var_b, var_c])

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"var_a = {solver.Value(var_a)}")
    print(f"var_b = {solver.Value(var_b)}")
    print(f"var_c = {solver.Value(var_c)}")
else:
    print("No solution found")
```

In this basic example, at least one of `var_a`, `var_b`, or `var_c` will be true in any solution, but it does not restrict other variables, which could potentially also be true. The solution could have multiple variables true.

**Example 2: Conditional Execution using `AddImplication`**

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Boolean variables
trigger = model.NewBoolVar('trigger')
action_a = model.NewBoolVar('action_a')

# If trigger is true, then action_a must also be true
model.AddImplication(trigger, action_a)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"trigger = {solver.Value(trigger)}")
    print(f"action_a = {solver.Value(action_a)}")
else:
    print("No solution found")
```

Here, if `trigger` is true, `action_a` is forced to be true as well. But if `trigger` is false, `action_a` can be either true or false. This demonstrates the conditional aspect of the implication.

**Example 3: Combining `AddBoolOr` and `AddImplication` for Mutual Exclusivity**

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Boolean variables
action_1 = model.NewBoolVar('action_1')
action_2 = model.NewBoolVar('action_2')
action_3 = model.NewBoolVar('action_3')
precondition = model.NewBoolVar('precondition')

# Ensure at least one action is taken.
model.AddBoolOr([action_1, action_2, action_3])

# If precondition is true, only action_1 is available
model.AddImplication(precondition, action_1)
model.AddImplication(precondition, action_2.Not())
model.AddImplication(precondition, action_3.Not())

# If precondition is false, action_2 or action_3 can be used
model.AddImplication(precondition.Not(), model.NewBoolOr([action_2, action_3]))


solver = cp_model.CpSolver()
status = solver.Solve(model)


if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"precondition = {solver.Value(precondition)}")
    print(f"action_1 = {solver.Value(action_1)}")
    print(f"action_2 = {solver.Value(action_2)}")
    print(f"action_3 = {solver.Value(action_3)}")
else:
    print("No solution found")
```

This example is more intricate. It shows how to force mutual exclusivity. If `precondition` is true, only `action_1` is allowed. Conversely, if `precondition` is false, either `action_2` or `action_3` (or both, in this case since mutual exclusivity wasn't enforced for them specifically) can happen. This pattern is common in many scheduling or resource allocation scenarios. Note the usage of `model.NewBoolOr([action_2, action_3])` in the last implication: while `action_2` and `action_3` can both be true, the `or` constraint on their combined values is only active when `precondition` is false.

For a deeper understanding, I strongly recommend delving into "Handbook of Satisfiability" by Armin Biere et al., which gives a strong theoretical background. Also, "Principles and Practice of Constraint Programming" by Kenneth R. Apt is invaluable for grasping the underlying constraint logic principles, which underpin all constraint programming solvers. Furthermore, the Google OR-Tools documentation is always a useful companion, but the theoretical foundation is equally important.

I hope these examples and explanations provide a clearer view of how `AddBoolOr` and `AddImplication` function and interact in the `cp_sat` solver. Remember, like any tool, proficiency comes from practice and a strong grasp of underlying concepts. Happy solving!
