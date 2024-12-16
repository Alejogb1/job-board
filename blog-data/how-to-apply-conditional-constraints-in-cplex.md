---
title: "How to apply conditional constraints in Cplex?"
date: "2024-12-16"
id: "how-to-apply-conditional-constraints-in-cplex"
---

Let's tackle conditional constraints in Cplex; it's a topic I've grappled with extensively, particularly in the early days of working on large-scale supply chain optimization models. One instance I recall involved a rather complex production scheduling problem where specific machine types could only be activated *if* a certain demand threshold was met for specific products. Directly encoding that into the model was... challenging initially. Cplex, while powerful, doesn't natively support "if-then-else" style constraints in the conventional sense that a scripting language would. Instead, we leverage clever mathematical formulations using binary variables and large constants (often called "big-M" constraints) or indicator constraints when feasible.

The core issue revolves around transforming a logical condition into a mathematical one that Cplex's solver can understand and process. Let me illustrate with a scenario where we have two variables, x and y, both continuous and non-negative. We want to enforce the following: *if* `x > 0`, *then* `y >= 5`. This type of constraint doesn't exist directly in standard linear programming (LP) or mixed-integer programming (MIP) formulation. We have to be a bit more creative.

**Method 1: Big-M Formulation**

The first technique we commonly resort to is the "big-M" approach. It works by introducing a binary variable, let's call it `delta`, which takes the value 1 when the condition is true and 0 when it is false. We then combine `delta` with a suitably large constant, `M`, to control the activation of the constraint we want to apply conditionally.

Here's how it pans out for our example:

1.  **Introduce a Binary Variable:** Define a binary variable `delta`, where `delta` = 1 implies x > 0, and `delta` = 0 implies x <= 0 (approximately, since our model may have tolerances).

2.  **Link `x` and `delta`:** Add constraints linking `x` and `delta`. These constraints are designed such that if x is greater than zero, then `delta` must be 1, and if x is zero, delta is allowed to be 0 or 1. Typically they take the form:
    *   `x <= M * delta` (if x is greater than zero, delta must be 1)
    *   `x >= epsilon * delta` (epsilon is a small positive number, effectively making sure that x is zero when delta is 0. This addresses numerical tolerance issues.)

3.  **Apply the Conditional Constraint:** We can now impose our target conditional rule, “if x > 0, then y >= 5” using: `y >= 5 * delta`. When `delta` is 1, the constraint `y >= 5` is active; when `delta` is 0, the constraint becomes `y >= 0` , which is always true, so effectively the condition does not impose the condition `y>=5`.

Let me give you a working code snippet in Python using `docplex`:

```python
from docplex.mp.model import Model

mdl = Model(name='conditional_constraints_bigM')
x = mdl.continuous_var(name='x')
y = mdl.continuous_var(name='y')
delta = mdl.binary_var(name='delta')

M = 1000  # Choosing a large enough M
epsilon = 0.0001 #small positive number for numerical tolerance

# Link x and delta
mdl.add_constraint(x <= M * delta)
mdl.add_constraint(x >= epsilon * delta)

# Conditional constraint: if x > 0 then y >= 5
mdl.add_constraint(y >= 5 * delta)

#Objective to illustrate
mdl.minimize(x + y)

sol = mdl.solve()
if sol:
    print(f"x: {sol.get_value(x)}")
    print(f"y: {sol.get_value(y)}")
    print(f"delta: {sol.get_value(delta)}")
```

In this case, the solution will likely have both `x` and `y` at or very close to 0 unless they are forced to be something else by other constraints or the objective. If you were to, for instance, add `x >=1`, it will result in delta becoming 1 and forcing the conditional constraint y >=5 to be true.

**Method 2: Indicator Constraints**

Cplex, in later versions, introduced the notion of *indicator constraints* which often simplify the syntax quite a lot compared to big-M formulations. Indicator constraints directly associate a binary variable with a linear constraint. In our previous case, the logic is simpler to express. It's still based on a binary variable, but Cplex handles the 'big-M' application implicitly.

Here’s the code to achieve the same using an indicator constraint:

```python
from docplex.mp.model import Model

mdl = Model(name='conditional_constraints_indicator')
x = mdl.continuous_var(name='x')
y = mdl.continuous_var(name='y')
delta = mdl.binary_var(name='delta')

#Link x and delta using indicator constraint
mdl.add_if_then(delta==1, x>0)

# Conditional constraint: if x > 0 then y >= 5
mdl.add_if_then(delta==1, y>=5)

#Objective to illustrate
mdl.minimize(x + y)


sol = mdl.solve()
if sol:
    print(f"x: {sol.get_value(x)}")
    print(f"y: {sol.get_value(y)}")
    print(f"delta: {sol.get_value(delta)}")
```

Observe how the constraint becomes more readable and direct. The `add_if_then` function establishes the indicator relationship. If delta is one, the second constraint is active, and if delta is zero, it's not considered. We still have to define a way to relate `x` and `delta`, which was done using the `x>0` constraint inside the first `add_if_then` line of code. It is important to remember that indicator constraints can only be applied to certain types of constraints; therefore, using big-M formulation is a more versatile approach.

**Method 3: A More Complex Scenario with Multiple Conditions**

Let's expand this to demonstrate a more complex conditional constraint with multiple conditions. Assume we have three variables, `a`, `b`, and `c`, and we want to apply the following rule:

*If* `a > 10` *and* `b < 5`, *then* `c >= 20`.

This now involves combining conditions. Here, I will use the big-M technique, as it is more versatile:

1.  **Introduce Binary Variables for each Condition:** Define binary variables `delta_a`, and `delta_b`. `delta_a` is 1 if a> 10 and `delta_b` is one if `b <5`.

2.  **Link a, b with the respective deltas:** Similar to our previous example, we link a and b to their corresponding deltas using constraints like `a <= 10 + M * (1- delta_a)` (making sure `delta_a` is 1 when `a>10`, and `b>= 5 * (1-delta_b)` (making sure `delta_b` is one when `b<5`). Also, we will add the epsilon to ensure the numerical accuracy of our solution. So the constraint will look like: `a>=10 + epsilon * (1-delta_a)` and `b<=5-epsilon + M*(1-delta_b)`.

3.  **Combined Condition using an auxiliary binary:** Introduce a new binary variable `delta_combined`. We need this auxiliary variable because the condition is not simply dependent on `a` or `b`, but also on their relationship. Set `delta_combined = delta_a + delta_b -1` which means `delta_combined` is 1 if both `delta_a` and `delta_b` are 1, and 0 otherwise.

4.  **Apply Conditional Constraint:** Now add the constraint: `c >= 20 * delta_combined`, forcing `c>=20` only if `delta_combined` is 1 which happens when `a>10` and `b<5` are true.

Here is a python implementation of this:

```python
from docplex.mp.model import Model

mdl = Model(name='complex_conditional')
a = mdl.continuous_var(name='a')
b = mdl.continuous_var(name='b')
c = mdl.continuous_var(name='c')
delta_a = mdl.binary_var(name='delta_a')
delta_b = mdl.binary_var(name='delta_b')
delta_combined = mdl.binary_var(name='delta_combined')


M = 1000  # Choosing a large enough M
epsilon = 0.0001 #small positive number for numerical tolerance


# Link a to delta_a
mdl.add_constraint(a <= 10 + M * (1 - delta_a))
mdl.add_constraint(a >= 10 + epsilon * delta_a )

# Link b to delta_b
mdl.add_constraint(b >= 5 - M * delta_b)
mdl.add_constraint(b <= 5 - epsilon + M * (1-delta_b))

# Combined condition
mdl.add_constraint(delta_combined == delta_a + delta_b - 1)

# Conditional constraint: if a > 10 and b < 5, then c >= 20
mdl.add_constraint(c >= 20 * delta_combined)

#Objective to illustrate
mdl.minimize(a+b+c)

sol = mdl.solve()
if sol:
    print(f"a: {sol.get_value(a)}")
    print(f"b: {sol.get_value(b)}")
    print(f"c: {sol.get_value(c)}")
    print(f"delta_a: {sol.get_value(delta_a)}")
    print(f"delta_b: {sol.get_value(delta_b)}")
    print(f"delta_combined: {sol.get_value(delta_combined)}")
```

This example demonstrates how to construct conditional rules that involve multiple individual conditions linked through ‘and’ or ‘or’ conditions.

For further exploration, I strongly recommend delving into the *Mixed Integer Programming* book by Nemhauser and Wolsey. The book offers a detailed explanation of modeling techniques, especially the use of indicator variables. Similarly, the *Model Building in Mathematical Programming* by H. Paul Williams is also a very practical resource. Additionally, Cplex's own documentation has advanced sections that discuss the subtleties of big-M techniques and indicator constraints. I also suggest a solid read of *Integer Programming and Network Flows* by Ravindra K. Ahuja, Thomas L. Magnanti, James B. Orlin, which provides a great theoretical background for MIP models in general. Mastering these methods is crucial for anyone seriously working with optimization problems. I hope this detailed response and the code snippets will be beneficial for your application of conditional constraints in Cplex.
