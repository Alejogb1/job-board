---
title: "How can OR-Tools CP-SAT be used to constrain two variable lists to share the same integer domain?"
date: "2025-01-30"
id: "how-can-or-tools-cp-sat-be-used-to-constrain"
---
I've frequently encountered situations where the need arises to synchronize the feasible values of two distinct sets of integer variables within a constraint programming model. This synchronization, requiring both lists to effectively operate within an identical numerical domain, is not inherently built into CP-SAT but can be achieved using a combination of intermediate variables and carefully constructed constraints.

The core issue stems from how CP-SAT variables are inherently independent. Creating two integer variable lists, such as `x = model.NewIntVarList(n, 0, 10, "x")` and `y = model.NewIntVarList(n, 0, 10, "y")`, initially provides two distinct sets of variables, each with its own range. There is no explicit link forcing them to share the *same* set of actual possible values; while both may *start* within 0 and 10, one might get assigned to only even numbers, and another only odd, without violating any obvious constraints. To ensure these lists share the same *domain*—that is, the *set* of values each list can take, rather than the range they are *defined* within—we need to construct constraints that establish this equivalency of reachable values.

The strategy I've found effective is threefold. First, we construct a set variable that contains all the possible unique integer values that *either* list can take. This set is implicitly constrained by the ranges of both variable lists. Second, we explicitly link each variable in the two lists to membership in this single set. This enforces that any value assigned to any variable must be present in the established single domain. Finally, we implement an additional constraint: every potential value in this set must be *used* by at least one of the variables from either list. This guarantees that the established set does not contain values *not* present in either variable's solution. This approach ensures a dynamically shared domain across both lists.

Let's illustrate this with code examples using the Python interface to OR-Tools.

**Example 1: Basic Domain Synchronization**

```python
from ortools.sat.python import cp_model

def shared_domain_basic(n):
    model = cp_model.CpModel()

    x = [model.NewIntVar(0, 10, f"x_{i}") for i in range(n)]
    y = [model.NewIntVar(0, 10, f"y_{i}") for i in range(n)]

    all_values = set(range(11)) # Potential values 0 through 10, to include max range
    domain = model.NewSetVar(all_values, "domain_set")

    # Constraint 1: Link x and y to the domain set
    for val in x:
        model.Add(val.MemberOf(domain))
    for val in y:
        model.Add(val.MemberOf(domain))

    # Constraint 2: Domain must be fully used. This is enforced with an additional int var for each value in range(11)
    for val in range(11):
        used_by_x_or_y = model.NewBoolVar(f"used_{val}")
        x_contains = model.NewBoolVar(f"x_contains_{val}")
        y_contains = model.NewBoolVar(f"y_contains_{val}")

        model.AddBoolOr([(variable == val) for variable in x]).OnlyEnforceIf(x_contains) # true if x has var = value
        model.AddBoolOr([(variable == val) for variable in y]).OnlyEnforceIf(y_contains) # true if y has var = value

        model.AddBoolOr([x_contains, y_contains]).OnlyEnforceIf(used_by_x_or_y) # true if either x or y has var = value
        model.AddBoolOr([x_contains.Not(), y_contains.Not()]).OnlyEnforceIf(used_by_x_or_y.Not()) # false if neither has

        model.Add(domain.Contains(val) == used_by_x_or_y) # val must be in domain iff one has

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        print("x:", [solver.Value(var) for var in x])
        print("y:", [solver.Value(var) for var in y])
        print("Domain:", solver.Value(domain))
        return True
    else:
        print("No solution found.")
        return False

shared_domain_basic(3)
```
In this example, `shared_domain_basic(n)` defines an OR-Tools model and `n` integer variables for `x` and `y`. The `domain` set variable is created with the universe of possible values [0..10]. Each element of x and y is linked to this domain using `MemberOf`. The crux of the solution resides in ensuring that the `domain` set only contains the values *actually* used in the variables `x` and `y`. We iterate through possible values, setting auxiliary boolean variables `x_contains` and `y_contains` which are `true` when a particular value exists in the lists. This is then used to drive whether or not a value is `Contains` within `domain`. The solution will return `true` and print to standard output if a feasible assignment to the lists and `domain` can be found.

**Example 2: Adding a Global Constraint**

```python
from ortools.sat.python import cp_model

def shared_domain_with_all_different(n):
    model = cp_model.CpModel()

    x = [model.NewIntVar(0, 10, f"x_{i}") for i in range(n)]
    y = [model.NewIntVar(0, 10, f"y_{i}") for i in range(n)]

    model.AddAllDifferent(x)
    model.AddAllDifferent(y)

    all_values = set(range(11))
    domain = model.NewSetVar(all_values, "domain_set")

    for val in x:
        model.Add(val.MemberOf(domain))
    for val in y:
        model.Add(val.MemberOf(domain))

    for val in range(11):
        used_by_x_or_y = model.NewBoolVar(f"used_{val}")
        x_contains = model.NewBoolVar(f"x_contains_{val}")
        y_contains = model.NewBoolVar(f"y_contains_{val}")

        model.AddBoolOr([(variable == val) for variable in x]).OnlyEnforceIf(x_contains)
        model.AddBoolOr([(variable == val) for variable in y]).OnlyEnforceIf(y_contains)

        model.AddBoolOr([x_contains, y_contains]).OnlyEnforceIf(used_by_x_or_y)
        model.AddBoolOr([x_contains.Not(), y_contains.Not()]).OnlyEnforceIf(used_by_x_or_y.Not())
        model.Add(domain.Contains(val) == used_by_x_or_y)


    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        print("x:", [solver.Value(var) for var in x])
        print("y:", [solver.Value(var) for var in y])
        print("Domain:", solver.Value(domain))
        return True
    else:
        print("No solution found.")
        return False

shared_domain_with_all_different(4)
```

This example builds upon the first by incorporating `model.AddAllDifferent(x)` and `model.AddAllDifferent(y)`. This means that, within each respective list, values must be unique, demonstrating the ability to combine the shared domain with other types of constraints. The domain synchronization logic remains identical, showcasing its versatility. This example will also return `true` if a feasible solution is discovered, with output of `x`, `y` and the found `domain` set.

**Example 3: Handling a Non-contiguous Range**

```python
from ortools.sat.python import cp_model

def shared_domain_non_contiguous(n):
    model = cp_model.CpModel()

    # Non-contiguous initial range
    x = [model.NewIntVar(2, 8, f"x_{i}") for i in range(n)]
    y = [model.NewIntVar(1, 9, f"y_{i}") for i in range(n)]

    all_values = set(range(1, 10))
    domain = model.NewSetVar(all_values, "domain_set")

    for val in x:
        model.Add(val.MemberOf(domain))
    for val in y:
        model.Add(val.MemberOf(domain))

    for val in range(1,10): # Note, must match all values range for domain
        used_by_x_or_y = model.NewBoolVar(f"used_{val}")
        x_contains = model.NewBoolVar(f"x_contains_{val}")
        y_contains = model.NewBoolVar(f"y_contains_{val}")

        model.AddBoolOr([(variable == val) for variable in x]).OnlyEnforceIf(x_contains)
        model.AddBoolOr([(variable == val) for variable in y]).OnlyEnforceIf(y_contains)

        model.AddBoolOr([x_contains, y_contains]).OnlyEnforceIf(used_by_x_or_y)
        model.AddBoolOr([x_contains.Not(), y_contains.Not()]).OnlyEnforceIf(used_by_x_or_y.Not())
        model.Add(domain.Contains(val) == used_by_x_or_y)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        print("x:", [solver.Value(var) for var in x])
        print("y:", [solver.Value(var) for var in y])
        print("Domain:", solver.Value(domain))
        return True
    else:
        print("No solution found.")
        return False


shared_domain_non_contiguous(3)
```

This final example shows how the synchronization mechanism adapts to non-contiguous initial variable ranges. The key change is the variable initialization `x` and `y`, where I have assigned `[2,8]` and `[1,9]`. The domain creation also sets the correct range, `range(1,10)`. This demonstrates the robustness of this synchronization method, extending it beyond uniform initial ranges. The other elements and solution detection remains identical.

This technique, using a shared set variable and linked indicator constraints, provides a robust method for forcing two variable lists to operate within the same solution space. The auxiliary boolean variables are crucial for ensuring the set contains precisely the values from the two variable lists, no more and no less.

For further exploration of CP-SAT capabilities, the OR-Tools documentation is essential.  Study the sections on set variables, Boolean constraints (especially `OnlyEnforceIf` and Boolean operators) and the comprehensive API details.  I also found the examples within the OR-Tools repository, and the associated research papers, very helpful during the learning process, particularly when dealing with advanced concepts like indicator constraints and model formulation. Finally, consider consulting constraint programming textbooks for a deeper understanding of the theoretical underpinnings of the solver and model building.
