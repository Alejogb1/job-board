---
title: "What does 'constrviolation' represent in Matlab's fmincon output?"
date: "2025-01-30"
id: "what-does-constrviolation-represent-in-matlabs-fmincon-output"
---
The `constrviolation` output from MATLAB's `fmincon` function represents the maximum constraint violation at the solution found by the solver.  It's crucial to understand that a non-zero value doesn't necessarily indicate a failed optimization; rather, it quantifies the extent to which the final solution point fails to satisfy all imposed constraints.  This is particularly important when dealing with constrained nonlinear problems where finding a perfectly feasible solution might be computationally intractable or even impossible. In my experience debugging complex simulations involving material parameter estimation, correctly interpreting this value has been paramount in ensuring the validity and reliability of results.

**1.  Clear Explanation:**

`fmincon` employs various algorithms to find a local minimum of a function subject to constraints.  These constraints can be linear equalities or inequalities, nonlinear equalities or inequalities, or bounds on the design variables.  The solver strives to satisfy all constraints; however, depending on the problem's complexity, the algorithm's settings, and the algorithm's inherent limitations (e.g., reaching a maximum iteration count before converging), it might terminate at a point that violates one or more constraints.

The `constrviolation` output provides a single scalar value representing the largest absolute constraint violation among all active constraints at the solution point. This value is calculated by evaluating each constraint function at the solution and taking the maximum of the absolute violations.  A constraint is considered violated if its value is outside the specified range (e.g., a value greater than zero for a less-than-or-equal-to constraint).  A value of zero implies that all constraints are satisfied within the solver's tolerance.  A positive value indicates constraint violations, with a larger value implying a greater degree of violation.

It is essential to distinguish `constrviolation` from the `exitflag`. The `exitflag` indicates the reason for termination, while `constrviolation` provides a metric quantifying the feasibility of the solution regardless of the termination reason.  A successful `exitflag` (e.g., 1, indicating a successful solution) does not necessarily guarantee `constrviolation` will be zero; it might simply mean the solver converged to a point with minimal constraint violation below a specified tolerance. Conversely, a negative `exitflag` (indicating a failure to converge) could still yield a relatively low `constrviolation`, suggesting the solver made progress towards a feasible solution but didn't reach it within the given resources.

Understanding the interplay between `constrviolation`, `exitflag`, and the solver's options (especially tolerances) is vital for properly interpreting `fmincon`'s results and deciding if the solution is acceptable for a specific application. In some cases, a small `constrviolation` might be acceptable, depending on the context and the criticality of the constraints. In other scenarios, even a small violation might be unacceptable, and adjustments to the problem formulation, constraints, or solver settings are required.

**2. Code Examples with Commentary:**


**Example 1:  Simple Bound Constraints**

```matlab
% Objective function
fun = @(x) x(1)^2 + x(2)^2;

% Bounds
lb = [0; 0];
ub = [5; 5];

% Optimization
[x,fval,exitflag,output] = fmincon(fun, [1;1], [], [], [], [], lb, ub);

% Display results
disp(['Solution: x = [', num2str(x(1)), ', ', num2str(x(2)), ']']);
disp(['Objective function value: ', num2str(fval)]);
disp(['Exitflag: ', num2str(exitflag)]);
disp(['Constraint violation: ', num2str(output.constrviolation)]);
```

This example minimizes a simple quadratic function subject to bound constraints.  The output will show the solution, the objective function value, the exitflag, and importantly, the `constrviolation`.  If the solver finds a solution within the bounds, `constrviolation` will be zero.  Any positive value indicates that at least one of the bound constraints is slightly violated (possibly due to numerical tolerances).


**Example 2:  Nonlinear Inequality Constraints**

```matlab
% Objective function
fun = @(x) x(1)^2 + x(2)^2;

% Nonlinear inequality constraint
nonlcon = @(x) x(1)^2 + x(2) - 1;

% Optimization
[x,fval,exitflag,output] = fmincon(fun,[1;1],[],[],[],[],[],[],nonlcon);

% Display results (same as Example 1)
disp(['Solution: x = [', num2str(x(1)), ', ', num2str(x(2)), ']']);
disp(['Objective function value: ', num2str(fval)]);
disp(['Exitflag: ', num2str(exitflag)]);
disp(['Constraint violation: ', num2str(output.constrviolation)]);
```

Here, a nonlinear inequality constraint (`x(1)^2 + x(2) - 1 <= 0`) is added.  The `nonlcon` function defines this constraint.  Again, the `constrviolation` indicates the degree of violation at the solution.  A non-zero value suggests the final solution doesn't perfectly satisfy `x(1)^2 + x(2) <= 1`.


**Example 3:  Handling Constraint Violations –  Increased Tolerance**

```matlab
% Objective function (same as Example 2)
fun = @(x) x(1)^2 + x(2)^2;

% Nonlinear inequality constraint (same as Example 2)
nonlcon = @(x) x(1)^2 + x(2) - 1;

% Options with increased constraint tolerance
options = optimoptions('fmincon','ConstraintTolerance',1e-2);

% Optimization with modified options
[x,fval,exitflag,output] = fmincon(fun,[1;1],[],[],[],[],[],[],nonlcon,options);

% Display results (same as Example 1)
disp(['Solution: x = [', num2str(x(1)), ', ', num2str(x(2)), ']']);
disp(['Objective function value: ', num2str(fval)]);
disp(['Exitflag: ', num2str(exitflag)]);
disp(['Constraint violation: ', num2str(output.constrviolation)]);
```

This example demonstrates how to adjust the solver's tolerance.  By increasing `ConstraintTolerance`, we allow the solver to accept solutions with larger constraint violations. This is useful if the problem is highly nonlinear or computationally expensive, prioritizing a near-optimal solution over perfect constraint satisfaction.  Observe how modifying the tolerance impacts the `constrviolation` value.

**3. Resource Recommendations:**

The MATLAB documentation for `fmincon` is essential. Carefully review the descriptions of the output parameters, options, and algorithms.  Explore the examples provided in the documentation to gain practical experience.  Additionally, consult a standard optimization textbook focusing on nonlinear programming for a more thorough theoretical understanding of constraint satisfaction and optimization algorithms.  Consider working through exercises related to constraint handling and numerical tolerance in optimization.  Finally, review the section on diagnosing convergence issues in the MATLAB optimization toolbox documentation.
