---
title: "What are MATLAB optimization stopping criteria?"
date: "2025-01-30"
id: "what-are-matlab-optimization-stopping-criteria"
---
MATLAB's optimization functions offer a suite of stopping criteria, crucial for balancing solution accuracy against computational cost.  My experience working on large-scale parameter estimation problems for petrophysical models highlighted the criticality of appropriately selecting these criteria to avoid premature termination or excessive runtime.  Improper selection can lead to suboptimal solutions or needlessly prolonged computation.  Understanding the interplay between these criteria is paramount for effective optimization.

The stopping criteria generally fall into categories based on the objective function, the change in the solution vector, and iteration limits.  They're often used in combination to ensure a robust and reliable optimization process.  Let's examine these in detail.

**1. Objective Function Criteria:**

These criteria focus on the value of the objective function being minimized or maximized.  The most common are:

* **`FunctionTolerance`:** This specifies the absolute tolerance on the change in the objective function value between iterations.  The algorithm terminates when the absolute difference between the objective function values of successive iterations falls below this tolerance.  For example, if `FunctionTolerance` is set to 1e-6, and the difference between the objective function values in two consecutive iterations is less than 1e-6, the optimization halts.  A smaller value demands higher accuracy but may increase computation time.  This is particularly useful when seeking high precision solutions, even if the parameter vector changes minimally.

* **`ObjectiveLimit`:** This criterion sets a target value for the objective function.  Optimization stops when the objective function value reaches or surpasses this limit. This is beneficial when a specific performance threshold is known beforehand.  For instance, in minimizing error, this could be set to a desired acceptable error level.  Using this criterion exclusively, however, risks premature termination if the algorithm is not converging towards the desired limit.

**2. Solution Vector Criteria:**

These criteria assess the change in the parameter vector (or decision variables) being optimized.

* **`StepTolerance`:** This tolerance focuses on the change in the solution vector between iterations. Optimization ceases when the largest absolute difference between the elements of the solution vectors of two successive iterations falls below this tolerance.  This measures the change in the parameters themselves, independent of the change in the objective function value. This is vital when the objective function is relatively flat in a region, meaning small changes in the solution might not lead to significant improvements in the objective function, yet represent a meaningful change in parameter space.  Care should be taken in setting this value, particularly in highly nonlinear problems where small changes in parameters can lead to significant changes in the objective function.


**3. Iteration Limits:**

These criteria prevent indefinite computation.

* **`MaxIterations`:**  This criterion simply limits the maximum number of iterations the optimization algorithm is allowed to perform. This is a safeguard against infinite loops or excessively long computation times.  A practical strategy involves initially running the optimization with a reasonably high `MaxIterations` value, carefully monitoring the convergence behaviour. Subsequently, the value can be adjusted based on the observed convergence rate for efficiency. It should be noted that using only this criterion can result in a premature termination, especially when the algorithm is close to a potential solution but hasn’t reached the other tolerance criteria.

**Code Examples and Commentary:**

Let me illustrate these concepts with three MATLAB examples using the `fmincon` function, focusing on different optimization problems and emphasizing the selection of stopping criteria.

**Example 1:  Minimizing a simple quadratic function**

```matlab
% Define the objective function
fun = @(x) x(1)^2 + x(2)^2;

% Define the initial point
x0 = [1; 2];

% Set options with multiple stopping criteria
options = optimoptions('fmincon','FunctionTolerance',1e-8,'StepTolerance',1e-6,'MaxIterations',1000,'Display','iter');

% Perform optimization
[x,fval,exitflag,output] = fmincon(fun,x0,[],[],[],[],[],[],[],options);

% Display results
disp(['Minimum found at x = ',num2str(x')]);
disp(['Minimum function value: ',num2str(fval)]);
disp(['Exitflag: ',num2str(exitflag)]);
disp(['Number of iterations: ',num2str(output.iterations)]);
```

This example uses a simple quadratic function.  The `optimoptions` function allows us to set multiple stopping criteria simultaneously: a high precision `FunctionTolerance` coupled with a moderate `StepTolerance`  and a generous `MaxIterations`.  The `Display` option provides detailed iterative output for monitoring convergence.  The `exitflag` and `output` structures provide valuable information on the optimization process' success and characteristics.


**Example 2:  A constrained optimization problem**

```matlab
% Objective function
fun = @(x) (x(1)-2)^2 + (x(2)-1)^2;

% Linear inequality constraint
A = [1, 2];
b = 3;

% Initial point
x0 = [0; 0];

% Options with objective limit and iteration limit
options = optimoptions('fmincon','ObjectiveLimit',0.1,'MaxIterations',500,'Display','iter');

% Optimization
[x,fval,exitflag,output] = fmincon(fun,x0,A,b,[],[],[],[],[],options);

% Display Results
disp(['Solution: ',num2str(x')]);
disp(['Objective function value: ',num2str(fval)]);
disp(['Exitflag: ',num2str(exitflag)]);
```

This example incorporates a linear inequality constraint, illustrating the application of `ObjectiveLimit`.  The optimization terminates when the objective function value reaches 0.1 or the maximum iteration count is reached.  This approach is particularly relevant when a satisfactory solution within a specified error bound is sufficient.  Observing the `exitflag` is crucial to determine if the optimization met the `ObjectiveLimit` or reached the `MaxIterations` limit.


**Example 3:  A problem prone to slow convergence**

```matlab
% Rosenbrock function (known for slow convergence)
fun = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;

% Initial point
x0 = [-1.2; 1];

% Options focusing on StepTolerance and MaxIterations
options = optimoptions('fmincon','StepTolerance',1e-4,'MaxIterations',5000,'Display','iter');

% Optimization
[x,fval,exitflag,output] = fmincon(fun,x0,[],[],[],[],[],[],[],options);

% Display results
disp(['Solution: ',num2str(x')]);
disp(['Objective function value: ',num2str(fval)]);
disp(['Exitflag: ',num2str(exitflag)]);
disp(['Number of iterations: ',num2str(output.iterations)]);

```

The Rosenbrock function is notorious for its slow convergence. This example highlights the importance of `StepTolerance` and a generous `MaxIterations` limit. Using a stringent `FunctionTolerance` might lead to excessive runtime without a significant improvement in solution quality. Prioritizing `StepTolerance` ensures termination when the solution vector changes minimally, even if the objective function still shows slight improvement.


**Resource Recommendations:**

The MATLAB documentation on optimization functions, particularly the descriptions of `optimoptions` and the various solver-specific options, provides comprehensive details on stopping criteria and their usage.  Consult the optimization toolbox documentation for detailed explanations of each algorithm's behavior and recommendations on suitable stopping criterion combinations.  Furthermore, textbooks on numerical optimization offer a theoretical foundation for understanding the significance of convergence criteria.  Finally, review articles comparing different optimization algorithms and their convergence properties can provide valuable insights into effective criterion selection based on problem characteristics.
