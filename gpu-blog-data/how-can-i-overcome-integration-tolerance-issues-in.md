---
title: "How can I overcome integration tolerance issues in fmincon for ODE optimization?"
date: "2025-01-30"
id: "how-can-i-overcome-integration-tolerance-issues-in"
---
The core difficulty in using `fmincon` for optimal control problems involving ordinary differential equations (ODEs) often stems from the inherent sensitivity of the numerical ODE solver to the control parameters.  Minor adjustments in the control vector can lead to significant changes in the ODE solution, resulting in an erratic objective function landscape that frustrates `fmincon`'s gradient-based optimization. This manifests as slow convergence, premature termination, or simply failure to find a satisfactory solution, despite the problem's theoretical solvability.  My experience in high-fidelity aircraft trajectory optimization has consistently highlighted this challenge, particularly when dealing with stiff ODE systems.

**1.  Addressing Integration Tolerance Issues:**

The integration tolerance, controlled through options like `RelTol` and `AbsTol` within MATLAB's ODE solvers (e.g., `ode45`, `ode15s`), directly impacts the accuracy of the ODE solution.  A tolerance that is too loose can introduce significant numerical error, leading to an inaccurate evaluation of the objective function and its gradients, deceiving `fmincon`.  Conversely, an overly stringent tolerance increases computational cost without necessarily improving the optimization result, potentially causing `fmincon` to fail due to excessive function evaluation time. The optimal balance requires careful consideration and iterative experimentation.

The first crucial step is separating the ODE integration from the optimization process.  Instead of embedding the ODE solver directly within the objective function, I strongly advocate for a modular approach.  This allows independent control over the integration parameters and facilitates debugging.  Furthermore, utilizing a more robust ODE solver appropriate for the specific stiffness characteristics of your system is paramount.  For stiff ODEs, `ode15s` or `ode23s` generally outperform `ode45`.

The second key aspect involves careful gradient calculation.  `fmincon` often requires gradient information for efficient optimization.  While finite differencing is available, it is computationally expensive and prone to error.  A far superior approach is to utilize the adjoint sensitivity method or similar techniques that analytically or semi-analytically compute the gradient. This reduces computational overhead and increases the robustness of the optimization process.  My experience with complex aerodynamic models has shown a significant improvement in convergence speed and solution quality when employing adjoint sensitivity methods.

Finally, scaling of variables is critical.  If the scales of your state variables, control variables, and objective function differ significantly, it can negatively influence `fmincon`'s performance.  Consider normalizing your variables to a similar range (e.g., -1 to 1) to improve numerical stability and gradient calculation accuracy.


**2. Code Examples and Commentary:**

**Example 1: Modular Approach with `ode45`:**

```matlab
function [obj, grad] = objectiveFunction(u)
  % Define ODE system
  [t, x] = ode45(@(t,x) odefun(t,x,u), tspan, x0, odeset('RelTol', 1e-6, 'AbsTol', 1e-8));

  % Evaluate objective function
  obj = calculateObjective(t, x);

  % Calculate gradient (using adjoint method or finite differences)
  grad = calculateGradient(t, x, u);
end

function dxdt = odefun(t, x, u)
  % Define your ODE system here
  % ...
end

function obj = calculateObjective(t, x)
  % Calculate the objective function value
  % ...
end

function grad = calculateGradient(t, x, u)
  % Calculate the gradient using adjoint sensitivity or finite differences
  % ...
end

% Optimization using fmincon
options = optimoptions('fmincon', 'Display', 'iter', 'GradObj', 'on');
[uOpt, fval] = fmincon(@objectiveFunction, u0, [], [], [], [], lb, ub, [], options);
```

This example showcases the modular design, separating ODE integration, objective function calculation, and gradient computation into distinct functions.  The `odeset` function allows explicit control over integration tolerances.  The use of `'GradObj', 'on'` informs `fmincon` that the gradient is provided.


**Example 2:  Utilizing `ode15s` for Stiff Systems:**

```matlab
function [obj, grad] = objectiveFunctionStiff(u)
  % Define ODE system
  options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'BDF', 'on'); % Use BDF for stiffness
  [t, x] = ode15s(@(t,x) odefunStiff(t,x,u), tspan, x0, options);

  % Evaluate objective function and gradient (as in Example 1)
  obj = calculateObjectiveStiff(t, x);
  grad = calculateGradientStiff(t, x, u);
end

function dxdt = odefunStiff(t, x, u)
    % Define stiff ODE system here
    % ...
end

function obj = calculateObjectiveStiff(t, x)
  % Calculate the objective function value for stiff system
  % ...
end

function grad = calculateGradientStiff(t, x, u)
  % Calculate gradient using adjoint method or finite differences for stiff system
  % ...
end

% Optimization using fmincon
options = optimoptions('fmincon', 'Display', 'iter', 'GradObj', 'on');
[uOpt, fval] = fmincon(@objectiveFunctionStiff, u0, [], [], [], [], lb, ub, [], options);
```

This example demonstrates the use of `ode15s`, a backward differentiation formula (BDF) solver particularly well-suited for stiff ODE systems. The `'BDF', 'on'` option explicitly selects the BDF method.  Tighter tolerances might be necessary for stiff problems.


**Example 3:  Variable Scaling:**

```matlab
function [obj, grad] = objectiveFunctionScaled(uScaled)
  % Unscale control variables
  u = unscaleVariables(uScaled);

  % ODE integration and objective/gradient calculation (as in previous examples)
  [t, x] = ode45(@(t,x) odefun(t,x,u), tspan, x0, odeset('RelTol', 1e-6, 'AbsTol', 1e-8));
  obj = calculateObjective(t, x);
  grad = calculateGradient(t, x, u);

  % Scale gradient back to scaled variable space
  grad = scaleGradient(grad);
end

function u = unscaleVariables(uScaled)
  % Unscale the variables (e.g., linear scaling, normalization)
  % ...
end

function gradScaled = scaleGradient(grad)
  % Scale the gradient back to scaled variable space
  % ...
end

%Optimization
options = optimoptions('fmincon','Display','iter','GradObj','on');
[uScaledOpt, fval] = fmincon(@objectiveFunctionScaled, u0Scaled, [], [], [], [], lbScaled, ubScaled, [], options);
```

This example incorporates variable scaling.  `unscaleVariables` and `scaleGradient` perform the necessary transformations, ensuring that `fmincon` operates on a well-conditioned problem.  Appropriate scaling methods depend on the problem’s specific characteristics.  Note that `u0`, `lb`, and `ub` would need to be similarly scaled.


**3. Resource Recommendations:**

"Numerical Optimization" by Jorge Nocedal and Stephen J. Wright.
"Solving Ordinary Differential Equations I: Nonstiff Problems" by Hairer, Norsett, and Wanner.
"Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems" by Hairer and Wanner.
MATLAB documentation on `fmincon` and ODE solvers.  Consult relevant textbooks on optimal control theory.


By diligently applying these strategies—modular code structure, appropriate ODE solver selection, accurate gradient calculation, and careful variable scaling—one can significantly improve the robustness and efficiency of `fmincon` in solving optimization problems involving ODEs.  Remember that iterative refinement of integration tolerances and optimization parameters is often necessary to achieve optimal performance for a specific problem.
