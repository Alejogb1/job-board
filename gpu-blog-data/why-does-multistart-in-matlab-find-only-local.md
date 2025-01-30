---
title: "Why does MultiStart in MATLAB find only local minima when fitting nonlinear functions?"
date: "2025-01-30"
id: "why-does-multistart-in-matlab-find-only-local"
---
The inherent limitation of MultiStart in MATLAB's `fmincon` function, and similar multi-start optimization algorithms, to locating only local minima stems directly from the nature of the employed optimization methods.  These methods, typically gradient-based, are susceptible to becoming trapped in local optima within the search space, particularly for nonlinear functions exhibiting complex landscapes with numerous valleys and peaks.  My experience optimizing complex chemical reaction kinetics models has repeatedly highlighted this characteristic. The algorithm's iterative nature, relying on successive improvements from an initial guess, inevitably converges to the nearest minimum, failing to explore the entire solution space exhaustively.

This behaviour is fundamentally distinct from global optimization methods, which explicitly aim to identify the global minimum.  MultiStart, by its design, iteratively performs local searches from various starting points, but each individual search remains a local optimization. While the strategy of multiple starting points increases the probability of finding a better minimum (closer to the global one), it offers no guarantee.  The effectiveness of MultiStart heavily depends on the chosen starting points and the nature of the objective function.  A poorly chosen set of starting points might lead to several instances of the algorithm converging to the same local minimum, providing a false sense of global optimality.  In contrast, a global optimization algorithm uses specific strategies like branch and bound, simulated annealing, or genetic algorithms to systematically explore the entire search space.

The crucial aspect to understand is the distinction between local and global minima. A local minimum is a point where the objective function value is smaller than its immediate neighbours, while a global minimum is the absolute lowest point across the entire feasible region.  For unimodal functions (having only one minimum), MultiStart might be sufficient, but for multimodal functions (with several minima), the probability of finding the global minimum reduces significantly, even with a large number of starting points.  Furthermore, the dimensionality of the problem exacerbates this challenge, exponentially increasing the search space and the likelihood of encountering and converging to less desirable local minima.

Let's examine this behaviour through code examples.  These examples use a simplified Rosenbrock function, known for its challenging non-linearity and numerous local minima.

**Example 1:  Illustrating Local Minimum Trapping**

```matlab
% Define the Rosenbrock function
rosenbrock = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;

% Single starting point near a local minimum
x0 = [1.2; 1.4];
options = optimoptions('fmincon','Display','iter');
[x,fval] = fmincon(rosenbrock,x0,[],[],[],[],[],[],[],options);

disp(['Minimum found at x = [', num2str(x(1)), ', ', num2str(x(2)), ']']);
disp(['Function value: ', num2str(fval)]);
```

This example demonstrates a single run of `fmincon`.  The starting point `x0` is chosen relatively close to a local minimum.  Consequently, the algorithm efficiently converges to this local minimum. Changing the starting point would lead to convergence towards a different local minimum.

**Example 2: MultiStart Attempt**

```matlab
% Define the Rosenbrock function (same as Example 1)
rosenbrock = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;

% Multiple starting points using MultiStart
nStarts = 10;
problem = createOptimProblem('fmincon','objective',rosenbrock,...
    'x0',randn(2,nStarts),'lb',[-5,-5],'ub',[5,5]);
ms = MultiStart('Display','iter');
[xmin,fmin] = run(ms,problem,nStarts);

disp('Minima found:');
disp(xmin);
disp('Function values:');
disp(fmin);
```

This example uses `MultiStart` with 10 random starting points within a specified range. Observe how different starting points lead to the algorithm converging to potentially different local minima.  While a few solutions might be closer to the global minimum ([1;1]), the lack of guarantee of global optimality remains.  Increasing `nStarts` might increase the probability of finding a better minimum, but the computational cost also increases linearly.

**Example 3:  Impact of Starting Point Distribution**

```matlab
% Define the Rosenbrock function (same as Example 1)
rosenbrock = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;

% Strategically selected starting points
x0 = [-1.2, -1.4; 1.2, 1.4; 0, 0; 2, 4; -2, -4];
options = optimoptions('fmincon','Display','iter');
results = cell(size(x0,1),1);

for i = 1:size(x0,1)
    [x,fval] = fmincon(rosenbrock,x0(i,:),[],[],[],[],[],[],[],options);
    results{i} = [x, fval];
end

disp('Results from different starting points:');
disp(cell2mat(results));
```

This example demonstrates the importance of starting point selection.  A more informed choice of starting points, potentially guided by prior knowledge or preliminary analyses of the function, can significantly improve the likelihood of finding minima closer to the global one.  However, such knowledge is not always available, especially for complex, high-dimensional problems.

In summary, while MultiStart improves the chances of finding a better local minimum compared to a single run of `fmincon`, it inherently remains a local optimization strategy.  The algorithm's success relies heavily on the starting points and the problem's characteristics.  For complex, multimodal problems, a global optimization technique is necessary to guarantee finding the global minimum, although these methods often come with increased computational cost and complexities.

For further understanding, I recommend consulting documentation on optimization algorithms, particularly those dealing with global optimization techniques.  A thorough understanding of gradient-based methods and their limitations is essential.   Finally, exploring different optimization toolboxes and their capabilities is beneficial to select the most appropriate method for the problem at hand.
