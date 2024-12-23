---
title: "How can portfolio return be maximized subject to a variance constraint using MATLAB?"
date: "2024-12-23"
id: "how-can-portfolio-return-be-maximized-subject-to-a-variance-constraint-using-matlab"
---

Okay, let's unpack this. Maximizing portfolio return while adhering to a specific variance constraint is a classic problem in quantitative finance, and it’s one I’ve encountered multiple times over the years—most recently while optimizing a risk-managed trading strategy back at my previous firm. It's a fairly elegant demonstration of constrained optimization, and MATLAB, with its robust optimization toolbox, is well-suited for handling it.

The core idea revolves around finding the portfolio weights that yield the highest expected return without exceeding a predetermined level of risk, represented by portfolio variance. We aren't just throwing darts here; it’s a structured approach. This isn't about seeking the absolutely highest return irrespective of volatility, but rather finding the optimal balance given your risk tolerance.

Here’s how I've typically approached it, broken into digestible steps.

First, we need the input data: a historical returns series for each asset in our portfolio. From this, we estimate two key parameters: the expected returns vector (let’s call it `mu`) and the covariance matrix (let's denote it as `sigma`). These are the foundations upon which our optimization will be built. A good place to review the theory behind this is Harry Markowitz's original work on Modern Portfolio Theory in the Journal of Finance (1952).

In MATLAB, you'd start by loading or calculating these values. Assuming you have historical returns in a variable called `returns_data` where each column represents a different asset:

```matlab
% Assuming returns_data is your matrix of historical returns
mu = mean(returns_data)'; % Transpose to make it a column vector
sigma = cov(returns_data); % Compute the covariance matrix
```

Next, we set our variance constraint. This value will vary depending on your acceptable risk. Let's say we want to keep the annualised variance of the portfolio below a given level, for example `max_variance`.

Now comes the crucial step: setting up the optimization problem. We'll employ MATLAB's `quadprog` function. `quadprog` is designed to solve quadratic programming problems, which fit our need precisely. Our objective function is to maximize the expected portfolio return, which translates to *minimizing* the negative of the expected return in the `quadprog` context. The variance constraint is formulated as an inequality constraint which must hold. Also, the weights of the portfolio must sum to one, and no asset can have a negative weight.

```matlab
% Define the number of assets
nAssets = length(mu);

% Define parameters for quadprog
H = 2*sigma; % The Hessian (covariance matrix doubled) is required to solve the quadratic optimization
f = -mu; % Negative expected return as quadprog is for minimization

% Define the equality constraint, weights sum to 1
Aeq = ones(1, nAssets);
beq = 1;

% Define inequality constraint due to portfolio variance
Aineq = sigma;
bineq = max_variance;

% Lower and upper bounds for weights
lb = zeros(nAssets, 1);
ub = ones(nAssets, 1);

% Solve the optimization problem using quadprog
options = optimoptions('quadprog','Display','off'); % Turn off display output
[weights, ~, exitflag, output] = quadprog(H,f,Aineq,bineq,Aeq,beq,lb,ub,[],options);

if exitflag == 1
   disp('Optimization Successful');
   disp('Optimized Portfolio Weights:');
   disp(weights);
   portfolioReturn = weights'*mu;
   portfolioVariance = weights'*sigma*weights;
   disp(['Portfolio Expected Return: ',num2str(portfolioReturn)]);
   disp(['Portfolio Variance: ',num2str(portfolioVariance)]);
else
   disp(['Optimization failed, exitflag=',num2str(exitflag)]);
end

```

In this snippet, `H` represents the Hessian matrix (twice the covariance matrix), `f` is the negative expected returns, and we’ve set up equality and inequality constraints using `Aeq`, `beq`, `Aineq`, and `bineq` respectively. `lb` and `ub` are the lower and upper bounds for the weights which in this instance are 0 and 1. We're also checking the `exitflag` value, it’s helpful to diagnose optimization issues. A value of 1 indicates successful convergence.

I've seen projects where we've had much more complex constraints, sometimes requiring the introduction of a `cplex` solver as well, but for the core of your request, `quadprog` tends to do the trick. A useful reference for constraint-solving problems is "Numerical Optimization" by Nocedal and Wright.

It’s important to note that in practice we would be using much more sophisticated input parameters than these, and these parameters would be revised often. The mean and covariance matrix of a financial time series can change. This can be accommodated by rerunning the optimization on an ongoing basis and/or using more sophisticated approaches such as Bayesian estimation to deal with these parameter uncertainty issues.

Finally, let's look at a slightly different variation that maximizes the Sharpe ratio (a measure of risk-adjusted return) instead, while keeping the variance constrained. This is another approach, especially if you are aiming for optimal risk-adjusted returns instead of just returns. To maximise the Sharpe ratio, we need to maximise the ratio of excess returns over standard deviation, a square root transformation is needed on the variance constraint to ensure we are constraining the standard deviation.

```matlab
% Assuming returns_data is your matrix of historical returns
mu = mean(returns_data)'; % Transpose to make it a column vector
sigma = cov(returns_data); % Compute the covariance matrix

% Define the number of assets
nAssets = length(mu);
% Define the risk free rate
rf = 0.02;
% Define parameters for quadprog
H = sigma; % The Hessian matrix (covariance matrix)
f = -(mu - rf); % Negative excess return is the Sharpe Ratio numerator

% Define the equality constraint, weights sum to 1
Aeq = ones(1, nAssets);
beq = 1;

% Define inequality constraint due to portfolio standard deviation
Aineq = sigma;
bineq = sqrt(max_variance);

% Lower and upper bounds for weights
lb = zeros(nAssets, 1);
ub = ones(nAssets, 1);

% Solve the optimization problem using quadprog
options = optimoptions('quadprog','Display','off'); % Turn off display output
[weights, ~, exitflag, output] = quadprog(H,f,Aineq,bineq,Aeq,beq,lb,ub,[],options);

if exitflag == 1
   disp('Optimization Successful');
   disp('Optimized Portfolio Weights:');
   disp(weights);
   portfolioReturn = weights'*mu;
   portfolioVariance = weights'*sigma*weights;
   portfolioStandardDeviation = sqrt(portfolioVariance);
   portfolioSharpeRatio = (portfolioReturn-rf)/portfolioStandardDeviation;
   disp(['Portfolio Expected Return: ',num2str(portfolioReturn)]);
    disp(['Portfolio Variance: ',num2str(portfolioVariance)]);
   disp(['Portfolio Standard Deviation: ',num2str(portfolioStandardDeviation)]);
   disp(['Portfolio Sharpe Ratio: ',num2str(portfolioSharpeRatio)]);

else
   disp(['Optimization failed, exitflag=',num2str(exitflag)]);
end
```

In this variation, we're focusing on excess returns over the risk-free rate. The objective function `f` has changed and the `bineq` parameter is now constraining standard deviation instead of variance. The `H` parameter also changed, as `quadprog` requires the Hessian to be the covariance matrix in the Sharpe ratio optimisation. The Sharpe ratio is a common risk-adjusted measure.

In all these implementations, remember to always validate your results using backtesting or historical simulations. The data you use is crucial for these optimisations. We also need to be aware of the limitations of historical data. What worked well in the past might not be the optimal result for the future.

I’ve found, from my own practical experience, that the real trick is not necessarily just getting the code working, but also to have a good understanding of the fundamental parameters and how these will affect the portfolio. The model is an abstraction of reality. You need to understand its limitations and not treat the output blindly as the truth. Understanding and being familiar with Modern Portfolio Theory is vital to using these optimisations effectively.
