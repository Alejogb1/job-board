---
title: "How to add constraints in python when optimizing a portfolio?"
date: "2024-12-15"
id: "how-to-add-constraints-in-python-when-optimizing-a-portfolio"
---

ah, portfolio optimization with constraints in python, yeah, i've been down that rabbit hole a few times. it’s one of those things that sounds simple at first but quickly spirals into a dense thicket of linear algebra and numerical methods. so, here's the lowdown, based on my own trials and errors, mostly errors to be honest.

the core issue is that when you’re trying to maximize returns or minimize risk (or some combination of both) for a portfolio, you usually can't just let the optimizer run wild. you need to impose real-world restrictions, like not shorting assets, having a maximum exposure to any given sector, or simply ensuring your portfolio's total weight adds up to one. these restrictions are what we call constraints. without them, the optimizer may give you a ‘perfect’ portfolio that’s utterly impractical, holding negative amounts of some assets or weights exceeding 100 percent. i've had the optimizer suggest things that defy both financial laws and common sense.

personally, my first encounter with this was when i was trying to build an automated trading bot, a while ago. i was using a basic quadratic programming solver, and it kept spitting out portfolios that were heavily leveraged, with some assets having weights way above one. it was like the optimizer was drunk and had complete disregard for trading rules. it dawned on me that adding constraints is not some optional add-on, but the backbone of any useful portfolio optimization.

let’s get to the python part. most often i see people using `scipy.optimize`, which is fantastic. it has tools like `minimize` that can be used to handle constrained optimization problems effectively. now, the most important thing for constraints is to think about them as functions and bounds.

here is an example of a simple constraint: preventing shorting. in terms of code, this translates to non-negative weight bounds on each asset.

```python
import numpy as np
from scipy.optimize import minimize

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights.T, expected_returns)
    portfolio_volatility = np.sqrt(portfolio_variance(weights, cov_matrix))
    return - (portfolio_return - risk_free_rate) / portfolio_volatility


def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate):
    num_assets = len(expected_returns)
    #initial guess
    init_weights = np.ones(num_assets) / num_assets
    #bounds are a tuple of pairs (min, max) for each asset weight
    bounds = tuple((0, 1) for _ in range(num_assets))
    #constraint that sum of weights is one
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    # optimization
    result = minimize(neg_sharpe_ratio, init_weights, args=(expected_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# dummy data
expected_returns = np.array([0.10, 0.12, 0.08, 0.15])
cov_matrix = np.array([[0.005, 0.002, 0.001, 0.003],
                        [0.002, 0.008, 0.001, 0.004],
                        [0.001, 0.001, 0.006, 0.002],
                        [0.003, 0.004, 0.002, 0.010]])
risk_free_rate = 0.02

optimized_weights = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate)
print("optimized weights:", optimized_weights)

```

in the code above, the `bounds` parameter in the `minimize` function ensures that each weight is between zero and one. this is quite a common constraint for long-only portfolios. the constraint that weights sum to one is handled as a dictionary. see that the sum of weights is constrained to be exactly 1.

a common thing i have seen people ask about is how to add linear inequality constraints, so, let's say, you want to limit your exposure to a particular sector, for example, you don’t want your weights on technology to go over a specific value. it can be done.

let's extend the previous example:

```python
import numpy as np
from scipy.optimize import minimize

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights.T, expected_returns)
    portfolio_volatility = np.sqrt(portfolio_variance(weights, cov_matrix))
    return - (portfolio_return - risk_free_rate) / portfolio_volatility


def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate, sector_limit_matrix, sector_max):
    num_assets = len(expected_returns)
    init_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))

    # constraint that sum of weights is one
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    #constraint that max sector exposure is within a limit
    constraints.append({'type': 'ineq', 'fun': lambda weights, sl=sector_limit_matrix, sm=sector_max:
                                               sm - np.dot(sl,weights) })

    result = minimize(neg_sharpe_ratio, init_weights, args=(expected_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


# dummy data
expected_returns = np.array([0.10, 0.12, 0.08, 0.15])
cov_matrix = np.array([[0.005, 0.002, 0.001, 0.003],
                        [0.002, 0.008, 0.001, 0.004],
                        [0.001, 0.001, 0.006, 0.002],
                        [0.003, 0.004, 0.002, 0.010]])

risk_free_rate = 0.02

#sector matrix
sector_limit_matrix = np.array([1,1,0,0]) # sector 1 composed by asset 1 and asset 2

sector_max = 0.5 #max exposure to sector 1 is 50%

optimized_weights = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate, sector_limit_matrix, sector_max)
print("optimized weights:", optimized_weights)
```

in this modified snippet, we have introduced `sector_limit_matrix` which dictates how to calculate exposure to sector 1 and sector_max that dictates the maximum exposure to this sector, `constraints.append` introduces a new constraint that is an inequality one. it now makes sure that the resulting weights for the sector don’t go over this maximum value. this can be extended to multiple sectors, just add them to the sector matrix.

a crucial tip is that the constraint function should return a positive value if it is within the constraint or zero if it is exactly at the edge of the constraint. if the constraint is violated it should return a negative number. this is the way `scipy` understands the constraints. it might sound a little counterintuitive, but it does simplify a lot of things under the hood. remember that the solvers we are using here are gradient based, so smoothness and well-defined behavior of the constraint functions is key, if you have functions with discontinuities or weird behaviors the solver is going to have a hard time and it will most likely not converge, or converge to a suboptimal solution.

now, the `SLSQP` method i used in the examples is just one of several options available in `scipy`. it is quite a solid choice for most situations but depending on the type of your problem and your data, you may need to use other methods such as 'trust-constr', `cobyla` or others. selecting the correct optimization method is quite important to achieve convergence, numerical stability, and solution optimality. i once spent three days figuring out that my problem did not converge because i selected the wrong method, i eventually changed it and it converged instantly, but i spent 3 full days thinking that my problem was fundamentally wrong. just so that `you know` that these things are not straightforward and require experience, there are not a lot of magical bullets for optimization. (here's your one joke, i just felt like i needed to say that. it is really not a trivial thing, and requires a lot of careful thought and testing)

another thing, i have seen that the way you structure your constraint functions is essential. often i see people trying to fit too much logic into one single constraint function, leading to very complicated and confusing code that is also very hard to debug. if you need to include multiple constraints, keep them separate. have a single function for every single constraint, that makes debugging and maintaining the code way easier. trust me. there's always that moment when a constraint is not working as expected, and separating them makes it more tractable.

and here’s another thing, you are gonna have to test your implementation extensively. the optimal portfolio weights change dramatically with the inputs and also with the constraints, if you change one of them or add a new one, the results might be completely different, and counterintuitive in some cases, always check, test, and double-check the results. i personally like to simulate a lot of scenarios and check my results using different parameters. this is especially true for more complex constraint conditions.

finally, if you are like me and you love to go deep into optimization details, some great resources on convex optimization are ‘convex optimization’ by boyd and vandenberghe, and also ‘numerical optimization’ by nocaedal and wright. there is no better way to learn this topic than starting with these resources. both are advanced books, and sometimes may seem overwhelming, but the effort is really worth it. if you want to understand the inner workings of these algorithms these books are essential.

in summary, adding constraints is more than just an afterthought in portfolio optimization. it is a central component to get meaningful and practical results. and while `scipy.optimize` handles most of the mathematical heavy lifting, the way you define the constraint functions, and choose the optimization method, can make or break your optimization process. always, always test your implementation thoroughly.
