---
title: "How can infeasible MILP solutions be addressed when using Gurobi for median regression estimation?"
date: "2025-01-30"
id: "how-can-infeasible-milp-solutions-be-addressed-when"
---
Median regression, formulated as a Mixed Integer Linear Program (MILP), often encounters infeasibility, particularly when dealing with noisy or highly variable datasets. This occurs primarily because the strict constraints defining the median fitting process, especially those involving absolute deviation, can become contradictory in the face of outliers or inconsistent data patterns. I've seen this quite often while developing statistical models for dynamic pricing algorithms, where real-time data can exhibit considerable unpredictability.

The core of the infeasibility problem arises from the nature of the MILP formulation itself. Typically, median regression using absolute deviation loss involves introducing auxiliary variables to represent the positive and negative deviations of predicted values from the actual values. Let *y<sub>i</sub>* represent the observed response for the *i*-th data point, and *x<sub>i</sub>* represent its corresponding feature vector. Our goal is to find a coefficient vector *β* such that we minimize the sum of the absolute differences between *y<sub>i</sub>* and *x<sub>i</sub><sup>T</sup>β*. To do this within a linear framework, we define positive deviation *p<sub>i</sub>* and negative deviation *n<sub>i</sub>*, both non-negative, such that *y<sub>i</sub>* - *x<sub>i</sub><sup>T</sup>β* = *p<sub>i</sub>* - *n<sub>i</sub>*. The objective then becomes minimizing the sum of *p<sub>i</sub>* + *n<sub>i</sub>*, subject to the constraint that *y<sub>i</sub>* - *x<sub>i</sub><sup>T</sup>β* = *p<sub>i</sub>* - *n<sub>i</sub>*. Introducing a binary variable *z<sub>i</sub>* which equals one when the residual is positive and zero when negative allows us to model the absolute value. Infeasible solutions appear when this system of linear constraints, coupled with the integer requirements on *z<sub>i</sub>*, has no solution space that satisfies all requirements simultaneously. This is generally due to the constraints of the problem.

When encountering infeasibility with Gurobi, the initial step should be to thoroughly examine the model formulation. One can begin by simplifying the dataset for testing purposes. Reducing the number of data points can isolate any issues related to the specific characteristics of the full dataset, and reducing the number of coefficients can highlight multicollinearity. Checking for constraint consistency is crucial – a common mistake is inadvertently creating inconsistent conditions, such as requiring both a positive and negative deviation for the same data point to be simultaneously zero or nonzero. Careful review of the constraints is the cornerstone to addressing any infeasibility issues.

Several strategies can be employed to address the infeasibility when working with Gurobi. One approach is to introduce *elastic constraints*. Instead of requiring the equality *y<sub>i</sub>* - *x<sub>i</sub><sup>T</sup>β* = *p<sub>i</sub>* - *n<sub>i</sub>* to hold strictly, one can introduce a slack variable, say *s<sub>i</sub>*, that allows for a small deviation. The constraint then becomes *y<sub>i</sub>* - *x<sub>i</sub><sup>T</sup>β* = *p<sub>i</sub>* - *n<sub>i</sub>* + *s<sub>i</sub>*. Adding a penalty term to the objective function, proportional to the sum of the absolute values of *s<sub>i</sub>*, ensures that the deviation is minimized. These slack variables should be non-negative. The coefficient of these slack variables in the objective function acts as a penalty, and must be chosen appropriately based on the size of the data and scale of the variables. I have found that the value used for this penalty must be tested and tuned.

Another effective technique is to reformulate the problem by applying constraint relaxation. The key concept is to identify and remove the most stringent constraints that likely contribute to infeasibility. If the original model requires a strict inequality for the residual, relaxing this to an approximate one can allow the model to find a feasible region. This needs to be done methodically, rather than arbitrarily changing constraints.

A third approach is to employ a different modeling method. Instead of requiring the median residual, consider using a linear regression with Huber loss. Huber loss provides a hybrid approach to minimization that limits the impact of outliers, and using a linear programming formulation with this loss function instead of absolute value can eliminate some of the inherent strictness of the median model. This is a shift in the statistical methodology, but can often address the infeasibility issue while still arriving at a robust result.

The following code examples demonstrate these approaches, with commentary.

**Example 1: Elastic Constraints**

```python
import gurobipy as gp
import numpy as np

def median_regression_elastic(X, y, penalty_factor=0.1):
    """
    Implements median regression with elastic constraints to address infeasibility.
    """
    n, p = X.shape  # n is number of datapoints, p is the number of features
    with gp.Env(empty=True) as env:
      env.start()
      model = gp.Model("median_regression", env=env)

      beta = model.addVars(p, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="beta")
      p_dev = model.addVars(n, lb=0, vtype=gp.GRB.CONTINUOUS, name="p")
      n_dev = model.addVars(n, lb=0, vtype=gp.GRB.CONTINUOUS, name="n")
      s = model.addVars(n, lb=0, vtype=gp.GRB.CONTINUOUS, name="s")


      for i in range(n):
        model.addConstr(y[i] - gp.quicksum(X[i,j] * beta[j] for j in range(p)) == p_dev[i] - n_dev[i] + s[i])


      obj = gp.quicksum(p_dev[i] + n_dev[i] + penalty_factor * s[i] for i in range(n))
      model.setObjective(obj, gp.GRB.MINIMIZE)

      model.optimize()

      if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
        return [v.x for v in beta.values()], model.objVal
      else:
        print("Model Infeasible")
        return None, None


# Example Usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 12]) # a slightly noisy example

beta_elastic, obj_elastic = median_regression_elastic(X, y, penalty_factor=0.1)
if beta_elastic is not None:
    print("Elastic Constraint Coefficients:", beta_elastic)
    print("Objective value:", obj_elastic)
```

Here, slack variables `s` are introduced to allow for slight deviations in the equality constraint, with a penalty controlled by the `penalty_factor`. This often allows the model to find a feasible, albeit slightly adjusted, solution.

**Example 2: Huber Loss Minimization**

```python
import gurobipy as gp
import numpy as np

def huber_regression(X, y, delta=1.0):
    """
    Implements Huber regression with a linear program to avoid strict equality constraint.
    """
    n, p = X.shape
    with gp.Env(empty=True) as env:
      env.start()
      model = gp.Model("huber_regression", env=env)

      beta = model.addVars(p, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="beta")
      loss_vars = model.addVars(n, lb=0, vtype=gp.GRB.CONTINUOUS, name="loss")

      for i in range(n):
        residual = y[i] - gp.quicksum(X[i,j] * beta[j] for j in range(p))
        model.addConstr(loss_vars[i] >= residual)
        model.addConstr(loss_vars[i] >= -residual)
        model.addConstr(loss_vars[i] >= delta * gp.abs_(residual) - (delta ** 2) / 2)

      obj = gp.quicksum(loss_vars[i] for i in range(n))
      model.setObjective(obj, gp.GRB.MINIMIZE)

      model.optimize()

      if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
        return [v.x for v in beta.values()], model.objVal
      else:
        print("Model Infeasible")
        return None, None



# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 12])

beta_huber, obj_huber = huber_regression(X, y, delta=1.0)

if beta_huber is not None:
    print("Huber Regression Coefficients:", beta_huber)
    print("Objective value:", obj_huber)
```

This example implements Huber loss, which is less sensitive to outliers than the absolute deviation loss used in standard median regression, and avoids the integer variables that often cause infeasibility.

**Example 3: Simplified Dataset Test**

```python
import gurobipy as gp
import numpy as np

def median_regression_simplified_test(X, y):
    """
    Implements standard median regression with a reduced dataset to test the problem
    """
    n, p = X.shape  # n is number of datapoints, p is the number of features
    with gp.Env(empty=True) as env:
        env.start()
        model = gp.Model("median_regression", env=env)

        beta = model.addVars(p, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="beta")
        p_dev = model.addVars(n, lb=0, vtype=gp.GRB.CONTINUOUS, name="p")
        n_dev = model.addVars(n, lb=0, vtype=gp.GRB.CONTINUOUS, name="n")

        for i in range(n):
          model.addConstr(y[i] - gp.quicksum(X[i,j] * beta[j] for j in range(p)) == p_dev[i] - n_dev[i])

        obj = gp.quicksum(p_dev[i] + n_dev[i] for i in range(n))
        model.setObjective(obj, gp.GRB.MINIMIZE)

        model.optimize()

        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
          return [v.x for v in beta.values()], model.objVal
        else:
          print("Model Infeasible")
          return None, None



# Example Usage
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

beta_simplified, obj_simplified = median_regression_simplified_test(X, y)
if beta_simplified is not None:
    print("Simplified Data Coefficients:", beta_simplified)
    print("Objective value:", obj_simplified)
```

This example shows the core structure of the MILP formulation with a much simplified data set. Comparing the results to the elastic and huber approaches can highlight whether the full problem is fundamentally flawed, or if there are simply constraints which are not being satisfied.

For further understanding and practical application, consult resources on mathematical programming, specifically focusing on the application of optimization techniques for statistical modeling. Textbooks covering linear and integer programming, particularly those with a practical focus, can be quite valuable. There are also numerous courses available online and in academic institutions that delve into optimization methods using Gurobi. Additionally, statistical modeling books dedicated to robust regression techniques are useful to understand when to best use each modeling strategy. Studying the documentation provided by Gurobi will also help in understanding how the solver is used, and how to diagnose any issues with model setup.
