---
title: "How to do a Quadratic constraint with the ROI package?"
date: "2024-12-14"
id: "how-to-do-a-quadratic-constraint-with-the-roi-package"
---

alright, so you're looking to wrangle quadratic constraints within the roi package, huh? i've been there, trust me. roi, for all its strengths, can feel a bit… particular… when it comes to anything beyond linear stuff. i remember back when i first started working with optimization problems, it was all nice and tidy linear objective functions and constraints. then, bam, real-world hits, and you get these wonderfully nonlinear things that the standard solvers just choke on. quadratic constraints are exactly that kind of beast.

so, the short answer, roi, the 'optimization infrastructure' package in r, doesn't directly handle quadratic constraints in its traditional function interfaces like `roi_solve` or `lpsolve`. this is not really surprising, honestly. it's designed for linear programming problems primarily.

but, we're not doomed. there’s always a workaround, usually involving some form of re-expression or using a different solver that does handle quadratic programming (qp) well. i usually prefer to use 'osqp' that is one of the best solvers for quadratic constraints.

let’s start with the core idea. a quadratic constraint essentially has this form:

`x' * q * x + a' * x <= b`

where:

*   `x` is your vector of decision variables.
*   `q` is a symmetric matrix, that forms the quadratic part.
*   `a` is a vector representing the linear part.
*   `b` is a scalar defining the constraint limit.

now, the trick with roi is that it’s structured around a linear, that is a sum of the variables times the weights paradigm. you can’t just plug in this general quadratic form directly. but, and here's the good stuff, we can transform some instances of quadratic constraints into equivalent linear ones, at least approximately. this will work for some limited number of quadratic constraints but the result will be that it will approximate the solution, it wont be exact. for example you can approximate `x^2` with a set of linear constraints, but that will not be a general solution.

let me explain, the most simple case that i have encountered is when we need `x1^2 <= limit` we can represent it with `-limit <= x1 <= limit`, but be careful this approximation will work well only if the value of `limit` is close to zero, otherwise the solution might be completely different.

ok, let’s get to some code snippets that may help. i'll start with the approximation method since i think it's the easiest to grasp and implement, but i need to warn you again this method has strong limitations and works on specific scenarios. i remember one time i was working on a supply chain optimization problem, trying to model the storage capacity nonlinearly. turns out, a linear approximation was good enough for the quick and dirty prototype, but not for the actual solution. this is where i understood the trade-offs.

```r
# Approximating x^2 <= limit with linear bounds
library(roi)
library(ROI.plugin.glpk) # or your preferred linear solver

# Let's say the quadratic constraint is x1^2 <= 9
# So we approximate with -3 <= x1 <= 3

objective <-  L_objective(c(1, 1)) # Simple objective for demo x1 + x2
constraint <-  L_constraint(
  L = matrix(c(1, 0,
               -1, 0,
               0,1,
               0,-1), ncol = 2, byrow = TRUE),
  dir = c("<=", "<=", "<=", "<="),
  rhs = c(10, 10, 10, 10)
)

bounds <-  V_bound(
  lb = c(-3, -Inf),
  ub = c(3,Inf)
)

lp <- OP(objective, constraint, bounds = bounds)
result <- roi_solve(lp, solver = "glpk") # or any linear solver

print(result)
```

in this example, i’m essentially replacing the `x1^2 <= 9` with`-3 <= x1 <= 3`, which can lead to a solution but this won't be accurate at all if `x1` is far away from 0. the objective function is also linear for simplicity. this kind of approximation is good for when you have a small neighborhood where the squared term doesn’t change dramatically, for example when modeling a non-linear response in a chemical process where you have small changes from the operation point. but if you are out of that small region then the approximation would not work well.

now, let's say you have a more complex case that is solvable with 'osqp'. imagine a portfolio optimization, trying to minimize risk, which often involves a covariance matrix that is, you know, quadratic. this is where ‘osqp’ shines because it can handle it.

```r
# Example with a simple quadratic constraint using 'osqp'

library(ROI)
library(ROI.plugin.osqp)

# Objective: minimize x1^2 + x2^2
objective <- Q_objective(
  Q = matrix(c(2,0,0,2), nrow = 2, byrow = TRUE),
  L = c(0, 0)
)

# Constraint: x1 + x2 <= 1
constraint <- L_constraint(
    L = matrix(c(1,1), ncol = 2),
    dir = c("<="),
    rhs = 1
)


# Bound: x1 >= 0, x2 >= 0
bounds <- V_bound(lb = c(0, 0), ub = c(Inf, Inf))


qp_problem <- OP(objective, constraint, bounds = bounds)

result <- roi_solve(qp_problem, solver = "osqp")

print(result)
```

this second snippet is using `q_objective`, you specify your quadratic coefficients in `q`. the `l` part in `q_objective` is still there, in this case with `c(0,0)`, because it might be the case that your objective has a linear term too. The solver that we use is `osqp`, which you will need to install and install the `roi.plugin.osqp`. this code is a simple example of how to use 'osqp' with `roi` and quadratic objectives. for this to work you need your objective function to be convex, in other words your q matrix should be positive semi-definite, otherwise it will not work and the solver would be unable to find a solution.

now a final code snippet, just to add more possibilities. you can also mix quadratic objectives with linear constraints. i remember when i was trying to solve a problem of power consumption in a microgrid and some variables were quadratic and other linear. this can be very usual.

```r
# Example with a quadratic objective and linear constraint

library(ROI)
library(ROI.plugin.osqp)

# Objective: minimize x1^2 + x2
objective <- Q_objective(
  Q = matrix(c(2,0,0,0), nrow = 2, byrow = TRUE),
  L = c(0, 1)
)

# Constraint: 2*x1 + x2 <= 4
constraint <- L_constraint(
  L = matrix(c(2, 1), ncol = 2),
  dir = c("<="),
  rhs = 4
)

# Bounds: x1 >= 0, x2 >= 0
bounds <- V_bound(lb = c(0, 0), ub = c(Inf, Inf))


qp_problem <- OP(objective, constraint, bounds = bounds)

result <- roi_solve(qp_problem, solver = "osqp")

print(result)
```

in this third example we have a mix of quadratic `x1^2` and linear `x2` in our objective function. the solver `osqp` can handle this without any problem. one thing you will notice, when `q=0` you might expect it to fall back to using the linear solvers in roi, but it does not, and it will still be solved as a quadratic problem, so that is good to keep in mind.

in terms of resources, i highly recommend the book "convex optimization" by stephen boyd and lieven vandenberghe. it’s basically the bible for understanding the theory behind all of this. if you are looking for something less theoretical, "numerical optimization" by jorge nocedal and stephen j. wright is a great book focusing on algorithms, and the implementation of the solvers. and of course, the documentation for 'osqp' is very good as well, the paper on it is “osqp: an operator splitting solver for quadratic programs” by stellato et al. (2020), this one is more advanced for research purposes.

one last piece of advice, always visualize your problems, it will help you to see if the results are correct and if your formulations are right. it can be done in `r` or any other software that you might use. one time i had a problem with a formulation that i was convinced was correct, but when visualizing the constraint area i saw that i had a typo, it took me two hours to figure it out, so save the time and plot your constraints.

i think this should be a good start. remember the devil is in the details. oh, one final joke, why did the programmer quit his job? because he didn't get arrays! (ba-dum-tss). good luck, and let me know if you have more questions.
