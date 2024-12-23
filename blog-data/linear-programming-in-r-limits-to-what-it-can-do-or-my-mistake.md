---
title: "linear programming in r limits to what it can do or my mistake?"
date: "2024-12-13"
id: "linear-programming-in-r-limits-to-what-it-can-do-or-my-mistake"
---

let's tackle this linear programming in R thing I've been there done that a few times and I can tell you it's usually not R that's the problem it's usually us coding or our model being slightly off

So the question hints at linear programming limitations in R or user error well I'll tell you first hand back in my early days trying to optimize some supply chain stuff I swear I spent days debugging and scratching my head only to find out it was a constraint I forgot about a misplaced plus sign things that are easy to oversee and make the solver throw weird results or fail outright I remember the time I was trying to minimize transportation cost using the `lpSolve` package and the solver would just return nonsense it turned out my cost matrix was inverted I mean who does that well I did that newbie mistake

The standard packages for linear programming in R are solid `lpSolve` which is a classic and `ROI` the R Optimization Infrastructure which can handle a variety of solvers. `ROI` can interface with a lot of different solvers including `glpk` `lpsolve` and `cplex` etc..

First thing when you're getting unexpected results from a linear program is to double check your input specifically:

*   **Objective Function:** Are you maximizing or minimizing correctly? Is the objective function mathematically right? I've seen people use min when they want max I was almost guilty of this myself I admit it
*   **Constraints:** This is where it mostly falls apart are all your constraints included? Are they in the correct direction inequality? Did you define both lower and upper bounds when needed?
*   **Input Data:** Is your coefficient matrix and vector correct? I'm speaking from experience here data entry errors are your kryptonite. For example the order of the variable coefficients in the objective function vector and the columns of your constraint matrix must match.
*   **Solver specific stuff:** each solver has its own quirks like tolerance levels or if it's an integer program there might be limits on the number of variables or constraints in the trial versions sometimes.

Here is a simple example using `lpSolve`:

```r
library(lpSolve)

# Objective function coefficients
objective_coef <- c(3, 2)

# Constraint matrix
constraint_matrix <- matrix(c(1, 1,
                             2, 1,
                             0, 1), nrow = 3, byrow = TRUE)

# Constraint directions
constraint_dir <- c("<=", "<=", "<=")

# Constraint right-hand side values
constraint_rhs <- c(10, 16, 8)

# Solve the linear program
lp_solution <- lp(objective = objective_coef,
                 direction = "max",
                 const.mat = constraint_matrix,
                 const.dir = constraint_dir,
                 const.rhs = constraint_rhs)

# Check results
print(paste("Objective Value:", lp_solution$objval))
print(paste("Solution:", paste(lp_solution$solution, collapse = ", ")))
```

This example covers a simple linear program with two decision variables and three constraints if you see that your result is not what you are expecting you have to start debugging and go back to the details of what you asked

Now let's assume your model is  and you are hitting a limitation R itself doesn't have a specific limit on linear programs. But what you might be running into is:

*   **Solver limits:** The underlying solvers have limits especially for the free versions. Large-scale problems can push the limits of free solvers like `glpk`. If you have thousands of variables or constraints a commercial solver like `cplex` or `gurobi` via `ROI` might be necessary and they have very good performance on complex issues and they are also usually very well coded with their own limitations and stuff but generally you can push the number of variables really high using those

*   **Memory:** Your computer's RAM can also be a bottleneck especially for large sparse matrices if the number of variables is in the order of several thousands you might face this. R itself is not really a memory-optimized environment and if you have a very large problem the data structures to support the matrix will take some RAM and the solver calculations will also consume RAM.

*   **Numerical Issues:** Sometimes the problem is mathematically ill-conditioned meaning your constraint matrix might be nearly singular which leads to numerical instability when the solver computes something or round off errors for the numbers leading to incorrect solutions. This happens with data of extremely high differences in terms of value the solver might not be able to resolve if one value is near to zero and another one is at the order of 100000000 for example this causes issues.

Here's an example using `ROI` this time which shows you how it works and can be used with different solvers.

```r
library(ROI)
library(ROI.plugin.glpk)

# Objective function coefficients
objective_coef <- c(3, 2)

# Constraint matrix
constraint_matrix <- matrix(c(1, 1,
                             2, 1,
                             0, 1), nrow = 3, byrow = TRUE)

# Constraint directions
constraint_dir <- c("<=", "<=", "<=")

# Constraint right-hand side values
constraint_rhs <- c(10, 16, 8)

# Create optimization problem object
lp_problem <- OP(objective = objective_coef,
                constraints = L_constraint(constraint_matrix,
                                          constraint_dir,
                                          constraint_rhs),
                maximum = TRUE)

# Solve the linear program
roi_solution <- solve(lp_problem, solver = "glpk")

# Check the results
print(paste("Objective Value:", roi_solution$objval))
print(paste("Solution:", paste(roi_solution$solution, collapse = ", ")))
```

If you still are hitting a brick wall and you tried the debug suggestions from above here are some points to verify

*   **Problem Formulation:** Are you sure your linear program is mathematically sound? Maybe you are missing some constraints? Is this a linear program or are you trying to solve a non linear problem using a linear program solver? if that is the case you're doomed you are solving the wrong thing from scratch. This is the biggest time waste I have seen a lot of my past colleagues doing this.
*   **Try other Solvers:** Sometimes switching to a different solver can give you a better numerical solution or work around a particular solver's quirks using `ROI` that is easy try `cplex` or `gurobi` or any other available solver.
*   **Scale your data:** If you have very large numbers in the coefficient matrix or the right hand side vector try scaling them so they are in the same order of magnitude. This can improve the solver precision. And also this can help avoid the numerical instability problem.
*   **Check your variables ranges**: It is very easy to make errors in specifying your variables ranges the ranges that are mathematically possible might not be the ranges you wanted so it is always better to check if the solution makes sense in the context of the problem you are trying to solve.

One time I had a model with a cost variable which was supposed to be the price per unit times the quantity of units bought and my constraint of the spending was lower than the quantity times the price which was the cost I was trying to optimize this made my model not feasible of course It was one of those long coding days it was friday evening when i realized my mistake (I should have checked earlier right) and I just decided to call it a day and fix it on monday

And just for some random fun fact linear programming is so important that you will be able to find it used in any area of supply chain for sure its everywhere and there are actually people who are paid just to know this and they are called operations research analysts. It is the same as in a movie with the nerd working in an excel sheet but in real life there are very capable people behind these operations. Ok that was my random joke.

Now one last example for an integer linear program using `lpSolve`:

```r
library(lpSolve)

# Objective function coefficients
objective_coef <- c(3, 2)

# Constraint matrix
constraint_matrix <- matrix(c(1, 1,
                             2, 1,
                             0, 1), nrow = 3, byrow = TRUE)

# Constraint directions
constraint_dir <- c("<=", "<=", "<=")

# Constraint right-hand side values
constraint_rhs <- c(10, 16, 8)

# Solve the integer linear program
lp_solution <- lp(objective = objective_coef,
                 direction = "max",
                 const.mat = constraint_matrix,
                 const.dir = constraint_dir,
                 const.rhs = constraint_rhs,
                 int.vec = 1:2)

# Check the results
print(paste("Objective Value:", lp_solution$objval))
print(paste("Solution:", paste(lp_solution$solution, collapse = ", ")))
```

Notice the `int.vec` parameter that makes the solution integers

For more in depth understanding you should check some books on linear programming or operations research specifically "Introduction to Operations Research" by Hillier and Lieberman it's a classic or if you want more code focused you can check "Linear Programming and Network Flows" by Bazaraa Jarvis and Sherali or you can delve into the specifics of the specific solver that you are using by looking into their documentation they might have some examples for what you are trying to do

Linear programming is a powerful tool and R gives you access to it but as it is for everything details are key in order to get results. So it is not an R problem it is most likely the problem that you created and that is absolutely  we all have been there
