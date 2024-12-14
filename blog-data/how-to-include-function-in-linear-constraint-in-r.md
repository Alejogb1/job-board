---
title: "How to Include function in linear constraint in r?"
date: "2024-12-14"
id: "how-to-include-function-in-linear-constraint-in-r"
---

alright, so you're looking to incorporate a function's output directly within a linear constraint when doing linear programming or optimization in r. i get it, i've been there, and it's not always immediately obvious how to structure things. this isn't just about plugging any old function in; we have to be mindful of the linearity we're supposed to maintain in these kinds of problems.

first, let's unpack what makes this tricky. standard linear programming solvers expect constraints in the form of inequalities or equalities involving linear combinations of variables. we can usually represent these as matrices and vectors. for example a constraint `2*x1 + 3*x2 <= 10` is a linear constraint. but if we introduce a function that calculates the output say `f(x1)`, and we have the constraint `f(x1) + 3*x2 <= 10` is now not linear if f(x1) itself is non linear.

the usual solvers, like those accessed by the `lpSolve` or `Rsymphony` packages, aren't built to handle arbitrary functions directly. think of it like this: they are very good at managing numbers and relationships like multiplication and addition between variables but not calculating them via functions, those calculations have to be done beforehand and results of these function can be used in the linear constraint. they work by repeatedly updating the values of the x vector in an iterative way and cannot update the function values as they do not operate as a function but as a set of matrix multiplications. so we will need to get the output from a function before we can do linear programming.

the core issue is how do we get the output of the function and introduce this result into a linear constraint in such a way so that we can use standard solvers. there's a couple of ways we can approach this, depending on the specific nature of your function and the precision we are looking for.

one method i’ve used frequently, and it's often the most practical approach, is to approximate the function's output through piecewise linear interpolation. we evaluate our function at several points and create a linear approximation based on those points. in fact many functions we see daily are handled by approximations. for instance the trigonometric functions, like sine and cosine are typically approximated using taylor series or other similar approximation techniques that are linear approximations and use several terms. so we have seen that approximations are extremely useful for non linear calculations.

let's imagine i’m optimizing a supply chain, trying to minimize total costs. one part of my cost was a weird function relating distance traveled by the truck and how much fuel was used `fuel_consumption(distance)` . i initially tried plugging the distance as a variable in a linear program and the function gave me the fuel consumed. this did not work, because `fuel_consumption` was a complex, non-linear function with multiple curves, representing speed and terrain. that’s when i started interpolating the curve.

here’s how we might apply piecewise linear approximation to your situation in r.

```r
# example of a function we wish to incorporate into linear constraint
my_function <- function(x) {
  return(sin(x) * x^2) #non linear function
}

# range of x and the number of segments
x_values <- seq(0, 5, length.out = 10) 
y_values <- sapply(x_values, my_function) #output for x values

# create an approximation to the non linear function
approx_function <- approxfun(x_values, y_values, method = "linear")

# lets see how well it approximates at different values
test_x <- c(1.5, 2.7, 4.1)
test_results_original <- sapply(test_x, my_function)
test_results_approx <- sapply(test_x, approx_function)
print("the original function results at different x values")
print(test_results_original)
print("the results for approximated values for the same x")
print(test_results_approx)

#now we use it with lpSolve
library(lpSolve)
# define our linear optimization problem
objective_coefficients <- c(1, 1) # example coefficients
constraint_matrix <- matrix(c(1, 2, 
                            2, 1), nrow = 2, byrow = TRUE)

constraint_rhs <- c(10, 10) # example constraints
constraint_direction <- c("<=", "<=")
#we are including our constraint via the approx_function in linear form
constraint_matrix_new <- rbind(constraint_matrix, c(approx_function(1.5) , 3) )
constraint_rhs_new <- c(constraint_rhs, 20)
constraint_direction_new <- c(constraint_direction, "<=")


#solve the linear problem
optimization_result <- lp("min", objective_coefficients, constraint_matrix_new, constraint_direction_new, constraint_rhs_new)
# display results
print("optimization result with approximated linear function:")
print(optimization_result)
print(paste("optimal values for x",optimization_result$solution ))
```

in the code, we’ve got a dummy function `my_function` (i’ve used a sin function just to demonstrate non-linearity), evaluated it for a range of values and then used `approxfun` to create a linear approximation function, and used in another linear constraint. this `approx_function` now gives us the approximate output values that is linear with respect to `x` in its approximation. the more points we sample the better the approximation. the resulting approximation can now be included in our linear problem as a constraint. as it is a linear approximation it can be used with standard lp solvers.

another, and very common approach that i used at my previous job doing power load optimization was to use binary variables to select between different regions of the function. you might think of this like a step-wise approximation. if the function has an abrupt change somewhere or if you have a special region where you are interested this is very useful. suppose your function jumps from 0 to 100 abruptly at some value and you need your optimization to account for this jump. one way is to introduce a binary variable that switches between the two function values. let us assume the `my_function` has the following behavior. the value is 0 when x < 2, and 100 when x >= 2. lets see this implemented in code

```r
# the function with abrupt change at x=2
my_function_step <- function(x) {
  if (x < 2) {
    return(0)
  } else {
    return(100)
  }
}
# range of values for binary variable selection
x_values <- seq(0, 5, length.out = 10)
y_values <- sapply(x_values, my_function_step)

# lets use it with lpSolve

library(lpSolve)

# objective coefficients
objective_coefficients <- c(1, 1, 0) # added a coefficient for binary variable z

# constraint matrix (two normal constraints, and one for function output)
constraint_matrix <- matrix(c(1, 2, 0,
                            2, 1, 0), nrow = 2, byrow = TRUE)
# right hand side
constraint_rhs <- c(10, 10)

# direction of constraint
constraint_direction <- c("<=", "<=")

# new constraint for the function output. z is the new binary variable
# x1 + 100*z <= 100
constraint_matrix_new <- rbind(constraint_matrix, c(1,0,100) )
constraint_rhs_new <- c(constraint_rhs, 100 )
constraint_direction_new <- c(constraint_direction, "<=")
# constraints for the binary variable, z
constraint_matrix_z <- matrix(c(0,0, 1),nrow = 1,byrow=TRUE)
constraint_rhs_z_up <- c(1)
constraint_rhs_z_low <- c(0)

constraint_direction_z <- c("<=")

constraint_matrix_new <- rbind(constraint_matrix_new, constraint_matrix_z, constraint_matrix_z)
constraint_rhs_new <- c(constraint_rhs_new, constraint_rhs_z_up, constraint_rhs_z_low)
constraint_direction_new <- c(constraint_direction_new, constraint_direction_z,">=")

# solve the optimization problem. z is now defined as an integer
optimization_result <- lp("min", objective_coefficients, constraint_matrix_new, constraint_direction_new, constraint_rhs_new, all.bin = 3)


# print result
print("optimization result with binary variable:")
print(optimization_result)
print(paste("optimal values for x and binary z: ",optimization_result$solution ))
```

in this example, we've introduced a new variable, 'z' which is a binary variable (either 0 or 1). based on the optimal values, if z takes a value of 1, it means that in our constraint we use the value 100 (when x >= 2) otherwise the constraint does not include this output, which is equivalent to saying we have a value of 0. this way we can use step functions to mimic a non-linear function in our linear program.

one more thing i have seen often is that sometimes i just had a large lookup table of values. this table of values was a result of an extensive simulation. in those situations i used the lookup table approach. say we have a table of results for the output of `my_function(x)`. where x is not necessarily a linear value. in this case we need to find some value close to the value we are interested in so we can use it in our linear constraint. we would have to do something similar to nearest neighbor lookup. let’s imagine i had the following simulation results of the function `my_function(x)` as `sim_results` shown below.

```r
# example simulation results
my_function <- function(x) {
  return(sin(x) * x^2)
}
x_values <- seq(0, 5, length.out = 20) # x for simulation
y_values <- sapply(x_values, my_function)
sim_results <- data.frame(x = x_values, y = y_values)

# lookup nearest output to a specified x
lookup_value <- function(target_x, sim_results) {
  closest_index <- which.min(abs(sim_results$x - target_x))
  return(sim_results$y[closest_index])
}

# lookup the approximate value
target_x <- 2.3
approx_y <- lookup_value(target_x, sim_results)
print(paste("the approximate y value from lookup table for x=", target_x, " is: ", approx_y))

# use it with lpSolve
library(lpSolve)
objective_coefficients <- c(1, 1) # example coefficients
constraint_matrix <- matrix(c(1, 2, 
                            2, 1), nrow = 2, byrow = TRUE)
constraint_rhs <- c(10, 10) # example constraints
constraint_direction <- c("<=", "<=")

# include our constraint using the lookup table value
constraint_matrix_new <- rbind(constraint_matrix, c(lookup_value(2.3,sim_results) , 3) )
constraint_rhs_new <- c(constraint_rhs, 20)
constraint_direction_new <- c(constraint_direction, "<=")

optimization_result <- lp("min", objective_coefficients, constraint_matrix_new, constraint_direction_new, constraint_rhs_new)

print("optimization result with lookup table value:")
print(optimization_result)
print(paste("optimal values for x",optimization_result$solution ))
```

in this example we take an output of the simulated function, which is not calculated inside the linear program, but it is the output of a simulation and included in the constraint. this could be a simulation or a measurement result from a previous experiment.

when picking between approximation method you have to take into account several things: if you have non smooth function use many points for piecewise linear approximation, if you have sudden jumps use the binary selection method, and if you have results of a simulation that you do not want to recalculate again use the lookup table method.

these methods are very common, and i have used them many times. it is useful to check optimization theory books, such as “convex optimization” by boyd and vandenberghe, they have very nice descriptions of these methods. if you need a more rigorous treatment of linear programming "linear programming and network flows" by bazaraa, jarvis and sherali is a good place to start. also sometimes the solution is simple enough, you might not even need to do anything fancy just a single lookup will do.

hope this helps, and happy optimizing!
