---
title: "How to Formulate linear constraint in AMPL?"
date: "2024-12-14"
id: "how-to-formulate-linear-constraint-in-ampl"
---

alright, so you're asking about formulating linear constraints in ampl, huh? been there, done that. i remember back in the day, during my phd, i was working on this huge supply chain optimization problem. the thing was massive, we're talking hundreds of variables and what felt like a million constraints. trying to code all of that directly in some imperative language was just a nightmare, i kid you not. it was slow, error-prone, and just plain painful. that's when i discovered ampl. changed my life, i tell you, when dealing with that type of problem.

ampl really shines when dealing with these kinds of model formulations because it allows you to express the problem in a way that's very close to the mathematical notation. the separation of the model from the data is a game changer and makes your code much more readable and maintainable. let's break down how we'd tackle linear constraints in ampl with a few examples so you can get the gist of it.

first things first, in ampl, a constraint is defined using the keyword `subject to` (or `s.t.`) followed by a name, a colon, and the actual constraint. the cool thing is that you can use the usual mathematical operators for inequalities (<=, >=) and equality (==). here’s a really basic example to give you the feel, suppose we have two variables x and y, and we want to say that x + y should be less than or equal to 10:

```ampl
var x >= 0;
var y >= 0;

subject to constraint1: x + y <= 10;

```

see how clear that is? it pretty much reads like the mathematical equation. and the declarations `var x >= 0;` and `var y >= 0;` define your variables as non-negative. that's often the case with optimization problems. when you have multiple constraints you can add as many `subject to` clauses as needed, and name them so you can later on refer to them, to display them or for debugging purposes.

let's look at something more complicated. suppose you have an array of variables and you want to create a constraint involving a sum of these variables. you can use a `sum` operator to do so. imagine you have some variables `production[i]` representing, let's say, the amount of product 'i' produced, and a parameter `demand[i]` representing demand, and you need to formulate the constraint that the total production does not surpass the total demand, well this is how you do it:

```ampl
set products; # a set of product indices (e.g., {1, 2, 3, ...})
param demand {products} >= 0; # demand for each product
var production {products} >= 0; # production of each product

subject to total_production_constraint:
    sum {i in products} production[i] <= sum {i in products} demand[i];
```

here, `products` is a `set` (think of it like a list of indices). this `set` allows you to use the mathematical notation of summation. if you have defined a set of products in your data file that contains products labeled as 'p1', 'p2', 'p3', etc, then `i in products` will loop through these labels automatically and compute the sum over them. in the data file, you will just declare the `demand` parameter for each `product`, in the following way:

```
set products := p1 p2 p3;

param demand :=
    p1 10
    p2 20
    p3 15;
```
the power here is how it abstracts from the number of products. if you add another item to the set `products` and provide its demand in the data file the `sum` will automatically include that product. that was awesome the first time i realized that.

and it is not only summation. you can have any valid mathematical expression on either side of the inequality/equality. for example you can have scalar multiplication of the variable as well. suppose you have a problem where you want to relate two variables with a weight or something like that:

```ampl
param weight >=0;

var  x >= 0;
var  y >= 0;

subject to constraint2: weight * x + y <= 10;
```

here `weight` is declared as a parameter, so you can pass its value in the data file, and it will be multiplied by x. this works with any other mathematical operation. just remember that the constraints should be linear. so no expressions with `x^2` or `sqrt(x)` as that would violate the linearity of the constraint.

in case you need constraints with “for all” kind of conditions, then you’ll be using the `forall` operator. for example if you have some capacity constraint per each product. let's say `capacity[i]` is the capacity of each product `i` and `production[i]` is the production variable. and you want production of each product to not go beyond its capacity. this is how you formulate it with a `forall` statement:

```ampl
set products; # a set of product indices (e.g., {1, 2, 3, ...})
param capacity {products} >= 0; # capacity for each product
var production {products} >= 0; # production of each product

subject to capacity_constraint {i in products}:
    production[i] <= capacity[i];
```
in this example `i in products` will loop over each element of the set `products` and apply the constraint for each of the product indexes.

now, a little more real-world example. during my research, i was working on a problem with production planning. we had different machines, each with a different efficiency, and we had to produce several products. the production of product `p` in machine `m` is given by variable `production[p, m]` and there is a capacity `capacity[m]` per machine and i also need to make the total production surpass a specific value `min_total_production`.

```ampl
set products; # set of products
set machines; # set of machines
param capacity {machines} >= 0; # capacity of each machine
param min_total_production >= 0; # minimum total production
var production {products, machines} >= 0; # production of each product on each machine

subject to machine_capacity {m in machines}:
    sum {p in products} production[p, m] <= capacity[m];

subject to minimum_production:
    sum {p in products, m in machines} production[p, m] >= min_total_production;
```

this example highlights the power of using sets and indexed parameters. it becomes easy to scale your problem by just adding products or machines and it automatically adjusts, avoiding lots of manual modification in code. that is the beauty of model declaration. when you get that feeling of not fighting with the code you are more focused on the problem rather than debugging.

and this brings me to a point: always be very careful with the indices you are using to declare your variables. it is quite easy to declare something that may result in nonsensical or incorrect models. that's the type of error that is difficult to spot in very large model, specially if you are new to the field and the modeling language. always go back and review your model definition. think what you want to model and then try to declare that in the model. do not assume that the model you made represents what you intended to model.

the way i usually do it, is to start from the simpler example, like the first one with just 2 variables and one constraint, and then add more complexity and more constraints until you reach a representation that is meaningful to your model. in that way you can pinpoint errors quite easily. and it is very useful when you are new to ampl, so you can be able to detect syntax errors early. and believe me that those syntax errors can be very difficult to spot at the beginning. like an instance that happened to me. it took me 2 hours to realize i was calling a set by the wrong name. very frustrating at the time, but i learned a lesson that day.

and speaking of frustrating situations. one time i spent three hours on a very complex assignment with a lot of constraints, and i realized that the solver was taking forever to find a solution. i started thinking that i was doing something wrong, so i reviewed the model many times, tried to use different solvers. nothing. and then, out of nowhere i realized that my problem was infeasible. well that taught me a lesson in life. now i always check that first. it is always good to look at the simplest solutions first. you could have constraints that clash with each other making the problem unsolvable or very difficult to solve, so sometimes a quick check on those constraint may help to realize this type of error.

as for resources, i wouldn't recommend jumping straight to some complex academic papers. you need to build a solid foundation first. i suggest you start with the ampl book, "ampl: a modeling language for mathematical programming" by fourer, gay, and kernighan. that’s the bible of ampl. it really covers the fundamentals well, and it shows you how to think about modeling. there are also some free online tutorials and example models in their official website. also, there are several communities on the web that are quite active for ampl. sometimes you may find code examples or discussions about some specific error in the models. stackoverflow could also be a good place to ask specific questions about issues that you may have with ampl, if you do not find any answer online. that is why i am here. good luck on your journey with ampl.
