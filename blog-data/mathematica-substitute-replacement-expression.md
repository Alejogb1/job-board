---
title: "mathematica substitute replacement expression?"
date: "2024-12-13"
id: "mathematica-substitute-replacement-expression"
---

 so you're asking about substitution in Mathematica right Like how to replace parts of an expression with something else using rules or patterns I've been wrestling with this kind of thing since I first laid eyes on the software back in the day and let me tell you it's both incredibly powerful and sometimes frustratingly opaque

I'll lay out the different substitution techniques I've bumped into over the years hopefully it makes things clearer for you and others who stumble on this thread

First off the most common way is with the ReplaceAll operator which is `/.` or its shorthand `->`. You just have the expression you want to modify on the left of this operator and on the right you give the rule or list of rules.

Let's say we have a simple symbolic expression like `x^2 + 2x + 1` and we want to replace `x` with 3

```mathematica
expr = x^2 + 2x + 1;
expr /. x -> 3
```

That will spit out `16` which is the result of substituting `x` by `3`. Pretty simple stuff this is just basic variable replacement. Now here's where things get more interesting you can substitute an entire sub-expression. Say we had an expression with `sin(y)` and we want to replace all occurences of it with `cos(y)` this would be done like so

```mathematica
expr2 = a + b * Sin[y] + c* Sin[y]^2;
expr2 /. Sin[y] -> Cos[y]
```

This would give us `a + b Cos[y] + c Cos[y]^2`

You can also use a list of rules in the form `expr /. {rule1, rule2, ...}` . Here's where things start to go a little nuts and pattern matching begins to really shine. Say you want to replace all powers of x with something else you can use the following

```mathematica
expr3 = x^2 + 3x^5 + 7x^1;
expr3 /. x^n_ :> replacementFunction[n]
```

This is where you need to understand pattern matching. The `x^n_` part is a pattern. It says match anything of the form x to a power and the `n_` part means match anything and bind it to the name `n`. The `:>` is delayed rule application meaning the replacement on the right side `replacementFunction[n]` is not evaluated until the rule is applied to the expression. So if for example `replacementFunction[n]` was defined as `2n` the above line would return `2 x^2 + 10 x^5 + 2 x^1`. Remember if the `n` is without `_` it would be interpreted as literal `n` and not pattern matching.

Now for a bit of a real world example. In some signal processing work I did back in the day I needed to implement a recursive digital filter. Let's say you have a transfer function in the z-domain which is a rational function with a numerator and a denominator. After some work you derive an input and an output sequence relationship with past and future time samples but you still need to write it in code and you need to have the equation ready to use in a program. After some algebra with pen and paper I obtained the following equation (which you would need to rewrite manually)

`y[n] = a*x[n] + b*x[n-1] - c*y[n-1] - d*y[n-2]`

Lets rewrite it using the `Subscript` function to make it look like how it is on paper

```mathematica
exprFilter = Subscript[y, n] == a Subscript[x, n] + b Subscript[x, n - 1] - c Subscript[y, n - 1] - d Subscript[y, n - 2];
```

Now lets say you need this equation with `n` shifted to `n-1` everywhere you would do this like this

```mathematica
exprFilter /. n -> n-1
```

and it would return

`Subscript[y, n - 1] == a Subscript[x, n - 1] + b Subscript[x, n - 2] - c Subscript[y, n - 2] - d Subscript[y, n - 3]`

I've found that it's best practice to always make sure I test each substitution incrementally that is substituting only one change at a time. Otherwise it would become a tangled mess. Now where do things become complicated? Well patterns can get super sophisticated real quick. You can have conditions on the matches or multiple patterns that interact. That's when things can start to seem like a black box. When you do pattern matching it always tries to be most specific as possible and if you define some conditions in patterns it may not work as expected. Also sometimes the order in which you list your rules in a list matters because if multiple rules can be applied they are evaluated from first to last. It is useful to use `//. {rule1, rule2}` instead of `/. rule1 /. rule2` because it applies the substitution in one go. If you substitute a variable for an expression and you have to substitute that variable somewhere else later it might not work as expected. You might need to substitute an expression for a variable instead.

Also something else I struggled with when I first started out was dealing with expressions inside of functions I thought it would work like this

```mathematica
ClearAll[f]
f[y_] := x^2 + y
f[3]
f[y] /. x -> 5
```

But that didn't work I was confused why `f[y]` did not update the `x`. That is because `f[y]` is only evaluated when it is called with `f[3]` or `f[y]`. Thus you would need to explicitly define the `x` inside the definition of the function like so

```mathematica
ClearAll[f]
f[y_, x_] := x^2 + y
f[3,5]
f[y,5]
```

I've had to debug that kind of thing a lot you just have to keep in mind how things are evaluated. I think some might say it can get tricky.

Now lets talk about other replacement functions that you could use namely `Replace` and `ReplacePart`. These are more specific for certain tasks. `Replace` allows you to specify levels within an expression that you want to work on. For example, you can target specifically the first level or second level or any specific level of the expression tree with options in `Replace`. `ReplacePart` on the other hand allows you to replace parts of expressions by their position in the expression tree which is handy for lists matrices or nested objects. For me I rarely used `ReplacePart` I always try to find an approach with patterns or something a little more abstract.

I once spent about 2 whole days on a single substitution because I had an expression with hundreds of variables and the pattern matching was going all wrong due to a small mistake in the pattern it was something I spent too much time on. After that day I had to take a long walk to clear my mind because the software made me feel dumb haha but I learned my lesson and since then I use small test cases.

To summarize use `/.` for most substitutions with variables or simple subexpressions use patterns if you need to match more complicated parts of the expression. For more fine-grained control over where and how substitution occurs use `Replace` and `ReplacePart`.

As for more resources that can help you out I recommend a really old book actually "Mathematica Programming: An Advanced Introduction" by Leonid Shifrin. Also you could read "An Introduction to Mathematica" by David J. G. and also be sure to read the documentation available directly inside mathematica it is pretty comprehensive. The tutorials are also great! I would start with the official online documentation it is really really good. I personally also enjoy reading Stackoverflow questions similar to yours you might find some good stuff there too.

Hope this long answer helps you out good luck with your substitutions!
