---
title: "nested anova in r statistical analysis?"
date: "2024-12-13"
id: "nested-anova-in-r-statistical-analysis"
---

 so nested ANOVA in R right Been there done that seen the t-shirt more like seen the script crash and burn a few times let me tell you

This isn't exactly rocket science but it has its quirks You know how it is with statistical modeling things can get tangled up pretty quick

So you're looking at a nested ANOVA situation that means you have factors that are hierarchical one factor is nested within another Think of it like departments within a company departments within a branch something like that In R well we need to spell that out for it

I remember this project I had back in my grad school days It was like comparing different teaching methods but each method was used in different schools and the schools were part of different districts A classic nested design nightmare it felt like at the time My data was a hot mess it looked like this initially:

```R
# My Data Back in the Day
set.seed(123)
data_bad <- data.frame(
  district = rep(paste0("District", 1:3), each = 15),
  school = rep(paste0("School", 1:5), times = 3, each = 3),
  method = rep(paste0("Method", 1:3), times = 15),
  score = rnorm(45, mean = 75, sd = 10)
)

head(data_bad)
```

Yeah it looks kinda messy right It had districts schools and different teaching methods within each of them Each school only used one method it wasn’t like some schools tried all of the methods That’s where the nesting comes in schools are nested within districts The first thing is to recognize that the way the data is structured isn't really amenable to running the model efficiently

The usual suspects like `aov` can handle it but its output is not easy to understand especially when it is unbalanced That’s why `lme4` comes in handy It gives you a more controlled and easier to read way of defining the nested structure especially for unbalanced nested ANOVA

For the longest time in my early career with R i kept writing complex codes to simulate unbalanced design issues then someone had a brain wave and created `lme4` This function allows for mixed models and it allows you to write nested factor structure easily

So here's how you actually do it lets create a more structured version of the data I had back then

```R
# A Better Data structure
set.seed(456)
data <- data.frame(
  district = factor(rep(paste0("District", 1:3), each = 15)),
  school = factor(rep(paste0("School", 1:5), times = 3, each = 3)),
  method = factor(rep(paste0("Method", 1:3), times = 15)),
  score = rnorm(45, mean = 75, sd = 10)
)

head(data)
```

Make sure you convert the factors in the data as factors to use them correctly with the `lme4` package

Now let’s get to the actual model I always prefer to install packages first just to be sure

```R
# Install and load lme4 package
if(!require(lme4)){install.packages("lme4")}
library(lme4)

# The Nested ANOVA Model
model <- lmer(score ~ method + (1 | district/school), data = data)

summary(model)
anova(model)

```

 so what’s happening here `lmer` is the function from `lme4` that allows mixed models The formula is `score ~ method + (1 | district/school)` the score is the dependent variable that you want to understand the effect of the method factor `method` is fixed effect we want to know if the mean is statistically different for each level

`(1 | district/school)` this part is the crucial bit Its telling R that schools are nested within districts The `1` means it’s a random intercept model for each school and each district We could have other random coefficients but for the purpose of our problem, a random intercept model works well

The `anova(model)` is important here Because it will provide you the statistical F test of the random effects and fixed effects

Now the output can be a little bit dense at first it takes a little getting used to But generally the important parts are the fixed effects (method) and the random effects (variability between districts and schools) The p values will tell you whether the method has a significant effect and how much variability is in the district and school factors

I once spent like three days trying to debug a nested ANOVA using the traditional `aov` approach only to find out the error was a simple data typing mistake I felt like throwing my computer out the window at that moment haha

The other thing to consider is that if you are going to deal with unbalanced nested ANOVA the design part it gets more difficult

Lets consider a variation of our data where some levels of the factors are not fully filled up

```R
# An unbalanced nested design
set.seed(789)
data_unbalanced <- data.frame(
  district = factor(rep(paste0("District", 1:3), each = 10)),
  school = factor(rep(c(paste0("School", 1:3), paste0("School", 1:2), paste0("School",1:4) ),each = 2)),
  method = factor(rep(paste0("Method", 1:3), length.out = 30)),
  score = rnorm(30, mean = 75, sd = 10)
)

head(data_unbalanced)

# Running the model with the unbalanced data
model_unbalanced <- lmer(score ~ method + (1 | district/school), data = data_unbalanced)

summary(model_unbalanced)
anova(model_unbalanced)

```

As you can see the number of levels for the schools are not consistent between districts This is what we call an unbalanced design and `lmer` can handle this easily But when you start to have a very unbalanced design or with missing data issues `lmer` can fail to estimate the fixed and random effects correctly In that case its good to consider alternative techniques that might be more robust

In the olden days well before `lmer` or `nlme` people used to rely a lot on the `aov` package but you have to define the structure very carefully especially for unbalanced designs it's possible to do but its much more messy and hard to interpret for unbalanced designs

As for resources well I would suggest looking into books like "Linear Mixed Models: A Practical Guide Using Statistical Software" by Brady T. West Paul E. Johnson and Ralf G.M. Swierzbinski this book is amazing for understanding the mathematical fundamentals of mixed model and also the best ways to use them in R software and for more practical examples "Mixed Effects Models and Extensions in Ecology with R" by Alain F. Zuur Elena N. Ieno Neal J. Walker Annette A. Saveliev and Gavin M. Smith this book focus on how to use mixed models in Ecology but it also provides excellent material for how to use mixed model in R and the mathematical and theoretical framework it uses if you like papers there are many online but you have to search properly for the problem your dealing with

I think that’s about it for the nested ANOVA in R if you have more specific questions well throw them my way I have seen a few things in the past decade or so working with mixed models
