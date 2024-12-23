---
title: "nested random effects mixed model?"
date: "2024-12-13"
id: "nested-random-effects-mixed-model"
---

 so nested random effects mixed model right I've been down this rabbit hole a few times let me tell you It's not exactly a walk in the park but definitely something you can get your head around with a bit of elbow grease.

First off when you say "nested" it's crucial we understand exactly what we're nesting. Essentially you've got groups within groups think of it like this you've got students within classrooms within schools that's a classic example. The key thing here is that the grouping at the lower level is completely nested within the grouping at the higher level a student isn't in multiple classrooms in this model for instance. It's not crossed or partially crossed it's purely nested. If its partially crossed then we need a different approach.

My first real run-in with this wasn't a student example of course. Back in my early days of data tinkering I was working on sensor data from a manufacturing plant. We had individual sensors mounted on different machines each machine itself being part of a production line. So the sensors were nested within machines and those machines were nested in the production line structure. Classic nested structure if I ever saw one. We were trying to figure out which part of the process was causing the most variation and just using fixed effects wasn't cutting it because it treated everything as being independent when really the sensor measurements within a machine were clearly correlated. We were in a bad spot till we realized that random effects could account for it.

The thing about random effects is they let you model variation that’s due to grouping and not some underlying fixed process that applies to all groups universally. So in our case there would be variation between sensor readings in a machine as random effect and the machines random effect would also vary within production lines and this nestedness would be modeled in the model. If you don't take that into account you're going to have inflated standard errors and end up with some very bad conclusions and that's not good for your data analysis at all.

Now when you're talking about a *mixed* model that just means you're handling both random and fixed effects. Fixed effects are your typical predictors things you believe will have some kind of consistent relationship with your outcome like perhaps some feature of the machine itself. Random effects as we’ve discussed earlier are the grouping structure. It's a really powerful way to model data that has that kind of hierarchical structure.

Implementation wise its easier to deal with modern packages than it used to be but you really have to think through it before you start coding. This is how we usually implement it in R using `lme4`. It's my goto for this kind of thing.

```r
library(lme4)

# Simulate some example data
set.seed(123)
num_schools <- 5
num_classrooms_per_school <- 4
num_students_per_classroom <- 20

# Create data frame
df <- data.frame(
    school = rep(1:num_schools, each = num_classrooms_per_school * num_students_per_classroom),
    classroom = rep(1:(num_classrooms_per_school * num_schools), each = num_students_per_classroom),
    student = 1:(num_schools * num_classrooms_per_school * num_students_per_classroom),
    fixed_effect = rnorm(num_schools * num_classrooms_per_school * num_students_per_classroom)
)

# Add random effect for each classroom and school
school_effects <- rnorm(num_schools, mean = 0, sd = 2)
classroom_effects <- rnorm(num_schools * num_classrooms_per_school, mean = 0, sd = 1)

df$classroom_random <- classroom_effects[df$classroom]
df$school_random <- school_effects[df$school]

# Generate the outcome varibale
df$outcome <- 5 + 2 * df$fixed_effect + df$school_random + df$classroom_random + rnorm(num_schools * num_classrooms_per_school * num_students_per_classroom, sd= 0.5)

# Fit the nested random effects model
model <- lmer(outcome ~ fixed_effect + (1 | school/classroom), data = df)

# model summary
summary(model)

```
This snippet shows the very basic model formula `outcome ~ fixed_effect + (1 | school/classroom)` which is really the heart of it. The `(1 | school/classroom)` syntax tells `lme4` that you want to include random intercepts for both schools and classrooms but since the classrooms are nested the syntax tells the software the same. This is the bread and butter of nested models in R. The `+` here is for fixed effects and `|` are for random effects. The slashes are the nesting. If its crossed just replace the slash for `+` or `*`.

You could also do it in python with `statsmodels` which I found out is great for mixed model implementations.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Simulation of the same data
np.random.seed(123)
num_schools = 5
num_classrooms_per_school = 4
num_students_per_classroom = 20

data = {
    'school': np.repeat(range(1, num_schools + 1), num_classrooms_per_school * num_students_per_classroom),
    'classroom': np.repeat(range(1, num_schools * num_classrooms_per_school + 1), num_students_per_classroom),
    'student': range(1, num_schools * num_classrooms_per_school * num_students_per_classroom + 1),
    'fixed_effect': np.random.randn(num_schools * num_classrooms_per_school * num_students_per_classroom)
}
df = pd.DataFrame(data)

school_effects = np.random.normal(0, 2, num_schools)
classroom_effects = np.random.normal(0, 1, num_schools * num_classrooms_per_school)
df['classroom_random'] = classroom_effects[df['classroom'] - 1]
df['school_random'] = school_effects[df['school'] - 1]

df['outcome'] = 5 + 2 * df['fixed_effect'] + df['school_random'] + df['classroom_random'] + np.random.normal(0, 0.5, num_schools * num_classrooms_per_school * num_students_per_classroom)


# Fit the nested random effects model
model = smf.mixedlm("outcome ~ fixed_effect", data=df, groups=df["school"],
                    re_formula="1", vc_formula={"classroom": "0 + C(classroom)"}).fit()


print(model.summary())
```
The most crucial thing here is specifying the groups with `groups=df["school"]` and random effects with `vc_formula={"classroom": "0 + C(classroom)"}` in the python implementation. It tells the model to allow random intercepts for classroom within school. The "0 +" part means no fixed effect for classroom just random intercepts.

Now one thing to watch out for is model convergence. Sometimes your model doesn't converge and that's a big red flag. When this happened to me the first time I had no idea why my random effects model was showing me non convergence warnings and after a while I realized I had way too many random parameters to estimate relative to the amount of data I had available. The remedy was to simplify the model by reducing the random effects structure. Less parameters are easier to converge which makes sense when you think about it. I had to try multiple different random effect structures to find one that converges. It's the same as model selection problem for fixed effects but now you have the additional problem of nested structure.

One more example lets say you want to use bayesian modeling to account for the nested effects then you would use Stan or pymc3 or other bayesian modeling packages. Here is an example in Stan.

```stan
data {
  int<lower=0> N;
  int<lower=1> num_schools;
  int<lower=1> num_classrooms;
  int<lower=1> num_students;
  int<lower=1> school[N];
  int<lower=1> classroom[N];
  real fixed_effect[N];
  real outcome[N];
}

parameters {
  real mu;
  real beta;
  vector[num_schools] school_effect;
  vector[num_classrooms] classroom_effect;
  real<lower=0> sigma_school;
  real<lower=0> sigma_classroom;
  real<lower=0> sigma_residual;

}

model {
  // Priors
  mu ~ normal(0, 10);
  beta ~ normal(0, 10);
  school_effect ~ normal(0, sigma_school);
  classroom_effect ~ normal(0, sigma_classroom);
  sigma_school ~ normal(0, 1);
  sigma_classroom ~ normal(0, 1);
  sigma_residual ~ normal(0, 1);


  // Likelihood
  for(i in 1:N){
    outcome[i] ~ normal(mu + beta * fixed_effect[i] + school_effect[school[i]] + classroom_effect[classroom[i]] , sigma_residual);
  }

}
```

This Stan model defines the likelihood and priors of the parameters. You will need to load the simulated data in R and compile the model and then fit the model using Stan function of `rstan` package.

For resources honestly Pinheiro and Bates' *Mixed Effects Models in S and S-PLUS* is a classic you’ll want to get your hands on. It goes deep into the theory and how to use it in R also. There is also the book *Data Analysis Using Regression and Multilevel/Hierarchical Models* by Andrew Gelman and Jennifer Hill and that also explains very well how to approach hierarchical modeling. If you are into bayesian you may want to consider *Bayesian Data Analysis* from Gelman et al. They all are great references if you want to really understand what you are doing instead of just plugging and chugging. It will definitely up your statistical modeling game if you read at least one of them.

And yeah that’s a pretty good overview of nested random effects mixed models. It's not magic just a nice way to model data when you have this nested structure. Oh and did you hear about the statistician who was terrible at poker? He always folded because he was afraid of getting random draws.   I'll see myself out. If you have more questions just ask I'm always happy to ramble more about this topic.
