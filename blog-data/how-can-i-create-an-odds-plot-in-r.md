---
title: "How can I create an odds plot in R?"
date: "2024-12-23"
id: "how-can-i-create-an-odds-plot-in-r"
---

Alright, let's talk odds plots in R. I recall one particularly challenging project back at Cyberdyne Systems – not the movie kind, thankfully – where we were tasked with analyzing a large clinical dataset. The goal was to visually represent the relationship between various risk factors and the odds of a specific outcome. A simple bar chart wouldn't cut it; we needed to see the magnitude of the odds ratios and their confidence intervals, which is precisely where odds plots shine. I've built quite a few of these over the years, so I can offer some useful insights.

The fundamental idea behind an odds plot is to graphically display odds ratios (or sometimes odds themselves) along with their corresponding confidence intervals. This kind of visualization is incredibly useful in logistic regression, survival analysis, or any situation where you're comparing the likelihood of an event between different groups. It moves beyond simply stating statistical significance and allows you to assess the practical importance of the effect. You'll often find them accompanying forest plots in meta-analyses, too, so understanding their structure is generally beneficial.

To construct these plots in R, we predominantly lean on packages like `ggplot2`, known for its robust graphics grammar, and `dplyr` for data manipulation. Although `plot` or base R graphics could technically do the job, their flexibility and aesthetic appeal are generally limited.

Let’s break down the process with a few examples. I'll start with a fairly straightforward illustration using simulated data, then move to a situation where you might need to wrangle data from a statistical model.

**Example 1: Basic Odds Plot with Simulated Data**

First, we'll generate some mock data and compute the odds ratios and their confidence intervals. This is a typical scenario where you might be evaluating the effect of different treatments or variables.

```R
library(dplyr)
library(ggplot2)

# Simulate some data
set.seed(123)
data <- data.frame(
  group = factor(rep(c("A", "B", "C"), each = 50)),
  outcome = rbinom(150, 1, prob = c(0.3, 0.5, 0.7))
)


# Calculate odds ratios
odds_data <- data %>%
  group_by(group) %>%
  summarise(
    n_outcome = sum(outcome),
    n_total = n(),
    odds = n_outcome / (n_total - n_outcome),
    lower = NA, upper = NA # placeholders for later interval calculation
    )


# Calculate confidence intervals assuming simple binomial proportions
for (i in 1:nrow(odds_data)){
  p <- odds_data$n_outcome[i] / odds_data$n_total[i]
  n <- odds_data$n_total[i]
  se <- sqrt(p*(1-p)/n) #Standard Error
  z_critical <- qnorm(0.975) # 95% CI
  
  lower_p <- p - z_critical * se
  upper_p <- p + z_critical * se
  
  odds_data$lower[i] <- lower_p/(1-lower_p)
  odds_data$upper[i] <- upper_p/(1-upper_p)
  
}

# Create the odds plot
ggplot(odds_data, aes(x = group, y = odds)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Odds of Outcome by Group",
       x = "Group",
       y = "Odds") +
    coord_flip() + # Makes it horizontal
  theme_minimal()
```

In this first example, the `dplyr` package is used to group the simulated data by groups A, B, and C, and then to calculate the odds of the binary outcome within each group. I've added a simple loop that calculates the standard error of the proportions in order to define 95% confidence intervals on the logit-scale, which I then transform to a normal odds scale. Then `ggplot2` is used to create the main plot, rendering the odds as points and the intervals as error bars. Crucially, the `geom_hline(yintercept = 1)` line acts as a reference—odds ratios above 1 indicate an increased odds, while those below 1 indicate a decreased odds compared to the baseline. In my experience, this reference line is often vital for interpreting such visuals.

**Example 2: Odds Plot from a Logistic Regression Model**

Now, let's look at the common situation where you have results from a logistic regression. Here's where things get a bit more involved, but the workflow remains broadly similar.

```R
library(dplyr)
library(ggplot2)

# Example data
set.seed(456)
data <- data.frame(
  age = rnorm(100, mean = 50, sd = 10),
  treatment = factor(sample(c("A", "B"), 100, replace = TRUE)),
  outcome = rbinom(100, 1, prob = ifelse(treatment == "A", plogis(-2+age/30), plogis(-1 + age/30)))
)

# Fit a logistic regression model
model <- glm(outcome ~ age + treatment, data = data, family = binomial())

# Extract odds ratios and confidence intervals
odds_data <- exp(coef(model)) %>%
  data.frame(odds = .) %>%
  rownames_to_column(var = "term") %>%
  filter(term != "(Intercept)") # Ignore Intercept

ci <- exp(confint(model)) %>%
  data.frame(lower = ., upper = .) %>%
  rownames_to_column(var = "term") %>%
  filter(term != "(Intercept)")

odds_data <- merge(odds_data, ci, by = "term")

# Create the odds plot
ggplot(odds_data, aes(x = term, y = odds)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Odds Ratios from Logistic Regression",
       x = "Predictor",
       y = "Odds Ratio") +
   coord_flip()+ # Makes it horizontal
  theme_minimal()
```

This code first fits a logistic regression model using `glm`. Crucially, we extract the odds ratios from the model’s coefficients by applying `exp()`. This function exponentiates the log-odds obtained from the regression model. The confidence intervals are also obtained by exponentiating the results of the `confint` function. Then these are merged together and rendered again as an odds plot. Note that unlike the first example, we have odds ratios here, not just the odds themselves, allowing us to look at how the change in `age` and the treatment affect the odds. The horizontal reference line is still vital for interpreting whether those ratios are above or below the baseline of 1.

**Example 3: Handling Multiple Variables from Logistic Regression**

Now, let’s examine a slightly more complex version, where we might want to represent several interactions.

```R
library(dplyr)
library(ggplot2)

# Example data with interaction
set.seed(789)
data <- data.frame(
    age = rnorm(200, mean = 50, sd = 10),
    treatment = factor(sample(c("A", "B"), 200, replace = TRUE)),
    gender = factor(sample(c("Male", "Female"), 200, replace = TRUE)),
    outcome = rbinom(200, 1, prob = ifelse(treatment == "A" & gender == "Male", plogis(-2 + age/30),
                                        ifelse(treatment == "B" & gender == "Female", plogis(-1 + age/30), plogis(-1.5 + age/30))))
  )

# Fit a logistic regression model with interaction
model <- glm(outcome ~ age + treatment * gender, data = data, family = binomial())

# Extract odds ratios and confidence intervals
odds_data <- exp(coef(model)) %>%
  data.frame(odds = .) %>%
  rownames_to_column(var = "term") %>%
  filter(term != "(Intercept)")

ci <- exp(confint(model)) %>%
  data.frame(lower = ., upper = .) %>%
  rownames_to_column(var = "term") %>%
  filter(term != "(Intercept)")


odds_data <- merge(odds_data, ci, by = "term")

# Create the odds plot
ggplot(odds_data, aes(x = term, y = odds)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Odds Ratios from Logistic Regression (with Interaction)",
       x = "Predictor",
       y = "Odds Ratio") +
    coord_flip() + # Makes it horizontal
  theme_minimal()
```

The code is structurally quite similar to the previous example, but now includes an interaction term between `treatment` and `gender`.  When you fit such models, the odds ratios for the main terms (e.g. treatment, gender) are conditioned on the interaction terms being zero. The coefficients (and by extension the odds ratios) associated with interactions need to be interpreted within this context.  The resulting odds plot can help discern the complex relationships that may exist within your data. Again, a horizontal presentation helps with the interpretation, and the `theme_minimal()` provides a clean visual layout.

For further study, I would recommend getting a firm grounding in the fundamentals of logistic regression, which is discussed in detail in *Applied Logistic Regression* by Hosmer, Lemeshow, and Sturdivant. To improve visualization skills, Hadley Wickham’s *ggplot2: Elegant Graphics for Data Analysis* is an essential resource. Additionally, *Regression Modeling Strategies* by Frank Harrell offers a wealth of information on using regression models for statistical analysis.

In summary, crafting compelling odds plots in R involves a clear understanding of what you aim to display and a proper methodology to do so. I've shown a few common approaches here, each emphasizing clear visualization and thoroughness. It’s a powerful tool to move past simply reporting numerical values and towards data-driven storytelling.
