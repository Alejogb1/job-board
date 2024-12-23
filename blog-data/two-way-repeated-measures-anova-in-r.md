---
title: "two way repeated measures anova in r?"
date: "2024-12-13"
id: "two-way-repeated-measures-anova-in-r"
---

 so you're asking about two-way repeated measures ANOVA in R right been there done that a few times let me tell you. It's not exactly a walk in the park when you first get into it. I remember my undergrad days wrestling with this stuff it felt like the stats gods were throwing spaghetti code at me. You need the right packages and to structure your data correctly or you'll be staring at error messages for hours. I swear it's happened to the best of us.

Let's break this down into something digestible. First off why two-way repeated measures ANOVA not a regular one way? Well you've got two factors influencing your dependent variable and the measurements are taken on the same subjects or units multiple times think of a medical trial where the same patients receive different treatments at different time points. It's about figuring out how much of the variability comes from each factor their interaction and the within-subject changes. It's a pretty useful tool when you need to account for individual variation in your research data.

So you need to handle within subject and between subject factors differently. You'll have repeated measures or within-subject factors which get measured multiple times on each subject and between subject factors which are like group level variables if any. I mean if you have only within subjects factors you would deal with a simpler repeated measures ANOVA. Data formatting is key here. It really is the number one thing you need to make this work properly.

In R we typically use the `rstatix` package which I have come to love for repeated measures. Let’s say we’re measuring some hypothetical performance score in three different conditions and two timepoints. If you don't have rstatix installed go ahead and run `install.packages("rstatix")` like a good R user.

Here’s how your data should be structured ideally in a tidy format a data frame where each row is one observation:

```R
library(tidyverse)
library(rstatix)

# Sample data - your data should look similar
data <- data.frame(
  id = rep(1:10, each = 6),
  condition = rep(c("A", "B", "C"), times = 20),
  time = rep(c("T1", "T2"), each = 30),
  score = rnorm(60, mean = 50, sd = 10) + #some random data
           rep(c(0, 5, 10), times = 20) + #condition difference
           rep(c(0, -3), each = 30)   #time difference

)

print(head(data))
```
Here id is the subject identifier condition is your first factor time is your second factor and score is your dependent variable. This data represents a balanced design which simplifies the analysis. I am not going to cover unbalanced designs because I do not want to open that can of worms right now.

Now for the actual ANOVA the `anova_test` function is your best friend. You will specify your model the between and within subjects factors and your dependent variable.

```R
# Perform two way repeated measures ANOVA
res.anova <- data %>%
  anova_test(
    dv = score,
    wid = id, #subject variable
    within = c(condition, time) #within subject factors
  )
print(res.anova)

```
What this will output is a table with results the F statistics degrees of freedom and p values. This will allow you to understand the significance of your factors and their interaction. Do not only look at the p value there are many resources about effect size and post hoc analysis if necessary.

Here's a little quirk I discovered back in my early days I was using some other packages and I always struggled with the output tables. They were never clean and I never understood what the heck they were telling me. This is why I always recommend `rstatix`. It cleans up the output and it makes everything more understandable which is very important if you have to present this somewhere in a paper or at a meeting. It even gives you partial eta squared as effect size which is pretty useful.

Now about that interaction effect. That is the most interesting part. You have to interpret that very carefully. If you have a significant interaction effect it means that the effect of one within-subject factor (e.g. condition) depends on the level of the other within-subject factor (e.g. time). In simple words the effect of condition differs across the time points. In that case looking at the main effects might be misleading. You should do further analysis such as simple effects to understand what is going on.

So what do you do if you find a significant interaction? My advice is to visualize your data especially the means for all condition and time combinations that is your best bet to understand what is really going on. Then you can explore simple effects which is a form of post-hoc test.

Here is a small example to produce the plot.
```R
# Plotting the means
library(ggplot2)
data %>%
  group_by(condition, time) %>%
  summarise(mean_score = mean(score)) %>%
  ggplot(aes(x = time, y = mean_score, color = condition, group=condition)) +
  geom_line() +
    geom_point() +
  labs(title = "Means Plot",
       x = "Time",
       y = "Mean Score") +
     theme_minimal()


```
This plot will show you what the means are and how they vary across the time points. If your interaction is significant then you should see that the lines are not parallel.

Now if your interaction is significant you need to dive into post hoc test. `emmeans` package is great for that:
```R
# Post-hoc tests (if interaction is significant)

library(emmeans)

# create the model
model <- lm(score ~ condition*time + Error(id/(condition*time)), data = data)
emm_model <- emmeans(model, ~ condition | time)
pairs(emm_model) # compare between conditions within each timepoint
```
This lets you compare the conditions at each time point to test what is going on and if they are significantly different from each other at each specific time point.

Regarding resources for this type of analysis I highly recommend you get your hands on Andy Field’s “Discovering Statistics using R”. That’s my go-to book on everything statistics and R. Also you might want to check “Linear Mixed Models: A Practical Guide Using Statistical Software” by Brady West. It dives deep into mixed models which is what you are actually doing in this case it will help you understand this process a lot better. Don't just rely on package documentation or stackoverflow answers read books to understand the underlying theory. You should also look up some good papers on the design you are considering. There are a number of classic experimental design books such as "Design and Analysis of Experiments" by Douglas Montgomery.

In summary make sure your data is formatted correctly run the ANOVA using `anova_test` interpret interaction carefully visualize your data and use emmeans if post hocs are needed. And do yourself a favor use `rstatix` and read Andy Field’s book.

Remember the p-value tells you that the results are unlikely to have happened by chance but I heard once that you should treat it as a suggestion not a verdict.
