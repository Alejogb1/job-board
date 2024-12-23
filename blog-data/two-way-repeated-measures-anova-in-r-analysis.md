---
title: "two way repeated measures anova in r analysis?"
date: "2024-12-13"
id: "two-way-repeated-measures-anova-in-r-analysis"
---

 so you're wrestling with a two way repeated measures ANOVA in R yeah I've been there done that got the t shirt and probably a few stress induced grey hairs from it lets dive right in

First things first you need to understand what you're dealing with a two way repeated measures ANOVA is for when you have two within subject factors basically two things you're measuring repeatedly on the same individuals its not some free for all ANOVA party it has strict requirements about what data is needed for the analysis

You are right that this isnt your basic ANOVA scenario you cannot just chuck your data into any `aov()` or `anova()` function and expect a correct answer you'll end up with garbage output at least that is what happened to me when I tried and that output made absolutely no sense no wonder I hated stats in undergrad now i love it

 let's get practical the most common package for repeated measures ANOVA in R is `ez` it handles the heavy lifting of setting up the model and gives you the ANOVA table that you need we can also use `rstatix` which is also very good

You need to install it first if you dont have it already with

```R
install.packages("ez")
install.packages("rstatix")
```

You'll also want `tidyverse` in your workflow for data manipulation so if you don't have it use this also

```R
install.packages("tidyverse")
```

Now let me show you a fake dataset and go through the process I've had this exact issue before and I ended up fixing it by just reading the correct papers so lets hope you learn quicker than I did back then

Let's say you are tracking reaction time to different stimuli under two conditions say noise level and stimulus complexity each participant does all combinations of noise and complexity levels which makes it repeated measures

Here's how the data might look in a csv file:

```csv
participant,noise_level,complexity,reaction_time
1,low,simple,300
1,low,complex,400
1,high,simple,350
1,high,complex,450
2,low,simple,280
2,low,complex,380
2,high,simple,330
2,high,complex,430
3,low,simple,310
3,low,complex,410
3,high,simple,360
3,high,complex,460
```

Now we need to get this into R and prep it for analysis this is where we use `tidyverse` because it makes life so much easier

```R
library(tidyverse)
library(ez)
library(rstatix)

data <- read_csv("your_data.csv")  # replace with your file path or read directly from clipboard etc

# Convert to factor for ez package
data$participant <- factor(data$participant)
data$noise_level <- factor(data$noise_level)
data$complexity <- factor(data$complexity)
```

Right so the `read_csv` part is self explanatory it loads the csv data we need the `factor` bit is crucial because `ez` needs categorical data to be factors or else it will treat it as numeric values and produce wrong results I learned this the hard way after hours of troubleshooting once the pain of finding that error will stay with me always

Next we use the `ezANOVA()` function and here's how I tend to set it up notice the `dv` (dependent variable), `wid` (within subjects variable or the id), `within` (the factors that you repeatedly measured) this is the most important part:

```R
# EZ package function
model_ez <- ezANOVA(
  data = data,
  dv = reaction_time,
  wid = participant,
  within = .(noise_level, complexity),
  detailed = TRUE
)

print(model_ez)
```
That `detailed = TRUE` argument will give you a bunch more useful info in the output besides the regular ANOVA table

Alternatively we can use `rstatix` which I also find very useful and does almost the same thing as `ez` but sometimes I have found myself in situations were I need both for specific cases which is why I show it to you here. First of all `rstatix` works better with tibbles so we convert first the data:
```R
data <- as_tibble(data)
```
Then we do the ANOVA model by using the `anova_test` function:
```R
# Rstatix Package Function
model_rstatix <- data %>%
  anova_test(
    dv = reaction_time,
    wid = participant,
    within = c(noise_level, complexity)
  )

print(model_rstatix)
```

Both approaches should give you a pretty similar ANOVA output including stuff like F values degrees of freedom and the all important p values which tell you if the effects are statistically significant

Now here's a little pro tip you often need to look at post hoc tests especially if you have more than two levels of your within subjects factor in case you need to explore the interactions more.

`ez` does not do post hoc tests directly so you need to use other packages or methods like the `emmeans` or `pairwise.t.test` package after the ANOVA model is calculated which is a pain point for me and this is where I think `rstatix` shines because it lets you do pairwise tests easily.

Here is a quick example of how to do it with `rstatix`:

```R
posthoc_rstatix <- data %>%
  pairwise_t_test(
    reaction_time ~ noise_level,
    paired = TRUE,
    p.adjust.method = "bonferroni"  # or other correction methods
  )
print(posthoc_rstatix)
posthoc_rstatix_complexity <- data %>%
  pairwise_t_test(
    reaction_time ~ complexity,
    paired = TRUE,
    p.adjust.method = "bonferroni"  # or other correction methods
  )

print(posthoc_rstatix_complexity)

```

I showed a simple posthoc test using paired t tests but there are many other posthoc tests you can do to explore the data further

Now for interpreting the output let's not forget that bit in stats the p value tells you if there is any statistical effect but be careful with the interpretation significance does not imply causality or effect size. To properly understand and report the findings you should report effect sizes like partial eta squared or cohens d for pairwise tests in order to have a complete picture. For effect size I usually just google it every time I need it because I always forget the formulas I'm lazy yeah I know.

Here is something I have used before to calculate effect sizes using `rstatix`:

```R
effect_size_noise_level <- data %>%
    cohens_d(reaction_time ~ noise_level, paired = TRUE)
print(effect_size_noise_level)

effect_size_complexity <- data %>%
  cohens_d(reaction_time ~ complexity, paired = TRUE)
print(effect_size_complexity)
```
You can do much more complex stuff but I am not going to get into that now you get the point

 this is where things get a little bit tricky and a lot of people get stuck and I did too that is the assumption checks for repeated measures ANOVA. You need to check things like sphericity and normality for the differences among your within-subject conditions. `ez` provides a way to check sphericity using Mauchly's test and you can find that in the `model_ez` output under the `Sphericity Corrections` part. If that assumption is violated (p < 0.05) you may need to apply corrections like the Greenhouse-Geisser or Huynh-Feldt corrections which you can specify in the `ezANOVA` function using the `correction` parameter. If not specified by default it uses the GG correction but I like to see it explicit there.
However if you want to check for normality I think it is best to use separate functions rather than the `ez` output for me `rstatix` has been my go to:
```R
normality_test <- data %>%
  group_by(noise_level, complexity) %>%
  shapiro_test(reaction_time)
print(normality_test)
```
You can see each groups normality by this function

For more information on sphericity you can check Maxwell & Delaney's book on "Designing Experiments and Analyzing Data" they have an excellent section on this topic with practical examples

 and one more thing if your data does not meet assumptions you may need to do transformations to the data for the analysis or consider alternatives to parametric analysis such as Friedman's test or similar nonparametric procedures

Just a final note as a word of caution be very careful with your data you must understand very well what is the design of your experiment is before analyzing it because the slightest misunderstanding can lead to a lot of headache for example I recall having a data where subjects were tested twice and I was trying to analyze it using repeated measures ANOVA and it was completely wrong because it was not repeated measure ANOVA it was another test, lets just say I wasted a lot of time troubleshooting that error.

 you asked for a techy tone that is it hope this helps you and now you have all the information you need to solve your problem if you are struggling again please post the code and the data and what have you tried.
