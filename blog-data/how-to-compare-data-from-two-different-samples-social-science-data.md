---
title: "How to compare data from two different samples (social science data)?"
date: "2024-12-15"
id: "how-to-compare-data-from-two-different-samples-social-science-data"
---

alright, so you've got two datasets, presumably from social science research, and you need to figure out if there are meaningful differences between them. this is a bread-and-butter problem in statistical analysis, and thankfully, there are quite a few tools at our disposal. i've been down this road a few times, mostly during my grad school days working on experimental psychology studies, and later when i was contributing to a project trying to model online community behavior. i can share some of what i’ve learned, sticking to practical approaches with minimal fluff.

first thing first, the method you choose really depends on the type of data you have. are we talking about numerical values like age or income? or categorical data, like survey responses (“strongly agree,” “disagree,” etc.)? we also need to think about the structure of the data; are your samples paired (like comparing pre-test and post-test scores of the same individuals) or independent (two separate groups)? lets start with numerical data and independent samples.

for independent numerical samples, a classic approach is the t-test. the t-test checks if the means (averages) of your two groups are statistically different. we are, of course, going to check the data for some assumptions first. before using a t-test you want to make sure your data is roughly normally distributed and that your samples have similar variances. there are statistical tests for that but i usually go for a simple histogram. if the histograms look roughly bell-shaped and the spread is not vastly different you are ok to proceed. i remember using a simple boxplot to test for equality of variance which helps too.

 here is some simple python code that shows how you would do this. i like scipy, its always there when i need it, plus numpy is a dependency so its win-win:

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# fictional data for demo purposes only
group_a = np.random.normal(loc=50, scale=15, size=100)
group_b = np.random.normal(loc=55, scale=17, size=120)

#visual inspection of data
plt.hist(group_a, alpha=0.5, label='group a')
plt.hist(group_b, alpha=0.5, label='group b')
plt.legend()
plt.show()


# perform the t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True) #if variances are not equal set to false.
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.3f}")
alpha = 0.05
if p_value < alpha:
  print("we reject the null hypothesis, groups are statistically different!")
else:
   print("we fail to reject the null hypothesis, groups are not different")
```

in that example, we generate some dummy data with numpy, do a quick visual inspection and then we calculate a t-statistic and a p-value. the p-value is the probability of getting results as extreme as those observed, assuming that there is no difference between the means. if the p-value is below a chosen significance level (typically 0.05, the alpha) then we can conclude that there is evidence to suggest the groups are different in the average.

now, what if the numerical data does not follow a normal distribution, and you’ve ruled out data transformation? then you might want to consider a non-parametric test like the mann-whitney u test.  it's an alternative to the t-test that doesn't rely on data normality. basically, it ranks all data points and then sees if ranks are distributed differently between groups, that is, it does not test the mean but the distribution, which is neat. it is a good test for when the shape of the distribution is far from a bell curve.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# fictional data
group_a = np.random.exponential(scale=20, size=100)
group_b = np.random.exponential(scale=15, size=120)

plt.hist(group_a, alpha=0.5, label='group a')
plt.hist(group_b, alpha=0.5, label='group b')
plt.legend()
plt.show()

# perform the mann-whitney u test
statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
print(f"statistic: {statistic:.2f}")
print(f"p-value: {p_value:.3f}")
alpha = 0.05
if p_value < alpha:
  print("we reject the null hypothesis, distributions are statistically different!")
else:
   print("we fail to reject the null hypothesis, distributions are not different")
```

this code runs the mann-whitney test, which gives you a statistic and a p-value, just like the t-test, interpreting this p-value is the same as in the t-test.

shifting gears to categorical data, like survey responses on a scale. if you have two independent groups and are trying to see if the distributions of responses are different, the chi-squared test is your friend. this test compares the observed frequency of responses in each category to what you would expect if the two groups were the same. when i was analyzing data for an online forum, i used this quite a bit, it helped me see if certain demographics reacted differently to changes in the forum. it's like a big crosstab with numbers and you see if the numbers in one group are bigger than the others.

```python
import numpy as np
from scipy import stats

# fictional data
group_a_responses = np.array([50, 30, 20]) # [agree, neutral, disagree]
group_b_responses = np.array([40, 40, 20]) # [agree, neutral, disagree]

# contingency table for chi-square test
observed = np.array([group_a_responses, group_b_responses])

# chi-square test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"chi-squared statistic: {chi2_stat:.2f}")
print(f"p-value: {p_value:.3f}")
alpha = 0.05
if p_value < alpha:
  print("we reject the null hypothesis, distributions are statistically different!")
else:
   print("we fail to reject the null hypothesis, distributions are not different")

```

in the above snippet, we create some fictional data for two groups and their responses to three possible categories. then we create the contingency table, and use scipy to run the chi-squared test, it returns a statistic, a p-value, the degrees of freedom, and the expected results under the null hypothesis. the expected results can be handy to look at if the test is statistically significant, it can help you see what category is being most influenced by the group difference.

i’ve only covered the basics here, and its all python, but there are other languages like r and sas, which are used by researchers too, that you might need to be acquainted with. depending on your field, you might need to dive into more complex techniques like anova for comparing more than two groups or multivariate methods like regression when you have multiple variables influencing your outcomes. the important thing to keep in mind when comparing groups of data is that it's not just about the test. it is equally important to think about the structure of your data and what questions you are trying to answer. also remember that correlation does not mean causation and that your conclusions are always within the scope of your experiment or research, meaning that the more variables you have the more likely you will find a statistical significance somewhere. it also helps to be mindful that any statistical conclusion needs to be discussed with the experiment context, and be as reproducible as possible.

if you want to go deeper, i'd suggest a couple of solid resources. "discovering statistics using r" by andy field is great for a general overview and practical applications, and it does not get lost in theoretical discussions, it also includes the r code for running the methods. "all of statistics: a concise course in statistical inference" by larry wasserman is more mathy, its more conceptual and has proofs and is a good companion for the book by field, a must read for any person dealing with data. also, just google 'choosing a statistical test', there are some nice charts that may help you select the proper test.

one time i was analyzing some data and after a few hours of head scratching i found out the problem was that my data had a single entry duplicated several times, and that is why i was getting some 'unexplained' results, that was a good laugh. remember to always check your data, data cleaning and checking is 90% of the work.
