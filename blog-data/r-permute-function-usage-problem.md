---
title: "r permute function usage problem?"
date: "2024-12-13"
id: "r-permute-function-usage-problem"
---

 so "r permute function usage problem" right I’ve seen this rodeo before let me walk you through it because I’ve spent way too much time untangling permutation issues particularly in R It's like a rite of passage for anyone doing any kind of serious data analysis there You think you’re good then boom permutation problems slap you in the face

So the question as I understand it seems to be about the `permute` function in R or maybe a more general issue about creating permutations The core of the problem is you’re trying to generate shuffled or rearranged versions of a dataset a vector a list anything really and it’s not quite working how you'd expect I've been there believe me I was knee-deep in a Monte Carlo simulation once trying to test some complex algorithm I thought my logic was perfect but then my permutations were either too repetitive or not random enough my p-values were all wrong and I spent like 3 days pulling my hair out trying to fix it It was a mess

First things first let’s talk about what the `permute` function usually does and I’m assuming we're talking about the one from the `permute` package because that’s the most common one It’s designed to create permutation indices which is just a fancy way of saying it gives you a new order of numbers from 1 to n that when applied to your original data will shuffle it around

Here's a really basic example

```R
library(permute)

my_vector <- 1:5
permutations_indices <- permute(my_vector)
# lets check what those indices are they will change every time you run the code
print(permutations_indices)
# let's actually reorder the original vector using these indices
shuffled_vector <- my_vector[permutations_indices]
print(shuffled_vector)

```
The output of `permutations_indices` will be a matrix where each row is a different set of permutation indices like `4 2 5 1 3`  applying this to my\_vector will reorder it. This is the fundamental concept if you’re not getting this part right things are gonna get messy

Now I have a feeling that you are experiencing a specific problem if you are posting here There are a couple of common things that people mess up and I’ve personally made each one of them at some point let me share my failures.

One big mistake people do is that they get confused between permutations with and without replacement This becomes crucial when dealing with smaller data sets If you're dealing with sampling or drawing a subset of your vector and then you're generating these permutations with the replacement you might end up drawing the same item multiple times within the same permutation and that is not what you usually want you want permutations where you change the order of the existing data not add or delete from it so the `permute` package uses permutations without replacement by default

For instance if you try to use `sample` with replacement and want permutations that can give you duplicated values and you expect no duplicates `sample(my_vector, size = length(my_vector), replace = TRUE)` will get you that but it’s not the same type of permutation

Another thing I often messed up was with stratified or blocked permutation If you have grouped data like in a repeated measure experiment or different conditions and you want to shuffle within the groups but not between them you can use `blocks` argument but you have to make sure it’s exactly what you’re expecting and you need to have the `strata` argument set to the right value it should be a factor type variable

Like imagine you have data from three different experiments and you want to permute data inside each experiment but not mix it between them

```R
experiment_data <- data.frame(
  value = 1:12,
  experiment = factor(rep(1:3, each = 4))
)
# here we will generate permutation indices that respect the experiment groupings
permutations_indices_blocked <- permute(experiment_data$value,blocks= experiment_data$experiment)

# let's apply these indices and see the new order
shuffled_blocked <- experiment_data$value[permutations_indices_blocked]
print(shuffled_blocked)
```

You can see that within each experiment group the data is shuffled but the experiment groupings are preserved
Also this is important if you just blindly use `permute` on a column within a data frame you’re going to lose all the row associations and that could lead to incorrect results in your analysis I've done this more times than I care to admit. You need to make sure the permutation is tied to the correct row

Speaking of data frames let’s talk about applying the permutations to the rest of data frame

You're not just shuffling one column usually right I learned the hard way that messing up the ordering of your rows when you’re trying to permute is super bad

```R
my_df <- data.frame(
  id = 1:5,
  data_a = sample(10:20,5),
  data_b = sample(30:40,5)
  )
# to apply a permution to the whole dataframe we keep the id's permuted
# then use that permutation in the data rows
permutation_id <- permute(my_df$id)
shuffled_df <- my_df[permutation_id, ]
print(shuffled_df)
```

See that we’re reordering the whole data frame not just one column at time this is crucial for many applications.

Now let's talk about why you might be seeing weird results even after all this let’s assume that you have your permutation indices correct and everything works fine You might be getting the feeling that the results are the same each time This usually happens because the random number generator is not set with a seed, which makes it non-deterministic. R’s random number generator is pseudo-random it means that the numbers seem random but in reality are generated from a mathematical algorithm and by setting the seed you make sure you generate the same sequence of random numbers each time which for a research is very important and allows replication
So always use `set.seed()` whenever you're doing anything that involves randomness especially when publishing code

Also you need to be sure that the number of permutations is large enough for your analysis A small number of permutations will result in a bad approximation of your null distribution If you are building a p-value or something you need a good permutation distribution for it You might need thousands or tens of thousands of permutations depending on the scale of the effect that you are measuring A small sample can lead to very unstable p-values

Oh and one funny story I got into once where I forgot about the `set.seed()` thing I was doing some random null model tests and every time I'd run my code the p-values were jumping around so much I thought I’d discovered some sort of fundamental law of quantum randomness Then I realised I just hadn't set the seed properly I should have remembered this rule I learned when I was just starting and now I always `set.seed()` before anything random It is a basic principle you think you will never forget but we are all humans

 resources you want resources not links and I agree with this So start with "All of Statistics" by Larry Wasserman it will give you the fundamental statistical concepts you need to really grasp what you are doing when creating permutation tests also it covers the importance of random sampling and data generation properly This is a very useful book but it requires some knowledge of mathematical and statistical notation "The Art of Statistics" by David Spiegelhalter is also a very good choice and provides all the fundamental concepts but it is less notation-heavy and more understandable for less technical people It is a perfect companion to "All of Statistics"
and if you need more specific details about resampling methods and Monte Carlo simulation I recommend you "An Introduction to Bootstrap Methods with Applications to R" by Davison and Hinkley It is an amazing book with plenty of code examples on R

So this is it You’ve been through my messy history with permutations and you’ve seen some working code Now I’m hoping that your permutation problem gets sorted out soon and you are on to the next research problem If you are still having issues please elaborate in the comments and I will happily try to help you further You can also attach reproducible code and data but before that please remember to set the seed good luck.
