---
title: "loss matrix in rs package rpart?"
date: "2024-12-13"
id: "loss-matrix-in-rs-package-rpart"
---

Okay so you're asking about the loss matrix in `rpart` from the R package right I get it this is something that trips up a lot of folks including my past self trust me I’ve been there done that and probably debugged more `rpart` trees than I’ve had hot meals ok maybe not but its up there

So `rpart` uses this thing called a loss matrix basically it tells the algorithm how to penalize misclassifications if you are doing classification that is If you just let it run with defaults it assumes all errors are equal which might be fine sometimes but what if misclassifying one class is way more costly than another say you are classifying spam and ham I mean it's pretty annoying when you miss spam but when you classify important business email as spam that's a nightmare or in medical diagnosis where a false negative is usually way worse than a false positive You can use a loss matrix to tell `rpart` that

Now before I jump into code I remember this one project I had way back in 2015 I was trying to build a classifier for customer churn prediction We had tons of data about demographics usage patterns etc and of course we had a big class imbalance like 90% non-churners and 10% churners and my first attempt with defaults resulted in a model that predicted almost everyone as non-churners because why not the accuracy was like 90% yay but it was totally useless because it missed all the churners We tried various techniques but the loss matrix made a big difference

So lets get to brass tacks what does the loss matrix look like It's a square matrix where rows represent the true class and columns represent the predicted class. The value at each cell (row, column) is the cost of classifying something as the column when it actually belongs to the row. Zero along the main diagonal and values greater than 0 elsewhere is what you are looking at

Here is how we can setup one

```R
loss_matrix <- matrix(c(0, 2, 1, 0), nrow = 2, byrow = TRUE)
rownames(loss_matrix) <- c("class1", "class2")
colnames(loss_matrix) <- c("class1", "class2")

print(loss_matrix)
```

This matrix says if the true class is `class1` and the prediction is `class2` then the cost is 2 If the true class is `class2` and predicted `class1` the cost is 1 all correct predictions are zero you can see that I'm pretty sure and those were the rules

Now lets use this in `rpart`

```R
library(rpart)

# Sample data
data <- data.frame(
  feature1 = rnorm(100),
  feature2 = rnorm(100),
  class = factor(sample(c("class1", "class2"), 100, replace = TRUE))
)


# Fit the rpart tree with the loss matrix
tree <- rpart(
  class ~ feature1 + feature2,
  data = data,
  method = "class",
  parms = list(loss = loss_matrix)
)

# Lets check what the tree looks like
print(tree)
```

Note the `parms = list(loss = loss_matrix)` this is where the magic happens. This tells `rpart` to use that specific loss matrix. You may wonder how `rpart` uses this well basically it tries to minimize the overall expected cost which is the sum of the costs for each classification multiplied by its probabilities.

Okay so here is the funny part one time I was so deep into `rpart` I started dreaming in decision trees. it was a nightmare. Ok no more jokes back to code

Now lets try another example with three classes. This is going to get a bit trickier but we will get through it

```R
library(rpart)

# Sample data with three classes
data <- data.frame(
  feature1 = rnorm(300),
  feature2 = rnorm(300),
  class = factor(sample(c("class1", "class2", "class3"), 300, replace = TRUE))
)

# Create a loss matrix with 3 classes
loss_matrix <- matrix(
  c(0, 1, 2,
    3, 0, 1,
    2, 2, 0),
  nrow = 3,
  byrow = TRUE
)

rownames(loss_matrix) <- c("class1", "class2", "class3")
colnames(loss_matrix) <- c("class1", "class2", "class3")

print(loss_matrix)

# Fit the tree
tree <- rpart(
  class ~ feature1 + feature2,
  data = data,
  method = "class",
  parms = list(loss = loss_matrix)
)

# Print to see what we have
print(tree)

```

This matrix implies that misclassifying class 1 as class 3 is the most expensive mistake and misclassifying class 2 as class 3 is less expensive than misclassifying class 1 as class 3. You can tailor the matrix to represent your use cases. So as you can see defining the loss matrix is very straightforward. you need to be careful about how you define the loss matrix since it depends heavily on your specific problem and needs.

If you are just starting with `rpart` it may help to plot your trees so that you have a better picture of how it actually works

```R
library(rpart.plot)

rpart.plot(tree, main = "Decision Tree with Loss Matrix")
```
This will help you visualize how your tree is working with your loss matrix parameters

Now when you use the `parms` argument `rpart` is also accepting several other parameters for `parms` I recommend looking at the `rpart` package documentation. Specifically look at the `control` parameter of `rpart` there are a bunch of parameters you can modify there including but not limited to `cp` or complexity parameter `minsplit` and `minbucket` there are many options there so play around

If you want to go deeper into theory of decision trees in general check out the book “The Elements of Statistical Learning” by Hastie Tibshirani and Friedman they really go deep into the mathematical background also you might find "Classification and Regression Trees" by Breiman et al its the original book on CART which is what `rpart` is based on and yes its a heavy read

Just a reminder the loss matrix is not going to magically solve all your classification problems You need to think about your features data quality feature engineering balancing your data and choosing the right parameters is key and I have learned that the hard way. I mean I had my fair share of misclassified data and I got really frustrated but eventually I got it

Finally remember when it comes to models its not all about accuracy accuracy is a very limited metric consider using F1 score precision and recall for imbalanced classes. The loss matrix plays into this since it allows you to optimize for specific misclassification costs. I mean you should experiment and test your results thoroughly cross validation is key and don't trust models blindly.

So yeah thats basically it when you are dealing with `rpart` and loss matrices think of it as a tool to fine-tune your algorithm’s decisions based on your cost structure. Now I’m going to go grab a coffee and probably debug some other code I hope this helps
