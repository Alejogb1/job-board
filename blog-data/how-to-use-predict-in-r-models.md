---
title: "how to use predict in r models?"
date: "2024-12-13"
id: "how-to-use-predict-in-r-models"
---

Okay so you wanna use `predict` in R models right Been there done that countless times let me tell you its not rocket science but it can be a bit fiddly if youre not careful I mean Ive been wrestling with R models since like R 2.15 days so you can trust me on this one

First things first `predict` is basically the universal translator for your trained model You feed it some data that looks like what the model was trained on and it spits out predictions So the most basic usage is incredibly simple

```r
#Lets say you have a model called my_model trained on data called train_data
#and you wanna predict on new data called new_data

predictions <- predict(my_model, newdata = new_data)

#now predictions contains the predicted values
```

See super easy right But the devil is in the details Lets break down some common scenarios Ive encountered and how I've tackled them

**Scenario 1: Different Column Names or Column Order**

Okay this is a classic one You train a model on a dataset with column names like `feature_1 feature_2 target` and then you try to predict on data with `input_1 input_2 result` Guess what R throws a fit Not the model's fault you just gave it the wrong recipe It expects the exact same column names and order as the training data

```r
#Let's say your train data has columns "height" "weight" "age"
train_data <- data.frame(height = c(170, 180, 165), weight = c(70, 85, 60), age = c(25, 30, 22))
my_model <- lm(height ~ weight + age, data=train_data)

#But your new data has "weight_kg" "age_years"

new_data_wrong <- data.frame(weight_kg = c(75, 90, 65), age_years = c(26, 31, 23))


#This will throw an error because the model is looking for "weight" and "age"
#predictions_wrong <- predict(my_model, newdata = new_data_wrong)

#The fix is to either rename columns or extract the columns in the same order as the model expects
new_data_fixed <- data.frame(weight = new_data_wrong$weight_kg, age = new_data_wrong$age_years)
predictions_fixed <- predict(my_model, newdata = new_data_fixed)

#Now predictions_fixed is what you want
```

The lesson here is: always double-check your column names and their order before using `predict` If you are like me and you are too lazy to remember the original column names from train data then use R’s model object to do that It stores all the info you need

**Scenario 2: Factors and Levels**

Factors man they are a source of endless frustration If your model uses categorical variables as factors it's vital that the new data has the exact same levels as the training data I've spent entire nights debugging issues related to this one

```r
#Let's say train data has a column "color" with levels "red" "blue"
train_data <- data.frame(color = factor(c("red", "blue", "red")), value = c(10, 20, 15))
my_model <- lm(value ~ color, data = train_data)

#New data with "green"
new_data_wrong <- data.frame(color = factor(c("green", "blue")))

#This will give you NA predictions because "green" is not in train data levels
#predictions_wrong <- predict(my_model, newdata = new_data_wrong)


#The fix is to make sure levels are consistent using factor() on new data with levels = levels(train_data$color)
new_data_fixed <- data.frame(color = factor(c("green", "blue"), levels = levels(train_data$color)))

#This now gives valid predictions
predictions_fixed <- predict(my_model, newdata = new_data_fixed)

#But here we still have an issue with NA predictions because we do not have the "green" category on our original train data

#To avoid this we have to re-factor the new data with both the original levels and the new ones:

new_data_all_levels <- data.frame(color = factor(c("green", "blue"), levels = c(levels(train_data$color), "green")))

#Now this should work
predictions_all_levels <- predict(my_model, newdata = new_data_all_levels)

```

So yeah make sure your new data has levels consistent with the original training set Otherwise `predict` throws up its hands

**Scenario 3: Missing Values**

Ah the bane of every data scientist’s existence Missing values (`NA`) can really mess with `predict` If your model was trained with missing data handled one way you can't just throw in a new dataset with missing values treated differently R will complain

```r
#Let's say your train data has some NAs and they were filled with mean for the model
train_data <- data.frame(feature1 = c(10, NA, 12), feature2 = c(20, 22, 24), target = c(1,2,3))
mean_feature1 <- mean(train_data$feature1, na.rm=TRUE)
train_data$feature1[is.na(train_data$feature1)] <- mean_feature1
my_model <- lm(target ~ feature1 + feature2, data=train_data)

#New data also has missing value
new_data_wrong <- data.frame(feature1= c(11,NA), feature2=c(21,23))

#This will give NA predictions on the NA input and probably a warning
#predictions_wrong <- predict(my_model, newdata=new_data_wrong)


#The fix is to treat new data with the same method we did the train data
#Impute missing values in new_data with the same mean as the train_data
new_data_fixed <- new_data_wrong

#Use the same mean used for training for the new data. VERY IMPORTANT.
new_data_fixed$feature1[is.na(new_data_fixed$feature1)] <- mean_feature1
predictions_fixed <- predict(my_model, newdata=new_data_fixed)
#Now prediction works fine
```

The point here is be consistent and make sure you have the same treatment for NAs if you have them in train and new data before running `predict` There was this time I was debugging for hours and it ended up that i forgot to impute NA values in my test set That was a humbling experience

**Important Model Specific Considerations**

Now every model type might have its own nuances This applies to decision trees random forest and neural networks for example and they require different treatment of missing values and even different way to interpret the results of a `predict` function

*   **Linear models (`lm`) and GLMs (`glm`):** These are pretty straightforward `predict` returns predicted values by default for regression models and predicted probabilities for classification models For GLMs remember the `type` argument It's what determines if you get probabilities or predictions on the original scale
*   **Tree-based models (`rpart` `randomForest`):** These can give predictions of either class probabilities or the predicted class depending on the model configuration and also your needs Also they may or may not be able to work with NA values and should be imputed beforehand if needed
*   **Support Vector Machines (`svm`):** SVMs can do both regression and classification so keep in mind what you are after when using `predict` For classification you need to decide if you want to get classification labels or predicted probabilities It is recommended to use predicted probabilities when possible to get a good idea of confidence and margins.
*   **Neural Networks (`nnet` `keras` `torch`):** Neural networks require a different treatment on `predict` functions and it is important to understand the framework used for the particular neural network. For example in `keras` it is often required to explicitly specify the output layer that we are using for prediction. The outputs are very often probabilities and require a separate step to get the classification labels.

**Resources**

Instead of throwing links at you try out these papers and books that are very helpful for understanding modeling and model predictions in R

*   "An Introduction to Statistical Learning" by Gareth James Daniela Witten Trevor Hastie and Robert Tibshirani this one is a must-have if you are into statistics and machine learning the explanations are top tier and you can learn about all of these models and their nuances.
*   "The R Book" by Michael J Crawley It is very helpful for learning R programming as a whole and how statistical methods are implemented in R.
*   "Applied Regression Modeling" by Iain Pardoe This one goes in deep into the world of regressions and it can be very helpful if you are using linear models or GLMs in R

**Final thoughts**

`predict` seems simple on the surface but its all about consistency and attention to detail Make sure that your training data structure and the new prediction data are as close as possible this includes not only column names and order but also factors and levels as well as missing values That is my 10 cents after suffering through all of these issues many times over. Hope this helps and you get to predict stuff smoothly from now on.

Oh and one last thing. Did you hear about the statistician who got lost in the woods? He found his way eventually he just kept calculating all the possible paths until one made sense. Okay I'm done now. Good luck and have fun with your R models.
