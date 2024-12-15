---
title: "How to do a Classification Task - Feature importance/selection in R (n-gram)?"
date: "2024-12-15"
id: "how-to-do-a-classification-task---feature-importanceselection-in-r-n-gram"
---

alright, so you're diving into classification with text data, specifically using n-grams, and you need to figure out which features matter most. been there, done that, got the t-shirt, and probably a few lingering headaches too. feature selection, especially with n-grams, can be a real pain if you don't approach it methodically. let me share what i’ve learned over the years, specifically focusing on how i’d tackle this in r, and how i’ve personally navigated similar situations in my previous work.

first off, let’s clarify n-grams. think of them as sequences of words. a unigram is a single word, bigrams are two words, trigrams are three, and so on. when working with text data for classification, using n-grams instead of individual words can capture more context and improve model performance, particularly when the sequence of words carries more meaning than the individual words themselves.

in my early days, i worked on a project trying to classify customer reviews – a classic example of this type of problem. i initially used just single words, but the performance was terrible, it couldn't differentiate between a happy customer and a very unhappy one. i was getting frustrated, and that’s where i realized the importance of n-grams and then the necessity of selection process. what good is using 100k features of n-grams without understanding their importance to the model? so i moved on to using bigrams and even trigrams, it improved the performance, but my feature space exploded and i was dealing with an unmanageable number of features and an overfitting model. that’s when i started learning more about feature importance and selection techniques. 

so, when you're dealing with potentially thousands of n-grams, you need a way to determine which ones are actually useful for classification and discard the rest. that's crucial for model performance and training time. feature importance helps you understand which features are contributing most to your model’s predictions, while feature selection helps you reduce your dataset's dimensionality. the main goal is to use the relevant n-grams only and remove the noise.

now, how do we do this in r? let's go step by step:

1. **text preprocessing and n-gram generation:**
before we even think about selection, we need to generate those n-grams. let’s assume you have a text dataset already, typically a vector of character strings representing your texts. the `tidytext` package is my go-to here. it plays very nicely with the `tidyverse`, making the whole process quite streamlined. it also makes it easy to convert from text to tidy format and do operations on a word-by-word basis.
```r
library(tidyverse)
library(tidytext)

# example text data
text_data <- tibble(
  id = 1:4,
  text = c("this is a very positive review",
           "this is a terrible negative review",
           "the product is fantastic",
           "i am really disappointed")
)

# function to generate n-grams
generate_ngrams <- function(data, n = 2) {
  data %>%
    unnest_tokens(ngram, text, token = "ngrams", n = n) %>%
    count(id, ngram, sort = TRUE) %>%
    ungroup()
}

# generate bigrams
bigram_counts <- generate_ngrams(text_data, n = 2)
print(bigram_counts)


# generate trigrams
trigram_counts <- generate_ngrams(text_data, n = 3)
print(trigram_counts)
```

what we are doing here is to tokenize the text data into n-grams and calculate how many times each n-gram appears in each document (id). in that process we also unnest the tokens so we can do this. this gives us a structure that’s easy to work with. as you can see above, you can easily change the value of n to generate different n-grams.

2. **feature importance:**
once we have the n-grams, we need to quantify their importance. one straightforward approach, if you're using tree based models like random forest or gradient boosting, is to use the feature importance directly provided by these models, which quantifies how much each n-gram has contributed to reducing the model's error during training. you can obtain that directly in r using the `caret` package, among others. also, packages like `xgboost` also provides feature importance natively if you are using it, and that is my preference, as `caret` can be a bit of a beast sometimes with different package versions. 

here is an example using `xgboost` assuming you have a binary classification problem. i prefer this model because it’s very robust and it can handle the high-dimensionality of n-gram data and also because it offers easy access to feature importance scores:
```r
library(xgboost)
library(Matrix) # for sparse matrix representation of n-grams

# assume text_data, bigram_counts and labels are already defined
# lets add some dummy labels (1 and 0 for positive and negative respectively)
text_data <- text_data %>%
  mutate(label = ifelse(grepl("positive|fantastic", text), 1, 0))

# first we need to reshape data to be a matrix representation
bigram_matrix <- bigram_counts %>%
  cast_sparse(id, ngram, n)

# get the labels for the rows
labels <- text_data$label[as.numeric(rownames(bigram_matrix))]

# prepare the xgboost matrix
dmatrix <- xgb.DMatrix(data = bigram_matrix, label = labels)

# train a model (parameters need tuning)
xgb_model <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "logloss"),
  data = dmatrix,
  nrounds = 10, # increase this in a real application
  verbose= 0
)

# get feature importance
importance_matrix <- xgb.importance(feature_names = colnames(bigram_matrix), model = xgb_model)

print(importance_matrix)
```
in this example, we are first preparing the data to be used in `xgboost` converting it to a sparse matrix using `cast_sparse`. then, we create a xgb.dmatrix object and we train the xgboost model. after that, we extract the feature importance matrix. this shows the importance score of each n-gram in the model. 

the `feature_names` argument is crucial so that you can understand which are the n-grams and the numbers associated with them. the `nrounds` parameter indicates the number of boosting iterations and should be increased in a real project to get a good model, i just used 10 here because it is an example.

3. **feature selection:**

once we have the feature importances, we can use it to select the most important n-grams. it will vary from project to project and the method that suits your needs the best will be up to you. the most basic is to select by importance thresholds, i.e. only keeping n-grams with importance higher than a particular value. or, you can pick the top n most important features. alternatively, more sophisticated techniques include recursive feature elimination (rfe) or using regularized models (like lasso or ridge regression) that automatically reduce the influence of unimportant features. i tend to prefer the simpler approaches, specially when starting, then if the performance is not up to what i’m expecting, i tend to start trying more advanced strategies.

here’s how you can select the top n features based on the feature importance we got previously with xgboost:

```r
# select top n features
select_top_n_features <- function(importance_matrix, n) {
  importance_matrix %>%
    as_tibble() %>%
    arrange(desc(Gain)) %>%
    head(n) %>%
    pull(Feature)
}

# lets select the top 5 features
top_features <- select_top_n_features(importance_matrix, 5)

print(top_features)
```
here we are using the tibble created from the importance matrix and then using the `arrange()` and `head()` functions to select the top n features (in this case, 5), and then getting the names of the features. now you can use only those n-grams that you selected to retrain a classification model. the main point of doing this is to reduce the dimensionality of your data, the model’s complexity and usually also improves the generalisation. 

**resources**

when i started with text mining, a book that really made a difference for me was "text mining with r: a tidy approach" by julia silge and david robinson. this book presents you with practical examples using tidyverse and tidytext approach. also, for more theoretical stuff, i would recommend you checking "the elements of statistical learning" by hastie, tibshirani, and friedman. there is a free pdf of this book on stanford website. this one is quite dense, but is a classic if you want to go deeper in statistical modelling and machine learning.

to add, i also tend to check blogs and stackoverflow for code snippets on specific problems. the official documentation of the r packages are usually very good and useful. 

remember, the specific steps and parameters will vary depending on the specifics of your dataset and the classification task itself. it requires a degree of experimentation, and there’s no single magic number. the n-gram size, the type of model, how many top features to select, the algorithm used to extract those, all of that needs to be adapted to your data. you can get lost in a rabbit hole of different approaches. there are a lot of options and sometimes it seems that all is the same. it’s not, they work differently in the different datasets, and usually requires some experimentation. don't be afraid to try different things and see what works best for your data and problem at hand.

i've always found that sometimes it’s best to start simple, validate performance and then if needed try more sophisticated techniques. i mean, why would you try something very complex if you can achieve similar results with something simple? sometimes less is more, you know?

oh, and one more thing, never ever trust a computer. i mean, they are just a dumb calculator. they only do what you tell them to, it’s your job to make sure you are telling it to do the correct thing. and debugging is always 90% of a machine learning engineer job.
