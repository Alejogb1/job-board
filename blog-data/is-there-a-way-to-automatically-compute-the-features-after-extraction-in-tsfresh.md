---
title: "Is there a way to automatically compute the features after extraction in tsfresh?"
date: "2024-12-15"
id: "is-there-a-way-to-automatically-compute-the-features-after-extraction-in-tsfresh"
---

here's my take on it.

yeah, i've been down this road with tsfresh before. it's a powerful tool, no doubt, but sometimes you hit a wall where you’re just buried under a mountain of extracted features. the question, as i see it, is how to automate the process of figuring out which of these features are actually *useful* after tsfresh has done its thing. it's not enough to just extract everything and hope for the best, trust me, i've learned that the hard way. 

so, first things first, we need to talk about why this is a problem in the first place. tsfresh, in its default mode, generates a ridiculously large number of features. depending on your data and the settings you use, we're talking hundreds or even thousands. most of these are going to be redundant, irrelevant, or just plain noisy. training a model on all of them is a recipe for disaster: overfitting, long training times, and poor generalization. been there, done that, got the t-shirt that says "i regret not doing feature selection".

i remember this one project, a few years back, working on some predictive maintenance stuff with sensor data from industrial machinery. we had time series data coming out of our ears, and i thought, "hey, tsfresh, let's do this." we were dealing with vibration, temperature, current draw, everything. we cranked it up, and it spat out over 2000 features. our model was performing like a toddler trying to build a skyscraper out of jello; it was all over the place, no reliable patterns, just random noise.

that's when i realized we had to get serious about feature selection. manually picking features is out of the question, it's just not feasible with that scale. plus, it introduces all sorts of bias. so, what can we do?

the key here is to treat this as a machine learning problem itself. we’ve got a bunch of features and a target variable, just like any other modeling task. we can use a variety of feature selection techniques that are built into most ml libraries.

one of the easiest ways, and one that i like using as a first step, is something called variance thresholding. the logic is simple: if a feature has very little variation across samples, it's probably not going to be useful for differentiating between them. it's essentially constant across the dataset.

here's how you can do it with scikit-learn:

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selection(dataframe, threshold=0.05):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(dataframe)
    selected_features = dataframe.columns[selector.get_support()]
    return dataframe[selected_features]

# example usage
# assuming you have features in a pandas dataframe called `extracted_features`
# and we set a threshold of 0.05
filtered_features = variance_threshold_selection(extracted_features, threshold=0.05)
print(f"number of features after filtering: {filtered_features.shape[1]}")
```
this function filters a data frame columns based on the variance threshold provided as parameter returning the filtered data frame.

this is a good starting point, but it only considers the features individually. what about relationships between features and their impact on the target variable? that's where things get more interesting.

another useful technique is using statistical methods like correlation analysis. you can calculate the correlation between each feature and your target variable. features with a very low correlation are likely not contributing much and can be discarded. it’s important here to not just remove all features but consider also the negative correlations. this requires to implement a absolute correlation value filter. i am sure you know that it's important to consider multicollinearity between the features in relation with the target variables also.

here's an example of how to do that:

```python
import pandas as pd

def correlation_selection(dataframe, target, correlation_threshold=0.1):
    correlations = dataframe.corrwith(target)
    selected_features = correlations[abs(correlations) >= correlation_threshold].index
    return dataframe[selected_features]

# example of use
# assuming you have features in a pandas dataframe called `extracted_features`
# and the target variable is in a pandas series called `target_variable`
filtered_features = correlation_selection(extracted_features, target_variable, correlation_threshold=0.1)
print(f"number of features after filtering: {filtered_features.shape[1]}")
```
this function filters a data frame columns based on a correlation value with the target column provided as parameter returning the filtered data frame.

now, a lot of the time, the best feature selection approach is iterative, and it involves using some form of model-based approach like recursive feature elimination (rfe). the idea here is to use a model, like a linear regression or a tree-based model, and repeatedly remove features that contribute the least to performance. it’s an approach that can really get you to the features that are more impactful for the model task, but it is generally more computationally expensive.

and here’s one final example of how to do that with a random forest:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def rfe_selection(dataframe, target, n_features_to_select=10):
    model = RandomForestClassifier(random_state=42)
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select)
    selector.fit(dataframe, target)
    selected_features = dataframe.columns[selector.support_]
    return dataframe[selected_features]
    
# example of use
# assuming you have features in a pandas dataframe called `extracted_features`
# and the target variable is in a pandas series called `target_variable`
filtered_features = rfe_selection(extracted_features, target_variable, n_features_to_select=10)
print(f"number of features after filtering: {filtered_features.shape[1]}")
```
this function filters a data frame columns based on the results of an RFE filter selecting the most important features based on the number of features to select provided as parameter returning the filtered data frame.

one last important point: feature selection isn't a one-time thing. as you collect more data or as your problem changes, you might need to re-evaluate your selected features, recompute them and re train your model. it's something to keep in mind as the model evolves.

also, remember that these methods are not mutually exclusive. you can combine them. you can start with variance thresholding to eliminate the obvious ones, move on to correlation analysis and later use something like rfe for a more precise selection.

for more in-depth understanding of feature selection techniques, i would suggest taking a look at the classic "the elements of statistical learning" by hastie, tibshirani, and friedman. it covers feature selection in-depth, as well as many others of the most common machine learning techniques. there is also an amazing book called "feature engineering for machine learning" by alice zheng and amanda casari. it's a hands-on guide that will guide you through many other ways of handling data and feature selection.

one more thing, before i go, keep in mind that this whole process can be a bit of a black art. the best settings, the best approach depends on the particular data set, the context, the model you plan to use and some luck. sometimes it is even more difficult to select the right features when you have multi class problems.

there is a very important issue here though; the number of features tsfresh extract by default makes this process hard for a beginner. i once saw a colleague who had generated over 5k features. when i asked why she said, i thought it would be better to have more data... that day i learned a new level of sadness, i swear, some people just never stop finding new ways to surprise me with the most obscure ideas.

hope this helps.
